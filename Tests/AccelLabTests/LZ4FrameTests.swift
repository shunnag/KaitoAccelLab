internal import Foundation
internal import XCTest
@testable import AccelLab

final class LZ4FrameTests: XCTestCase {
    func testParallelAndSerialSizesAgreeOnMultipleBlocks() throws {
        let input = LZ4TestSupport.corpus(count: LZ4TestSupport.corpusSize + 17)
        let source = try LZ4Frame.encodeFrame(input, blockSize: 4_096)
        let frame = try LZ4Frame(source)
        let expected = [Int](repeating: 4_096, count: LZ4TestSupport.corpusSize / 4_096) + [17]
        XCTAssertTrue(frame.blocks.contains { $0.isStored })
        XCTAssertTrue(frame.blocks.contains { !$0.isStored })
        XCTAssertEqual(try frame.expectedDecodedSizesSerial(), expected)
        XCTAssertEqual(try frame.expectedDecodedSizes(), expected)
        for count in [0, 1, 15, 16, 17] {
            let small = try LZ4Frame(LZ4TestSupport.frame(blocks: Array(repeating: ([0x10, 65], false), count: count)))
            XCTAssertEqual(try small.expectedDecodedSizes(), try small.expectedDecodedSizesSerial())
        }
    }

    func testParallelScanPreservesSerialErrorOrderAndSizeValidation() throws {
        var blocks: [(bytes: [UInt8], stored: Bool)] = Array(repeating: ([0x10, 65], false), count: 33)
        blocks[2] = ([0x10, 65, 2, 0, 0], false)
        blocks[19] = ([0xF0], false)
        let corrupt = try LZ4Frame(LZ4TestSupport.frame(blocks: blocks))
        XCTAssertThrowsError(try corrupt.expectedDecodedSizesSerial()) { XCTAssertEqual($0 as? LZ4Error, .invalidOffset) }
        XCTAssertThrowsError(try corrupt.expectedDecodedSizes()) { XCTAssertEqual($0 as? LZ4Error, .invalidOffset) }
        let mismatch = try LZ4Frame(LZ4TestSupport.frame(blocks: [([0x10, 65], false)], contentSize: 2))
        XCTAssertThrowsError(try mismatch.expectedDecodedSizesSerial()) { XCTAssertEqual($0 as? LZ4Error, .contentSizeMismatch) }
        XCTAssertThrowsError(try mismatch.expectedDecodedSizes()) { XCTAssertEqual($0 as? LZ4Error, .contentSizeMismatch) }
    }

    func testUniformSizesRoundUpAndUseFinalRemainder() throws {
        let source = Data(LZ4TestSupport.frame(blocks: [([1, 2, 3], true), ([4, 5, 6], true), ([7, 8], true)], contentSize: 8))
        let frame = try LZ4Frame(source)
        let sizes = try frame.assumedUniformDecodedSizes()
        XCTAssertEqual(sizes, [3, 3, 2])
        var output = [UInt8](repeating: 0, count: 8)
        _ = try output.withUnsafeMutableBytes {
            try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, sizes: sizes, into: $0, lanes: 16)
        }
        XCTAssertEqual(output, [1, 2, 3, 4, 5, 6, 7, 8])
        let empty = try LZ4Frame(LZ4TestSupport.frame(blocks: [], contentSize: 0))
        XCTAssertEqual(try empty.assumedUniformDecodedSizes(), [])
    }

    func testUniformSizesRequireContentSizeAndRejectImpossibleLayouts() throws {
        let missing = try LZ4Frame(LZ4TestSupport.frame(blocks: [([0], false)]))
        XCTAssertThrowsError(try missing.assumedUniformDecodedSizes()) {
            XCTAssertEqual($0 as? LZ4Error, .invalidArgument("--assume-uniform requires contentSize"))
        }
        let negativeTail = try LZ4Frame(LZ4TestSupport.frame(blocks: Array(repeating: ([0], false), count: 3), contentSize: 1))
        XCTAssertThrowsError(try negativeTail.assumedUniformDecodedSizes()) {
            XCTAssertEqual($0 as? LZ4Error, .contentSizeMismatch)
        }
        let overflow = try LZ4Frame(LZ4TestSupport.frame(blocks: [([0], false)], contentSize: UInt64.max))
        XCTAssertThrowsError(try overflow.assumedUniformDecodedSizes()) { XCTAssertEqual($0 as? LZ4Error, .invalidBlockSize) }
        let empty = try LZ4Frame(LZ4TestSupport.frame(blocks: [], contentSize: 1))
        XCTAssertThrowsError(try empty.assumedUniformDecodedSizes()) { XCTAssertEqual($0 as? LZ4Error, .contentSizeMismatch) }
    }

    func testUniformAssumptionDoesNotScanCompressedSequences() throws {
        let corrupt = try LZ4Frame(LZ4TestSupport.frame(blocks: [([0x10, 65, 2, 0, 0], false)], contentSize: 4))
        XCTAssertEqual(try corrupt.assumedUniformDecodedSizes(), [4])
        XCTAssertThrowsError(try corrupt.expectedDecodedSizesSerial()) { XCTAssertEqual($0 as? LZ4Error, .invalidOffset) }
        XCTAssertThrowsError(try corrupt.expectedDecodedSizes()) { XCTAssertEqual($0 as? LZ4Error, .invalidOffset) }
    }

    func testWrongUniformAssumptionFailsCPUVerification() throws {
        let large = try LZ4TestSupport.compress([UInt8](repeating: 65, count: 32))
        let small = try LZ4TestSupport.compress([UInt8](repeating: 66, count: 2))
        let source = Data(LZ4TestSupport.frame(blocks: [(large, false), (small, false)], contentSize: 34))
        let frame = try LZ4Frame(source)
        let sizes = try frame.assumedUniformDecodedSizes()
        var output = [UInt8](repeating: 0, count: 34)
        XCTAssertThrowsError(try output.withUnsafeMutableBytes {
            try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, sizes: sizes, into: $0, lanes: 16)
        }) {
            guard let error = $0 as? LZ4Error, case .appleDecodeFailed = error else {
                return XCTFail("Unexpected error: \($0)")
            }
        }
    }

    func testBlockTableWithOptionalSizesAndChecksums() throws {
        let full = LZ4TestSupport.overlap(offset: 1, matchLength: 65_530)
        let stored: [UInt8] = [9, 8, 7]
        for hasSize in [false, true] {
            for checksums in [false, true] {
                let bytes = LZ4TestSupport.frame(
                    blocks: [(full.compressed, false), (full.compressed, false), (stored, true)],
                    contentSize: hasSize ? 131_075 : nil, blockChecksums: checksums, contentChecksum: checksums)
                let frame = try LZ4Frame(bytes)
                XCTAssertEqual(frame.blockMaxSize, 65_536)
                XCTAssertEqual(frame.contentSize, hasSize ? 131_075 : nil)
                XCTAssertTrue(frame.isIndependent)
                XCTAssertEqual(frame.headerChecksum, 0xA5)
                XCTAssertEqual(frame.hasBlockChecksum, checksums)
                XCTAssertEqual(frame.hasContentChecksum, checksums)
                XCTAssertEqual(frame.blocks.count, 3)
                XCTAssertEqual(frame.blocks.map(\.isStored), [false, false, true])
                let sizes = try frame.expectedDecodedSizes()
                XCTAssertEqual(sizes, [65_536, 65_536, 3])
                XCTAssertTrue(sizes.dropLast().allSatisfy { $0 == frame.blockMaxSize })
                let firstOffset = (hasSize ? 15 : 7) + 4
                XCTAssertEqual(frame.blocks[0].compressedOffset, firstOffset)
                XCTAssertEqual(frame.blocks[1].compressedOffset, firstOffset + full.compressed.count + 4 + (checksums ? 4 : 0))
                for (index, payload) in [full.compressed, full.compressed, stored].enumerated() {
                    let block = frame.blocks[index]
                    XCTAssertEqual(block.compressedLength, payload.count)
                    XCTAssertEqual(Array(bytes[block.compressedOffset..<(block.compressedOffset + block.compressedLength)]), payload)
                    XCTAssertEqual(try frame.decodedLength(of: block), sizes[index])
                }
                XCTAssertEqual(try LZ4FrameDecoder.decodeCPU(frame: frame, source: Data(bytes), lanes: 16),
                               full.decoded + full.decoded + stored)
            }
        }
    }

    func testAllBlockSizeCodesAndDictionaryParsing() throws {
        for code in UInt8(4)...7 {
            let frame = try LZ4Frame(LZ4TestSupport.frame(blocks: [], sizeCode: code))
            XCTAssertEqual(frame.blockMaxSize, LZ4Frame.blockSizes[Int(code) - 4])
        }
        let dependent = try LZ4Frame(LZ4TestSupport.frame(blocks: [], independent: false))
        XCTAssertFalse(dependent.isIndependent)
        XCTAssertThrowsError(try dependent.expectedDecodedSizes()) {
            XCTAssertEqual($0 as? LZ4Error, .unsupportedDependency)
        }
        let dictionary = try LZ4Frame(LZ4TestSupport.frame(blocks: [], contentSize: 0, dictionaryID: 0x1234_5678))
        XCTAssertEqual(dictionary.dictionaryID, 0x1234_5678)
        XCTAssertEqual(dictionary.contentSize, 0)
        XCTAssertThrowsError(try dictionary.expectedDecodedSizes()) {
            XCTAssertEqual($0 as? LZ4Error, .unsupportedDictionary)
        }
    }

    func testMalformedHeadersAndBlockSizes() throws {
        let valid = LZ4TestSupport.frame(blocks: [([0x10, 65], false)])
        for (offset, value, expected) in [(0, UInt8(0), LZ4Error.invalidMagic), (4, 0x20, .invalidVersion),
                                           (4, 0x62, .reservedBits), (5, 0x41, .reservedBits),
                                           (5, 0xC0, .reservedBits), (5, 0x30, .invalidBlockSize),
                                           (5, 0, .invalidBlockSize)] {
            var bytes = valid
            bytes[offset] = value
            XCTAssertThrowsError(try LZ4Frame(bytes)) { XCTAssertEqual($0 as? LZ4Error, expected) }
        }
        var tooLarge = valid
        tooLarge.replaceSubrange(7..<11, with: [1, 0, 1, 0])
        XCTAssertThrowsError(try LZ4Frame(tooLarge)) { XCTAssertEqual($0 as? LZ4Error, .invalidBlockSize) }
        XCTAssertThrowsError(try LZ4Frame(valid + [0])) { XCTAssertEqual($0 as? LZ4Error, .trailingData) }
    }

    func testEveryTruncatedPrefixThrows() throws {
        let bytes = LZ4TestSupport.frame(blocks: [([0x10, 65], false), ([1, 2, 3], true)],
                                        contentSize: 4, blockChecksums: true, contentChecksum: true, dictionaryID: 12)
        for length in 0..<bytes.count {
            XCTAssertThrowsError(try LZ4Frame(Array(bytes.prefix(length)))) {
                XCTAssertEqual($0 as? LZ4Error, .truncatedInput, "length=\(length)")
            }
        }
    }

    func testDecodedSizeValidation() throws {
        let mismatch = try LZ4Frame(LZ4TestSupport.frame(blocks: [([0x10, 65], false)], contentSize: UInt64.max))
        XCTAssertThrowsError(try mismatch.expectedDecodedSizes()) { XCTAssertEqual($0 as? LZ4Error, .contentSizeMismatch) }
        let corrupt = try LZ4Frame(LZ4TestSupport.frame(blocks: [([0x10, 65, 2, 0, 0], false)]))
        XCTAssertThrowsError(try corrupt.expectedDecodedSizes()) { XCTAssertEqual($0 as? LZ4Error, .invalidOffset) }
        let oversized = LZ4TestSupport.overlap(offset: 1, matchLength: 65_531)
        let frame = try LZ4Frame(LZ4TestSupport.frame(blocks: [(oversized.compressed, false)]))
        XCTAssertThrowsError(try frame.expectedDecodedSizes()) { XCTAssertEqual($0 as? LZ4Error, .outputTooSmall) }
    }
}
