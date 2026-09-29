internal import Foundation
internal import XCTest
@testable import AccelLab

final class LZ4FrameDecoderTests: XCTestCase {
    func testPreallocatedStaticAndDynamicDecodingReusesDestination() throws {
        let input = LZ4TestSupport.corpus(count: 200_003)
        let source = try LZ4Frame.encodeFrame(input, blockSize: 16_384)
        let frame = try LZ4Frame(source)
        let sizes = try frame.expectedDecodedSizes()
        let plan = try LZ4FrameDecoder.Plan(frame: frame, source: source, sizes: sizes)
        var output = [UInt8](repeating: 0xA5, count: input.count + 2)
        try output.withUnsafeMutableBytes { bytes in
            let destination = UnsafeMutableRawBufferPointer(rebasing: bytes[1..<(1 + input.count)])
            for scheduling in [LZ4FrameDecoder.Scheduling.staticPartition, .dynamic] {
                for algorithm in [LZ4FrameDecoder.Algorithm.apple, .swift] {
                    let seconds = try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, sizes: sizes,
                        into: destination, lanes: 16, algorithm: algorithm, scheduling: scheduling)
                    XCTAssertGreaterThan(seconds, 0)
                    XCTAssertTrue(destination.elementsEqual(input))
                    destination.initializeMemory(as: UInt8.self, repeating: 0)
                    try LZ4FrameDecoder.decodeCPU(plan: plan, into: destination, lanes: 16,
                                                 algorithm: algorithm, scheduling: scheduling)
                    XCTAssertTrue(destination.elementsEqual(input))
                }
            }
        }
        XCTAssertEqual(output.first, 0xA5)
        XCTAssertEqual(output.last, 0xA5)
    }

    func testPrecomputedSizeAndDestinationValidation() throws {
        let input = [UInt8](repeating: 42, count: 4_096)
        let source = try LZ4Frame.encodeFrame(input, blockSize: 4_096)
        let frame = try LZ4Frame(source)
        var destination = [UInt8](repeating: 0, count: input.count)
        for sizes in [[], [-1], [Int.max], [input.count - 1]] {
            XCTAssertThrowsError(try destination.withUnsafeMutableBytes {
                try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, sizes: sizes, into: $0, lanes: 16)
            })
        }
        XCTAssertThrowsError(try destination.withUnsafeMutableBytes {
            try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, sizes: [input.count],
                into: UnsafeMutableRawBufferPointer(rebasing: $0[..<(input.count - 1)]), lanes: 1)
        }) { XCTAssertEqual($0 as? LZ4Error, .outputTooSmall) }
        XCTAssertThrowsError(try destination.withUnsafeMutableBytes {
            try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, sizes: [input.count], into: $0, lanes: 0)
        })
    }

    func testThreeMiBFrameWithOneAndSixteenLanes() throws {
        let input = LZ4TestSupport.corpus()
        let source = try LZ4Frame.encodeFrame(input, blockSize: 65_536)
        let frame = try LZ4Frame(source)
        XCTAssertEqual(frame.contentSize, UInt64(input.count))
        XCTAssertEqual(frame.blocks.count, 48)
        XCTAssertTrue(frame.blocks.contains { $0.isStored })
        XCTAssertTrue(frame.blocks.contains { !$0.isStored })
        XCTAssertEqual(try frame.expectedDecodedSizes(), [Int](repeating: frame.blockMaxSize, count: 48))
        for lanes in [1, 16] {
            XCTAssertEqual(try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, lanes: lanes), input)
        }
        XCTAssertEqual(try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, lanes: 1, algorithm: .swift), input)
    }

    func testAllBlockSizesAndPartialFinalBlock() throws {
        for blockSize in LZ4Frame.blockSizes {
            let input = [UInt8](repeating: 42, count: blockSize + 17)
            let source = try LZ4Frame.encodeFrame(input, blockSize: blockSize)
            let frame = try LZ4Frame(source)
            XCTAssertEqual(try frame.expectedDecodedSizes(), [blockSize, 17])
            XCTAssertEqual(try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, lanes: 16), input)
        }
    }

    func testEmptyFramesAndZeroLengthBlocks() throws {
        for bytes in [Array(try LZ4Frame.encodeFrame([], blockSize: 65_536)),
                      LZ4TestSupport.frame(blocks: [([0], false), ([], true)], contentSize: 0)] {
            let source = Data(bytes)
            let frame = try LZ4Frame(source)
            for algorithm in [LZ4FrameDecoder.Algorithm.apple, .swift] {
                XCTAssertEqual(try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, lanes: 16, algorithm: algorithm), [])
            }
        }
    }

    func testInvalidArgumentsAndUnsupportedFrames() throws {
        let source = Data(LZ4TestSupport.frame(blocks: [([0x10, 65], false)], contentSize: 1))
        let frame = try LZ4Frame(source)
        for lanes in [0, -1] {
            XCTAssertThrowsError(try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, lanes: lanes))
        }
        var changed = Array(source)
        changed[changed.count - 5] ^= 1
        XCTAssertThrowsError(try LZ4FrameDecoder.decodeCPU(frame: frame, source: Data(changed), lanes: 1)) {
            XCTAssertEqual($0 as? LZ4Error, .sourceMismatch)
        }
        for (bytes, error) in [
            (LZ4TestSupport.frame(blocks: [], independent: false), LZ4Error.unsupportedDependency),
            (LZ4TestSupport.frame(blocks: [], dictionaryID: 1), .unsupportedDictionary),
            (LZ4TestSupport.frame(blocks: [([0x10, 65, 2, 0, 0], false)]), .invalidOffset),
            (LZ4TestSupport.frame(blocks: [], contentSize: 1), .contentSizeMismatch),
        ] {
            let data = Data(bytes)
            let parsed = try LZ4Frame(data)
            XCTAssertThrowsError(try LZ4FrameDecoder.decodeCPU(frame: parsed, source: data, lanes: 16)) {
                XCTAssertEqual($0 as? LZ4Error, error)
            }
        }
    }
}
