internal import XCTest
@testable import AccelLab

final class LZ4BlockDecoderTests: XCTestCase {
    func testAppleEncodedRandomAndRepetitiveBlocks() throws {
        for size in [1, 14, 15, 255, 1_024, 65_536] {
            for input in [LZ4TestSupport.corpus(count: size), [UInt8](repeating: 42, count: size)] {
                let compressed = try LZ4TestSupport.compress(input)
                try assertDecoders(compressed, expected: input)
            }
        }
    }

    func testOverlappingMatches() throws {
        for offset in [1, 2, 3, 7, 31, 32, 33, 65_535] {
            let sample = LZ4TestSupport.overlap(offset: offset, matchLength: 1_029)
            try assertDecoders(sample.compressed, expected: sample.decoded)
        }
    }

    func testTruncatedInputAndInvalidOffsetsThrow() throws {
        let cases: [([UInt8], LZ4Error)] = [
            ([], .truncatedInput), ([0xF0], .truncatedInput), ([0xF0, 255], .truncatedInput),
            ([0x30, 1, 2], .truncatedInput), ([0x10, 65, 1], .truncatedInput),
            ([0x10, 65, 0, 0, 0], .invalidOffset), ([0x10, 65, 2, 0, 0], .invalidOffset),
            ([0x1F, 65, 1, 0], .truncatedInput), ([0x1F, 65, 1, 0, 255], .truncatedInput),
            ([0x10, 65, 1, 0], .truncatedInput),
        ]
        for (input, expected) in cases {
            var output = [UInt8](repeating: 0, count: 1_024)
            XCTAssertThrowsError(try input.withUnsafeBytes { source in
                try output.withUnsafeMutableBytes { try LZ4BlockDecoder.decodeSwift(source, into: $0) }
            }) { XCTAssertEqual($0 as? LZ4Error, expected, "\(input)") }
            XCTAssertThrowsError(try input.withUnsafeBytes {
                try LZ4BlockDecoder.decodedLength($0, limit: 1_024)
            }) { XCTAssertEqual($0 as? LZ4Error, expected, "\(input)") }
        }
    }

    func testOutputBoundsAndEmptyBlock() throws {
        for input in [[UInt8(0x20), 1, 2], [0x10, 65, 1, 0, 0x00], [0x1F, 65, 1, 0, 255, 0, 0]] {
            var output = [UInt8](repeating: 0xA5, count: 3)
            XCTAssertThrowsError(try input.withUnsafeBytes { source in
                try output.withUnsafeMutableBytes { destination in
                    try LZ4BlockDecoder.decodeSwift(source, into: UnsafeMutableRawBufferPointer(rebasing: destination[1..<2]))
                }
            }) { XCTAssertEqual($0 as? LZ4Error, .outputTooSmall) }
            XCTAssertEqual(output.first, 0xA5)
            XCTAssertEqual(output.last, 0xA5)
        }
        let empty: [UInt8] = [0]
        XCTAssertEqual(try empty.withUnsafeBytes {
            try LZ4BlockDecoder.decodeSwift($0, into: UnsafeMutableRawBufferPointer(start: nil, count: 0))
        }, 0)
    }

    private func assertDecoders(_ compressed: [UInt8], expected: [UInt8]) throws {
        var swift = [UInt8](repeating: 0xA5, count: expected.count + 1)
        var apple = swift
        let swiftCount = try compressed.withUnsafeBytes { source in
            try swift.withUnsafeMutableBytes { try LZ4BlockDecoder.decodeSwift(source, into: $0) }
        }
        let appleCount = compressed.withUnsafeBytes { source in
            apple.withUnsafeMutableBytes { LZ4BlockDecoder.decodeApple(source, into: $0) }
        }
        XCTAssertEqual(swiftCount, expected.count)
        XCTAssertEqual(appleCount, expected.count)
        XCTAssertEqual(Array(swift.prefix(swiftCount)), expected)
        XCTAssertEqual(Array(apple.prefix(appleCount)), expected)
        XCTAssertEqual(swift.last, 0xA5)
        XCTAssertEqual(try compressed.withUnsafeBytes {
            try LZ4BlockDecoder.decodedLength($0, limit: expected.count)
        }, expected.count)
    }
}
