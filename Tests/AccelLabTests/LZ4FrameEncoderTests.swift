internal import Foundation
internal import XCTest
@testable import AccelLab

final class LZ4FrameEncoderTests: XCTestCase {
    func testArbitraryBlockSizesChooseSmallestDescriptor() throws {
        for blockSize in [4_096, 16_384, 65_535, 65_536, 65_537, 262_144, 262_145, 1_048_577, 4_194_304] {
            let input = [UInt8](repeating: 65, count: blockSize + 17)
            let source = try LZ4Frame.encodeFrame(input, blockSize: blockSize)
            let frame = try LZ4Frame(source)
            XCTAssertTrue(frame.isIndependent)
            XCTAssertEqual(frame.contentSize, UInt64(input.count))
            XCTAssertEqual(frame.blockMaxSize, LZ4Frame.blockSizes.first { $0 >= blockSize })
            XCTAssertEqual(try frame.expectedDecodedSizes(), [blockSize, 17])
            XCTAssertFalse(frame.hasBlockChecksum)
            XCTAssertFalse(frame.hasContentChecksum)
            XCTAssertEqual(try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, lanes: 16), input)
        }
    }

    func testIncompressibleBlocksAreStoredAndSmallTailIsPreserved() throws {
        let input = LZ4TestSupport.corpus(count: 32_769)
        let source = try LZ4Frame.encodeFrame(input, blockSize: 16_384)
        let frame = try LZ4Frame(source)
        XCTAssertEqual(frame.blocks.map(\.isStored), [true, true, true])
        XCTAssertEqual(try frame.expectedDecodedSizes(), [16_384, 16_384, 1])
        XCTAssertEqual(try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, lanes: 1), input)
    }

    func testEmptyHeaderChecksumAndBlockSizeRange() throws {
        let source = try LZ4Frame.encodeFrame([], blockSize: 4_096)
        let frame = try LZ4Frame(source)
        XCTAssertEqual(source.count, 19)
        XCTAssertEqual(frame.headerChecksum, 0x05)
        XCTAssertEqual(frame.contentSize, 0)
        XCTAssertTrue(frame.blocks.isEmpty)
        for blockSize in [Int.min, 0, 4_095, 4_194_305, Int.max] {
            XCTAssertThrowsError(try LZ4Frame.encodeFrame([], blockSize: blockSize)) {
                XCTAssertEqual($0 as? LZ4Error, .invalidBlockSize)
            }
        }
    }

    func testFileEncoderReportsSizesAndBlockCount() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let inputURL = directory.appendingPathComponent("input.txt")
        let outputURL = directory.appendingPathComponent("output.lz4")
        let input = Data(LZ4TestSupport.corpus(count: 32_769))
        try input.write(to: inputURL)
        let result = try LZ4FrameEncoder.encodeFile(inputPath: inputURL.path, outputPath: outputURL.path, blockSize: 16_384)
        let output = try Data(contentsOf: outputURL)
        XCTAssertEqual(result.inputBytes, input.count)
        XCTAssertEqual(result.outputBytes, output.count)
        XCTAssertEqual(result.blockCount, 3)
        XCTAssertEqual(try LZ4FrameDecoder.decodeCPU(frame: LZ4Frame(output), source: output, lanes: 1), Array(input))
    }
}
