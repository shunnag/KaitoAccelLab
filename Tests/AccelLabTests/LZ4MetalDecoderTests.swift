internal import Foundation
internal import Metal
internal import XCTest
@testable import AccelLab

final class LZ4MetalDecoderTests: XCTestCase {
    func testThreeMiBFrameBothVariants() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal device unavailable") }
        let input = LZ4TestSupport.corpus()
        let source = try LZ4Frame.encodeFrame(input, blockSize: 65_536)
        let frame = try LZ4Frame(source)
        let expected = try LZ4FrameDecoder.decodeCPU(frame: frame, source: source, lanes: 16)
        let decoder = try LZ4MetalDecoder(device: device)
        try decoder.prepare(frame: frame, source: source, sizes: frame.expectedDecodedSizes())
        let uploadSeconds = decoder.uploadSeconds
        XCTAssertGreaterThan(uploadSeconds, 0)
        for variant in LZ4MetalDecoder.Variant.allCases {
            var previousOutput: (any MTLBuffer)?
            // 端数の SIMD グループも同じ結果になることを確かめる。
            for groups in [4, 7] {
                let result = try decoder.decode(frame: frame, source: source, variant: variant,
                                                simdgroupsPerThreadgroup: groups)
                XCTAssertEqual(result.output, expected)
                XCTAssertEqual(result.statuses, [UInt32](repeating: 0, count: frame.blocks.count))
                XCTAssertGreaterThan(result.cpuWallSeconds, 0)
                XCTAssertGreaterThan(result.gpuSeconds, 0)
                XCTAssertGreaterThan(result.uploadSeconds, 0)
                XCTAssertEqual(result.uploadSeconds, uploadSeconds)
                if let previousOutput { XCTAssertTrue(previousOutput === result.outputBuffer) }
                previousOutput = result.outputBuffer
            }
        }
        for blockIndex in [0, 1] {
            let times = try decoder.decodeSingleBlock(frame: frame, source: source, blockIndex: blockIndex, rounds: 3)
            XCTAssertEqual(times.count, 3)
            XCTAssertTrue(times.allSatisfy { $0 > 0 })
        }
        XCTAssertEqual(decoder.uploadSeconds, uploadSeconds)
        let otherSource = try LZ4Frame.encodeFrame([42], blockSize: 4_096)
        XCTAssertThrowsError(try decoder.prepare(frame: LZ4Frame(otherSource), source: otherSource, sizes: [1])) {
            XCTAssertEqual($0 as? LZ4Error, .sourceMismatch)
        }
        XCTAssertThrowsError(try decoder.decode(frame: frame, source: source, variant: .simdPerBlock,
                                                simdgroupsPerThreadgroup: Int.max))
    }

    func testOverlappingMatchesAndEmptyFrame() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal device unavailable") }
        let decoder = try LZ4MetalDecoder(device: device)
        let samples = [1, 2, 3, 7, 31, 32, 33].map { LZ4TestSupport.overlap(offset: $0, matchLength: 1_029) }
        let source = Data(LZ4TestSupport.frame(blocks: samples.map { ($0.compressed, false) }))
        let frame = try LZ4Frame(source)
        let expected = samples.flatMap(\.decoded)
        let empty = try LZ4Frame.encodeFrame([], blockSize: 65_536)
        let emptyDecoder = try LZ4MetalDecoder(device: device)
        for variant in LZ4MetalDecoder.Variant.allCases {
            let result = try decoder.decode(frame: frame, source: source, variant: variant, simdgroupsPerThreadgroup: 3)
            XCTAssertEqual(result.output, expected)
            XCTAssertTrue(result.statuses.allSatisfy { $0 == 0 })
            let emptyResult = try emptyDecoder.decode(frame: LZ4Frame(empty), source: empty, variant: variant)
            XCTAssertEqual(emptyResult.output, [])
            XCTAssertEqual(emptyResult.statuses, [])
        }
    }
}
