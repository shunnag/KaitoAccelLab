public import XCTest
@testable public import AccelLab

final class RangeCoderTests: XCTestCase {
    func testEmptyInputAndZeroPadding() throws {
        var encoder = RangeEncoder()
        XCTAssertEqual(encoder.finish(), [0, 0, 0, 0, 0])
        XCTAssertEqual(encoder.finish(), [0, 0, 0, 0, 0])
        var decoder = RangeDecoder([])
        let uniform = FrequencyQuantizer.cumulativeFrequencies(for: [Float](repeating: 1, count: 256))
        for _ in 0..<32 { XCTAssertEqual(try decoder.decode(cumulative: uniform), 0) }
    }

    func testUniformKnownBytes() throws {
        let uniform = FrequencyQuantizer.cumulativeFrequencies(for: [Float](repeating: 1, count: 256))
        var encoder = RangeEncoder()
        encoder.encode(255, cumulative: uniform)
        XCTAssertEqual(encoder.finish(), [0, 254, 255, 1, 0, 0])
        var decoder = RangeDecoder(encoder.finish())
        XCTAssertEqual(try decoder.decode(cumulative: uniform), 255)
    }

    func testFixedSkewedDistributionsAndRandomStreams() throws {
        var weights = [Float](repeating: 0, count: 256)
        weights[0] = 60_000
        weights[1] = 5_000
        weights[255] = 1
        let skewed = FrequencyQuantizer.cumulativeFrequencies(for: weights)
        let uniform = FrequencyQuantizer.cumulativeFrequencies(for: [Float](repeating: 1, count: 256))
        let random = NeuralTestData.random(count: 16_384)
        for cumulative in [skewed, uniform] {
            for stream in [random, [UInt8](repeating: 0, count: 8_192), [UInt8](repeating: 255, count: 8_192)] {
                var encoder = RangeEncoder()
                for byte in stream { encoder.encode(byte, cumulative: cumulative) }
                var decoder = RangeDecoder(encoder.finish())
                let restored = try stream.map { _ in try decoder.decode(cumulative: cumulative) }
                XCTAssertEqual(restored, stream)
            }
        }
    }

    func testChangingDistributionsAndCarryPropagation() throws {
        let stream = NeuralTestData.random(count: 20_000, seed: 0xFFFF_FFFF)
        let tables = (0..<16).map { row in
            FrequencyQuantizer.cumulativeFrequencies(for: (0..<256).map { symbol in
                symbol % 16 == row ? 10_000 : Float(symbol + 1)
            })
        }
        var encoder = RangeEncoder()
        for (index, byte) in stream.enumerated() { encoder.encode(byte, cumulative: tables[index % tables.count]) }
        var decoder = RangeDecoder(encoder.finish())
        for (index, byte) in stream.enumerated() {
            XCTAssertEqual(try decoder.decode(cumulative: tables[index % tables.count]), byte, "position \(index)")
        }
    }
}
