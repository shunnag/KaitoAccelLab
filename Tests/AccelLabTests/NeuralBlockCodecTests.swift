public import Foundation
public import XCTest
@testable public import AccelLab

final class NeuralBlockCodecTests: XCTestCase {
    func testSerialAndParallelBytesMatch() throws {
        let factories: [(String, () -> any BytePredictor)] = [
            ("uniform", { UniformPredictor() }), ("order0", { OrderZeroPredictor() }), ("order1", { OrderOnePredictor() }),
        ]
        for (tag, makePredictor) in factories {
            for blocks in [7, 64] {
                for length in [0, 1, blocks - 1, blocks * 3 + 5, 2_048] {
                    let input = Data(NeuralTestData.random(count: length))
                    var timings = NeuralCodecTimings()
                    let serial = try NeuralBlockCodec.encode(data: input, blockCount: blocks, predictor: makePredictor(),
                                                            predictorTag: tag, lanes: 1, timings: &timings)
                    let parallel = try NeuralBlockCodec.encode(data: input, blockCount: blocks, predictor: makePredictor(),
                                                              predictorTag: tag, lanes: 4, timings: &timings)
                    XCTAssertEqual(parallel, serial, "\(tag), blocks=\(blocks), length=\(length)")
                    let steps = length / blocks + (length.isMultiple(of: blocks) ? 0 : 1)
                    XCTAssertEqual(timings.steps, steps)
                    for lanes in [1, 4] {
                        let decoded = try NeuralBlockCodec.decode(parallel, predictor: makePredictor(), lanes: lanes, timings: &timings)
                        XCTAssertEqual(decoded.payload, input)
                        XCTAssertEqual(timings.steps, steps)
                        if steps == 0 {
                            XCTAssertEqual(timings.predictorSeconds, 0)
                            XCTAssertEqual(timings.coderSeconds, 0)
                        } else {
                            XCTAssertGreaterThan(timings.predictorSeconds, 0)
                            XCTAssertGreaterThan(timings.coderSeconds, 0)
                        }
                    }
                }
            }
        }
    }

    func testParallelDecoderPropagatesLaneFailureBeforeObservation() throws {
        let predictor = RecordingPredictor()
        let encoded = try NeuralBlockCodec.encode(data: Data(1...7), blockCount: 7, predictor: predictor, predictorTag: "uniform")
        var bytes = Array(encoded)
        let (header, payloadOffset) = try Header.parse(bytes)
        let damagedBlock = 4
        let start = payloadOffset + header.payloadLengths.prefix(damagedBlock).reduce(0) { $0 + Int($1) }
        for index in start..<(start + Int(header.payloadLengths[damagedBlock])) { bytes[index] = .max }
        var timings = NeuralCodecTimings()
        for lanes in [1, 4] {
            XCTAssertThrowsError(try NeuralBlockCodec.decode(Data(bytes), predictor: predictor, lanes: lanes, timings: &timings)) {
                XCTAssertEqual($0.localizedDescription, "Invalid range-coded payload")
            }
            XCTAssertEqual(predictor.predictions, 1)
            XCTAssertTrue(predictor.observations.isEmpty)
        }
    }

    func testRoundTripsWithAllCPUReferences() throws {
        let text = NeuralTestData.text()
        let factories: [(String, () -> any BytePredictor)] = [
            ("uniform", { UniformPredictor() }), ("order0", { OrderZeroPredictor() }), ("order1", { OrderOnePredictor() }),
        ]
        for (tag, makePredictor) in factories {
            for blockCount in [1, 7, 64] {
                for length in [0, 1, blockCount - 1, blockCount * 3 + 5, 4_096] {
                    let textBytes = (0..<length).map { text[$0 % text.count] }
                    for bytes in [NeuralTestData.random(count: length), textBytes] {
                        let input = Data(bytes)
                        let encoded = try NeuralBlockCodec.encode(data: input, blockCount: blockCount,
                                                                 predictor: makePredictor(), predictorTag: tag)
                        let decoded = try NeuralBlockCodec.decode(encoded, predictor: makePredictor())
                        XCTAssertEqual(decoded.payload, input, "\(tag), blocks=\(blockCount), length=\(length)")
                        XCTAssertEqual(decoded.header.version, 1)
                        XCTAssertEqual(decoded.header.blockCount, UInt32(blockCount))
                        XCTAssertEqual(decoded.header.originalLength, UInt64(length))
                        XCTAssertEqual(decoded.header.predictorTag, tag)
                        XCTAssertEqual(decoded.header.payloadLengths.count, blockCount)
                        XCTAssertEqual(try Header.decode(encoded), decoded.header)
                    }
                }
            }
        }
    }

    func testLockStepAndZeroObservationsForEndedBlocks() throws {
        let predictor = RecordingPredictor()
        let input = Data(1...8)
        let encoded = try NeuralBlockCodec.encode(data: input, blockCount: 7, predictor: predictor, predictorTag: "uniform")
        let expected: [[UInt8]] = [[1, 3, 5, 7, 0, 0, 0], [2, 4, 6, 8, 0, 0, 0]]
        XCTAssertEqual(predictor.predictions, 2)
        XCTAssertEqual(predictor.observations, expected)
        XCTAssertEqual(try NeuralBlockCodec.decode(encoded, predictor: predictor).payload, input)
        XCTAssertEqual(predictor.predictions, 2)
        XCTAssertEqual(predictor.observations, expected)

        let partial = try NeuralBlockCodec.encode(data: Data(1...7), blockCount: 3, predictor: predictor, predictorTag: "uniform")
        XCTAssertEqual(predictor.observations, [[1, 4, 7], [2, 5, 0], [3, 6, 0]])
        XCTAssertEqual(try NeuralBlockCodec.decode(partial, predictor: predictor).payload, Data(1...7))
        XCTAssertEqual(predictor.observations, [[1, 4, 7], [2, 5, 0], [3, 6, 0]])
    }

    func testHeaderLayoutUTF8AndDeterministicReset() throws {
        let predictor = OrderZeroPredictor()
        let input = Data(NeuralTestData.random(count: 1_024))
        let tag = "mlp:文脈:ane"
        let encoded = try NeuralBlockCodec.encode(data: input, blockCount: 7, predictor: predictor, predictorTag: tag)
        XCTAssertEqual(Array(encoded.prefix(17)), [78, 66, 67, 49, 1, 7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0])
        XCTAssertEqual(try Header.decode(encoded).predictorTag, tag)
        XCTAssertEqual(try NeuralBlockCodec.encode(data: input, blockCount: 7, predictor: predictor, predictorTag: tag), encoded)
        XCTAssertEqual(try NeuralBlockCodec.decode(encoded, predictor: predictor).payload, input)
    }

    func testInvalidHeadersAndOptionsThrow() throws {
        let predictor = UniformPredictor()
        for count in [0, -1, Int.max] {
            XCTAssertThrowsError(try NeuralBlockCodec.encode(data: Data(), blockCount: count, predictor: predictor, predictorTag: "uniform"))
        }
        XCTAssertThrowsError(try NeuralBlockCodec.encode(data: Data(), blockCount: 1, predictor: predictor,
                                                       predictorTag: String(repeating: "x", count: 65_536)))
        let encoded = try NeuralBlockCodec.encode(data: Data([1, 2, 3]), blockCount: 7, predictor: predictor, predictorTag: "uniform")
        for length in 0..<encoded.count {
            XCTAssertThrowsError(try NeuralBlockCodec.decode(Data(encoded.prefix(length)), predictor: predictor))
        }
        for (offset, value): (Int, UInt8) in [(0, 0), (4, 2), (5, 0), (19, 255)] {
            var malformed = Array(encoded)
            malformed[offset] = value
            XCTAssertThrowsError(try Header.decode(Data(malformed)))
        }
        XCTAssertThrowsError(try Header.decode(Data(Array(encoded) + [0])))
    }
}
