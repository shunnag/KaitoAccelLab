internal import CoreML
internal import Foundation
internal import XCTest
@testable import AccelLab

final class CoreMLRecurrentPredictorTests: XCTestCase {
    func testStatefulPredictionsMatchExplicitStateAndReset() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let models = repository.appendingPathComponent("Models")
        let explicitURL = models.appendingPathComponent("gru-words-h1024-b1024.mlmodelc")
        let statefulURL = models.appendingPathComponent("gru-words-h1024-b1024-stateful.mlmodelc")
        guard FileManager.default.fileExists(atPath: explicitURL.path),
              FileManager.default.fileExists(atPath: statefulURL.path) else {
            throw XCTSkip("Generate the explicit and stateful GRU models with Scripts/train_gru_predictor.py")
        }
        let allUnits: [MLComputeUnits] = [.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine]
        let variants = [
            (statefulURL, allUnits),
            (models.appendingPathComponent("gru-words-h1024-b1024-stateful-select.mlmodelc"), allUnits),
            (models.appendingPathComponent("gru-words-h1024-b1024-stateful-direct.mlmodelc"), [.cpuOnly, .cpuAndGPU]),
        ].filter { FileManager.default.fileExists(atPath: $0.0.path) }
        let observed = (0..<1024).map { UInt8(truncatingIfNeeded: $0 &* 37 &+ 11) }
        for (statefulURL, units) in variants {
            for unit in units {
                let explicit = try CoreMLRecurrentPredictor(modelURL: explicitURL, computeUnits: unit)
                let stateful = try CoreMLRecurrentPredictor(modelURL: statefulURL, computeUnits: unit)
                try explicit.begin(blockCount: 1024)
                try stateful.begin(blockCount: 1024)
                var first: [Float] = []
                for _ in 0..<3 {
                    let expected = try explicit.predictNext()
                    let actual = try stateful.predictNext()
                    XCTAssertEqual(actual, expected, statefulURL.lastPathComponent)
                    if first.isEmpty { first = actual }
                    try explicit.observe(observed)
                    try stateful.observe(observed)
                }
                try stateful.begin(blockCount: 1024)
                XCTAssertEqual(try stateful.predictNext(), first)
                XCTAssertThrowsError(try stateful.begin(blockCount: 31))
            }
        }
    }

    func testInt8StatefulRoundTripAndReset() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let modelURL = repository.appendingPathComponent("Models/gru-words-h1024-b1024-stateful-int8.mlmodelc")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("Generate the int8 model with Scripts/quantize_gru_model.py")
        }
        let input = Data(NeuralTestData.random(count: 4_096))
        for units in [MLComputeUnits.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine] {
            let predictor = try CoreMLRecurrentPredictor(modelURL: modelURL, computeUnits: units)
            let encoded = try NeuralBlockCodec.encode(
                data: input, blockCount: 1024, predictor: predictor, predictorTag: "gru-int8-test"
            )
            let decoded = try NeuralBlockCodec.decode(encoded, predictor: predictor)
            XCTAssertEqual(decoded.payload, input)
        }
    }
}
