internal import AccelLab
private import CoreML
internal import Foundation

/// CLI 間で予測器の構築とヘッダータグを揃える。
internal struct NeuralPredictor {
    let predictor: any BytePredictor
    let tag: String

    static func make(selection: String, modelURL: URL?, units: String) async throws -> NeuralPredictor {
        let computeUnits: MLComputeUnits
        switch units {
        case "cpu": computeUnits = .cpuOnly
        case "gpu": computeUnits = .cpuAndGPU
        case "all": computeUnits = .all
        default: computeUnits = .cpuAndNeuralEngine
        }
        let predictor: any BytePredictor
        let tag: String
        switch selection {
        case "uniform": predictor = UniformPredictor(); tag = selection
        case "order0": predictor = OrderZeroPredictor(); tag = selection
        case "order1": predictor = OrderOnePredictor(); tag = selection
        case "mlp":
            let modelURL = modelURL!
            let coreML = try CoreMLContextPredictor(modelURL: modelURL, computeUnits: computeUnits)
            print(try await coreML.computePlanSummary())
            predictor = coreML
            tag = "mlp:\(modelURL.deletingPathExtension().lastPathComponent):\(units)"
        default:
            let modelURL = modelURL!
            let coreML = try CoreMLRecurrentPredictor(modelURL: modelURL, computeUnits: computeUnits)
            print(try await coreML.computePlanSummary())
            predictor = coreML
            tag = "gru:\(modelURL.deletingPathExtension().lastPathComponent):\(units)"
        }
        return NeuralPredictor(predictor: predictor, tag: tag)
    }
}
