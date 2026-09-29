public import CoreML
public import Foundation

/// fp16 の文脈モデル。符号化と復号には同一のモデルファイル・計算ユニット・バッチ形状が必要。
public final class CoreMLContextPredictor: BytePredictor {
    private let model: MLModel
    private let modelURL: URL
    private let computeUnits: MLComputeUnits
    private let inputConstraint: MLMultiArrayConstraint
    private let outputConstraint: MLMultiArrayConstraint
    private let context: Int
    private var blockCount = 0
    private var state: ByteContextState?
    private var input: MLMultiArray?
    private var provider: MLDictionaryFeatureProvider?
    private var probabilities: [Float] = []

    public init(modelURL: URL, computeUnits: MLComputeUnits) throws {
        self.modelURL = modelURL
        self.computeUnits = computeUnits
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
        inputConstraint = try CoreMLTensor.constraint(model.modelDescription.inputDescriptionsByName, name: "x")
        outputConstraint = try CoreMLTensor.constraint(model.modelDescription.outputDescriptionsByName,
                                                       name: "probabilities", width: FrequencyQuantizer.symbolCount)
        context = inputConstraint.shape[1].intValue
    }

    public func begin(blockCount: Int) throws {
        try CoreMLTensor.validate(inputConstraint, batch: blockCount, width: context, name: "x")
        try CoreMLTensor.validate(outputConstraint, batch: blockCount, width: FrequencyQuantizer.symbolCount, name: "probabilities")
        let input = try CoreMLTensor.make(batch: blockCount, width: context)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: input)])
        state = try ByteContextState(blockCount: blockCount, context: context)
        self.input = input
        self.provider = provider
        self.blockCount = blockCount
        probabilities = [Float](repeating: 0, count: blockCount * FrequencyQuantizer.symbolCount)
    }

    public func predictNext() throws -> [Float] {
        guard let input, let provider, let state else { throw NeuralCodecError("Call begin(blockCount:) before prediction") }
        input.withUnsafeMutableBytes { bytes, strides in
            state.write(to: bytes.bindMemory(to: Float16.self), rowStride: strides[0], columnStride: strides[1])
        }
        let result = try model.prediction(from: provider)
        let output = try CoreMLTensor.output(result, name: "probabilities", batch: blockCount, width: FrequencyQuantizer.symbolCount)
        CoreMLTensor.readProbabilities(output, into: &probabilities)
        return probabilities
    }

    public func observe(_ bytes: [UInt8]) throws {
        try PredictorValidation.observation(bytes, blockCount: blockCount)
        try state?.observe(bytes)
    }

    public func computePlanSummary() async throws -> String {
        try await CoreMLComputePlanSummary.load(modelURL: modelURL, computeUnits: computeUnits)
    }
}
