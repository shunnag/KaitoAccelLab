public import CoreML
public import Foundation

/// fp16 の再帰モデル。符号化と復号には同一のモデルファイル・計算ユニット・バッチ形状が必要。
public final class CoreMLRecurrentPredictor: BytePredictor {
    private let model: MLModel
    private let modelURL: URL
    private let computeUnits: MLComputeUnits
    private let inputConstraint: MLMultiArrayConstraint
    private let hiddenConstraint: MLMultiArrayConstraint
    private let probabilityConstraint: MLMultiArrayConstraint
    private let nextHiddenConstraint: MLMultiArrayConstraint
    private let hidden: Int
    private var blockCount = 0
    private var state: RecurrentByteState?
    private var input: MLMultiArray?
    private var inputPointer: UnsafeMutablePointer<Float16>?
    private var inputRowStride = 0
    private var inputColumnStride = 0
    private var hiddenInput: MLMultiArray?
    private var provider: MLDictionaryFeatureProvider?
    private var probabilities: [Float] = []

    public init(modelURL: URL, computeUnits: MLComputeUnits) throws {
        self.modelURL = modelURL
        self.computeUnits = computeUnits
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
        let inputs = model.modelDescription.inputDescriptionsByName
        let outputs = model.modelDescription.outputDescriptionsByName
        inputConstraint = try CoreMLTensor.constraint(inputs, name: "x_onehot", width: FrequencyQuantizer.symbolCount)
        hiddenConstraint = try CoreMLTensor.constraint(inputs, name: "h_in")
        hidden = hiddenConstraint.shape[1].intValue
        probabilityConstraint = try CoreMLTensor.constraint(outputs, name: "probabilities", width: FrequencyQuantizer.symbolCount)
        nextHiddenConstraint = try CoreMLTensor.constraint(outputs, name: "h_out", width: hidden)
    }

    public func begin(blockCount: Int) throws {
        try CoreMLTensor.validate(inputConstraint, batch: blockCount, width: FrequencyQuantizer.symbolCount, name: "x_onehot")
        try CoreMLTensor.validate(hiddenConstraint, batch: blockCount, width: hidden, name: "h_in")
        try CoreMLTensor.validate(probabilityConstraint, batch: blockCount, width: FrequencyQuantizer.symbolCount, name: "probabilities")
        try CoreMLTensor.validate(nextHiddenConstraint, batch: blockCount, width: hidden, name: "h_out")
        let input = try CoreMLTensor.make(batch: blockCount, width: FrequencyQuantizer.symbolCount)
        let hiddenInput = try CoreMLTensor.make(batch: blockCount, width: hidden)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "x_onehot": MLFeatureValue(multiArray: input), "h_in": MLFeatureValue(multiArray: hiddenInput),
        ])
        state = try RecurrentByteState(blockCount: blockCount)
        self.input = input
        inputPointer = input.dataPointer.assumingMemoryBound(to: Float16.self)
        inputRowStride = input.strides[0].intValue
        inputColumnStride = input.strides[1].intValue
        self.hiddenInput = hiddenInput
        self.provider = provider
        self.blockCount = blockCount
        probabilities = [Float](repeating: 0, count: blockCount * FrequencyQuantizer.symbolCount)
    }

    public func predictNext() throws -> [Float] {
        guard let inputPointer, let hiddenInput, let provider, let state else {
            throw NeuralCodecError("Call begin(blockCount:) before prediction")
        }
        state.write(to: inputPointer, rowStride: inputRowStride, columnStride: inputColumnStride)
        let result = try model.prediction(from: provider)
        let output = try CoreMLTensor.output(result, name: "probabilities", batch: blockCount, width: FrequencyQuantizer.symbolCount)
        let nextHidden = try CoreMLTensor.output(result, name: "h_out", batch: blockCount, width: hidden)
        CoreMLTensor.readProbabilities(output, into: &probabilities)
        CoreMLTensor.copyState(nextHidden, to: hiddenInput)
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
