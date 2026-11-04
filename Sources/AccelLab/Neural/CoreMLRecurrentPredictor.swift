public import CoreML
public import Foundation

/// fp16 の再帰モデル。符号化と復号には同一のモデルファイル・計算ユニット・バッチ形状が必要。
public final class CoreMLRecurrentPredictor: BytePredictor {
    private let model: MLModel
    private let modelURL: URL
    private let computeUnits: MLComputeUnits
    private let inputConstraint: MLMultiArrayConstraint
    private let hiddenConstraint: MLMultiArrayConstraint?
    private let stateConstraint: MLStateConstraint?
    private let stateName: String?
    private let probabilityConstraint: MLMultiArrayConstraint
    private let nextHiddenConstraint: MLMultiArrayConstraint?
    private let hidden: Int
    private var blockCount = 0
    private var state: RecurrentByteState?
    private var mlState: MLState?
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
        probabilityConstraint = try CoreMLTensor.constraint(outputs, name: "probabilities", width: FrequencyQuantizer.symbolCount)
        if inputs["h_in"] != nil {
            let constraint = try CoreMLTensor.constraint(inputs, name: "h_in")
            hiddenConstraint = constraint
            stateConstraint = nil
            stateName = nil
            hidden = constraint.shape[1].intValue
            nextHiddenConstraint = try CoreMLTensor.constraint(outputs, name: "h_out", width: hidden)
        } else {
            guard let constraint = model.modelDescription.stateDescriptionsByName["hidden_state"]?.stateConstraint,
                  constraint.dataType == .float16 else {
                throw NeuralCodecError("Core ML model requires fp16 h_in or a hidden_state MLState")
            }
            let shape = constraint.bufferShape
            guard (shape.count == 2 && shape[0] > 0 && shape[1] > 0)
                    || (shape.count == 3 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0) else {
                throw NeuralCodecError("Core ML model requires fp16 h_in or a hidden_state MLState")
            }
            hiddenConstraint = nil
            stateConstraint = constraint
            stateName = "hidden_state"
            hidden = shape[shape.count - 1]
            nextHiddenConstraint = nil
        }
    }

    public func begin(blockCount: Int) throws {
        try CoreMLTensor.validate(inputConstraint, batch: blockCount, width: FrequencyQuantizer.symbolCount, name: "x_onehot")
        try CoreMLTensor.validate(probabilityConstraint, batch: blockCount, width: FrequencyQuantizer.symbolCount, name: "probabilities")
        let input = try CoreMLTensor.make(batch: blockCount, width: FrequencyQuantizer.symbolCount)
        var features: [String: MLFeatureValue] = ["x_onehot": MLFeatureValue(multiArray: input)]
        if stateName != nil {
            try CoreMLTensor.validate(stateConstraint!, batch: blockCount, width: hidden, name: "hidden_state")
            mlState = model.makeState()
            hiddenInput = nil
        } else {
            try CoreMLTensor.validate(hiddenConstraint!, batch: blockCount, width: hidden, name: "h_in")
            try CoreMLTensor.validate(nextHiddenConstraint!, batch: blockCount, width: hidden, name: "h_out")
            let hiddenInput = try CoreMLTensor.make(batch: blockCount, width: hidden)
            features["h_in"] = MLFeatureValue(multiArray: hiddenInput)
            self.hiddenInput = hiddenInput
            mlState = nil
        }
        provider = try MLDictionaryFeatureProvider(dictionary: features)
        state = try RecurrentByteState(blockCount: blockCount)
        self.input = input
        inputPointer = input.dataPointer.assumingMemoryBound(to: Float16.self)
        inputRowStride = input.strides[0].intValue
        inputColumnStride = input.strides[1].intValue
        self.blockCount = blockCount
        probabilities = [Float](repeating: 0, count: blockCount * FrequencyQuantizer.symbolCount)
    }

    public func predictNext() throws -> [Float] {
        guard let inputPointer, let provider, let state else {
            throw NeuralCodecError("Call begin(blockCount:) before prediction")
        }
        state.write(to: inputPointer, rowStride: inputRowStride, columnStride: inputColumnStride)
        let result: any MLFeatureProvider
        if let mlState {
            result = try model.prediction(from: provider, using: mlState)
        } else {
            result = try model.prediction(from: provider)
        }
        let output = try CoreMLTensor.output(result, name: "probabilities", batch: blockCount, width: FrequencyQuantizer.symbolCount)
        CoreMLTensor.readProbabilities(output, into: &probabilities)
        if stateName == nil, let hiddenInput {
            let nextHidden = try CoreMLTensor.output(result, name: "h_out", batch: blockCount, width: hidden)
            CoreMLTensor.copyState(nextHidden, to: hiddenInput)
        }
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
