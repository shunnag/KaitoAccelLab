public final class UniformPredictor: BytePredictor {
    private var blockCount = 0
    private var probabilities: [Float] = []

    public init() {}

    public func begin(blockCount: Int) throws {
        let count = try PredictorValidation.elementCount(blockCount: blockCount, width: FrequencyQuantizer.symbolCount)
        self.blockCount = blockCount
        probabilities = [Float](repeating: 1, count: count)
    }

    public func predictNext() throws -> [Float] {
        try PredictorValidation.started(blockCount)
        return probabilities
    }

    public func observe(_ bytes: [UInt8]) throws {
        try PredictorValidation.observation(bytes, blockCount: blockCount)
    }
}
