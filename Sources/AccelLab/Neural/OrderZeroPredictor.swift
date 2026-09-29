/// ブロックごとに独立した加算平滑化付きの次数 0 予測器。
public final class OrderZeroPredictor: BytePredictor {
    private var blockCount = 0
    private var counts: [UInt64] = []
    private var probabilities: [Float] = []

    public init() {}

    public func begin(blockCount: Int) throws {
        let count = try PredictorValidation.elementCount(blockCount: blockCount, width: FrequencyQuantizer.symbolCount)
        self.blockCount = blockCount
        counts = [UInt64](repeating: 1, count: count)
        probabilities = [Float](repeating: 1, count: count)
    }

    public func predictNext() throws -> [Float] {
        try PredictorValidation.started(blockCount)
        for index in counts.indices { probabilities[index] = Float(counts[index]) }
        return probabilities
    }

    public func observe(_ bytes: [UInt8]) throws {
        try PredictorValidation.observation(bytes, blockCount: blockCount)
        for block in 0..<blockCount { counts[block * FrequencyQuantizer.symbolCount + Int(bytes[block])] += 1 }
    }
}
