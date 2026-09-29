/// 直前のバイトを条件とする次数 1 予測器。初期の条件は 0。
public final class OrderOnePredictor: BytePredictor {
    private var blockCount = 0
    private var counts: [UInt64] = []
    private var previous: [UInt8] = []
    private var probabilities: [Float] = []
    private static let rowWidth = FrequencyQuantizer.symbolCount
    private static let blockWidth = rowWidth * rowWidth

    public init() {}

    public func begin(blockCount: Int) throws {
        let count = try PredictorValidation.elementCount(blockCount: blockCount, width: Self.blockWidth)
        self.blockCount = blockCount
        counts = [UInt64](repeating: 1, count: count)
        previous = [UInt8](repeating: 0, count: blockCount)
        probabilities = [Float](repeating: 1, count: blockCount * Self.rowWidth)
    }

    public func predictNext() throws -> [Float] {
        try PredictorValidation.started(blockCount)
        for block in 0..<blockCount {
            let row = block * Self.blockWidth + Int(previous[block]) * Self.rowWidth
            let destination = block * Self.rowWidth
            for symbol in 0..<Self.rowWidth { probabilities[destination + symbol] = Float(counts[row + symbol]) }
        }
        return probabilities
    }

    public func observe(_ bytes: [UInt8]) throws {
        try PredictorValidation.observation(bytes, blockCount: blockCount)
        for block in 0..<blockCount {
            let row = block * Self.blockWidth + Int(previous[block]) * Self.rowWidth
            counts[row + Int(bytes[block])] += 1
            previous[block] = bytes[block]
        }
    }
}
