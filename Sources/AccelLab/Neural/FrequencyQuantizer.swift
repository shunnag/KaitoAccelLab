/// 非有限値と負値をゼロとして、総和 65536 の累積頻度に量子化する。
public enum FrequencyQuantizer {
    public static let symbolCount = 256
    public static let total: UInt32 = 1 << 16

    public static func cumulativeFrequencies(for probabilities: [Float]) -> [UInt32] {
        precondition(probabilities.count == symbolCount)
        var cumulative = [UInt32](repeating: 0, count: symbolCount + 1)
        probabilities.withUnsafeBufferPointer { probabilities in
            cumulative.withUnsafeMutableBufferPointer { cumulative in
                fill(probabilities.baseAddress!, cumulative: cumulative.baseAddress!)
            }
        }
        return cumulative
    }

    // 呼び出し側の累積頻度バッファを再利用する。
    internal static func fill(_ probabilities: UnsafePointer<Float>, cumulative: UnsafeMutablePointer<UInt32>) {
        var sum = 0.0
        var largest = 0
        var largestWeight: Float = 0
        for symbol in 0..<symbolCount {
            let value = weight(probabilities[symbol])
            sum += Double(value)
            if value > largestWeight { largest = symbol; largestWeight = value }
        }
        guard sum > 0 else {
            let frequency = total / UInt32(symbolCount)
            for symbol in 0...symbolCount { cumulative[symbol] = UInt32(symbol) * frequency }
            return
        }
        // 各記号の 1 を先に予約するため、切り捨て後の総和は total を超えない。
        let available = Double(total - UInt32(symbolCount))
        var allocated: UInt32 = 0
        for symbol in 0..<symbolCount {
            let frequency = 1 + UInt32(Double(weight(probabilities[symbol])) / sum * available)
            cumulative[symbol + 1] = frequency
            allocated += frequency
        }
        cumulative[largest + 1] += total - allocated
        cumulative[0] = 0
        for symbol in 0..<symbolCount { cumulative[symbol + 1] += cumulative[symbol] }
    }

    private static func weight(_ value: Float) -> Float { value.isFinite && value > 0 ? value : 0 }
}
