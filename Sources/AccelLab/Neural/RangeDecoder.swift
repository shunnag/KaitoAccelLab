/// ペイロード末尾以降をゼロとして読む range decoder。
public struct RangeDecoder {
    private let payload: ArraySlice<UInt8>
    private var offset: Int
    private var code: UInt32 = 0
    private var range = UInt32.max

    public init(_ payload: [UInt8]) {
        self.init(payload[...])
    }

    internal init(_ payload: ArraySlice<UInt8>) {
        self.payload = payload
        offset = payload.startIndex
        for _ in 0..<RangeCoder.flushByteCount {
            code = (code << RangeCoder.byteBits) | UInt32(readByte())
        }
    }

    /// cumulative の契約は RangeEncoder.encode と同じ。
    public mutating func decode(cumulative: [UInt32]) throws -> UInt8 {
        precondition(cumulative.count == FrequencyQuantizer.symbolCount + 1)
        return try cumulative.withUnsafeBufferPointer { try decode(cumulative: $0.baseAddress!) }
    }

    internal mutating func decode(cumulative: UnsafePointer<UInt32>) throws -> UInt8 {
        range /= FrequencyQuantizer.total
        let value = code / range
        guard value < FrequencyQuantizer.total else {
            throw NeuralCodecError("Invalid range-coded payload")
        }
        var lower = 0
        var upper = FrequencyQuantizer.symbolCount
        while lower + 1 < upper {
            let middle = (lower + upper) / 2
            if cumulative[middle] <= value { lower = middle } else { upper = middle }
        }
        code -= cumulative[lower] * range
        range *= cumulative[lower + 1] - cumulative[lower]
        while range < RangeCoder.normalizationThreshold {
            range <<= RangeCoder.byteBits
            code = (code << RangeCoder.byteBits) | UInt32(readByte())
        }
        return UInt8(lower)
    }

    private mutating func readByte() -> UInt8 {
        guard offset < payload.endIndex else { return 0 }
        defer { offset += 1 }
        return payload[offset]
    }
}
