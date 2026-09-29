/// 64 ビットの low で桁上がりを伝播する range encoder。
public struct RangeEncoder {
    private var low: UInt64 = 0
    private var range = UInt32.max
    private var cache: UInt8 = 0
    private var pendingBytes = 1
    private var payload: [UInt8] = []
    private var finished = false

    public init() {}

    /// cumulative は 0 から 65536 まで単調増加する 257 要素。
    public mutating func encode(_ symbol: UInt8, cumulative: [UInt32]) {
        precondition(!finished && cumulative.count == FrequencyQuantizer.symbolCount + 1)
        cumulative.withUnsafeBufferPointer { encode(symbol, cumulative: $0.baseAddress!) }
    }

    internal mutating func encode(_ symbol: UInt8, cumulative: UnsafePointer<UInt32>) {
        precondition(!finished)
        let index = Int(symbol)
        range /= FrequencyQuantizer.total
        low += UInt64(cumulative[index]) * UInt64(range)
        range *= cumulative[index + 1] - cumulative[index]
        while range < RangeCoder.normalizationThreshold {
            range <<= RangeCoder.byteBits
            shiftLow()
        }
    }

    public mutating func finish() -> [UInt8] {
        if !finished {
            for _ in 0..<RangeCoder.flushByteCount { shiftLow() }
            finished = true
        }
        return payload
    }

    private mutating func shiftLow() {
        let lower = UInt32(truncatingIfNeeded: low)
        let carry = UInt8(low >> UInt32.bitWidth)
        if lower < RangeCoder.carryThreshold || carry != 0 {
            var byte = cache
            repeat {
                payload.append(byte &+ carry)
                byte = .max
                pendingBytes -= 1
            } while pendingBytes != 0
            cache = UInt8(truncatingIfNeeded: lower >> (UInt32.bitWidth - RangeCoder.byteBits))
        }
        pendingBytes += 1
        low = UInt64(lower << RangeCoder.byteBits)
    }
}
