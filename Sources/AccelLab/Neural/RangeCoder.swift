/// 32 ビット range coder の共通定数。
internal enum RangeCoder {
    static let normalizationThreshold: UInt32 = 1 << 24
    static let carryThreshold: UInt32 = 0xFF00_0000
    static let flushByteCount = 5
    static let byteBits = 8
}
