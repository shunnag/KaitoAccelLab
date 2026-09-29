public import Foundation

/// NBC1 のヘッダー。整数はリトルエンディアン、version は UInt8。
public struct Header: Sendable, Equatable {
    public let version: UInt8
    public let blockCount: UInt32
    public let originalLength: UInt64
    public let predictorTag: String
    public let payloadLengths: [UInt32]

    private static let magic = Array("NBC1".utf8)
    internal static let currentVersion: UInt8 = 1

    public static func decode(_ data: Data) throws -> Header {
        try parse(Array(data)).header
    }

    internal func append(to bytes: inout [UInt8]) {
        bytes.append(contentsOf: Self.magic)
        Self.appendInteger(version, to: &bytes)
        Self.appendInteger(blockCount, to: &bytes)
        Self.appendInteger(originalLength, to: &bytes)
        let tag = Array(predictorTag.utf8)
        Self.appendInteger(UInt16(tag.count), to: &bytes)
        bytes.append(contentsOf: tag)
        for length in payloadLengths { Self.appendInteger(length, to: &bytes) }
    }

    internal static func parse(_ bytes: [UInt8]) throws -> (header: Header, payloadOffset: Int) {
        guard bytes.starts(with: magic) else { throw NeuralCodecError("Invalid NBC1 magic") }
        var offset = magic.count
        let version: UInt8 = try readInteger(bytes, offset: &offset)
        guard version == currentVersion else { throw NeuralCodecError("Unsupported NBC1 version: \(version)") }
        let blockCount: UInt32 = try readInteger(bytes, offset: &offset)
        guard blockCount > 0 else { throw NeuralCodecError("NBC1 blockCount must be positive") }
        let originalLength: UInt64 = try readInteger(bytes, offset: &offset)
        guard originalLength <= UInt64(Int.max) else { throw NeuralCodecError("NBC1 originalLength is too large") }
        let tagLength: UInt16 = try readInteger(bytes, offset: &offset)
        guard Int(tagLength) <= bytes.count - offset else { throw NeuralCodecError("Truncated NBC1 predictor tag") }
        guard let tag = String(bytes: bytes[offset..<(offset + Int(tagLength))], encoding: .utf8) else {
            throw NeuralCodecError("Invalid UTF-8 in NBC1 predictor tag")
        }
        offset += Int(tagLength)
        guard Int(blockCount) <= (bytes.count - offset) / MemoryLayout<UInt32>.size else {
            throw NeuralCodecError("Truncated NBC1 payload length table")
        }
        var lengths: [UInt32] = []
        lengths.reserveCapacity(Int(blockCount))
        var total: UInt64 = 0
        for _ in 0..<blockCount {
            let length: UInt32 = try readInteger(bytes, offset: &offset)
            lengths.append(length)
            total += UInt64(length)
        }
        guard total == UInt64(bytes.count - offset) else {
            throw NeuralCodecError("NBC1 payload lengths do not match container size")
        }
        return (Header(version: version, blockCount: blockCount, originalLength: originalLength,
                       predictorTag: tag, payloadLengths: lengths), offset)
    }

    private static func appendInteger<T: FixedWidthInteger>(_ value: T, to bytes: inout [UInt8]) {
        for shift in stride(from: 0, to: T.bitWidth, by: RangeCoder.byteBits) {
            bytes.append(UInt8(truncatingIfNeeded: value >> shift))
        }
    }

    private static func readInteger<T: FixedWidthInteger>(_ bytes: [UInt8], offset: inout Int) throws -> T {
        guard MemoryLayout<T>.size <= bytes.count - offset else { throw NeuralCodecError("Truncated NBC1 header") }
        var value: T = 0
        for shift in stride(from: 0, to: T.bitWidth, by: RangeCoder.byteBits) {
            value |= T(bytes[offset]) << shift
            offset += 1
        }
        return value
    }
}
