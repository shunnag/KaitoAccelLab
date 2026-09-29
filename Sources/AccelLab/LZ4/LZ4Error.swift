internal import Foundation

enum LZ4Error: Error, Equatable, LocalizedError {
    case truncatedInput
    case invalidMagic
    case invalidVersion
    case reservedBits
    case invalidBlockSize
    case trailingData
    case invalidOffset
    case outputTooSmall
    case sizeOverflow
    case contentSizeMismatch
    case sourceMismatch
    case unsupportedDependency
    case unsupportedDictionary
    case invalidArgument(String)
    case appleDecodeFailed(Int)
    case metalUnavailable
    case metalFailure(String)

    var errorDescription: String? {
        switch self {
        case .truncatedInput: "Truncated LZ4 input"
        case .invalidMagic: "Invalid LZ4 frame magic"
        case .invalidVersion: "Unsupported LZ4 frame version"
        case .reservedBits: "Reserved LZ4 header bits are set"
        case .invalidBlockSize: "Invalid LZ4 block size"
        case .trailingData: "Unexpected bytes after the LZ4 frame"
        case .invalidOffset: "Invalid LZ4 match offset"
        case .outputTooSmall: "LZ4 output exceeds its buffer or block limit"
        case .sizeOverflow: "LZ4 size exceeds the supported range"
        case .contentSizeMismatch: "LZ4 content size does not match decoded blocks"
        case .sourceMismatch: "LZ4 source differs from the parsed frame"
        case .unsupportedDependency: "LZ4 decoding requires independent blocks"
        case .unsupportedDictionary: "LZ4 external dictionaries are unsupported"
        case .invalidArgument(let message): message
        case .appleDecodeFailed(let block): "Apple LZ4 decode failed for block \(block)"
        case .metalUnavailable: "Metal device unavailable"
        case .metalFailure(let message): "Metal LZ4 decode failed: \(message)"
        }
    }
}
