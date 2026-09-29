internal import Compression
internal import Foundation
@testable import AccelLab

enum LZ4TestSupport {
    static let corpusSize = 3 * 1_048_576

    static func corpus(count: Int = corpusSize) -> [UInt8] {
        var seed: UInt64 = 0x4C5A_3454_6573_7473
        return (0..<count).map { index in
            if (index / 65_536).isMultiple(of: 3) {
                seed ^= seed << 13
                seed ^= seed >> 7
                seed ^= seed << 17
                return UInt8(truncatingIfNeeded: seed)
            }
            return UInt8((index % 251) % 7 + 65)
        }
    }

    static func compress(_ input: [UInt8]) throws -> [UInt8] {
        let encoderHeadroom = 65_536
        var output = [UInt8](repeating: 0, count: input.count + input.count / 255 + encoderHeadroom)
        let count = output.withUnsafeMutableBufferPointer { destination in
            input.withUnsafeBufferPointer { source in
                compression_encode_buffer(destination.baseAddress!, destination.count,
                                          source.baseAddress!, source.count, nil, COMPRESSION_LZ4_RAW)
            }
        }
        guard count > 0 else { throw LZ4Error.invalidArgument("Test compression failed for \(input.count) bytes") }
        return Array(output.prefix(count))
    }

    static func appendLE(_ value: UInt64, bytes count: Int, to output: inout [UInt8]) {
        for index in 0..<count { output.append(UInt8(truncatingIfNeeded: value >> (index * 8))) }
    }

    static func frame(
        blocks: [(bytes: [UInt8], stored: Bool)], sizeCode: UInt8 = 4, contentSize: UInt64? = nil,
        blockChecksums: Bool = false, contentChecksum: Bool = false,
        independent: Bool = true, dictionaryID: UInt32? = nil
    ) -> [UInt8] {
        var output: [UInt8] = []
        appendLE(UInt64(LZ4Frame.magic), bytes: 4, to: &output)
        var flags: UInt8 = 0x40
        if independent { flags |= 0x20 }
        if contentSize != nil { flags |= 0x08 }
        if blockChecksums { flags |= 0x10 }
        if contentChecksum { flags |= 0x04 }
        if dictionaryID != nil { flags |= 0x01 }
        output += [flags, sizeCode << 4]
        if let contentSize { appendLE(contentSize, bytes: 8, to: &output) }
        if let dictionaryID { appendLE(UInt64(dictionaryID), bytes: 4, to: &output) }
        // パーサーのチェックサム非検証契約を確かめるため、任意の値を使う。
        output.append(0xA5)
        for block in blocks {
            appendLE(UInt64(block.bytes.count) | (block.stored ? 0x8000_0000 : 0), bytes: 4, to: &output)
            output += block.bytes
            if blockChecksums { output += [1, 2, 3, 4] }
        }
        output += [0, 0, 0, 0]
        if contentChecksum { output += [5, 6, 7, 8] }
        return output
    }

    static func overlap(offset: Int, matchLength: Int) -> (compressed: [UInt8], decoded: [UInt8]) {
        let literals = (0..<offset).map { UInt8($0 % 251) }
        var compressed = [UInt8(min(offset, 15) << 4 | min(matchLength - 4, 15))]
        func appendExtension(_ length: Int) {
            var remaining = length
            while remaining >= 255 { compressed.append(255); remaining -= 255 }
            compressed.append(UInt8(remaining))
        }
        if offset >= 15 { appendExtension(offset - 15) }
        compressed += literals
        appendLE(UInt64(offset), bytes: 2, to: &compressed)
        if matchLength - 4 >= 15 { appendExtension(matchLength - 4 - 15) }
        let tail: [UInt8] = [90, 91, 92, 93, 94]
        compressed.append(UInt8(tail.count << 4))
        compressed += tail
        var decoded = literals
        for index in 0..<matchLength { decoded.append(decoded[index]) }
        decoded += tail
        return (compressed, decoded)
    }
}
