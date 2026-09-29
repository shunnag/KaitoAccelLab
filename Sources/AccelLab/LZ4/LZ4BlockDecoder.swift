private import Compression

enum LZ4BlockDecoder {
    private static let extendedLength = 15
    private static let extensionContinuation = 255
    private static let minimumMatchLength = 4

    static func decodeSwift(
        _ src: UnsafeRawBufferPointer, into dst: UnsafeMutableRawBufferPointer
    ) throws -> Int {
        try scan(src, into: dst, limit: dst.count)
    }

    static func decodeApple(
        _ src: UnsafeRawBufferPointer, into dst: UnsafeMutableRawBufferPointer
    ) -> Int {
        guard let source = src.baseAddress, let destination = dst.baseAddress,
              !src.isEmpty, !dst.isEmpty else { return 0 }
        return compression_decode_buffer(
            destination.assumingMemoryBound(to: UInt8.self), dst.count,
            source.assumingMemoryBound(to: UInt8.self), src.count, nil, COMPRESSION_LZ4_RAW)
    }

    // 書き込みを省略しても、参照距離と長さの検証は復元時と同じにする。
    static func decodedLength(_ src: UnsafeRawBufferPointer, limit: Int) throws -> Int {
        try scan(src, into: nil, limit: limit)
    }

    private static func scan(
        _ src: UnsafeRawBufferPointer, into dst: UnsafeMutableRawBufferPointer?, limit: Int
    ) throws -> Int {
        var cursor = 0
        var written = 0
        func readLength(_ initial: Int) throws -> Int {
            var length = initial
            if initial == extendedLength {
                while true {
                    guard cursor < src.count else { throw LZ4Error.truncatedInput }
                    let byte = Int(src[cursor])
                    cursor += 1
                    let sum = length.addingReportingOverflow(byte)
                    guard !sum.overflow else { throw LZ4Error.sizeOverflow }
                    length = sum.partialValue
                    if byte != extensionContinuation { break }
                }
            }
            return length
        }

        while true {
            guard cursor < src.count else { throw LZ4Error.truncatedInput }
            let token = Int(src[cursor])
            cursor += 1
            let literals = try readLength(token >> 4)
            guard literals <= src.count - cursor else { throw LZ4Error.truncatedInput }
            guard literals <= limit - written else { throw LZ4Error.outputTooSmall }
            if let dst, literals > 0 {
                dst.baseAddress!.advanced(by: written).copyMemory(
                    from: src.baseAddress!.advanced(by: cursor), byteCount: literals)
            }
            cursor += literals
            written += literals
            if cursor == src.count { return written }

            guard src.count - cursor >= 2 else { throw LZ4Error.truncatedInput }
            let offset = Int(src[cursor]) | (Int(src[cursor + 1]) << 8)
            cursor += 2
            guard offset > 0, offset <= written else { throw LZ4Error.invalidOffset }
            let match = try readLength(token & extendedLength)
            guard limit - written >= minimumMatchLength,
                  match <= limit - written - minimumMatchLength else { throw LZ4Error.outputTooSmall }
            let length = match + minimumMatchLength
            if let dst {
                // 重なる一致列は前方から逐次コピーする。
                for index in 0..<length { dst[written + index] = dst[written + index - offset] }
            }
            written += length
        }
    }
}
