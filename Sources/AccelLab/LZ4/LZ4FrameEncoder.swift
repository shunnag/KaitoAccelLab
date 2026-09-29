internal import Foundation
private import Compression

package enum LZ4FrameEncoder {
    static let minimumBlockSize = 4_096
    static let maximumBlockSize = 4_194_304
    private static let firstSizeCode = 4
    private static let independentContentSizeFlags: UInt8 = 0x68
    private static let storedMask: UInt64 = 0x8000_0000
    private static let encoderHeadroom = 65_536

    package static func encodeFile(
        inputPath: String, outputPath: String, blockSize: Int
    ) throws -> (inputBytes: Int, outputBytes: Int, blockCount: Int) {
        try validate(blockSize: blockSize)
        let input = try Data(contentsOf: URL(fileURLWithPath: inputPath))
        let output = try encodeFrame(Array(input), blockSize: blockSize)
        try output.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
        let blocks = input.count / blockSize + (input.count.isMultiple(of: blockSize) ? 0 : 1)
        return (input.count, output.count, blocks)
    }

    // 独立ブロックと内容サイズを記録し、必須のヘッダーチェックサムだけを付ける。
    static func encodeFrame(_ input: [UInt8], blockSize: Int) throws -> Data {
        try validate(blockSize: blockSize)
        let sizeIndex = LZ4Frame.blockSizes.firstIndex { $0 >= blockSize }!
        var header: [UInt8] = [independentContentSizeFlags, UInt8(sizeIndex + firstSizeCode) << 4]
        appendLE(UInt64(input.count), bytes: 8, to: &header)
        var output: [UInt8] = []
        appendLE(UInt64(LZ4Frame.magic), bytes: 4, to: &output)
        output += header
        output.append(headerChecksum(header))
        var compressed = [UInt8](repeating: 0, count: blockSize + blockSize / 255 + encoderHeadroom)
        var scratch = [UInt8](repeating: 0, count: max(1, compression_encode_scratch_buffer_size(COMPRESSION_LZ4_RAW)))
        try input.withUnsafeBufferPointer { source in
            try compressed.withUnsafeMutableBufferPointer { destination in
                try scratch.withUnsafeMutableBytes { workspace in
                    for offset in stride(from: 0, to: input.count, by: blockSize) {
                        let count = min(blockSize, input.count - offset)
                        let encoded = compression_encode_buffer(
                            destination.baseAddress!, destination.count, source.baseAddress!.advanced(by: offset),
                            count, workspace.baseAddress, COMPRESSION_LZ4_RAW)
                        guard encoded > 0 else { throw LZ4Error.invalidArgument("Apple LZ4 encoding failed") }
                        let stored = encoded >= count
                        let payloadCount = stored ? count : encoded
                        appendLE(UInt64(payloadCount) | (stored ? storedMask : 0), bytes: 4, to: &output)
                        if stored { output.append(contentsOf: source[offset..<(offset + count)]) }
                        else { output.append(contentsOf: destination[..<encoded]) }
                    }
                }
            }
        }
        appendLE(0, bytes: 4, to: &output)
        return Data(output)
    }

    private static func validate(blockSize: Int) throws {
        guard (minimumBlockSize...maximumBlockSize).contains(blockSize) else { throw LZ4Error.invalidBlockSize }
    }

    private static func appendLE(_ value: UInt64, bytes count: Int, to output: inout [UInt8]) {
        for index in 0..<count { output.append(UInt8(truncatingIfNeeded: value >> (index * 8))) }
    }

    // この固定ヘッダーは 16 バイト未満なので、XXH32 の短い入力の経路だけを使う。
    private static func headerChecksum(_ header: [UInt8]) -> UInt8 {
        let prime1: UInt32 = 2_654_435_761
        let prime2: UInt32 = 2_246_822_519
        let prime3: UInt32 = 3_266_489_917
        let prime4: UInt32 = 668_265_263
        let prime5: UInt32 = 374_761_393
        func rotate(_ value: UInt32, _ bits: UInt32) -> UInt32 { (value << bits) | (value >> (32 - bits)) }
        var hash = prime5 &+ UInt32(header.count)
        var index = 0
        while index + 4 <= header.count {
            var word: UInt32 = 0
            for byte in 0..<4 { word |= UInt32(header[index + byte]) << (byte * 8) }
            hash = rotate(hash &+ word &* prime3, 17) &* prime4
            index += 4
        }
        while index < header.count {
            hash = rotate(hash &+ UInt32(header[index]) &* prime5, 11) &* prime1
            index += 1
        }
        hash ^= hash >> 15
        hash &*= prime2
        hash ^= hash >> 13
        hash &*= prime3
        hash ^= hash >> 16
        return UInt8(truncatingIfNeeded: hash >> 8)
    }
}
