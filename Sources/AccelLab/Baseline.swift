import CommonCrypto
import Compression
import Darwin
import Dispatch
import Foundation
import zlib

// macOS SDK には bz2 の Swift モジュールがないため、公開 C ABI を直接宣言する。
@_silgen_name("BZ2_bzBuffToBuffCompress")
private func bzCompress(
    _ destination: UnsafeMutableRawPointer, _ destinationLength: UnsafeMutablePointer<UInt32>,
    _ source: UnsafeRawPointer, _ sourceLength: UInt32,
    _ blockSize: Int32, _ verbosity: Int32, _ workFactor: Int32
) -> Int32

@_silgen_name("BZ2_bzBuffToBuffDecompress")
private func bzDecompress(
    _ destination: UnsafeMutableRawPointer, _ destinationLength: UnsafeMutablePointer<UInt32>,
    _ source: UnsafeRawPointer, _ sourceLength: UInt32, _ small: Int32, _ verbosity: Int32
) -> Int32

/// GPU・NPU の実装と比較する CPU の基準値を測定する。
public enum Baseline {
    public struct Measurement: Sendable {
        public let name: String
        public let bytes: Int
        public let medianSeconds: Double
        public var throughputGBps: Double { Double(bytes) / medianSeconds / 1e9 }
        public let note: String

        // テストから実際の全バイト比較の結果を確認する。非圧縮処理では nil。
        let roundTripMatchesInput: Bool?
    }

    private struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    // 計測前に確保・初期化し、Swift 配列のコピーや再確保を計測区間から除く。
    private final class Buffer {
        let count: Int
        let pointer: UnsafeMutablePointer<UInt8>

        init(count: Int) {
            self.count = count
            pointer = .allocate(capacity: count)
            pointer.initialize(repeating: 0, count: count)
        }

        deinit {
            pointer.deinitialize(count: count)
            pointer.deallocate()
        }
    }

    // 入力は不変、各レーンの書き込み先は独立。同期的な concurrentPerform の終了まで
    // 呼び出し元が全バッファを保持するので、この限定的な共有は安全である。
    private struct CRCChunks: @unchecked Sendable {
        let input: UnsafePointer<UInt8>
        let results: UnsafeMutablePointer<uLong>
        let chunkSize: Int

        func run(lane: Int) {
            results[lane] = crc32(0, input.advanced(by: lane * chunkSize), uInt(chunkSize))
        }
    }

    private static let corpusPath = "/private/tmp/claude-501/-Users-nagash-Github-KaitoFinder/404391ed-5c96-4f47-afa7-7911dab682d1/scratchpad/corpora/text256.txt"
    private static let fallbackText = """
        The morning light falls across the quiet library. A researcher opens a book,
        reads a story about rivers and mountains, and writes careful notes for a friend.
        Every experiment begins with a question. We measure the work, compare the results,
        and repeat the steps so that another person can understand what happened.

        """

    public static func run(sizeMiB: Int = 256, rounds: Int = 5) throws -> [Measurement] {
        // zlib・bzip2・CommonCrypto の一括 API が扱える長さに制限する。
        guard sizeMiB > 0, sizeMiB <= 2_047, rounds > 0 else {
            throw Failure(message: "sizeMiB must be in 1...2047 and rounds must be positive")
        }
        let byteCount = sizeMiB * 1_048_576
        let random = Buffer(count: byteCount)
        var seed: UInt64 = 0x4B61_6974_6F4C_6162
        for index in 0..<byteCount {
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            random.pointer[index] = UInt8(truncatingIfNeeded: seed)
        }
        let text = Buffer(count: byteCount)
        let textSource = try fillText(text)
        let output = Buffer(count: byteCount)
        var measurements: [Measurement] = []
        measurements.reserveCapacity(18)

        func record(_ name: String, _ seconds: Double, _ note: String) {
            measurements.append(Measurement(name: name, bytes: byteCount, medianSeconds: seconds,
                                            note: note, roundTripMatchesInput: nil))
        }

        let expectedCRC = crc32(0, random.pointer, uInt(byteCount))
        let crcSeconds = try median(rounds: rounds, operation: {
            crc32(0, random.pointer, uInt(byteCount))
        }, validate: { try require($0 == expectedCRC, "crc32-zlib mismatch") })
        record("crc32-zlib", crcSeconds, "random; single-thread; crc32=\(expectedCRC)")

        let table = (0..<256).map { index -> UInt32 in
            var value = UInt32(index)
            for _ in 0..<8 { value = (value >> 1) ^ (value & 1 == 0 ? 0 : 0xEDB8_8320) }
            return value
        }
        let tableSeconds = try table.withUnsafeBufferPointer { table in
            try median(rounds: rounds, operation: {
                tableCRC(random.pointer, count: byteCount, table: table.baseAddress!)
            }, validate: { try require(uLong($0) == expectedCRC, "crc32-table mismatch") })
        }
        record("crc32-table", tableSeconds, "random; single-thread; 8-bit table; crc32=\(expectedCRC)")

        let expectedAdler = adler32(1, random.pointer, uInt(byteCount))
        let adlerSeconds = try median(rounds: rounds, operation: {
            adler32(1, random.pointer, uInt(byteCount))
        }, validate: { try require($0 == expectedAdler, "adler32-zlib mismatch") })
        record("adler32-zlib", adlerSeconds, "random; single-thread; adler32=\(expectedAdler)")

        // 固定の鍵と IV はベンチマーク専用で、アプリケーションの暗号化には使わない。
        let key = Buffer(count: kCCKeySizeAES256)
        let iv = Buffer(count: kCCBlockSizeAES128)
        memcpy(key.pointer, random.pointer, key.count)
        memcpy(iv.pointer, random.pointer.advanced(by: key.count), iv.count)
        let ctrSeconds = try median(rounds: rounds, operation: {
            cryptCTR(input: random, output: output, key: key, iv: iv)
        }, validate: {
            try require($0.status == kCCSuccess && $0.count == byteCount, "AES-CTR failed")
        })
        // CTR は同じ鍵・IV でもう一度適用すると元に戻る。
        let recovered = Buffer(count: byteCount)
        let ctrResult = cryptCTR(input: output, output: recovered, key: key, iv: iv)
        try require(ctrResult.status == kCCSuccess && ctrResult.count == byteCount
                    && memcmp(random.pointer, recovered.pointer, byteCount) == 0, "AES-CTR round-trip failed")
        record("aes-ctr-commoncrypto", ctrSeconds,
               "random; single-thread; AES-256; CCCryptorCreateWithMode/Update/Final/Release (CCCrypt has no CTR)")

        let cbcSeconds = try median(rounds: rounds, operation: {
            var written = 0
            let status = CCCrypt(CCOperation(kCCDecrypt), CCAlgorithm(kCCAlgorithmAES), 0,
                                 key.pointer, key.count, iv.pointer, random.pointer, byteCount,
                                 output.pointer, output.count, &written)
            return (status, written)
        }, validate: {
            try require($0.0 == kCCSuccess && $0.1 == byteCount, "AES-CBC decrypt failed")
        })
        var cbcWritten = 0
        let cbcStatus = CCCrypt(CCOperation(kCCEncrypt), CCAlgorithm(kCCAlgorithmAES), 0,
                                key.pointer, key.count, iv.pointer, output.pointer, byteCount,
                                recovered.pointer, recovered.count, &cbcWritten)
        try require(cbcStatus == kCCSuccess && cbcWritten == byteCount
                    && memcmp(random.pointer, recovered.pointer, byteCount) == 0, "AES-CBC round-trip failed")
        record("aes-cbc-decrypt-commoncrypto", cbcSeconds, "random; single-thread; AES-256; CCCrypt; no padding")

        let digest = Buffer(count: Int(CC_SHA256_DIGEST_LENGTH))
        let expectedDigest = Buffer(count: digest.count)
        CC_SHA256(random.pointer, CC_LONG(byteCount), expectedDigest.pointer)
        let shaSeconds = try median(rounds: rounds, operation: {
            CC_SHA256(random.pointer, CC_LONG(byteCount), digest.pointer)
        }, validate: {
            try require($0 != nil && memcmp(digest.pointer, expectedDigest.pointer, digest.count) == 0,
                        "SHA-256 failed")
        })
        record("sha256-commoncrypto", shaSeconds, "random; single-thread; SHA-256")

        measurements += try compressionPair(
            compressName: "deflate6-zlib-compress", decompressName: "inflate-zlib", input: text,
            capacity: Int(compressBound(uLong(byteCount))), rounds: rounds, note: "\(textSource); level=6",
            encode: { destination, capacity in
                var length = uLongf(capacity)
                let status = compress2(destination, &length, text.pointer, uLong(byteCount), 6)
                return (status, Int(length))
            }, decode: { source, count, destination, capacity in
                var length = uLongf(capacity)
                let status = uncompress(destination, &length, source, uLong(count))
                return (status, Int(length))
            })

        measurements += try compressionPair(
            compressName: "bzip2-compress", decompressName: "bzip2-decompress", input: text,
            capacity: byteCount + byteCount / 100 + 601, rounds: rounds, note: "\(textSource); block=900k",
            encode: { destination, capacity in
                var length = UInt32(capacity)
                let status = bzCompress(destination, &length, text.pointer, UInt32(byteCount), 9, 0, 30)
                return (status, Int(length))
            }, decode: { source, count, destination, capacity in
                var length = UInt32(capacity)
                let status = bzDecompress(destination, &length, source, UInt32(count), 0, 0)
                return (status, Int(length))
            })

        for (name, algorithm) in [("lzfse", COMPRESSION_LZFSE), ("lz4", COMPRESSION_LZ4), ("lzma", COMPRESSION_LZMA)] {
            let scratchSize = max(compression_encode_scratch_buffer_size(algorithm),
                                  compression_decode_scratch_buffer_size(algorithm))
            let scratch = Buffer(count: max(1, scratchSize))
            measurements += try compressionPair(
                compressName: "\(name)-compress", decompressName: "\(name)-decompress", input: text,
                capacity: byteCount + byteCount / 4 + 65_536, rounds: rounds, note: textSource,
                encode: { destination, capacity in
                    let length = compression_encode_buffer(destination, capacity, text.pointer, byteCount,
                                                           scratch.pointer, algorithm)
                    return (length > 0 ? 0 : -1, length)
                }, decode: { source, count, destination, capacity in
                    let length = compression_decode_buffer(destination, capacity, source, count, scratch.pointer, algorithm)
                    return (length > 0 ? 0 : -1, length)
                })
        }

        let laneResults = UnsafeMutablePointer<uLong>.allocate(capacity: 16)
        laneResults.initialize(repeating: 0, count: 16)
        defer { laneResults.deinitialize(count: 16); laneResults.deallocate() }
        let chunks = CRCChunks(input: UnsafePointer(random.pointer), results: laneResults, chunkSize: byteCount / 16)
        let laneOperation: @Sendable (Int) -> Void = { chunks.run(lane: $0) }
        let parallelSeconds = try median(rounds: rounds, operation: {
            DispatchQueue.concurrentPerform(iterations: 16, execute: laneOperation)
            var combined = laneResults[0]
            for lane in 1..<16 {
                combined = crc32_combine(combined, laneResults[lane], chunks.chunkSize)
            }
            return combined
        }, validate: { try require($0 == expectedCRC, "parallel CRC-32 mismatch") })
        record("parallel-crc32-zlib-16", parallelSeconds, "random; 16 equal chunks; crc32_combine included; crc32=\(expectedCRC)")

        let copySeconds = try median(rounds: rounds, operation: {
            copyBytes(from: random.pointer, to: output.pointer, count: byteCount)
        }, validate: { _ in
            try require(memcmp(random.pointer, output.pointer, byteCount) == 0, "memcpy mismatch")
        })
        record("memcpy", copySeconds, "random; single-thread; bytes=copied bytes (read+write traffic is 2x)")
        return measurements
    }

    private static func fillText(_ buffer: Buffer) throws -> String {
        let corpus: Data
        let source: String
        if FileManager.default.fileExists(atPath: corpusPath) {
            let file = try FileHandle(forReadingFrom: URL(fileURLWithPath: corpusPath))
            defer { try? file.close() }
            corpus = try file.read(upToCount: buffer.count) ?? Data()
            try require(!corpus.isEmpty, "Text corpus is empty: \(corpusPath)")
            source = "text=\(corpusPath)"
        } else {
            corpus = Data(fallbackText.utf8)
            source = "text=generated-English"
        }
        corpus.withUnsafeBytes { bytes in
            var offset = 0
            while offset < buffer.count {
                let count = min(bytes.count, buffer.count - offset)
                memcpy(buffer.pointer.advanced(by: offset), bytes.baseAddress!, count)
                offset += count
            }
        }
        return source
    }

    // ウォームアップと検証、配列確保、並べ替えはすべて計測区間の外で行う。
    private static func median<Result>(
        rounds: Int, operation: () -> Result, validate: (Result) throws -> Void
    ) throws -> Double {
        try validate(operation())
        let clock = ContinuousClock()
        var seconds = [Double](repeating: 0, count: rounds)
        for round in 0..<rounds {
            let start = clock.now
            let result = operation()
            let elapsed = start.duration(to: clock.now).components
            try validate(result)
            seconds[round] = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
            try require(seconds[round] > 0, "Clock did not advance during measurement")
        }
        seconds.sort()
        let middle = rounds / 2
        return rounds.isMultiple(of: 2) ? (seconds[middle - 1] + seconds[middle]) / 2 : seconds[middle]
    }

    @inline(never)
    private static func tableCRC(_ input: UnsafePointer<UInt8>, count: Int, table: UnsafePointer<UInt32>) -> UInt32 {
        var value: UInt32 = 0xFFFF_FFFF
        for index in 0..<count {
            value = table[Int((value ^ UInt32(input[index])) & 0xFF)] ^ (value >> 8)
        }
        return value ^ 0xFFFF_FFFF
    }

    @inline(never)
    private static func copyBytes(from input: UnsafePointer<UInt8>, to output: UnsafeMutablePointer<UInt8>, count: Int) {
        memcpy(output, input, count)
    }

    // CCCrypt は CTR モードを公開しないため、同等の一括処理をモード指定 API で行う。
    private static func cryptCTR(input: Buffer, output: Buffer, key: Buffer, iv: Buffer) -> (status: Int32, count: Int) {
        var cryptor: CCCryptorRef?
        let created = CCCryptorCreateWithMode(CCOperation(kCCEncrypt), CCMode(kCCModeCTR), CCAlgorithm(kCCAlgorithmAES),
                                             CCPadding(ccNoPadding), iv.pointer, key.pointer, key.count,
                                             nil, 0, 0, CCModeOptions(kCCModeOptionCTR_BE), &cryptor)
        guard created == kCCSuccess, let cryptor else { return (created, 0) }
        defer { CCCryptorRelease(cryptor) }
        var written = 0
        let updated = CCCryptorUpdate(cryptor, input.pointer, input.count, output.pointer, output.count, &written)
        guard updated == kCCSuccess else { return (updated, written) }
        var finalWritten = 0
        let finalized = CCCryptorFinal(cryptor, output.pointer.advanced(by: written), output.count - written, &finalWritten)
        return (finalized, written + finalWritten)
    }

    private static func compressionPair(
        compressName: String, decompressName: String, input: Buffer, capacity: Int, rounds: Int, note: String,
        encode: (UnsafeMutablePointer<UInt8>, Int) -> (Int32, Int),
        decode: (UnsafePointer<UInt8>, Int, UnsafeMutablePointer<UInt8>, Int) -> (Int32, Int)
    ) throws -> [Measurement] {
        let compressed = Buffer(count: capacity)
        // 余分な 1 バイトで、出力打ち切りを正常な復元と取り違えないようにする。
        let restored = Buffer(count: input.count + 1)
        var compressedCount = 0
        let encodeSeconds = try median(rounds: rounds, operation: {
            encode(compressed.pointer, compressed.count)
        }, validate: { status, count in
            try require(status == 0 && count > 0 && count <= compressed.count, "\(compressName) failed (status \(status))")
            compressedCount = count
            // 各圧縮結果を実際に復元し、計測の外で全バイトを比較する。
            let decoded = decode(compressed.pointer, count, restored.pointer, restored.count)
            try require(decoded.0 == 0 && decoded.1 == input.count
                        && memcmp(input.pointer, restored.pointer, input.count) == 0, "\(compressName) round-trip failed")
        })
        let decodeSeconds = try median(rounds: rounds, operation: {
            decode(compressed.pointer, compressedCount, restored.pointer, restored.count)
        }, validate: { status, count in
            try require(status == 0 && count == input.count
                        && memcmp(input.pointer, restored.pointer, input.count) == 0, "\(decompressName) round-trip failed")
        })
        let ratio = String(format: "%.6f", locale: Locale(identifier: "en_US_POSIX"), Double(compressedCount) / Double(input.count))
        let details = "\(note); single-thread; compressed_bytes=\(compressedCount); ratio=\(ratio) (compressed/original); bytes=original; round-trip=OK"
        return [
            Measurement(name: compressName, bytes: input.count, medianSeconds: encodeSeconds, note: details, roundTripMatchesInput: true),
            Measurement(name: decompressName, bytes: input.count, medianSeconds: decodeSeconds, note: details, roundTripMatchesInput: true),
        ]
    }

    private static func require(_ condition: Bool, _ message: @autoclosure () -> String) throws {
        if !condition { throw Failure(message: message()) }
    }
}
