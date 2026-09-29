internal import XCTest
@testable import AccelLab

final class BaselineTests: XCTestCase {
    func testBaselineThroughputAndCompressionRoundTrips() throws {
        let measurements = try Baseline.run(sizeMiB: 4, rounds: 1)
        let compressionNames: Set<String> = [
            "deflate6-zlib-compress", "inflate-zlib", "bzip2-compress", "bzip2-decompress",
            "lzfse-compress", "lzfse-decompress", "lz4-compress", "lz4-decompress",
            "lzma-compress", "lzma-decompress",
        ]
        let expectedNames = compressionNames.union([
            "crc32-zlib", "crc32-table", "adler32-zlib", "aes-ctr-commoncrypto",
            "aes-cbc-decrypt-commoncrypto", "sha256-commoncrypto", "parallel-crc32-zlib-16", "memcpy",
        ])
        XCTAssertEqual(measurements.count, expectedNames.count)
        XCTAssertEqual(Set(measurements.map(\.name)), expectedNames)
        for measurement in measurements {
            XCTAssertEqual(measurement.bytes, 4 * 1_048_576, measurement.name)
            XCTAssertGreaterThan(measurement.medianSeconds, 0, measurement.name)
            XCTAssertTrue(measurement.throughputGBps.isFinite, measurement.name)
            XCTAssertGreaterThan(measurement.throughputGBps, 0, measurement.name)
            XCTAssertEqual(measurement.throughputGBps, Double(measurement.bytes) / measurement.medianSeconds / 1e9)
            if compressionNames.contains(measurement.name) {
                // run 内で各回の復元結果を入力と全バイト比較した事実を検証する。
                XCTAssertEqual(measurement.roundTripMatchesInput, true, measurement.name)
                XCTAssertTrue(measurement.note.contains("ratio="), measurement.name)
            } else {
                XCTAssertNil(measurement.roundTripMatchesInput, measurement.name)
            }
        }
    }
}
