import Foundation
import Metal
import XCTest
@testable import GyoshukuKit

final class LZMABucketMetalTests: XCTestCase {
    func testBucketedTreeMatchesSequential() throws {
        let input = LZMAEncoderCorpus.text(size: 64 << 10)
        XCTAssertTrue(try assertMatches(input: input, dictionary: input.count))
    }

    func testDictionaryWrap() throws {
        let input = LZMAEncoderCorpus.text(size: 20_000)
        XCTAssertTrue(try assertMatches(input: input, dictionary: 4_096))
    }

    func testMetalTraversalBeforeDictionaryWrapFailure() throws {
        let input = LZMAEncoderCorpus.text(size: 20_000)
        let bytes = [UInt8](input)
        let properties = LZMAEncoderProperties.preset(6)
        let dictionary = 4_096, position = 4_171
        var reference = try LZMAMatchFinder(properties: properties, dictionary: dictionary)
        defer { reference.release() }
        let matches = UnsafeMutablePointer<LZMAMatch>.allocate(capacity: 274)
        defer { matches.deallocate() }
        var candidate: UInt32 = 0, expectedSon: [UInt32] = []
        bytes.withUnsafeBufferPointer { buffer in
            let base = buffer.baseAddress!
            for index in 0..<position {
                _ = reference.matches(base + index, available: input.count - index,
                    into: matches, extendMatches: false, recordShortMatches: false)
            }
            let word = UInt32(littleEndian: UnsafeRawPointer(base + position).loadUnaligned(as: UInt32.self))
            let crc0 = reference.crc[Int(word & 255)]
            let h4 = 1024 + 65536 + Int(((crc0 ^ ((word >> 8) & 65535))
                ^ (reference.crc[Int(word >> 24)] << 5)) & reference.hashMask)
            candidate = reference.hash[h4]
            expectedSon = Array(UnsafeBufferPointer(start: reference.son, count: reference.cyclicSize * 2))
        }
        let actual = try LZMABucketMetalEncoder(properties: properties, dictionary: dictionary).encode(input, through: position)
        func trace(_ son: [UInt32]) -> String {
            var candidate = candidate, len0 = 0, len1 = 0, result: [String] = []
            for _ in 0..<reference.cutValue where candidate != 0 {
                let delta = position + 1 - Int(candidate)
                if delta > dictionary { break }
                let index = (position % reference.cyclicSize - delta + reference.cyclicSize) % reference.cyclicSize
                let c0 = son[index * 2], c1 = son[index * 2 + 1]
                var length = min(len0, len1)
                let limit = min(input.count - position, reference.niceLen)
                while length < limit && bytes[position + length] == bytes[position - delta + length] { length += 1 }
                result.append("candidate=\(candidate) length=\(length) c0=\(c0) c1=\(c1)")
                if length == limit { break }
                if bytes[position - delta + length] < bytes[position + length] {
                    candidate = c1; len1 = length
                } else {
                    candidate = c0; len0 = length
                }
            }
            return result.joined(separator: "\n")
        }
        XCTAssertEqual(candidate, 3_444)
        XCTAssertEqual(trace(actual.son), trace(expectedSon))
    }

    func testShortInputs() throws {
        for count in [0, 1, 3, 4, 7, 64] {
            let input = Data(LZMAEncoderCorpus.text(size: 64).prefix(count))
            XCTAssertTrue(try assertMatches(input: input, dictionary: max(4, input.count)))
        }
    }

    func testLargeBenchmark() throws {
        try OptInGate.flag("GYOSHUKU_LZMA_GPU_BENCHMARK")
        #if DEBUG
        XCTFail("Run LZMABucketMetalTests with -c release")
        #else
        let input = LZMAEncoderCorpus.text(size: 4 << 20)
        let properties = LZMAEncoderProperties.preset(6)
        let dictionary = min(properties.dictSize, input.count)
        let repeats = max(1, Int(ProcessInfo.processInfo.environment["GYOSHUKU_LZMA_GPU_BENCHMARK_REPEATS"] ?? "3") ?? 3)
        var wall = Double.infinity, baseline = Double.infinity, finderBaseline = Double.infinity
        var bucketSeconds = 0.0, gpuSeconds = 0.0
        for _ in 0..<repeats {
            let encoder = try LZMABucketMetalEncoder(properties: properties, dictionary: dictionary)
            let start = ProcessInfo.processInfo.systemUptime
            let result = try encoder.encode(input)
            wall = min(wall, ProcessInfo.processInfo.systemUptime - start)
            bucketSeconds = result.bucketSeconds
            gpuSeconds = result.gpuSeconds
            let baselineStart = ProcessInfo.processInfo.systemUptime
            let encoded = try LZMAEncoder.encode(input, properties: properties)
            baseline = min(baseline, ProcessInfo.processInfo.systemUptime - baselineStart)
            XCTAssertFalse(encoded.isEmpty)
            do {
                var finder = try LZMAMatchFinder(properties: properties, dictionary: dictionary)
                defer { finder.release() }
                let capacity = input.count * finder.cutValue
                let matches = UnsafeMutablePointer<LZMAMatch>.allocate(capacity: capacity)
                defer { matches.deallocate() }
                let finderStart = ProcessInfo.processInfo.systemUptime
                input.withUnsafeBytes { bytes in
                    let base = bytes.baseAddress!.assumingMemoryBound(to: UInt8.self)
                    for index in 0..<input.count {
                        _ = finder.matches(base + index, available: input.count - index,
                            into: matches + index * finder.cutValue, extendMatches: false, recordShortMatches: false)
                    }
                }
                finderBaseline = min(finderBaseline, ProcessInfo.processInfo.systemUptime - finderStart)
                XCTAssertGreaterThanOrEqual(matches[input.count / 2 * finder.cutValue].length, 0)
            }
        }
        TestSupport.report("LZMA-GPU-BENCH\tbucket_s=\(bucketSeconds)\tgpu_s=\(gpuSeconds)\twall_s=\(wall)\tfinder_s=\(finderBaseline)\tfinder_speedup=\(finderBaseline / wall)\traw_s=\(baseline)\traw_speedup=\(baseline / wall)")
        XCTAssertTrue(try assertMatches(input: input, dictionary: dictionary))
        #endif
    }

    func testEveryPositionMatchesSequential() throws {
        let input = LZMAEncoderCorpus.text(size: 600)
        let properties = LZMAEncoderProperties.preset(6)
        let dictionary = 4_096
        let encoder = try LZMABucketMetalEncoder(properties: properties, dictionary: dictionary)
        let matches = UnsafeMutablePointer<LZMAMatch>.allocate(capacity: 274)
        defer { matches.deallocate() }
        try input.withUnsafeBytes { bytes in
            let base = bytes.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for index in 0..<input.count {
                var reference = try LZMAMatchFinder(properties: properties, dictionary: dictionary)
                defer { reference.release() }
                for position in 0...index {
                    _ = reference.matches(base + position, available: index + 1 - position,
                        into: matches, extendMatches: false, recordShortMatches: false)
                }
                do {
                    encoder.reset()
                    let actual = try encoder.encode(Data(input.prefix(index + 1)))
                    let expectedSon = Array(UnsafeBufferPointer(start: reference.son, count: reference.cyclicSize * 2))
                    let expectedHash = Array(UnsafeBufferPointer(start: reference.hash, count: reference.hashCount))
                    if actual.son != expectedSon || actual.hash != expectedHash {
                        let sonDiff = zip(actual.son, expectedSon).enumerated().first { $0.element.0 != $0.element.1 }
                        let hashDiff = zip(actual.hash, expectedHash).enumerated().first { $0.element.0 != $0.element.1 }
                        XCTFail("state at \(index), son=\(String(describing: sonDiff)), hash=\(String(describing: hashDiff))")
                        return
                    }
                } catch {
                    XCTFail("error at \(index): \(error)")
                    return
                }
            }
        }
    }

    func testCPUBucketedTreeMatchesSequential() throws {
        XCTAssertTrue(try assertCPUBucketMatches(input: LZMAEncoderCorpus.text(size: 64 << 10), dictionary: 64 << 10))
        XCTAssertTrue(try assertCPUBucketMatches(input: LZMAEncoderCorpus.text(size: 20_000), dictionary: 4_096))
    }

    private func assertCPUBucketMatches(input: Data, dictionary: Int) throws -> Bool {
        let properties = LZMAEncoderProperties.preset(6)
        var reference = try LZMAMatchFinder(properties: properties, dictionary: dictionary)
        defer { reference.release() }
        var bucketed = try LZMAMatchFinder(properties: properties, dictionary: dictionary)
        defer { bucketed.release() }
        var buckets: [Int: [Int]] = [:]
        var firstCandidates: [Int: UInt32] = [:]
        var shortHeads: [UInt32] = []
        var bucketHashes: [Int] = []
        input.withUnsafeBytes { bytes in
            let base = bytes.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for index in 0..<max(0, input.count - 3) {
                let word = UInt32(littleEndian: UnsafeRawPointer(base + index).loadUnaligned(as: UInt32.self))
                let crc0 = bucketed.crc[Int(word & 255)]
                let h2 = Int((crc0 ^ ((word >> 8) & 255)) & 1023)
                let h3 = 1024 + Int((crc0 ^ ((word >> 8) & 65535)) & 65535)
                let h4 = 1024 + 65536 + Int(((crc0 ^ ((word >> 8) & 65535))
                    ^ (bucketed.crc[Int(word >> 24)] << 5)) & bucketed.hashMask)
                let bucket = h4 - 1024 - 65536
                if firstCandidates[bucket] == nil { firstCandidates[bucket] = bucketed.hash[h4] }
                buckets[bucket, default: []].append(index)
                shortHeads.append(bucketed.hash[h2])
                shortHeads.append(bucketed.hash[h3])
                bucketHashes.append(contentsOf: [h2, h3, h4])
                bucketed.hash[h2] = UInt32(index + 1)
                bucketed.hash[h3] = UInt32(index + 1)
                bucketed.hash[h4] = UInt32(index + 1)
            }
        }
        let capacity = input.count * reference.cutValue
        let expected = UnsafeMutablePointer<LZMAMatch>.allocate(capacity: capacity)
        let actual = UnsafeMutablePointer<LZMAMatch>.allocate(capacity: capacity)
        defer { expected.deallocate(); actual.deallocate() }
        var expectedCounts = [Int](repeating: 0, count: input.count)
        var matched = true
        input.withUnsafeBytes { bytes in
            let base = bytes.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for index in 0..<input.count {
                expectedCounts[index] = reference.matches(base + index, available: input.count - index,
                    into: expected + index * reference.cutValue, extendMatches: false, recordShortMatches: false)
            }
            for bucket in buckets.keys.sorted() {
                for (offset, index) in buckets[bucket]!.enumerated() {
                    let h2 = bucketHashes[index * 3], h3 = bucketHashes[index * 3 + 1]
                    if offset == 0 { bucketed.hash[bucketHashes[index * 3 + 2]] = firstCandidates[bucket]! }
                    bucketed.hash[h2] = shortHeads[index * 2]
                    bucketed.hash[h3] = shortHeads[index * 2 + 1]
                    bucketed.position = UInt32(index + 1)
                    bucketed.cyclic = index % bucketed.cyclicSize
                    let count = bucketed.matches(base + index, available: input.count - index,
                        into: actual + index * reference.cutValue, extendMatches: false, recordShortMatches: false)
                    guard count == expectedCounts[index] else {
                        XCTFail("CPU count at \(index): \(count) != \(expectedCounts[index])")
                        matched = false
                        return
                    }
                    for matchIndex in 0..<count {
                        let start = index * reference.cutValue + matchIndex
                        guard actual[start].length == expected[start].length,
                              actual[start].distance == expected[start].distance else {
                            XCTFail("CPU match at \(index):\(matchIndex): actual=(\(actual[start].length),\(actual[start].distance)) expected=(\(expected[start].length),\(expected[start].distance))")
                            matched = false
                            return
                        }
                    }
                }
            }
        }
        return matched
    }

    private func assertMatches(input: Data, dictionary: Int) throws -> Bool {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal is unavailable") }
        let properties = LZMAEncoderProperties.preset(6)
        var reference = try LZMAMatchFinder(properties: properties, dictionary: dictionary)
        defer { reference.release() }
        let encoder = try LZMABucketMetalEncoder(properties: properties, dictionary: dictionary)
        let actual = try encoder.encode(input)
        let capacity = input.count * reference.cutValue
        let matches = UnsafeMutablePointer<LZMAMatch>.allocate(capacity: capacity)
        matches.initialize(repeating: LZMAMatch(length: 0, distance: 0), count: capacity)
        defer {
            matches.deinitialize(count: capacity)
            matches.deallocate()
        }
        var counts = [UInt32](repeating: 0, count: input.count)
        input.withUnsafeBytes { bytes in
            let base = bytes.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for index in 0..<input.count {
                counts[index] = UInt32(reference.matches(base + index, available: input.count - index,
                    into: matches + index * reference.cutValue, extendMatches: false, recordShortMatches: false))
            }
        }
        guard actual.counts == counts else {
            let index = (0..<input.count).first { actual.counts[$0] != counts[$0] }!
            let start = index * reference.cutValue
            XCTFail("count at \(index): \(actual.counts[index]) != \(counts[index]); actual extra=(\(actual.matches[start + 2].length),\(actual.matches[start + 2].distance)); expected first=(\(matches[start + 2].length),\(matches[start + 2].distance)); expected next=(\(matches[start + 3].length),\(matches[start + 3].distance)); candidate=\(actual.candidates[index])")
            return false
        }
        for index in 0..<input.count {
            let count = Int(counts[index])
            let start = index * reference.cutValue
            for offset in 0..<count {
                let expected = matches[start + offset]
                let actualMatch = actual.matches[start + offset]
                guard actualMatch.length == expected.length, actualMatch.distance == expected.distance else {
                    XCTFail("match at \(index):\(offset) actual=(\(actualMatch.length),\(actualMatch.distance)) expected=(\(expected.length),\(expected.distance)) candidate=\(actual.candidates[index])")
                    return false
                }
            }
        }
        XCTAssertEqual(actual.son, Array(UnsafeBufferPointer(start: reference.son, count: reference.cyclicSize * 2)))
        XCTAssertEqual(actual.hash, Array(UnsafeBufferPointer(start: reference.hash, count: reference.hashCount)))
        return true
    }
}
