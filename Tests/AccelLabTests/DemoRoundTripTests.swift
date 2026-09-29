internal import Foundation
internal import XCTest
@testable import AccelLab

final class DemoRoundTripTests: XCTestCase {
    private static let textBytes = 300 * 1_024
    private static let neuralBlocks = 8
    private static let lz4BlockSize = 4_096

    func testOrderOneRoundTrip() throws {
        try roundTrip(pack: .neural(OrderOnePredictor(), blocks: Self.neuralBlocks, predictorTag: "order1"),
                      unpack: .neural(OrderOnePredictor(), predictorTag: "order1"), kind: .neural)
    }

    func testLZ4CPURoundTrip() throws {
        try roundTrip(pack: .lz4(blockSize: Self.lz4BlockSize), unpack: .cpu, kind: .lz4)
    }

    func testContainerHeaderAndMalformedInputs() throws {
        let payload = Data([10, 20, 30])
        for kind in [DemoContainer.Kind.neural, .lz4] {
            let encoded = DemoContainer(kind: kind, payload: payload).encode()
            XCTAssertEqual(Array(encoded), [75, 65, 68, 77, 1, kind.rawValue, 10, 20, 30])
            let parsed = try DemoContainer.decode(encoded)
            XCTAssertEqual(parsed.kind, kind)
            XCTAssertEqual(parsed.payload, payload)
            // 開始インデックスがゼロでない Data もヘッダーとして読める。
            let slice = Data([0] + Array(encoded)).dropFirst()
            XCTAssertEqual(try DemoContainer.decode(slice).payload, payload)
            for length in 0...6 {
                XCTAssertThrowsError(try DemoContainer.decode(Data(encoded.prefix(length))))
            }
            for (offset, value): (Int, UInt8) in [(0, 0), (4, 0), (4, 2), (5, 0), (5, 3), (5, 255)] {
                var bytes = Array(encoded)
                bytes[offset] = value
                XCTAssertThrowsError(try DemoContainer.decode(Data(bytes)))
            }
        }
    }

    func testEmptyDirectoryRoundTrip() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let input = root.appendingPathComponent("input")
        let output = root.appendingPathComponent("output")
        let archive = root.appendingPathComponent("empty.kadm")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        let packed = try DemoPacker.pack(directory: input, output: archive, method: .lz4(blockSize: Self.lz4BlockSize))
        let unpacked = try DemoUnpacker.unpack(archive: archive, into: output, method: .cpu)
        XCTAssertEqual(packed.entries, 0)
        XCTAssertEqual(unpacked.entries, 0)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: output.path), [])
    }

    func testSkipsSymlinksAndOutputInsideInput() throws {
        let manager = FileManager.default
        let root = try temporaryDirectory()
        defer { try? manager.removeItem(at: root) }
        let input = root.appendingPathComponent("input")
        let output = root.appendingPathComponent("output")
        try manager.createDirectory(at: input.appendingPathComponent("empty-dir"), withIntermediateDirectories: true)
        try Data("hidden text".utf8).write(to: input.appendingPathComponent(".hidden"))
        try manager.createSymbolicLink(at: input.appendingPathComponent("file-link"),
                                       withDestinationURL: input.appendingPathComponent(".hidden"))
        try manager.createSymbolicLink(at: input.appendingPathComponent("directory-link"), withDestinationURL: input)
        let archive = input.appendingPathComponent("output.kadm")
        try Data("previous archive".utf8).write(to: archive)
        let packed = try DemoPacker.pack(directory: input, output: archive, method: .lz4(blockSize: Self.lz4BlockSize))
        let unpacked = try DemoUnpacker.unpack(archive: archive, into: output, method: .cpu)
        XCTAssertEqual(packed.entries, 2)
        XCTAssertEqual(unpacked.entries, 2)
        let tree = try tree(at: output)
        XCTAssertEqual(tree.directories, ["empty-dir"])
        XCTAssertEqual(tree.files, [".hidden": Data("hidden text".utf8)])
        XCTAssertFalse(try manager.contentsOfDirectory(atPath: input.path).contains { $0.hasPrefix(".kadm-") })
    }

    func testTemporaryTarRemovedOnCompressionAndWriteErrors() throws {
        let manager = FileManager.default
        let root = try temporaryDirectory()
        defer { try? manager.removeItem(at: root) }
        let input = root.appendingPathComponent("input")
        let archive = root.appendingPathComponent("output.kadm")
        try manager.createDirectory(at: input, withIntermediateDirectories: true)
        let original = Set(try manager.contentsOfDirectory(atPath: root.path))
        XCTAssertThrowsError(try DemoPacker.pack(directory: input, output: archive, method: .lz4(blockSize: 0)))
        XCTAssertEqual(Set(try manager.contentsOfDirectory(atPath: root.path)), original)
        try manager.createDirectory(at: archive, withIntermediateDirectories: true)
        XCTAssertThrowsError(try DemoPacker.pack(directory: input, output: archive, method: .lz4(blockSize: Self.lz4BlockSize)))
        XCTAssertEqual(Set(try manager.contentsOfDirectory(atPath: root.path)), original.union(["output.kadm"]))
    }

    func testWrongMethodDoesNotCreateExtractionDirectory() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let archive = root.appendingPathComponent("input.kadm")
        let output = root.appendingPathComponent("output")
        let payload = try NeuralBlockCodec.encode(data: Data(), blockCount: 1,
                                                  predictor: OrderOnePredictor(), predictorTag: "order1")
        try DemoContainer(kind: .neural, payload: payload).encode().write(to: archive)
        XCTAssertThrowsError(try DemoUnpacker.unpack(archive: archive, into: output, method: .cpu))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
    }

    private func roundTrip(pack: DemoPacker.Method, unpack: DemoUnpacker.UnpackMethod,
                           kind: DemoContainer.Kind) throws {
        let manager = FileManager.default
        let root = try temporaryDirectory()
        defer { try? manager.removeItem(at: root) }
        let input = root.appendingPathComponent("input")
        let output = root.appendingPathComponent("output")
        let archive = root.appendingPathComponent("archive.kadm")
        try manager.createDirectory(at: input.appendingPathComponent("subdirectory"), withIntermediateDirectories: true)
        try Data().write(to: input.appendingPathComponent("empty.txt"))
        try Data([65]).write(to: input.appendingPathComponent("one.txt"))
        let alphabet = Array("The quiet library has stories about rivers and mountains.\n".utf8)
        let text = NeuralTestData.random(count: Self.textBytes).map { alphabet[Int($0) % alphabet.count] }
        try Data(text).write(to: input.appendingPathComponent("subdirectory/pseudo-text.txt"))

        let packed = try DemoPacker.pack(directory: input, output: archive, method: pack)
        let bytes = try Data(contentsOf: archive)
        let container = try DemoContainer.decode(bytes)
        XCTAssertEqual(container.kind, kind)
        if kind == .neural {
            let header = try Header.decode(container.payload)
            XCTAssertEqual(header.blockCount, UInt32(Self.neuralBlocks))
            XCTAssertEqual(header.predictorTag, "order1")
            XCTAssertEqual(header.originalLength, UInt64(packed.tarBytes))
        } else {
            let frame = try LZ4Frame(container.payload)
            XCTAssertTrue(frame.isIndependent)
            XCTAssertEqual(frame.contentSize, UInt64(packed.tarBytes))
            let sizes = try frame.expectedDecodedSizes()
            XCTAssertTrue(sizes.dropLast().allSatisfy { $0 == Self.lz4BlockSize })
            XCTAssertEqual(sizes.reduce(0, +), packed.tarBytes)
        }
        let unpacked = try DemoUnpacker.unpack(archive: archive, into: output, method: unpack)
        let original = try tree(at: input)
        let restored = try tree(at: output)
        XCTAssertEqual(restored.directories, original.directories)
        XCTAssertEqual(Set(restored.files.keys), Set(original.files.keys))
        for (path, contents) in original.files {
            XCTAssertEqual(restored.files[path]?.count, contents.count, path)
            XCTAssertEqual(restored.files[path], contents, path)
        }
        XCTAssertEqual(packed.entries, 4)
        XCTAssertEqual(unpacked.entries, packed.entries)
        XCTAssertEqual(packed.archiveBytes, bytes.count)
        XCTAssertEqual(unpacked.archiveBytes, packed.archiveBytes)
        XCTAssertEqual(unpacked.tarBytes, packed.tarBytes)
        XCTAssertEqual(packed.ratio, Double(bytes.count) / Double(packed.tarBytes))
        XCTAssertNil(unpacked.gpuSeconds)
        for seconds in [packed.tarSeconds, packed.compressSeconds, packed.writeSeconds,
                        unpacked.readSeconds, unpacked.decodeSeconds, unpacked.extractSeconds] {
            XCTAssertGreaterThanOrEqual(seconds, 0)
        }
        XCTAssertEqual(Set(try manager.contentsOfDirectory(atPath: root.path)), ["input", "output", "archive.kadm"])
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("demo-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func tree(at directory: URL) throws -> (directories: Set<String>, files: [String: Data]) {
        let root = directory.standardizedFileURL
        let keys: Set<URLResourceKey> = [.isDirectoryKey, .isRegularFileKey]
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(at: root, includingPropertiesForKeys: Array(keys)))
        var directories: Set<String> = []
        var files: [String: Data] = [:]
        for case let url as URL in enumerator {
            let path = url.standardizedFileURL.pathComponents.dropFirst(root.pathComponents.count).joined(separator: "/")
            let values = try url.resourceValues(forKeys: keys)
            if values.isDirectory == true { directories.insert(path) }
            else {
                XCTAssertEqual(values.isRegularFile, true, path)
                files[path] = try Data(contentsOf: url)
            }
        }
        return (directories, files)
    }
}
