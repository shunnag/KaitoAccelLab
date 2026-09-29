package import Foundation
private import GyoshukuKit

package enum DemoPacker {
    package enum Method {
        case neural(any BytePredictor, blocks: Int, predictorTag: String)
        case lz4(blockSize: Int)
    }

    package struct Report: Sendable {
        package let entries: Int
        package let tarBytes: Int
        package let archiveBytes: Int
        package var ratio: Double { tarBytes > 0 ? Double(archiveBytes) / Double(tarBytes) : 0 }
        package let tarSeconds: Double
        package let compressSeconds: Double
        package let writeSeconds: Double
    }

    package static func pack(directory: URL, output: URL, method: Method) throws -> Report {
        let clock = ContinuousClock()
        let tarStart = clock.now
        let manager = FileManager.default
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        guard try root.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else {
            throw DemoError("Input must be a directory")
        }
        let outputURL = output.standardizedFileURL.resolvingSymlinksInPath()
        let keys: Set<URLResourceKey> = [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey]
        var enumerationError: (any Error)?
        guard let enumerator = manager.enumerator(at: root, includingPropertiesForKeys: Array(keys),
                                                  errorHandler: { _, error in
            enumerationError = error
            return false
        }) else { throw DemoError("Could not enumerate input directory") }
        var entries: [(url: URL, path: String, directory: Bool)] = []
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: keys)
            if values.isSymbolicLink == true {
                enumerator.skipDescendants()
                continue
            }
            guard url.standardizedFileURL != outputURL,
                  values.isDirectory == true || values.isRegularFile == true else { continue }
            let path = url.standardizedFileURL.pathComponents.dropFirst(root.pathComponents.count).joined(separator: "/")
            entries.append((url, path, values.isDirectory == true))
        }
        if let enumerationError { throw enumerationError }
        entries.sort { $0.path < $1.path }

        // 列挙後に同じボリュームへ作り、入力内の出力先でも一時 tar を取り込まない。
        let temporary = outputURL.deletingLastPathComponent().appendingPathComponent(".kadm-\(UUID().uuidString).tar")
        defer { try? manager.removeItem(at: temporary) }
        let writer = try ArchiveWriter.create(url: temporary, format: .tar)
        for entry in entries {
            if entry.directory { try writer.addDirectory(entry.path) }
            else { try writer.add(contentsOf: entry.url, as: entry.path) }
        }
        try writer.finish()
        let tar = try Data(contentsOf: temporary)
        let tarSeconds = seconds(tarStart.duration(to: clock.now))

        let compressStart = clock.now
        let container: DemoContainer
        switch method {
        case let .neural(predictor, blocks, tag):
            let payload = try NeuralBlockCodec.encode(data: tar, blockCount: blocks,
                                                      predictor: predictor, predictorTag: tag)
            container = DemoContainer(kind: .neural, payload: payload)
        case let .lz4(blockSize):
            container = DemoContainer(kind: .lz4, payload: try LZ4FrameEncoder.encodeFrame(Array(tar), blockSize: blockSize))
        }
        let compressSeconds = seconds(compressStart.duration(to: clock.now))

        let writeStart = clock.now
        let archive = container.encode()
        try archive.write(to: outputURL, options: .atomic)
        let writeSeconds = seconds(writeStart.duration(to: clock.now))
        return Report(entries: entries.count, tarBytes: tar.count, archiveBytes: archive.count,
                      tarSeconds: tarSeconds, compressSeconds: compressSeconds, writeSeconds: writeSeconds)
    }

    private static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
