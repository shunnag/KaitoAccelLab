package import Foundation
private import KaitoKit

package enum DemoUnpacker {
    package enum GPUVariant: String { case thread, simd }

    package enum UnpackMethod {
        case neural(any BytePredictor, predictorTag: String)
        case gpu(GPUVariant)
        case cpu
    }

    package struct Report: Sendable {
        package let entries: Int
        package let tarBytes: Int
        package let archiveBytes: Int
        package let readSeconds: Double
        package let decodeSeconds: Double
        package let extractSeconds: Double
        package let gpuSeconds: Double?
    }

    private static let cpuLanes = 16
    private static let tarRecordBytes = 512
    private static let emptyTarBytes = 2 * tarRecordBytes

    package static func unpack(archive: URL, into directory: URL, method: UnpackMethod) throws -> Report {
        let clock = ContinuousClock()
        let readStart = clock.now
        let data = try Data(contentsOf: archive)
        let container = try DemoContainer.decode(data)
        let readSeconds = seconds(readStart.duration(to: clock.now))

        let decodeStart = clock.now
        let tar: Data
        var gpuSeconds: Double?
        switch (container.kind, method) {
        case let (.neural, .neural(predictor, tag)):
            let header = try Header.decode(container.payload)
            if header.predictorTag != tag {
                FileHandle.standardError.write(Data("warning: header predictor tag '\(header.predictorTag)' differs from chosen predictor '\(tag)'\n".utf8))
            }
            tar = try NeuralBlockCodec.decode(container.payload, predictor: predictor).payload
        case (.lz4, .cpu):
            let frame = try LZ4Frame(container.payload)
            tar = Data(try LZ4FrameDecoder.decodeCPU(frame: frame, source: container.payload, lanes: cpuLanes))
        case let (.lz4, .gpu(variant)):
            let frame = try LZ4Frame(container.payload)
            let sizes = try frame.expectedDecodedSizes()
            let decoder = try LZ4MetalDecoder()
            try decoder.prepare(frame: frame, source: container.payload, sizes: sizes)
            let result = try decoder.decode(variant: variant == .thread ? .threadPerBlock : .simdPerBlock)
            for (index, status) in result.statuses.enumerated() where status != 0 {
                throw LZ4Error.metalFailure("Block \(index) status \(status)")
            }
            tar = Data(result.output)
            gpuSeconds = result.gpuSeconds
        case (.neural, _):
            throw DemoError("Neural KADM payload requires --gru, --mlp, or --order1")
        case (.lz4, .neural):
            throw DemoError("LZ4 KADM payload requires a GPU or CPU LZ4 decoder")
        }
        let decodeSeconds = seconds(decodeStart.duration(to: clock.now))

        let extractStart = clock.now
        let reader: ArchiveReader
        if tar.count >= emptyTarBytes, tar.count.isMultiple(of: tarRecordBytes), tar.allSatisfy({ $0 == 0 }) {
            // 終端だけの空 tar は識別用ヘッダーがないため、拡張子のヒントを渡す。
            reader = try ArchiveReader.open(source: DataByteSource(data: tar), sourceURL: archive.appendingPathExtension("tar"))
        } else {
            reader = try ArchiveReader.open(data: tar)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for entry in reader.entries { _ = try reader.extract(entry, to: directory) }
        let extractSeconds = seconds(extractStart.duration(to: clock.now))
        return Report(entries: reader.entries.count, tarBytes: tar.count, archiveBytes: data.count,
                      readSeconds: readSeconds, decodeSeconds: decodeSeconds,
                      extractSeconds: extractSeconds, gpuSeconds: gpuSeconds)
    }

    private static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
