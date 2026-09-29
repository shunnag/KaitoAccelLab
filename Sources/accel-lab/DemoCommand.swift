private import AccelLab
private import Foundation

internal struct DemoCommand {
    private static let blockSizeRange = 4_096...4_194_304
    private let packing: Bool
    private let inputURL: URL
    private let outputURL: URL
    private let neural: Bool
    private let blocks: Int?
    private let blockSize: Int?
    private let selection: String?
    private let modelURL: URL?
    private let units: String
    private let gpuVariant: String

    private init(arguments: [String]) throws {
        guard arguments.count >= 3, !arguments[1].isEmpty, !arguments[2].isEmpty,
              !arguments[1].hasPrefix("--"), !arguments[2].hasPrefix("--") else {
            throw argumentError("Demo commands require input and output paths")
        }
        packing = arguments[0] == "demo-pack"
        inputURL = URL(fileURLWithPath: arguments[1])
        outputURL = URL(fileURLWithPath: arguments[2])
        var neural: Bool?
        var blocks: Int?
        var blockSize: Int?
        var selection: String?
        var modelURL: URL?
        var units = "ane"
        var gpuVariant = "thread"
        var seen: Set<String> = []
        var index = 3
        while index < arguments.count {
            let option = arguments[index]
            guard seen.insert(option).inserted else { throw argumentError("Duplicate option: \(option)") }
            switch option {
            case "--neural", "--lz4":
                guard packing else { throw argumentError("demo-unpack reads the payload kind from the header") }
                guard neural == nil else { throw argumentError("Choose exactly one of --neural or --lz4") }
                neural = option == "--neural"
                index += 1
                continue
            case "--order1":
                guard selection == nil else { throw argumentError("Choose exactly one predictor") }
                selection = "order1"
                index += 1
                continue
            default: break
            }
            guard ["--blocks", "--block-size", "--mlp", "--gru", "--units", "--gpu-variant"].contains(option) else {
                throw argumentError("Unknown option: \(option)")
            }
            guard index + 1 < arguments.count, !arguments[index + 1].isEmpty,
                  !arguments[index + 1].hasPrefix("--") else { throw argumentError("Missing value for \(option)") }
            let value = arguments[index + 1]
            switch option {
            case "--blocks":
                guard packing else { throw argumentError("demo-unpack reads the block count from the header") }
                guard let count = Int(value), count > 0, UInt32(exactly: count) != nil else {
                    throw argumentError("--blocks must be an integer in 1...\(UInt32.max)")
                }
                blocks = count
            case "--block-size":
                guard packing else { throw argumentError("--block-size is only valid for demo-pack") }
                guard let size = Int(value), Self.blockSizeRange.contains(size) else {
                    throw argumentError("--block-size must be an integer in 4096...4194304")
                }
                blockSize = size
            case "--units":
                guard ["ane", "cpu", "gpu", "all"].contains(value) else {
                    throw argumentError("--units must be ane, cpu, gpu, or all")
                }
                units = value
            case "--gpu-variant":
                guard !packing else { throw argumentError("--gpu-variant is only valid for demo-unpack") }
                guard ["thread", "simd", "cpu"].contains(value) else {
                    throw argumentError("--gpu-variant must be thread, simd, or cpu")
                }
                gpuVariant = value
            default:
                guard selection == nil else { throw argumentError("Choose exactly one predictor") }
                selection = String(option.dropFirst(2))
                modelURL = URL(fileURLWithPath: value)
            }
            index += 2
        }
        if packing {
            guard let neural else { throw argumentError("Choose exactly one of --neural or --lz4") }
            if neural {
                guard blocks != nil, selection != nil else {
                    throw argumentError("--neural requires --blocks N and one predictor")
                }
                guard blockSize == nil else { throw argumentError("--block-size requires --lz4") }
            } else {
                guard blockSize != nil else { throw argumentError("--lz4 requires --block-size <bytes>") }
                guard blocks == nil, selection == nil, !seen.contains("--units") else {
                    throw argumentError("--lz4 does not accept neural predictor options")
                }
            }
        } else if seen.contains("--units"), selection == nil {
            throw argumentError("--units requires a neural predictor")
        }
        self.neural = neural ?? (selection != nil)
        self.blocks = blocks
        self.blockSize = blockSize
        self.selection = selection
        self.modelURL = modelURL
        self.units = units
        self.gpuVariant = gpuVariant
    }

    static func run(arguments: [String]) async throws {
        let command = try DemoCommand(arguments: arguments)
        try await command.run()
    }

    private func run() async throws {
        let selected: NeuralPredictor?
        if let selection {
            selected = try await NeuralPredictor.make(selection: selection, modelURL: modelURL, units: units)
        } else { selected = nil }
        if packing {
            let method: DemoPacker.Method
            if neural {
                method = .neural(selected!.predictor, blocks: blocks!, predictorTag: selected!.tag)
            } else { method = .lz4(blockSize: blockSize!) }
            let report = try DemoPacker.pack(directory: inputURL, output: outputURL, method: method)
            print("entries: \(report.entries)")
            print("tar_bytes: \(report.tarBytes)")
            print("archive_bytes: \(report.archiveBytes)")
            print("ratio: \(decimal(report.ratio))")
            print("stage\tseconds")
            print("tar\t\(decimal(report.tarSeconds))")
            print("compress\t\(decimal(report.compressSeconds))")
            print("write\t\(decimal(report.writeSeconds))")
        } else {
            let method: DemoUnpacker.UnpackMethod
            if let selected { method = .neural(selected.predictor, predictorTag: selected.tag) }
            else if gpuVariant == "cpu" { method = .cpu }
            else { method = .gpu(gpuVariant == "thread" ? .thread : .simd) }
            let report = try DemoUnpacker.unpack(archive: inputURL, into: outputURL, method: method)
            print("entries: \(report.entries)")
            print("tar_bytes: \(report.tarBytes)")
            print("archive_bytes: \(report.archiveBytes)")
            if let seconds = report.gpuSeconds { print("gpu_seconds: \(decimal(seconds))") }
            print("stage\tseconds")
            print("read\t\(decimal(report.readSeconds))")
            print("decode\t\(decimal(report.decodeSeconds))")
            print("extract\t\(decimal(report.extractSeconds))")
        }
    }

    private func decimal(_ value: Double) -> String {
        String(format: "%.6f", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}
