private import AccelLab
private import CoreML
private import Foundation

internal struct NeuralCommand {
    private let encoding: Bool
    private let inputURL: URL
    private let outputURL: URL
    private let blockCount: Int?
    private let selection: String
    private let modelURL: URL?
    private let units: String

    private init(arguments: [String]) throws {
        guard arguments.count >= 4 else { throw argumentError("Missing neural codec paths or predictor") }
        encoding = arguments[0] == "neural-encode"
        inputURL = URL(fileURLWithPath: arguments[1])
        outputURL = URL(fileURLWithPath: arguments[2])
        var blockCount: Int?
        var selection: String?
        var modelURL: URL?
        var units = "ane"
        var seen: Set<String> = []
        var index = 3
        while index < arguments.count {
            let option = arguments[index]
            guard seen.insert(option).inserted else { throw argumentError("Duplicate option: \(option)") }
            if ["--uniform", "--order0", "--order1"].contains(option) {
                guard selection == nil else { throw argumentError("Choose exactly one predictor") }
                selection = String(option.dropFirst(2))
                index += 1
                continue
            }
            guard ["--blocks", "--mlp", "--gru", "--units"].contains(option) else {
                throw argumentError("Unknown option: \(option)")
            }
            guard index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--"), !arguments[index + 1].isEmpty else {
                throw argumentError("Missing value for \(option)")
            }
            let value = arguments[index + 1]
            switch option {
            case "--blocks":
                guard encoding else { throw argumentError("neural-decode reads the block count from the header") }
                guard let count = Int(value), count > 0, UInt32(exactly: count) != nil else {
                    throw argumentError("--blocks must be an integer in 1...\(UInt32.max)")
                }
                blockCount = count
            case "--units":
                guard ["ane", "cpu", "gpu", "all"].contains(value) else {
                    throw argumentError("--units must be ane, cpu, gpu, or all")
                }
                units = value
            default:
                guard selection == nil else { throw argumentError("Choose exactly one predictor") }
                selection = String(option.dropFirst(2))
                modelURL = URL(fileURLWithPath: value)
            }
            index += 2
        }
        guard let selection else { throw argumentError("Choose exactly one predictor") }
        guard !encoding || blockCount != nil else { throw argumentError("neural-encode requires --blocks N") }
        self.blockCount = blockCount
        self.selection = selection
        self.modelURL = modelURL
        self.units = units
    }

    static func run(arguments: [String]) async throws {
        let command = try NeuralCommand(arguments: arguments)
        try await command.run()
    }

    private func run() async throws {
        let computeUnits: MLComputeUnits
        switch units {
        case "cpu": computeUnits = .cpuOnly
        case "gpu": computeUnits = .cpuAndGPU
        case "all": computeUnits = .all
        default: computeUnits = .cpuAndNeuralEngine
        }
        let predictor: any BytePredictor
        let tag: String
        switch selection {
        case "uniform": predictor = UniformPredictor(); tag = selection
        case "order0": predictor = OrderZeroPredictor(); tag = selection
        case "order1": predictor = OrderOnePredictor(); tag = selection
        case "mlp":
            let modelURL = modelURL!
            let coreML = try CoreMLContextPredictor(modelURL: modelURL, computeUnits: computeUnits)
            print(try await coreML.computePlanSummary())
            predictor = coreML
            tag = "mlp:\(modelURL.deletingPathExtension().lastPathComponent):\(units)"
        default:
            let modelURL = modelURL!
            let coreML = try CoreMLRecurrentPredictor(modelURL: modelURL, computeUnits: computeUnits)
            print(try await coreML.computePlanSummary())
            predictor = coreML
            tag = "gru:\(modelURL.deletingPathExtension().lastPathComponent):\(units)"
        }
        let input = try Data(contentsOf: inputURL)
        if !encoding {
            let header = try Header.decode(input)
            if header.predictorTag != tag {
                FileHandle.standardError.write(Data("warning: header predictor tag '\(header.predictorTag)' differs from chosen predictor '\(tag)'\n".utf8))
            }
        }
        let clock = ContinuousClock()
        let start = clock.now
        let output: Data
        var timings = NeuralCodecTimings()
        if encoding {
            output = try NeuralBlockCodec.encode(data: input, blockCount: blockCount!, predictor: predictor,
                                                 predictorTag: tag, timings: &timings)
        } else {
            output = try NeuralBlockCodec.decode(input, predictor: predictor, timings: &timings).payload
        }
        let elapsed = start.duration(to: clock.now).components
        let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
        try output.write(to: outputURL, options: .atomic)
        // 比率と処理速度の分母は符号化・復号とも元の非圧縮バイト数。
        let originalBytes = encoding ? input.count : output.count
        let compressedBytes = encoding ? output.count : input.count
        let ratio = originalBytes > 0 ? Double(compressedBytes) / Double(originalBytes) : 0
        let speed = seconds > 0 ? Double(originalBytes) / seconds : 0
        print("bytes in: \(input.count)")
        print("bytes out: \(output.count)")
        print(String(format: "ratio: %.6f; bits/byte: %.6f", locale: Locale(identifier: "en_US_POSIX"), ratio, ratio * 8))
        print(String(format: "wall seconds: %.6f; bytes/s: %.3f", locale: Locale(identifier: "en_US_POSIX"), seconds, speed))
        print(String(format: "predictor seconds: %.6f; coder seconds: %.6f", locale: Locale(identifier: "en_US_POSIX"),
                     timings.predictorSeconds, timings.coderSeconds) + "; steps: \(timings.steps)")
        let steps = Double(max(1, timings.steps))
        print(String(format: "predictor ms/step: %.6f; coder ms/step: %.6f", locale: Locale(identifier: "en_US_POSIX"),
                     timings.predictorSeconds * 1_000 / steps, timings.coderSeconds * 1_000 / steps))
    }
}
