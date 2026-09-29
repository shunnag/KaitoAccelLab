import AccelLab
import Foundation

// 引数なしでは従来どおり probe を実行する。
let arguments = Array(CommandLine.arguments.dropFirst())
let usage = "Usage: accel-lab probe | baseline [--size MiB] [--rounds N] [--out path]"

func argumentError(_ message: String) -> NSError {
    NSError(domain: "accel-lab", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(message)\n\(usage)"])
}

do {
    switch arguments.first ?? "probe" {
    case "probe":
        guard arguments.count <= 1 else { throw argumentError("probe accepts no options") }
        let report = try MetalProbe.run()
        print("device: \(report.deviceName)")
        print("maxThreadsPerThreadgroup: \(report.maxThreadsPerThreadgroup), threadgroupMemory: \(report.threadgroupMemoryLength), unifiedMemory: \(report.hasUnifiedMemory)")
        print("runtime MSL compile: \(report.compiledAtRuntime), sum-of-squares check: \(report.sumOfSquares != 0 ? "OK" : "MISMATCH")")
    case "baseline":
        var sizeMiB = 256
        var rounds = 5
        var outputPath: String?
        var seen: Set<String> = []
        var index = 1
        while index < arguments.count {
            let option = arguments[index]
            guard ["--size", "--rounds", "--out"].contains(option) else {
                throw argumentError("Unknown option: \(option)")
            }
            guard seen.insert(option).inserted else { throw argumentError("Duplicate option: \(option)") }
            guard index + 1 < arguments.count else { throw argumentError("Missing value for \(option)") }
            let value = arguments[index + 1]
            switch option {
            case "--size":
                guard let size = Int(value), (1...2047).contains(size) else {
                    throw argumentError("--size must be an integer in 1...2047")
                }
                sizeMiB = size
            case "--rounds":
                guard let count = Int(value), count > 0 else { throw argumentError("--rounds must be a positive integer") }
                rounds = count
            default:
                guard !value.isEmpty, !value.hasPrefix("--") else { throw argumentError("Missing path for --out") }
                outputPath = value
            }
            index += 2
        }

        let output: URL
        if let outputPath {
            output = URL(fileURLWithPath: outputPath)
        } else {
            // 実行時の作業ディレクトリに依存せず、ビルド元リポジトリの Results に保存する。
            let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
            let timestamp = DateFormatter()
            timestamp.locale = Locale(identifier: "en_US_POSIX")
            timestamp.calendar = Calendar(identifier: .gregorian)
            timestamp.dateFormat = "yyyyMMdd-HHmm"
            output = repository.appendingPathComponent("Results/baseline-\(timestamp.string(from: Date())).tsv")
        }
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        let measurements = try Baseline.run(sizeMiB: sizeMiB, rounds: rounds)
        let locale = Locale(identifier: "en_US_POSIX")
        var rows = ["name\tbytes\tmedian_s\tGB_per_s\tnote"]
        for measurement in measurements {
            let seconds = String(format: "%.9f", locale: locale, measurement.medianSeconds)
            let speed = String(format: "%.6f", locale: locale, measurement.throughputGBps)
            let note = measurement.note.replacingOccurrences(of: "\t", with: " ")
                .replacingOccurrences(of: "\n", with: " ").replacingOccurrences(of: "\r", with: " ")
            rows.append("\(measurement.name)\t\(measurement.bytes)\t\(seconds)\t\(speed)\t\(note)")
        }
        let table = rows.joined(separator: "\n") + "\n"
        try table.write(to: output, atomically: true, encoding: .utf8)
        print(table, terminator: "")
        print("Wrote \(output.path)")
    default:
        throw argumentError("Unknown command: \(arguments[0])")
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
    exit(1)
}
