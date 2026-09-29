import CoreML
import Foundation

// 使い方: ane-probe <compiled.mlmodelc> [batch] [rounds]
// Core ML の compute plan で op ごとの配置（ANE / GPU / CPU）を読み、推論 latency を測る。
let args = CommandLine.arguments
let url = URL(fileURLWithPath: args[1])
let batch = args.count > 2 ? Int(args[2])! : 1
let rounds = args.count > 3 ? Int(args[3])! : 20
let context = args.count > 4 ? Int(args[4])! : 16

func run(_ units: MLComputeUnits, label: String) async throws {
    let config = MLModelConfiguration()
    config.computeUnits = units
    let model = try MLModel(contentsOf: url, configuration: config)
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
    var counts: [String: Int] = [:]
    var lines: [String] = []
    if case .program(let program) = plan.modelStructure {
        for (_, function) in program.functions {
            for op in function.block.operations {
                let usage = plan.deviceUsage(for: op)
                let preferred = usage.map { String(describing: $0.preferred) } ?? "n/a"
                let supported = usage.map { $0.supported.map { String(describing: $0) }.joined(separator: ",") } ?? "n/a"
                counts[preferred, default: 0] += 1
                if op.operatorName != "const" { lines.append("  \(op.operatorName): preferred=\(preferred) supported=[\(supported)]") }
            }
        }
    }
    let input = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: context)], dataType: .float16)
    for i in 0..<(batch * context) { input[i] = NSNumber(value: Float(i % 256) / 255) }
    let provider = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: input)])
    _ = try await model.prediction(from: provider)
    var times: [Double] = []
    for _ in 0..<rounds {
        let t0 = ContinuousClock.now
        _ = try await model.prediction(from: provider)
        let d = ContinuousClock.now - t0
        times.append(Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18)
    }
    times.sort()
    let median = times[times.count / 2]
    print("[\(label)] preferred-device histogram: \(counts) ; batch=\(batch) median=\(String(format: "%.3f", median * 1000)) ms => \(String(format: "%.1f", Double(batch) / median)) predictions/s")
    for l in lines { print(l) }
}

let semaphore = DispatchSemaphore(value: 0)
Task {
    do {
        try await run(.cpuAndNeuralEngine, label: "cpuAndNeuralEngine")
        try await run(.cpuOnly, label: "cpuOnly")
        try await run(.cpuAndGPU, label: "cpuAndGPU")
    } catch { print("error: \(error)") }
    semaphore.signal()
}
semaphore.wait()
