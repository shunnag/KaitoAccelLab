private import AccelLab
private import CoreML
private import Foundation

internal struct NeuralBenchmarkCommand {
    static func run(arguments: [String]) async throws {
        guard arguments.count >= 3 else { throw argumentError("Two GRU model paths are required") }
        var blockCount = 1024
        var steps = 256
        var rounds = 5
        var units = "ane"
        var seen: Set<String> = []
        var index = 3
        while index < arguments.count {
            let option = arguments[index]
            guard ["--blocks", "--steps", "--rounds", "--units"].contains(option),
                  seen.insert(option).inserted, index + 1 < arguments.count else {
                throw argumentError("Invalid or duplicate neural-predictor-bench option: \(option)")
            }
            let value = arguments[index + 1]
            if option == "--units" {
                guard ["ane", "cpu", "gpu", "all"].contains(value) else { throw argumentError("Invalid compute units") }
                units = value
            } else {
                guard let number = Int(value), number > 0 else { throw argumentError("\(option) must be positive") }
                switch option {
                case "--blocks": blockCount = number
                case "--steps": steps = number
                default: rounds = number
                }
            }
            index += 2
        }
        let computeUnits: MLComputeUnits
        switch units {
        case "cpu": computeUnits = .cpuOnly
        case "gpu": computeUnits = .cpuAndGPU
        case "all": computeUnits = .all
        default: computeUnits = .cpuAndNeuralEngine
        }
        let explicit = try CoreMLRecurrentPredictor(modelURL: URL(fileURLWithPath: arguments[1]), computeUnits: computeUnits)
        let stateful = try CoreMLRecurrentPredictor(modelURL: URL(fileURLWithPath: arguments[2]), computeUnits: computeUnits)
        let explicitRepeat = try CoreMLRecurrentPredictor(modelURL: URL(fileURLWithPath: arguments[1]), computeUnits: computeUnits)
        print("explicit: " + (try await explicit.computePlanSummary()))
        print("stateful: " + (try await stateful.computePlanSummary()))

        var seed: UInt64 = 0x4B61_6974_6F53_7461
        let observations = (0..<steps).map { _ in
            (0..<blockCount).map { _ -> UInt8 in
                seed ^= seed << 13
                seed ^= seed >> 7
                seed ^= seed << 17
                return UInt8(truncatingIfNeeded: seed)
            }
        }
        let clock = ContinuousClock()
        func trace(_ predictor: any BytePredictor, expected: [Data]?) throws -> (seconds: Double, rows: [Data]) {
            try predictor.begin(blockCount: blockCount)
            var elapsed = 0.0
            var rows: [Data] = []
            if expected == nil { rows.reserveCapacity(steps) }
            for step in 0..<steps {
                let start = clock.now
                let probabilities = try predictor.predictNext()
                elapsed += seconds(start.duration(to: clock.now))
                let bytes = probabilities.withUnsafeBytes { Data($0) }
                if let expected {
                    guard bytes == expected[step] else { throw argumentError("Prediction mismatch at step \(step)") }
                } else {
                    rows.append(bytes)
                }
                try predictor.observe(observations[step])
            }
            return (elapsed, rows)
        }
        func pair(_ left: any BytePredictor, _ right: any BytePredictor) throws -> (left: Double, right: Double) {
            let leftTrace = try trace(left, expected: nil)
            let rightTrace = try trace(right, expected: leftTrace.rows)
            return (leftTrace.seconds, rightTrace.seconds)
        }
        func orderedPair(round: Int) throws -> (explicit: Double, stateful: Double) {
            if round.isMultiple(of: 2) {
                let elapsed = try pair(explicit, stateful)
                return (elapsed.left, elapsed.right)
            }
            let elapsed = try pair(stateful, explicit)
            return (elapsed.right, elapsed.left)
        }

        _ = try pair(explicit, stateful)
        print("match: OK (every fp16-derived Float probability, warmup)")
        var explicitAA: [Double] = []
        var repeatAA: [Double] = []
        for round in 0..<min(rounds, 3) {
            if round.isMultiple(of: 2) {
                let elapsed = try pair(explicit, explicitRepeat)
                explicitAA.append(elapsed.left)
                repeatAA.append(elapsed.right)
            } else {
                let elapsed = try pair(explicitRepeat, explicit)
                explicitAA.append(elapsed.right)
                repeatAA.append(elapsed.left)
            }
        }
        var explicitTimes: [Double] = []
        var statefulTimes: [Double] = []
        for round in 0..<rounds {
            let elapsed = try orderedPair(round: round)
            explicitTimes.append(elapsed.explicit)
            statefulTimes.append(elapsed.stateful)
        }
        print("match: OK (every fp16-derived Float probability, all rounds)")
        print("name\tblock_count\tsteps\tmedian_s\tMB_per_s\tms_per_step")
        row("explicit-A/A-1", times: explicitAA, blocks: blockCount, steps: steps)
        row("explicit-A/A-2", times: repeatAA, blocks: blockCount, steps: steps)
        row("explicit", times: explicitTimes, blocks: blockCount, steps: steps)
        row("MLState", times: statefulTimes, blocks: blockCount, steps: steps)
    }

    private static func row(_ name: String, times: [Double], blocks: Int, steps: Int) {
        let sorted = times.sorted()
        let middle = sorted.count / 2
        let median = sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
        let locale = Locale(identifier: "en_US_POSIX")
        print(String(format: "%@\t%d\t%d\t%.9f\t%.3f\t%.6f", locale: locale,
                     name, blocks, steps, median, Double(blocks * steps) / median / 1e6, median * 1_000 / Double(steps)))
    }

    private static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
