internal import CoreML
public import Foundation

internal enum CoreMLComputePlanSummary {
    static func load(modelURL: URL, computeUnits: MLComputeUnits) async throws -> String {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let plan = try await MLComputePlan.load(contentsOf: modelURL, configuration: configuration)
        var counts: [String: Int] = [:]

        func deviceName(_ usage: MLComputePlan.DeviceUsage?) -> String {
            guard let usage else { return "n/a" }
            switch usage.preferred {
            case .cpu: return "CPU"
            case .gpu: return "GPU"
            case .neuralEngine: return "ANE"
            @unknown default: return "unknown"
            }
        }

        func visit(_ block: MLModelStructure.Program.Block) {
            for operation in block.operations {
                counts[deviceName(plan.deviceUsage(for: operation)), default: 0] += 1
                for nested in operation.blocks { visit(nested) }
            }
        }

        func visitStructure(_ structure: MLModelStructure) {
            switch structure {
            case .program(let program):
                for function in program.functions.values { visit(function.block) }
            case .neuralNetwork(let network):
                for layer in network.layers { counts[deviceName(plan.deviceUsage(for: layer)), default: 0] += 1 }
            case .pipeline(let pipeline):
                for model in pipeline.subModels { visitStructure(model) }
            default: break
            }
        }
        visitStructure(plan.modelStructure)
        return "preferred-device histogram: [" + counts.keys.sorted().map { "\($0): \(counts[$0]!)" }.joined(separator: ", ") + "]"
    }
}
