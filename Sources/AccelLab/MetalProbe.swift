internal import Foundation
internal import Metal

/// GPU の有無と runtime での MSL compile を確かめる最小の probe。
public enum MetalProbe {
    public struct Report: Sendable {
        public let deviceName: String
        public let maxThreadsPerThreadgroup: Int
        public let threadgroupMemoryLength: Int
        public let hasUnifiedMemory: Bool
        public let compiledAtRuntime: Bool
        public let sumOfSquares: UInt64
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void square_sum(device const uint *input [[buffer(0)]],
                           device atomic_uint *output [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
        uint v = input[id];
        atomic_fetch_add_explicit(output, v * v, memory_order_relaxed);
    }
    """

    public static func run(count: Int = 1 << 16) throws -> Report {
        guard let device = MTLCreateSystemDefaultDevice() else { throw NSError(domain: "AccelLab", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal device unavailable"]) }
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "square_sum") else { throw NSError(domain: "AccelLab", code: 2) }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue(),
              let input = device.makeBuffer(length: count * 4, options: .storageModeShared),
              let output = device.makeBuffer(length: 4, options: .storageModeShared) else { throw NSError(domain: "AccelLab", code: 3) }
        let values = input.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count { values[i] = UInt32(i % 1000) }
        output.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        let commands = queue.makeCommandBuffer()!, encoder = commands.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, count)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding(); commands.commit(); commands.waitUntilCompleted()
        let sum = UInt64(output.contents().load(as: UInt32.self))
        var expected: UInt32 = 0
        for i in 0..<count { let v = UInt32(i % 1000); expected &+= v &* v }
        return Report(deviceName: device.name, maxThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup,
                      threadgroupMemoryLength: device.maxThreadgroupMemoryLength, hasUnifiedMemory: device.hasUnifiedMemory,
                      compiledAtRuntime: true, sumOfSquares: sum == UInt64(expected) ? max(sum, 1) : 0)
    }
}
