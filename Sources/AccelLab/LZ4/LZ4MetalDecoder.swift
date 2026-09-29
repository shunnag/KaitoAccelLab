internal import Foundation
internal import Metal

final class LZ4MetalDecoder {
    enum Variant: CaseIterable { case threadPerBlock, simdPerBlock }

    struct Result {
        let outputBuffer: any MTLBuffer
        let outputCount: Int
        // 同じ variant の次の復元までは、この共有領域を直接比較できる。
        var output: [UInt8] {
            Array(UnsafeBufferPointer(start: outputBuffer.contents().assumingMemoryBound(to: UInt8.self),
                                      count: outputCount))
        }
        let cpuWallSeconds: Double
        let gpuSeconds: Double
        let uploadSeconds: Double
        let statuses: [UInt32]
    }

    private struct Buffers {
        let input: any MTLBuffer
        let descriptors: any MTLBuffer
        let output: any MTLBuffer
        let status: any MTLBuffer
        let blockCount: Int
        let outputCount: Int
    }

    private struct Prepared {
        let plan: LZ4FrameDecoder.Plan
        let input: any MTLBuffer
    }

    private static let simdWidth = 32
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let threadPipeline: any MTLComputePipelineState
    private let simdPipeline: any MTLComputePipelineState
    private var prepared: Prepared?
    private var variantBuffers: [Variant: Buffers] = [:]
    private var singleBlockBuffers: [Int: Buffers] = [:]
    private(set) var uploadSeconds = 0.0

    init(device: (any MTLDevice)? = MTLCreateSystemDefaultDevice()) throws {
        guard let device else { throw LZ4Error.metalUnavailable }
        self.device = device
        let library = try device.makeLibrary(source: LZ4MetalKernels.source, options: nil)
        guard let thread = library.makeFunction(name: "lz4_decode_thread_per_block"),
              let simd = library.makeFunction(name: "lz4_decode_simd_per_block"),
              let queue = device.makeCommandQueue() else {
            throw LZ4Error.metalFailure("Could not create functions or command queue")
        }
        self.queue = queue
        threadPipeline = try device.makeComputePipelineState(function: thread)
        simdPipeline = try device.makeComputePipelineState(function: simd)
    }

    func decode(
        frame: LZ4Frame, source: Data, variant: Variant, simdgroupsPerThreadgroup: Int = 4
    ) throws -> Result {
        try validate(variant: variant, groups: simdgroupsPerThreadgroup)
        try ensurePrepared(frame: frame, source: source)
        return try decode(variant: variant, simdgroupsPerThreadgroup: simdgroupsPerThreadgroup)
    }

    func decode(variant: Variant, simdgroupsPerThreadgroup: Int = 4) throws -> Result {
        try validate(variant: variant, groups: simdgroupsPerThreadgroup)
        guard let prepared else { throw LZ4Error.invalidArgument("Prepare the Metal decoder before decoding") }
        let buffers: Buffers
        if let cached = variantBuffers[variant] { buffers = cached }
        else {
            buffers = try makeBuffers(prepared: prepared, blockIndices: Array(prepared.plan.blocks.indices))
            variantBuffers[variant] = buffers
        }
        let timing = try execute(buffers, variant: variant, groups: simdgroupsPerThreadgroup)
        let statuses = Array(UnsafeBufferPointer(
            start: buffers.status.contents().assumingMemoryBound(to: UInt32.self), count: buffers.blockCount))
        return Result(outputBuffer: buffers.output, outputCount: buffers.outputCount,
                      cpuWallSeconds: timing.wall, gpuSeconds: timing.gpu,
                      uploadSeconds: uploadSeconds, statuses: statuses)
    }

    func decodeSingleBlock(frame: LZ4Frame, source: Data, blockIndex: Int, rounds: Int) throws -> [Double] {
        guard frame.blocks.indices.contains(blockIndex), rounds > 0 else {
            throw LZ4Error.invalidArgument("A valid block index and positive rounds are required")
        }
        try ensurePrepared(frame: frame, source: source)
        let buffers: Buffers
        if let cached = singleBlockBuffers[blockIndex] { buffers = cached }
        else {
            buffers = try makeBuffers(prepared: prepared!, blockIndices: [blockIndex])
            singleBlockBuffers[blockIndex] = buffers
        }
        var times: [Double] = []
        for _ in 0..<rounds {
            // 記述子は一つだけなので、dispatchThreads の幅も必ず一つになる。
            let timing = try execute(buffers, variant: .threadPerBlock, groups: 1)
            let status = buffers.status.contents().load(as: UInt32.self)
            guard status == 0 else { throw LZ4Error.metalFailure("Single block status \(status)") }
            times.append(timing.gpu)
        }
        return times
    }

    private func validate(variant: Variant, groups: Int) throws {
        guard groups > 0 else { throw LZ4Error.invalidArgument("simdgroups must be positive") }
        if variant == .simdPerBlock {
            guard simdPipeline.threadExecutionWidth == Self.simdWidth,
                  groups <= simdPipeline.maxTotalThreadsPerThreadgroup / Self.simdWidth else {
                throw LZ4Error.invalidArgument("Unsupported SIMD width or simdgroups per threadgroup")
            }
        }
    }

    private func makeBuffer(length: Int) throws -> any MTLBuffer {
        guard length <= device.maxBufferLength,
              let buffer = device.makeBuffer(length: max(length, 1), options: .storageModeShared) else {
            throw LZ4Error.metalFailure("Could not allocate \(length) bytes")
        }
        return buffer
    }

    // 一つのインスタンスは一つの入力を保持し、全 variant と単一ブロック計測で共有する。
    func prepare(frame: LZ4Frame, source: Data, sizes: [Int]) throws {
        if let prepared {
            try frame.validate(source: source)
            guard prepared.plan.source == source, prepared.plan.sizes == sizes else { throw LZ4Error.sourceMismatch }
            return
        }
        let plan = try LZ4FrameDecoder.Plan(frame: frame, source: source, sizes: sizes)
        guard source.count <= Int(UInt32.max), plan.outputCount <= Int(UInt32.max),
              plan.blocks.count <= Int(UInt32.max) else {
            throw LZ4Error.sizeOverflow
        }
        let start = ContinuousClock.now
        let input = try makeBuffer(length: source.count)
        source.withUnsafeBytes { bytes in
            if !bytes.isEmpty { input.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count) }
        }
        uploadSeconds = seconds(start.duration(to: .now))
        prepared = Prepared(plan: plan, input: input)
    }

    private func ensurePrepared(frame: LZ4Frame, source: Data) throws {
        if let prepared {
            try frame.validate(source: source)
            guard source == prepared.plan.source else { throw LZ4Error.sourceMismatch }
        } else {
            try prepare(frame: frame, source: source, sizes: frame.expectedDecodedSizes())
        }
    }

    private func makeBuffers(prepared: Prepared, blockIndices: [Int]) throws -> Buffers {
        var descriptors: [BlockDesc] = []
        var total = 0
        for index in blockIndices {
            let block = prepared.plan.blocks[index]
            let size = prepared.plan.sizes[index]
            descriptors.append(BlockDesc(srcOffset: UInt32(block.compressedOffset),
                                         srcLength: UInt32(block.compressedLength), dstOffset: UInt32(total),
                                         dstLength: UInt32(size), isStored: block.isStored ? 1 : 0))
            total += size
        }
        let descriptorBuffer = try makeBuffer(length: descriptors.count * MemoryLayout<BlockDesc>.stride)
        descriptors.withUnsafeBytes { bytes in
            if !bytes.isEmpty { descriptorBuffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count) }
        }
        let output = try makeBuffer(length: total)
        // 最初の計測より前に全ページを書き込み、以後のラウンドでも同じ領域を使う。
        output.contents().initializeMemory(as: UInt8.self, repeating: 0, count: output.length)
        let status = try makeBuffer(length: descriptors.count * MemoryLayout<UInt32>.stride)
        // 未実行のブロックは成功扱いにしない。
        status.contents().initializeMemory(as: UInt8.self, repeating: 0xFF, count: status.length)
        return Buffers(input: prepared.input, descriptors: descriptorBuffer, output: output, status: status,
                       blockCount: descriptors.count, outputCount: total)
    }

    private func execute(_ buffers: Buffers, variant: Variant, groups: Int) throws -> (wall: Double, gpu: Double) {
        if buffers.blockCount == 0 { return (0, 0) }
        buffers.status.contents().initializeMemory(as: UInt8.self, repeating: 0xFF, count: buffers.status.length)
        let start = ContinuousClock.now
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw LZ4Error.metalFailure("Could not create command buffer or encoder")
        }
        let pipeline = variant == .threadPerBlock ? threadPipeline : simdPipeline
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffers.input, offset: 0, index: 0)
        encoder.setBuffer(buffers.descriptors, offset: 0, index: 1)
        encoder.setBuffer(buffers.output, offset: 0, index: 2)
        encoder.setBuffer(buffers.status, offset: 0, index: 3)
        var count = UInt32(buffers.blockCount)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
        if variant == .threadPerBlock {
            let width = min(buffers.blockCount, pipeline.maxTotalThreadsPerThreadgroup)
            encoder.dispatchThreads(MTLSize(width: buffers.blockCount, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        } else {
            var groupCount = UInt32(groups)
            encoder.setBytes(&groupCount, length: MemoryLayout<UInt32>.size, index: 5)
            let threadgroups = (buffers.blockCount + groups - 1) / groups
            encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Self.simdWidth * groups, height: 1, depth: 1))
        }
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        let wall = seconds(start.duration(to: .now))
        guard command.status == .completed else {
            throw LZ4Error.metalFailure(command.error?.localizedDescription ?? "Command did not complete")
        }
        return (wall, command.gpuEndTime - command.gpuStartTime)
    }

    private func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
