import Foundation
import Metal

private struct LZMABucketTask {
    var start: UInt32
    var count: UInt32
    var candidate: UInt32
    var pad: UInt32
}

private struct LZMABucketConstants {
    var dictionarySize: UInt32
    var cyclicSize: UInt32
    var niceLen: UInt32
    var cutValue: UInt32
    var inputCount: UInt32
    var taskCount: UInt32
    var positionLimit: UInt32
    var waveStart: UInt32
    var waveParity: UInt32
}

struct LZMABucketMetalResult {
    let matches: [LZMAMatch]
    let counts: [UInt32]
    let candidates: [UInt32]
    let son: [UInt32]
    let hash: [UInt32]
    let bucketCount: Int
    let bucketSeconds: Double
    let gpuSeconds: Double
    let wallSeconds: Double
}

final class LZMABucketMetalEncoder {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState
    private var finder: LZMAMatchFinder

    init(properties: LZMAEncoderProperties, dictionary: Int) throws {
        guard properties.matchFinder == .bt4, dictionary >= 4, dictionary < Int(UInt32.max) - 1 else {
            throw WriterError.invalidOption("LZMA bucket encoder requires BT4 and a UInt32 dictionary")
        }
        finder = try LZMAMatchFinder(properties: properties, dictionary: dictionary)
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            finder.release()
            throw WriterError.compression(-1)
        }
        self.device = device
        self.queue = queue
        do {
            let library = try device.makeLibrary(source: LZMABucketMetalKernels.source, options: nil)
            guard let function = library.makeFunction(name: "lzma_bucket_decode") else {
                throw WriterError.compression(-1)
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            finder.release()
            throw error
        }
    }

    deinit { finder.release() }

    func reset() {
        UnsafeMutableRawPointer(finder.hash).initializeMemory(as: UInt8.self, repeating: 0,
            count: finder.hashCount * MemoryLayout<UInt32>.stride)
        UnsafeMutableRawPointer(finder.son).initializeMemory(as: UInt8.self, repeating: 0,
            count: finder.cyclicSize * 2 * MemoryLayout<UInt32>.stride)
        finder.position = 1
        finder.cyclic = 0
    }

    func encode(_ input: Data, through positionLimit: Int? = nil) throws -> LZMABucketMetalResult {
        let started = ContinuousClock.now
        guard input.count >= 4 else {
            return LZMABucketMetalResult(matches: [], counts: [UInt32](repeating: 0, count: input.count),
                candidates: [], son: Array(UnsafeBufferPointer(start: finder.son, count: finder.cyclicSize * 2)),
                hash: Array(UnsafeBufferPointer(start: finder.hash, count: finder.hashCount)),
                bucketCount: 0, bucketSeconds: 0, gpuSeconds: 0, wallSeconds: seconds(started.duration(to: .now)))
        }
        var buckets: [Int: [UInt32]] = [:]
        var nextCandidates: [Int: UInt32] = [:]
        var shortHeads: [UInt32] = []
        shortHeads.reserveCapacity(input.count * 2)
        let bucketStarted = ContinuousClock.now
        input.withUnsafeBytes { bytes in
            let base = bytes.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for index in 0..<input.count {
                guard index + 4 <= input.count else { continue }
                let word = UInt32(littleEndian: UnsafeRawPointer(base + index).loadUnaligned(as: UInt32.self))
                let crc0 = finder.crc[Int(word & 255)]
                let h2 = Int((crc0 ^ ((word >> 8) & 255)) & 1023)
                let h3 = 1024 + Int((crc0 ^ ((word >> 8) & 65535)) & 65535)
                let h4 = 1024 + 65536 + Int(((crc0 ^ ((word >> 8) & 65535))
                    ^ (finder.crc[Int(word >> 24)] << 5)) & finder.hashMask)
                let bucket = h4 - 1024 - 65536
                if nextCandidates[bucket] == nil { nextCandidates[bucket] = finder.hash[h4] }
                buckets[bucket, default: []].append(UInt32(index + 1))
                shortHeads.append(finder.hash[h2])
                shortHeads.append(finder.hash[h3])
                let position = UInt32(index + 1)
                finder.hash[h2] = position
                finder.hash[h3] = position
                finder.hash[h4] = position
            }
        }
        let bucketSeconds = seconds(bucketStarted.duration(to: .now))
        let bucketKeys = buckets.keys.sorted()
        var bucketOffsets = Dictionary(uniqueKeysWithValues: bucketKeys.map { ($0, 0) })
        var tasks: [LZMABucketTask] = []
        var positions: [UInt32] = []
        tasks.reserveCapacity(bucketKeys.count)
        positions.reserveCapacity(min(input.count, finder.cyclicSize))
        var constants = LZMABucketConstants(
            dictionarySize: UInt32(finder.dictionarySize), cyclicSize: UInt32(finder.cyclicSize),
            niceLen: UInt32(finder.niceLen), cutValue: UInt32(finder.cutValue),
            inputCount: UInt32(input.count), taskCount: 0, positionLimit: 0, waveStart: 0, waveParity: 0
        )
        let capacity = input.count * finder.cutValue
        let inputBuffer = try makeBuffer(data: input)
        let positionBuffer = try makeBuffer(length: input.count * MemoryLayout<UInt32>.stride)
        let taskBuffer = try makeBuffer(length: bucketKeys.count * MemoryLayout<LZMABucketTask>.stride)
        let treeSpan = finder.cyclicSize * 2
        let sonBuffer = try makeBuffer(length: treeSpan * 2 * MemoryLayout<UInt32>.stride)
        sonBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0,
            count: treeSpan * 2 * MemoryLayout<UInt32>.stride)
        let matchBuffer = try makeBuffer(length: capacity * MemoryLayout<LZMAMatch>.stride)
        let countBuffer = try makeBuffer(length: input.count * MemoryLayout<UInt32>.stride)
        let candidateBuffer = try makeBuffer(length: input.count * MemoryLayout<UInt32>.stride)
        let shortBuffer = try makeBuffer(elements: shortHeads)
        countBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: input.count * MemoryLayout<UInt32>.stride)
        var gpuSeconds = 0.0
        let limit = min(input.count, max(0, positionLimit ?? input.count))
        for waveStart in stride(from: 0, to: limit, by: finder.cyclicSize) {
            let waveEnd = min(limit, waveStart + finder.cyclicSize)
            tasks.removeAll(keepingCapacity: true)
            positions.removeAll(keepingCapacity: true)
            for bucket in bucketKeys {
                let values = buckets[bucket]!
                var offset = bucketOffsets[bucket]!
                while offset < values.count && values[offset] <= UInt32(waveStart) { offset += 1 }
                let first = offset
                while offset < values.count && values[offset] <= UInt32(waveEnd) { offset += 1 }
                if first == offset { continue }
                tasks.append(LZMABucketTask(start: UInt32(positions.count), count: UInt32(offset - first),
                                            candidate: nextCandidates[bucket]!, pad: 0))
                positions.append(contentsOf: values[first..<offset])
                nextCandidates[bucket] = values[offset - 1]
                bucketOffsets[bucket] = offset
            }
            if tasks.isEmpty { continue }
            positions.withUnsafeBytes { bytes in
                positionBuffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
            tasks.withUnsafeBytes { bytes in
                taskBuffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
            constants.taskCount = UInt32(tasks.count)
            constants.positionLimit = UInt32(waveEnd)
            constants.waveStart = UInt32(waveStart)
            constants.waveParity = UInt32((waveStart / finder.cyclicSize) & 1)
            guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                throw WriterError.compression(-1)
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(positionBuffer, offset: 0, index: 1)
            encoder.setBuffer(taskBuffer, offset: 0, index: 2)
            encoder.setBuffer(sonBuffer, offset: 0, index: 3)
            encoder.setBuffer(matchBuffer, offset: 0, index: 4)
            encoder.setBuffer(countBuffer, offset: 0, index: 5)
            encoder.setBuffer(candidateBuffer, offset: 0, index: 7)
            encoder.setBuffer(shortBuffer, offset: 0, index: 8)
            encoder.setBytes(&constants, length: MemoryLayout<LZMABucketConstants>.stride, index: 6)
            let width = min(tasks.count, pipeline.maxTotalThreadsPerThreadgroup)
            encoder.dispatchThreads(MTLSize(width: tasks.count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else { throw WriterError.compression(-1) }
            gpuSeconds += command.gpuEndTime - command.gpuStartTime
        }
        let matches = Array(UnsafeBufferPointer(
            start: matchBuffer.contents().assumingMemoryBound(to: LZMAMatch.self), count: capacity))
        let counts = Array(UnsafeBufferPointer(
            start: countBuffer.contents().assumingMemoryBound(to: UInt32.self), count: input.count))
        let candidates = Array(UnsafeBufferPointer(
            start: candidateBuffer.contents().assumingMemoryBound(to: UInt32.self), count: input.count))
        let rawSon = sonBuffer.contents().assumingMemoryBound(to: UInt32.self)
        var son = [UInt32](repeating: 0, count: treeSpan)
        if limit >= 4 {
            let last = limit - 4
            for slot in 0..<finder.cyclicSize {
                let latest = last - (last + finder.cyclicSize - slot) % finder.cyclicSize
                if latest < 0 { continue }
                let source = ((latest / finder.cyclicSize) & 1) * treeSpan + slot * 2
                son[slot * 2] = rawSon[source]
                son[slot * 2 + 1] = rawSon[source + 1]
            }
        }
        let hash = Array(UnsafeBufferPointer(start: finder.hash, count: finder.hashCount))
        return LZMABucketMetalResult(matches: matches, counts: counts, candidates: candidates, son: son, hash: hash,
                                     bucketCount: bucketKeys.count, bucketSeconds: bucketSeconds,
                                     gpuSeconds: gpuSeconds, wallSeconds: seconds(started.duration(to: .now)))
    }

    private func makeBuffer<T>(elements: [T]) throws -> any MTLBuffer {
        try elements.withUnsafeBytes { bytes in
            guard let buffer = device.makeBuffer(length: max(bytes.count, 1), options: .storageModeShared) else {
                throw WriterError.compression(-1)
            }
            if !bytes.isEmpty {
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
            return buffer
        }
    }

    private func makeBuffer(data: Data) throws -> any MTLBuffer {
        try data.withUnsafeBytes { bytes in
            guard let buffer = device.makeBuffer(length: max(bytes.count, 1), options: .storageModeShared) else {
                throw WriterError.compression(-1)
            }
            if !bytes.isEmpty {
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
            return buffer
        }
    }

    private func makeBuffer(bytes: UnsafeRawBufferPointer) throws -> any MTLBuffer {
        guard let buffer = device.makeBuffer(length: max(bytes.count, 1), options: .storageModeShared) else {
            throw WriterError.compression(-1)
        }
        if !bytes.isEmpty {
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        return buffer
    }

    private func makeBuffer(length: Int) throws -> any MTLBuffer {
        guard let buffer = device.makeBuffer(length: max(length, 1), options: .storageModeShared) else {
            throw WriterError.compression(-1)
        }
        return buffer
    }

    private func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
