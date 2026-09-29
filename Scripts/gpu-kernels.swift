// GPU 側の byte 単位カーネル（histogram、chunk CRC-32、読み出し帯域）を CPU 16 レーンと比べる実験。
// 使い方: gpu-kernels <file> [rounds] [crcChunkBytes]   （file の先頭 256 MiB を使う）
import Foundation
import Metal
import zlib

let path = CommandLine.arguments[1]
let rounds = CommandLine.arguments.count > 2 ? Int(CommandLine.arguments[2])! : 5
let byteCount = 256 << 20
let chunkSize = CommandLine.arguments.count > 3 ? Int(CommandLine.arguments[3])! : 4096
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void read_sum(const device uint4 *input, device atomic_uint *out, uint gid [[thread_position_in_grid]], uint count [[threads_per_grid]]) {
    uint4 acc = 0;
    for (uint i = gid; i < \(byteCount / 16); i += count) acc += input[i];
    uint s = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (s == 0x12345678u) atomic_fetch_add_explicit(out, 1u, memory_order_relaxed); // 最適化で消えないための依存
}
kernel void histogram(const device uchar4 *input, device atomic_uint *out, uint gid [[thread_position_in_grid]], uint count [[threads_per_grid]], uint lid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]]) {
    threadgroup atomic_uint local[256];
    for (uint i = lid; i < 256; i += tsize) atomic_store_explicit(&local[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = gid; i < \(byteCount / 4); i += count) {
        uchar4 v = input[i];
        atomic_fetch_add_explicit(&local[v.x], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&local[v.y], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&local[v.z], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&local[v.w], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < 256; i += tsize) atomic_fetch_add_explicit(&out[i], atomic_load_explicit(&local[i], memory_order_relaxed), memory_order_relaxed);
}
// 1 thread が 1 chunk (4 KiB) の CRC-32 を slice-by-4 で計算する。結合は CPU の crc32_combine。
kernel void crc32_chunks(const device uchar *input, device uint *out, const device uint *tables, uint gid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]]) {
    threadgroup uint t[1024];
    for (uint i = lid; i < 1024; i += tsize) t[i] = tables[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uint *words = (const device uint *)(input + gid * \(chunkSize));
    uint crc = 0xFFFFFFFFu;
    for (uint i = 0; i < \(chunkSize / 4); i++) {
        uint w = words[i] ^ crc;
        crc = t[768 + (w & 0xFF)] ^ t[512 + ((w >> 8) & 0xFF)] ^ t[256 + ((w >> 16) & 0xFF)] ^ t[(w >> 24) & 0xFF];
    }
    out[gid] = crc ^ 0xFFFFFFFFu;
}
"""
func median(_ xs: [Double]) -> Double { let s = xs.sorted(); return s[s.count / 2] }
func now() -> Double { CFAbsoluteTimeGetCurrent() }

let data = try! FileHandle(forReadingFrom: URL(fileURLWithPath: path)).read(upToCount: byteCount)!
precondition(data.count == byteCount)
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let library = try! device.makeLibrary(source: source, options: nil)
let input = device.makeBuffer(length: byteCount, options: .storageModeShared)!
data.withUnsafeBytes { input.contents().copyMemory(from: $0.baseAddress!, byteCount: byteCount) }
let inputBytes = input.contents().assumingMemoryBound(to: UInt8.self)

// slice-by-4 の表
var tables = [UInt32](repeating: 0, count: 1024)
for i in 0..<256 {
    var c = UInt32(i)
    for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1 }
    tables[i] = c
}
for k in 1..<4 { for i in 0..<256 { let prev = tables[(k - 1) * 256 + i]; tables[k * 256 + i] = tables[Int(prev & 0xFF)] ^ (prev >> 8) } }
let tableBuffer = device.makeBuffer(bytes: tables, length: 4096, options: .storageModeShared)!

func run(_ name: String, threads: Int, threadgroup: Int, outLength: Int, setup: (MTLComputeCommandEncoder) -> Void) -> (gpu: Double, wall: Double, out: MTLBuffer) {
    let pipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
    let out = device.makeBuffer(length: outLength, options: .storageModeShared)!
    var gpuTimes: [Double] = [], wallTimes: [Double] = []
    for _ in 0..<rounds {
        memset(out.contents(), 0, outLength)
        let t0 = now()
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        setup(enc)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadgroup, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        wallTimes.append(now() - t0)
        gpuTimes.append(cb.gpuEndTime - cb.gpuStartTime)
    }
    return (median(gpuTimes), median(wallTimes), out)
}
func row(_ name: String, _ seconds: Double, _ note: String) {
    print("\(name)\t\(byteCount)\t\(String(format: "%.6f", seconds))\t\(String(format: "%.2f", Double(byteCount) / seconds / 1e9))\t\(note)")
}
print("name\tbytes\tmedian_s\tGB_per_s\tnote")

// GPU 読み出し帯域
let occupancy = 40 * 1024 * 8
let rs = run("read_sum", threads: occupancy, threadgroup: 256, outLength: 4) { $0.setBuffer(input, offset: 0, index: 0) }
row("gpu-read-sum", rs.gpu, "gpu time; wall \(String(format: "%.6f", rs.wall))")

// histogram
let hg = run("histogram", threads: occupancy, threadgroup: 256, outLength: 1024) { $0.setBuffer(input, offset: 0, index: 0) }
let gpuHist = Array(UnsafeBufferPointer(start: hg.out.contents().assumingMemoryBound(to: UInt32.self), count: 256))
var cpuTimes: [Double] = []
var cpuHist = [UInt32](repeating: 0, count: 256)
for _ in 0..<rounds {
    let lanes = 16
    var partial = [[UInt32]](repeating: [UInt32](repeating: 0, count: 256), count: lanes)
    let t0 = now()
    partial.withUnsafeMutableBufferPointer { p in
        DispatchQueue.concurrentPerform(iterations: lanes) { lane in
            var h = [UInt32](repeating: 0, count: 256)
            let start = byteCount / lanes * lane, end = byteCount / lanes * (lane + 1)
            h.withUnsafeMutableBufferPointer { hb in for i in start..<end { hb[Int(inputBytes[i])] &+= 1 } }
            p[lane] = h
        }
    }
    cpuHist = (0..<256).map { b in partial.reduce(0) { $0 &+ $1[b] } }
    cpuTimes.append(now() - t0)
}
row("gpu-histogram", hg.gpu, "gpu time; wall \(String(format: "%.6f", hg.wall)); match \(gpuHist == cpuHist ? "OK" : "MISMATCH")")
row("cpu-histogram-16lane", median(cpuTimes), "concurrentPerform 16; private counts")

// CRC-32: GPU per-chunk + CPU combine vs zlib 16 lanes
let chunks = byteCount / chunkSize
let cr = run("crc32_chunks", threads: chunks, threadgroup: 256, outLength: chunks * 4) { enc in
    enc.setBuffer(input, offset: 0, index: 0); enc.setBuffer(tableBuffer, offset: 0, index: 2)
}
let chunkCRCs = UnsafeBufferPointer(start: cr.out.contents().assumingMemoryBound(to: UInt32.self), count: chunks)
var combineTimes: [Double] = []
var combined: UInt = 0
for _ in 0..<rounds {
    // 結合は結合律を満たすので 16 lane で部分結合してから直列に 15 回結合する。
    let lanes = 16, per = chunks / lanes
    var parts = [UInt](repeating: 0, count: lanes)
    let t0 = now()
    parts.withUnsafeMutableBufferPointer { p in
        DispatchQueue.concurrentPerform(iterations: lanes) { lane in
            var acc = UInt(chunkCRCs[lane * per])
            for i in 1..<per { acc = crc32_combine(acc, UInt(chunkCRCs[lane * per + i]), chunkSize) }
            p[lane] = acc
        }
    }
    combined = parts[0]
    for i in 1..<lanes { combined = crc32_combine(combined, parts[i], per * chunkSize) }
    combineTimes.append(now() - t0)
}
let reference = data.withUnsafeBytes { crc32(0, $0.baseAddress!.assumingMemoryBound(to: Bytef.self), uInt(byteCount)) }
row("gpu-crc32-chunks", cr.gpu, "gpu time; wall \(String(format: "%.6f", cr.wall)); \(chunks) chunks × \(chunkSize) B; slice-by-4 in threadgroup memory")
row("cpu-crc32-combine-of-gpu-chunks", median(combineTimes), "crc32_combine 16 lane tree; match \(combined == reference ? "OK" : "MISMATCH")")
row("gpu-crc32-total", cr.gpu + median(combineTimes), "gpu kernel + combine")
var zl: [Double] = []
for _ in 0..<rounds {
    let lanes = 16
    var parts = [UInt](repeating: 0, count: lanes)
    let t0 = now()
    parts.withUnsafeMutableBufferPointer { p in
        DispatchQueue.concurrentPerform(iterations: lanes) { lane in
            let n = byteCount / lanes
            p[lane] = crc32(0, inputBytes + lane * n, uInt(n))
        }
    }
    var acc = parts[0]
    for i in 1..<lanes { acc = crc32_combine(acc, parts[i], byteCount / lanes) }
    zl.append(now() - t0)
    precondition(acc == reference)
}
row("cpu-crc32-zlib-16lane", median(zl), "reference")
