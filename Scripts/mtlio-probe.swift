// Metal 4 の MTLIO（MTLIOCompressionContext で圧縮した file を MTLIOCommandQueue で読み込む）が CPU 以外で展開するかを調べる。
// 使い方: mtlio-probe <input-file(256 MiB 以下)> [rounds]
import Foundation
import Metal
import Compression

let path = CommandLine.arguments[1]
let rounds = CommandLine.arguments.count > 2 ? Int(CommandLine.arguments[2])! : 5
let data = try! Data(contentsOf: URL(fileURLWithPath: path))
let device = MTLCreateSystemDefaultDevice()!
func median(_ xs: [Double]) -> Double { let s = xs.sorted(); return s[s.count / 2] }
func now() -> Double { CFAbsoluteTimeGetCurrent() }
print("name\tbytes\tmedian_s\tGB_per_s\tnote")
for (method, name) in [(MTLIOCompressionMethod.lz4, "lz4"), (.lzfse, "lzfse"), (.zlib, "zlib"), (.lzBitmap, "lzbitmap")] {
    let out = NSTemporaryDirectory() + "mtlio-probe-\(name).bin"
    let chunk = MTLIOCompressionContextDefaultChunkSize()
    let t0 = now()
    guard let ctx = MTLIOCreateCompressionContext(out, method, chunk) else { print("\(name)\tcontext failed"); continue }
    data.withUnsafeBytes { MTLIOCompressionContextAppendData(ctx, $0.baseAddress!, data.count) }
    let status = MTLIOFlushAndDestroyCompressionContext(ctx)
    let compressTime = now() - t0
    let compressed = (try? FileManager.default.attributesOfItem(atPath: out)[.size] as? Int) ?? 0
    print("mtlio-compress-\(name)\t\(data.count)\t\(String(format: "%.6f", compressTime))\t\(String(format: "%.3f", Double(data.count) / compressTime / 1e9))\tstatus=\(status.rawValue); chunk=\(chunk); compressed_bytes=\(compressed); ratio=\(String(format: "%.3f", Double(compressed) / Double(data.count)))")
    let queueDesc = MTLIOCommandQueueDescriptor()
    queueDesc.type = .concurrent
    let queue = try! device.makeIOCommandQueue(descriptor: queueDesc)
    let handle = try! device.makeIOFileHandle(url: URL(fileURLWithPath: out), compressionMethod: method)
    let buffer = device.makeBuffer(length: data.count, options: .storageModeShared)!
    var times: [Double] = []
    for _ in 0..<rounds {
        memset(buffer.contents(), 0, data.count)
        let t1 = now()
        let cb = queue.makeCommandBuffer()
        cb.load(buffer, offset: 0, size: data.count, sourceHandle: handle, sourceHandleOffset: 0)
        cb.commit(); cb.waitUntilCompleted()
        times.append(now() - t1)
        precondition(cb.status == .complete, "MTLIO load failed: \(String(describing: cb.error))")
    }
    let same = data.withUnsafeBytes { memcmp($0.baseAddress!, buffer.contents(), data.count) == 0 }
    print("mtlio-load-\(name)\t\(data.count)\t\(String(format: "%.6f", median(times)))\t\(String(format: "%.3f", Double(data.count) / median(times) / 1e9))\twall (commit+wait, file cached); match \(same ? "OK" : "MISMATCH")")
    // 参照: 同じ元 data を Apple Compression の 16 lane で（chunk ごとに）展開する時間は gpu-kernels / lz4-gpu の CPU 行を見る。
    try? FileManager.default.removeItem(atPath: out)
}
