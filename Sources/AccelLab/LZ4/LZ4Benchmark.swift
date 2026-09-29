private import Foundation

package enum LZ4Benchmark {
    package enum VariantSelection: String {
        case thread, simd, both

        fileprivate var variants: [LZ4MetalDecoder.Variant] {
            switch self {
            case .thread: [.threadPerBlock]
            case .simd: [.simdPerBlock]
            case .both: [.threadPerBlock, .simdPerBlock]
            }
        }
    }

    private final class OutputBuffer {
        let bytes: UnsafeMutableRawBufferPointer

        init(count: Int) {
            let allocation = UnsafeMutableRawBufferPointer.allocate(byteCount: max(count, 1), alignment: 64)
            // 全ページへのゼロ書き込みを計測前に済ませる。
            allocation.initializeMemory(as: UInt8.self, repeating: 0)
            bytes = UnsafeMutableRawBufferPointer(start: allocation.baseAddress, count: count)
        }

        deinit { bytes.deallocate() }
    }

    package static func run(
        path: String, rounds: Int, simdgroups: Int, variant: VariantSelection = .both, assumeUniform: Bool = false
    ) throws {
        do {
            try runBenchmark(path: path, rounds: rounds, simdgroups: simdgroups, variant: variant, assumeUniform: assumeUniform)
        } catch let error as LZ4Error {
            // 仮定した長さが不正なら、比較前の復元失敗も不一致として報告する。
            if assumeUniform {
                switch error {
                case .invalidBlockSize, .contentSizeMismatch, .outputTooSmall, .appleDecodeFailed:
                    print("match: MISMATCH (assume-uniform decoded sizes; \(error.localizedDescription))")
                default: break
                }
            }
            throw error
        }
    }

    private static func runBenchmark(
        path: String, rounds: Int, simdgroups: Int, variant: VariantSelection, assumeUniform: Bool
    ) throws {
        guard rounds > 0, simdgroups > 0 else {
            throw LZ4Error.invalidArgument("rounds and simdgroups must be positive")
        }
        let source = try Data(contentsOf: URL(fileURLWithPath: path))
        let frame = try LZ4Frame(source)
        let sizes: [Int]
        let scanTimes: (serial: Double, parallel: Double)?
        if assumeUniform {
            sizes = try frame.assumedUniformDecodedSizes()
            scanTimes = nil
        } else {
            let serialStart = ContinuousClock.now
            let serialSizes = try frame.expectedDecodedSizesSerial()
            let serialSeconds = seconds(serialStart.duration(to: .now))
            let parallelStart = ContinuousClock.now
            sizes = try frame.expectedDecodedSizes()
            let parallelSeconds = seconds(parallelStart.duration(to: .now))
            guard sizes == serialSizes else { throw LZ4Error.invalidArgument("Serial and parallel decoded sizes differ") }
            scanTimes = (serialSeconds, parallelSeconds)
        }
        let plan = try LZ4FrameDecoder.Plan(frame: frame, source: source, sizes: sizes)
        print("blocks: \(frame.blocks.count), blockMaxSize: \(frame.blockMaxSize), independence: \(frame.isIndependent), contentSize: \(frame.contentSize.map(String.init) ?? "unknown")")
        print("name\tbytes\tmedian_s\tGB_per_s\tnote")
        if let scanTimes {
            let exclusions = "file I/O/frame parse/output-offset table excluded"
            row("scan-serial", bytes: plan.outputCount, times: [scanTimes.serial],
                note: "once; serial decoded-size scan+size-table allocation+validation; \(exclusions)")
            row("scan-16lane", bytes: plan.outputCount, times: [scanTimes.parallel],
                note: "once; 16 static lanes; dispatch+decoded-size scan+size-table allocation+validation; \(exclusions)")
        } else {
            print("scan-skipped (assume-uniform)")
        }
        guard !frame.blocks.isEmpty else { throw LZ4Error.invalidArgument("Benchmark requires at least one block") }

        let reference = try measureCPU(plan: plan, lanes: 16, rounds: rounds)
        // 比較用の生ポインタを使い終わるまで参照出力を保持する。
        defer { withExtendedLifetime(reference.output) {} }
        let oneLane = try measureCPU(plan: plan, lanes: 1, rounds: rounds, reference: reference.output)
        let exclusions = "scan/allocation/prefault/verification excluded; output reused"
        row("cpu-1lane-apple", bytes: plan.outputCount, times: oneLane.times,
            note: "serial decode/copy loop only; \(exclusions); rounds=\(oneLane.times.count)")
        row("cpu-16lane-apple", bytes: plan.outputCount, times: reference.times,
            note: "16 static lanes; dispatch+decode/copy only; \(exclusions); rounds=\(reference.times.count)")
        let dynamic = try measureCPU(plan: plan, lanes: 16, scheduling: .dynamic, rounds: rounds, reference: reference.output)
        row("cpu-16lane-apple-dynamic", bytes: plan.outputCount, times: dynamic.times,
            note: "16 dynamic lanes; dispatch+locked block counter+decode/copy only; \(exclusions); rounds=\(dynamic.times.count)")
        let swift = try measureCPU(plan: plan, lanes: 1, algorithm: .swift, rounds: rounds, reference: reference.output)
        row("cpu-1lane-swift", bytes: plan.outputCount, times: swift.times,
            note: "serial decode/copy loop only; \(exclusions); rounds=\(swift.times.count)")

        let singleTimes = try measureSingleCPU(frame: frame, source: source, size: sizes[0],
                                               reference: reference.output, rounds: rounds)
        row("single-block-cpu-1core-apple", bytes: sizes[0], times: singleTimes,
            note: "block=0; decode/copy call only; \(exclusions); stored=\(frame.blocks[0].isStored); rounds=\(rounds)")

        // デバイスがなくても、ここまでの CPU 計測結果は出力する。
        let decoder = try LZ4MetalDecoder()
        try decoder.prepare(frame: frame, source: source, sizes: sizes)
        row("upload", bytes: plan.outputCount, times: [decoder.uploadSeconds],
            note: "once per decoder; input MTLBuffer allocation+copy; input_bytes=\(source.count); bytes=decoded")
        var allMatched = true
        for selected in variant.variants {
            let name = selected == .threadPerBlock ? "gpu-thread-per-block" : "gpu-simd-per-block"
            var gpuTimes: [Double] = []
            var wallTimes: [Double] = []
            for round in 0..<rounds {
                let result = try decoder.decode(variant: selected, simdgroupsPerThreadgroup: simdgroups)
                gpuTimes.append(result.gpuSeconds)
                wallTimes.append(result.cpuWallSeconds)
                let output = UnsafeRawBufferPointer(start: result.outputBuffer.contents(), count: result.outputCount)
                if let offset = firstDifference(output, UnsafeRawBufferPointer(reference.output.bytes)) {
                    print("match: MISMATCH (first differing offset: \(offset); \(name); round=\(round + 1))")
                    allMatched = false
                } else {
                    print("match: OK (\(name); round=\(round + 1))")
                }
                for (index, status) in result.statuses.enumerated() where status != 0 {
                    print("status: block=\(index) code=\(status) (\(name))")
                    allMatched = false
                }
            }
            row(name, bytes: plan.outputCount, times: gpuTimes,
                note: "GPU execution only; scan/upload/buffer allocation/prefault/verification excluded; buffers reused; simdgroups=\(simdgroups)")
            row(name + "-cpu-wall", bytes: plan.outputCount, times: wallTimes,
                note: "command/encoder creation+encode+commit+wait only; scan/upload/buffer allocation/prefault/status reset/verification excluded; buffers reused")
        }
        if variant != .simd {
            let singleGPU = try decoder.decodeSingleBlock(frame: frame, source: source, blockIndex: 0, rounds: rounds)
            row("single-block-gpu-1thread", bytes: sizes[0], times: singleGPU,
                note: "GPU execution only; block=0; scan/upload/buffer allocation/prefault/verification excluded; buffers reused; rounds=\(rounds)")
        }
        guard allMatched else { throw LZ4Error.metalFailure("Output or status mismatch") }
    }

    private static func measureCPU(
        plan: LZ4FrameDecoder.Plan, lanes: Int, algorithm: LZ4FrameDecoder.Algorithm = .apple,
        scheduling: LZ4FrameDecoder.Scheduling = .staticPartition, rounds: Int, reference: OutputBuffer? = nil
    ) throws -> (output: OutputBuffer, times: [Double]) {
        let output = OutputBuffer(count: plan.outputCount)
        var times: [Double] = []
        let swiftRoundLimitSeconds = 5.0
        for round in 0..<rounds {
            let elapsed = try LZ4FrameDecoder.decodeCPU(plan: plan, into: output.bytes, lanes: lanes,
                                                       algorithm: algorithm, scheduling: scheduling)
            times.append(elapsed)
            if let reference {
                try withExtendedLifetime((output, reference)) {
                    if let offset = firstDifference(UnsafeRawBufferPointer(output.bytes), UnsafeRawBufferPointer(reference.bytes)) {
                        throw LZ4Error.invalidArgument("CPU output mismatch at offset \(offset)")
                    }
                }
            }
            if algorithm == .swift, round == 0, elapsed > swiftRoundLimitSeconds { break }
        }
        return (output, times)
    }

    private static func measureSingleCPU(
        frame: LZ4Frame, source: Data, size: Int, reference: OutputBuffer, rounds: Int
    ) throws -> [Double] {
        let block = frame.blocks[0]
        let output = OutputBuffer(count: size)
        defer { withExtendedLifetime((output, reference)) {} }
        return try source.withUnsafeBytes { bytes in
            let input = UnsafeRawBufferPointer(rebasing:
                bytes[block.compressedOffset..<(block.compressedOffset + block.compressedLength)])
            let destination = output.bytes
            var times: [Double] = []
            for _ in 0..<rounds {
                let start = ContinuousClock.now
                let count: Int
                if block.isStored {
                    if size > 0 { destination.baseAddress!.copyMemory(from: input.baseAddress!, byteCount: size) }
                    count = size
                } else {
                    count = LZ4BlockDecoder.decodeApple(input, into: destination)
                }
                times.append(seconds(start.duration(to: .now)))
                guard count == size, destination.elementsEqual(reference.bytes.prefix(size)) else {
                    throw LZ4Error.appleDecodeFailed(0)
                }
            }
            return times
        }
    }

    private static func firstDifference(_ output: UnsafeRawBufferPointer, _ expected: UnsafeRawBufferPointer) -> Int? {
        for index in 0..<min(output.count, expected.count) where output[index] != expected[index] { return index }
        return output.count == expected.count ? nil : min(output.count, expected.count)
    }

    private static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }

    private static func row(_ name: String, bytes: Int, times: [Double], note: String) {
        let sorted = times.sorted()
        let middle = sorted.count / 2
        let median = sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
        let locale = Locale(identifier: "en_US_POSIX")
        let elapsed = String(format: "%.9f", locale: locale, median)
        let speed = String(format: "%.6f", locale: locale, median > 0 ? Double(bytes) / median / 1e9 : 0)
        print("\(name)\t\(bytes)\t\(elapsed)\t\(speed)\t\(note)")
    }
}
