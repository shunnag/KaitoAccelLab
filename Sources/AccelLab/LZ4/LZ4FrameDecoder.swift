internal import Foundation
private import Dispatch
private import os

enum LZ4FrameDecoder {
    enum Algorithm: Sendable { case apple, swift }
    enum Scheduling: Sendable { case staticPartition, dynamic }

    struct Plan: Sendable {
        let source: Data
        let blocks: [LZ4Frame.Block]
        let sizes: [Int]
        let offsets: [Int]
        let outputCount: Int

        // sizes には、このフレームの expectedDecodedSizes() の結果を渡す。
        init(frame: LZ4Frame, source: Data, sizes: [Int]) throws {
            try frame.validate(source: source)
            guard sizes.count == frame.blocks.count else { throw LZ4Error.invalidBlockSize }
            var offsets: [Int] = []
            var total = 0
            for (index, size) in sizes.enumerated() {
                guard (0...frame.blockMaxSize).contains(size),
                      !frame.blocks[index].isStored || size == frame.blocks[index].compressedLength else {
                    throw LZ4Error.invalidBlockSize
                }
                offsets.append(total)
                let sum = total.addingReportingOverflow(size)
                guard !sum.overflow else { throw LZ4Error.sizeOverflow }
                total = sum.partialValue
            }
            if let contentSize = frame.contentSize, contentSize != UInt64(total) {
                throw LZ4Error.contentSizeMismatch
            }
            self.source = source
            blocks = frame.blocks
            self.sizes = sizes
            self.offsets = offsets
            outputCount = total
        }
    }

    // 各レーンは別々のブロックと結果欄だけを書き換える。ポインタの寿命は同期呼び出し内に限る。
    private struct Work: @unchecked Sendable {
        let source: UnsafeRawBufferPointer
        let destination: UnsafeMutableRawBufferPointer
        let plan: Plan
        let failures: UnsafeMutablePointer<LZ4Error?>
        let lanes: Int
        let algorithm: Algorithm

        func run(lane: Int) {
            let quotient = plan.blocks.count / lanes
            let remainder = plan.blocks.count % lanes
            let start = lane * quotient + min(lane, remainder)
            let end = start + quotient + (lane < remainder ? 1 : 0)
            for index in start..<end { decode(index: index) }
        }

        func decode(index: Int) {
            let block = plan.blocks[index]
            let input = UnsafeRawBufferPointer(rebasing:
                source[block.compressedOffset..<(block.compressedOffset + block.compressedLength)])
            let output = UnsafeMutableRawBufferPointer(rebasing:
                destination[plan.offsets[index]..<(plan.offsets[index] + plan.sizes[index])])
            if block.isStored {
                if !input.isEmpty { output.baseAddress!.copyMemory(from: input.baseAddress!, byteCount: input.count) }
            } else {
                do {
                    let count: Int
                    switch algorithm {
                    case .apple: count = LZ4BlockDecoder.decodeApple(input, into: output)
                    case .swift: count = try LZ4BlockDecoder.decodeSwift(input, into: output)
                    }
                    if count != plan.sizes[index] { failures[index] = .appleDecodeFailed(index) }
                } catch let error as LZ4Error {
                    failures[index] = error
                } catch {
                    failures[index] = .invalidArgument(error.localizedDescription)
                }
            }
        }
    }

    static func decodeCPU(
        frame: LZ4Frame, source: Data, lanes: Int, algorithm: Algorithm = .apple
    ) throws -> [UInt8] {
        guard lanes > 0 else { throw LZ4Error.invalidArgument("lanes must be positive") }
        try frame.validate(source: source)
        let plan = try Plan(frame: frame, source: source, sizes: frame.expectedDecodedSizes())
        var output = [UInt8](repeating: 0, count: plan.outputCount)
        try output.withUnsafeMutableBytes {
            _ = try decodeCPU(plan: plan, into: $0, lanes: lanes, algorithm: algorithm)
        }
        return output
    }

    // 準備済みサイズと出力領域を使い、復元ループまたは dispatch の経過秒数だけを返す。
    @discardableResult
    static func decodeCPU(
        frame: LZ4Frame, source: Data, sizes: [Int], into destination: UnsafeMutableRawBufferPointer,
        lanes: Int, algorithm: Algorithm = .apple, scheduling: Scheduling = .staticPartition
    ) throws -> Double {
        let plan = try Plan(frame: frame, source: source, sizes: sizes)
        return try decodeCPU(plan: plan, into: destination, lanes: lanes, algorithm: algorithm, scheduling: scheduling)
    }

    @discardableResult
    static func decodeCPU(
        plan: Plan, into destination: UnsafeMutableRawBufferPointer,
        lanes: Int, algorithm: Algorithm = .apple, scheduling: Scheduling = .staticPartition
    ) throws -> Double {
        guard lanes > 0 else { throw LZ4Error.invalidArgument("lanes must be positive") }
        guard destination.count >= plan.outputCount else { throw LZ4Error.outputTooSmall }
        if plan.blocks.isEmpty { return 0 }
        let failures = UnsafeMutablePointer<LZ4Error?>.allocate(capacity: plan.sizes.count)
        failures.initialize(repeating: nil, count: plan.sizes.count)
        defer { failures.deinitialize(count: plan.sizes.count); failures.deallocate() }
        let counter = OSAllocatedUnfairLock(initialState: 0)
        let elapsed = plan.source.withUnsafeBytes { input in
            let work = Work(source: input, destination: destination, plan: plan,
                            failures: failures, lanes: lanes, algorithm: algorithm)
            // 検証・メタデータ確保・ポインタ取得は計測区間に含めない。
            let start = ContinuousClock.now
            if scheduling == .dynamic {
                DispatchQueue.concurrentPerform(iterations: lanes) { _ in
                    while let index = counter.withLock({ next -> Int? in
                        guard next < plan.blocks.count else { return nil }
                        defer { next += 1 }
                        return next
                    }) {
                        work.decode(index: index)
                    }
                }
            } else if lanes == 1 {
                work.run(lane: 0)
            } else {
                DispatchQueue.concurrentPerform(iterations: lanes) { work.run(lane: $0) }
            }
            return start.duration(to: .now)
        }
        for index in plan.sizes.indices { if let failure = failures[index] { throw failure } }
        let parts = elapsed.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
