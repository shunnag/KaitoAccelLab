/// 最初は全ゼロ、観測後は直前のバイトの one-hot を生成する CPU 参照状態。
internal final class RecurrentByteState {
    private let blockCount: Int
    private let last: UnsafeMutablePointer<UInt8>
    private let written: UnsafeMutablePointer<UInt8>
    private var hasObservation = false
    private var initialized = false

    init(blockCount: Int) throws {
        _ = try PredictorValidation.elementCount(blockCount: blockCount, width: FrequencyQuantizer.symbolCount)
        self.blockCount = blockCount
        last = .allocate(capacity: blockCount * 2)
        last.initialize(repeating: 0, count: blockCount * 2)
        written = last.advanced(by: blockCount)
    }

    deinit {
        last.deinitialize(count: blockCount * 2)
        last.deallocate()
    }

    func observe(_ bytes: [UInt8]) throws {
        try PredictorValidation.observation(bytes, blockCount: blockCount)
        bytes.withUnsafeBufferPointer { [last, blockCount] source in
            last.update(from: source.baseAddress!, count: blockCount)
        }
        hasObservation = true
    }

    // 同じ入力バッファを再利用し、初回以外は前回と今回の列だけを更新する。
    func write(to destination: UnsafeMutablePointer<Float16>, rowStride: Int, columnStride: Int) {
        if !initialized {
            for block in 0..<blockCount {
                let row = destination.advanced(by: block * rowStride)
                for symbol in 0..<FrequencyQuantizer.symbolCount { row[symbol * columnStride] = 0 }
            }
            initialized = true
        }
        guard hasObservation else { return }
        for block in 0..<blockCount {
            let row = destination.advanced(by: block * rowStride)
            row[Int(written[block]) * columnStride] = 0
            row[Int(last[block]) * columnStride] = 1
            written[block] = last[block]
        }
    }
}
