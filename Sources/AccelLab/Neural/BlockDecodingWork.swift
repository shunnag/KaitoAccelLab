// 全ポインタは同期的な concurrentPerform の終了まで有効。各レーンは独立した
// coder・出力位置・観測値・累積頻度・エラー枠を所有し、復号失敗も枠ごとに保存する。
internal struct BlockDecodingWork: @unchecked Sendable {
    let lengths: UnsafeBufferPointer<Int>
    let probabilities: UnsafeBufferPointer<Float>
    let decoders: UnsafeMutableBufferPointer<RangeDecoder>
    let output: UnsafeMutableBufferPointer<UInt8>
    let observed: UnsafeMutableBufferPointer<UInt8>
    let cumulative: UnsafeMutableBufferPointer<UInt32>
    let failures: UnsafeMutableBufferPointer<(any Error)?>
    let laneCount: Int
    let blockSize: Int
    let position: Int

    func run(lane: Int) {
        let blocks = NeuralBlockCodec.blockRange(lane: lane, lanes: laneCount, blockCount: decoders.count)
        let frequencies = cumulative.baseAddress!.advanced(by: lane * (FrequencyQuantizer.symbolCount + 1))
        let rows = probabilities.baseAddress!
        let coders = decoders.baseAddress!
        let observations = observed.baseAddress!
        let bytes = output.baseAddress!
        let counts = lengths.baseAddress!
        do {
            for block in blocks {
                observations[block] = 0
                guard position < counts[block] else { continue }
                FrequencyQuantizer.fill(rows.advanced(by: block * FrequencyQuantizer.symbolCount), cumulative: frequencies)
                let byte = try coders[block].decode(cumulative: UnsafePointer(frequencies))
                bytes[block * blockSize + position] = byte
                observations[block] = byte
            }
        } catch {
            failures[lane] = error
        }
    }
}
