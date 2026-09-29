// 全ポインタは同期的な concurrentPerform の終了まで有効。各レーンは互いに
// 重ならない coder・観測値・累積頻度だけを書き換え、入力と確率は読み取り専用。
internal struct BlockEncodingWork: @unchecked Sendable {
    let input: UnsafeBufferPointer<UInt8>
    let lengths: UnsafeBufferPointer<Int>
    let probabilities: UnsafeBufferPointer<Float>
    let encoders: UnsafeMutableBufferPointer<RangeEncoder>
    let observed: UnsafeMutableBufferPointer<UInt8>
    let cumulative: UnsafeMutableBufferPointer<UInt32>
    let laneCount: Int
    let blockSize: Int
    let position: Int

    func run(lane: Int) {
        let blocks = NeuralBlockCodec.blockRange(lane: lane, lanes: laneCount, blockCount: encoders.count)
        let frequencies = cumulative.baseAddress!.advanced(by: lane * (FrequencyQuantizer.symbolCount + 1))
        let rows = probabilities.baseAddress!
        let coders = encoders.baseAddress!
        let observations = observed.baseAddress!
        let bytes = input.baseAddress!
        let counts = lengths.baseAddress!
        for block in blocks {
            observations[block] = 0
            guard position < counts[block] else { continue }
            let byte = bytes[block * blockSize + position]
            FrequencyQuantizer.fill(rows.advanced(by: block * FrequencyQuantizer.symbolCount), cumulative: frequencies)
            coders[block].encode(byte, cumulative: UnsafePointer(frequencies))
            observations[block] = byte
        }
    }
}
