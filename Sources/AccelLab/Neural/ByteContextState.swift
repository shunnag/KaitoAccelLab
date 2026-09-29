/// 左側をゼロで埋め、古い順に文脈を取り出す CPU 参照状態。
internal struct ByteContextState {
    let blockCount: Int
    let context: Int
    private var bytes: [UInt8]
    private var cursor = 0
    private static let byteScale: Float = 255

    init(blockCount: Int, context: Int) throws {
        let count = try PredictorValidation.elementCount(blockCount: blockCount, width: context)
        self.blockCount = blockCount
        self.context = context
        bytes = [UInt8](repeating: 0, count: count)
    }

    mutating func observe(_ observed: [UInt8]) throws {
        try PredictorValidation.observation(observed, blockCount: blockCount)
        for block in 0..<blockCount { bytes[block * context + cursor] = observed[block] }
        cursor = (cursor + 1) % context
    }

    func write(to destination: UnsafeMutableBufferPointer<Float16>, rowStride: Int, columnStride: Int) {
        for block in 0..<blockCount {
            for column in 0..<context {
                let byte = bytes[block * context + (cursor + column) % context]
                destination[block * rowStride + column * columnStride] = Float16(Float(byte) / Self.byteScale)
            }
        }
    }
}
