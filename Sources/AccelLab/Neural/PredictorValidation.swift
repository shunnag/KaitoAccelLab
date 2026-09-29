internal enum PredictorValidation {
    static func elementCount(blockCount: Int, width: Int) throws -> Int {
        let (count, overflow) = blockCount.multipliedReportingOverflow(by: width)
        guard blockCount > 0, width > 0, !overflow, count <= Int.max / MemoryLayout<UInt64>.stride else {
            throw NeuralCodecError("Invalid predictor block count or shape: \(blockCount) x \(width)")
        }
        return count
    }

    static func started(_ blockCount: Int) throws {
        guard blockCount > 0 else { throw NeuralCodecError("Call begin(blockCount:) before prediction or observation") }
    }

    static func observation(_ bytes: [UInt8], blockCount: Int) throws {
        try started(blockCount)
        guard bytes.count == blockCount else {
            throw NeuralCodecError("Expected \(blockCount) observed bytes, got \(bytes.count)")
        }
    }
}
