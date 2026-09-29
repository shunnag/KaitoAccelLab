public import Foundation
public import Dispatch

/// 各バイト位置で全ブロックを一括予測し、独立した range coder に渡す。
public enum NeuralBlockCodec {
    public static func encode(data: Data, blockCount: Int, predictor: any BytePredictor,
                              predictorTag: String) throws -> Data {
        var timings = NeuralCodecTimings()
        return try encode(data: data, blockCount: blockCount, predictor: predictor,
                          predictorTag: predictorTag, lanes: nil, timings: &timings)
    }

    public static func encode(data: Data, blockCount: Int, predictor: any BytePredictor,
                              predictorTag: String, timings: inout NeuralCodecTimings) throws -> Data {
        try encode(data: data, blockCount: blockCount, predictor: predictor,
                   predictorTag: predictorTag, lanes: nil, timings: &timings)
    }

    internal static func encode(data: Data, blockCount: Int, predictor: any BytePredictor,
                                predictorTag: String, lanes: Int?, timings: inout NeuralCodecTimings) throws -> Data {
        guard blockCount > 0, UInt32(exactly: blockCount) != nil else {
            throw NeuralCodecError("blockCount must be in 1...\(UInt32.max)")
        }
        guard predictorTag.utf8.count <= Int(UInt16.max) else { throw NeuralCodecError("Predictor tag is too long") }
        let rowCount = try PredictorValidation.elementCount(blockCount: blockCount, width: FrequencyQuantizer.symbolCount)
        let input = Array(data)
        let blockSize = size(length: input.count, blockCount: blockCount)
        let lengths = blockLengths(length: input.count, blockCount: blockCount, blockSize: blockSize)
        let laneCount = try laneCount(blockCount: blockCount, requested: lanes)
        var encoders = [RangeEncoder](repeating: RangeEncoder(), count: blockCount)
        var observed = [UInt8](repeating: 0, count: blockCount)
        var cumulative = [UInt32](repeating: 0, count: laneCount * (FrequencyQuantizer.symbolCount + 1))
        timings = NeuralCodecTimings()
        try predictor.begin(blockCount: blockCount)
        let clock = ContinuousClock()
        try input.withUnsafeBufferPointer { input in
            try lengths.withUnsafeBufferPointer { lengths in
                try encoders.withUnsafeMutableBufferPointer { encoders in
                    try cumulative.withUnsafeMutableBufferPointer { cumulative in
                        for position in 0..<blockSize {
                            let predictionStart = clock.now
                            let probabilities = try predictor.predictNext()
                            timings.predictorDuration += predictionStart.duration(to: clock.now)
                            guard probabilities.count == rowCount else {
                                throw NeuralCodecError("Predictor returned an invalid probability count")
                            }
                            let coderStart = clock.now
                            probabilities.withUnsafeBufferPointer { rows in
                                observed.withUnsafeMutableBufferPointer { observed in
                                    let work = BlockEncodingWork(input: input, lengths: lengths, probabilities: rows,
                                                                 encoders: encoders, observed: observed, cumulative: cumulative,
                                                                 laneCount: laneCount, blockSize: blockSize, position: position)
                                    DispatchQueue.concurrentPerform(iterations: laneCount) { work.run(lane: $0) }
                                }
                            }
                            timings.coderDuration += coderStart.duration(to: clock.now)
                            timings.steps += 1
                            try predictor.observe(observed)
                        }
                    }
                }
            }
        }
        var payloadLengths: [UInt32] = []
        payloadLengths.reserveCapacity(blockCount)
        for block in 0..<blockCount {
            guard let length = UInt32(exactly: encoders[block].finish().count) else {
                throw NeuralCodecError("Encoded block exceeds UInt32 payload length")
            }
            payloadLengths.append(length)
        }
        let header = Header(version: Header.currentVersion, blockCount: UInt32(blockCount),
                            originalLength: UInt64(input.count), predictorTag: predictorTag, payloadLengths: payloadLengths)
        var container: [UInt8] = []
        header.append(to: &container)
        for block in 0..<blockCount { container.append(contentsOf: encoders[block].finish()) }
        return Data(container)
    }

    public static func decode(_ data: Data, predictor: any BytePredictor) throws -> (payload: Data, header: Header) {
        var timings = NeuralCodecTimings()
        return try decode(data, predictor: predictor, lanes: nil, timings: &timings)
    }

    public static func decode(_ data: Data, predictor: any BytePredictor,
                              timings: inout NeuralCodecTimings) throws -> (payload: Data, header: Header) {
        try decode(data, predictor: predictor, lanes: nil, timings: &timings)
    }

    internal static func decode(_ data: Data, predictor: any BytePredictor, lanes: Int?,
                                timings: inout NeuralCodecTimings) throws -> (payload: Data, header: Header) {
        let container = Array(data)
        let (header, payloadOffset) = try Header.parse(container)
        let blockCount = Int(header.blockCount)
        let rowCount = try PredictorValidation.elementCount(blockCount: blockCount, width: FrequencyQuantizer.symbolCount)
        let length = Int(header.originalLength)
        let blockSize = size(length: length, blockCount: blockCount)
        let lengths = blockLengths(length: length, blockCount: blockCount, blockSize: blockSize)
        let laneCount = try laneCount(blockCount: blockCount, requested: lanes)
        var decoders: [RangeDecoder] = []
        decoders.reserveCapacity(blockCount)
        var offset = payloadOffset
        for payloadLength in header.payloadLengths {
            let end = offset + Int(payloadLength)
            decoders.append(RangeDecoder(container[offset..<end]))
            offset = end
        }
        var output = [UInt8](repeating: 0, count: length)
        var observed = [UInt8](repeating: 0, count: blockCount)
        var cumulative = [UInt32](repeating: 0, count: laneCount * (FrequencyQuantizer.symbolCount + 1))
        var failures = [(any Error)?](repeating: nil, count: laneCount)
        timings = NeuralCodecTimings()
        try predictor.begin(blockCount: blockCount)
        let clock = ContinuousClock()
        try lengths.withUnsafeBufferPointer { lengths in
            try decoders.withUnsafeMutableBufferPointer { decoders in
                try output.withUnsafeMutableBufferPointer { output in
                    try cumulative.withUnsafeMutableBufferPointer { cumulative in
                        try failures.withUnsafeMutableBufferPointer { failures in
                            for position in 0..<blockSize {
                                let predictionStart = clock.now
                                let probabilities = try predictor.predictNext()
                                timings.predictorDuration += predictionStart.duration(to: clock.now)
                                guard probabilities.count == rowCount else {
                                    throw NeuralCodecError("Predictor returned an invalid probability count")
                                }
                                let coderStart = clock.now
                                probabilities.withUnsafeBufferPointer { rows in
                                    observed.withUnsafeMutableBufferPointer { observed in
                                        let work = BlockDecodingWork(lengths: lengths, probabilities: rows, decoders: decoders,
                                                                     output: output, observed: observed, cumulative: cumulative,
                                                                     failures: failures, laneCount: laneCount,
                                                                     blockSize: blockSize, position: position)
                                        DispatchQueue.concurrentPerform(iterations: laneCount) { work.run(lane: $0) }
                                    }
                                }
                                // レーン順に調べ、並列実行でも最初のブロックのエラーを返す。
                                for failure in failures { if let failure { throw failure } }
                                timings.coderDuration += coderStart.duration(to: clock.now)
                                timings.steps += 1
                                try predictor.observe(observed)
                            }
                        }
                    }
                }
            }
        }
        return (Data(output), header)
    }

    private static func size(length: Int, blockCount: Int) -> Int {
        length / blockCount + (length.isMultiple(of: blockCount) ? 0 : 1)
    }

    private static func laneCount(blockCount: Int, requested: Int?) throws -> Int {
        let count = requested ?? ProcessInfo.processInfo.activeProcessorCount
        guard count > 0 else { throw NeuralCodecError("lanes must be positive") }
        return min(blockCount, count)
    }

    internal static func blockRange(lane: Int, lanes: Int, blockCount: Int) -> Range<Int> {
        let width = blockCount / lanes
        let extra = blockCount % lanes
        let start = lane * width + min(lane, extra)
        return start..<(start + width + (lane < extra ? 1 : 0))
    }

    private static func blockLengths(length: Int, blockCount: Int, blockSize: Int) -> [Int] {
        var remaining = length
        return (0..<blockCount).map { _ in
            let count = min(remaining, blockSize)
            remaining -= count
            return count
        }
    }
}
