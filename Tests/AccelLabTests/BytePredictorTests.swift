public import XCTest
@testable public import AccelLab

final class BytePredictorTests: XCTestCase {
    func testOrderZeroCountsAreIndependentAndReset() throws {
        let predictor = OrderZeroPredictor()
        try predictor.begin(blockCount: 2)
        XCTAssertEqual(try predictor.predictNext(), [Float](repeating: 1, count: 512))
        try predictor.observe([7, 8])
        try predictor.observe([7, 9])
        let probabilities = try predictor.predictNext()
        XCTAssertEqual(probabilities[7], 3)
        XCTAssertEqual(probabilities[8], 1)
        XCTAssertEqual(probabilities[256 + 7], 1)
        XCTAssertEqual(probabilities[256 + 8], 2)
        XCTAssertEqual(probabilities[256 + 9], 2)
        try predictor.begin(blockCount: 1)
        XCTAssertEqual(try predictor.predictNext(), [Float](repeating: 1, count: 256))
    }

    func testOrderOneUsesPreviousBytePerBlock() throws {
        let predictor = OrderOnePredictor()
        try predictor.begin(blockCount: 2)
        try predictor.observe([10, 20])
        XCTAssertEqual(try predictor.predictNext(), [Float](repeating: 1, count: 512))
        try predictor.observe([30, 40])
        try predictor.observe([10, 20])
        let probabilities = try predictor.predictNext()
        XCTAssertEqual(probabilities[30], 2)
        XCTAssertEqual(probabilities[40], 1)
        XCTAssertEqual(probabilities[256 + 40], 2)
        XCTAssertEqual(probabilities[256 + 30], 1)
        try predictor.begin(blockCount: 2)
        XCTAssertEqual(try predictor.predictNext(), [Float](repeating: 1, count: 512))
    }

    func testPredictorLifecycleAndObservationCounts() throws {
        for predictor: any BytePredictor in [UniformPredictor(), OrderZeroPredictor(), OrderOnePredictor()] {
            XCTAssertThrowsError(try predictor.predictNext())
            XCTAssertThrowsError(try predictor.observe([]))
            XCTAssertThrowsError(try predictor.begin(blockCount: 0))
            XCTAssertThrowsError(try predictor.begin(blockCount: Int.max))
            try predictor.begin(blockCount: 2)
            XCTAssertThrowsError(try predictor.observe([1]))
            try predictor.observe([1, 2])
            XCTAssertEqual(try predictor.predictNext().count, 512)
        }
    }

    func testContextPaddingRingWrapAndStrides() throws {
        var state = try ByteContextState(blockCount: 2, context: 3)
        var values = [Float16](repeating: -1, count: 12)
        func read() {
            values.withUnsafeMutableBufferPointer { state.write(to: $0, rowStride: 6, columnStride: 2) }
        }
        read()
        XCTAssertEqual(values, [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1])
        try state.observe([255, 128])
        read()
        XCTAssertEqual(values[0], 0)
        XCTAssertEqual(values[2], 0)
        XCTAssertEqual(values[4], 1)
        XCTAssertEqual(values[10], Float16(Float(128) / 255))
        try state.observe([1, 2])
        try state.observe([3, 4])
        try state.observe([5, 6])
        read()
        XCTAssertEqual([values[0], values[2], values[4]], [1, 3, 5].map { Float16(Float($0) / 255) })
        XCTAssertEqual([values[6], values[8], values[10]], [2, 4, 6].map { Float16(Float($0) / 255) })
        XCTAssertThrowsError(try state.observe([1]))
    }

    func testRecurrentOneHotInitialStateAndObservations() throws {
        let state = try RecurrentByteState(blockCount: 2)
        var values = [Float16](repeating: -1, count: 1_024)
        values.withUnsafeMutableBufferPointer { state.write(to: $0.baseAddress!, rowStride: 512, columnStride: 2) }
        for index in stride(from: 0, to: values.count, by: 2) { XCTAssertEqual(values[index], 0) }
        try state.observe([0, 255])
        values.withUnsafeMutableBufferPointer { state.write(to: $0.baseAddress!, rowStride: 512, columnStride: 2) }
        for block in 0..<2 {
            for symbol in 0..<256 {
                XCTAssertEqual(values[block * 512 + symbol * 2], symbol == (block == 0 ? 0 : 255) ? 1 : 0)
                XCTAssertEqual(values[block * 512 + symbol * 2 + 1], -1)
            }
        }
        try state.observe([7, 8])
        values.withUnsafeMutableBufferPointer { state.write(to: $0.baseAddress!, rowStride: 512, columnStride: 2) }
        XCTAssertEqual(values[0], 0)
        XCTAssertEqual(values[14], 1)
        XCTAssertEqual(values[512 + 16], 1)
        XCTAssertEqual(values[512 + 510], 0)
        // 予測を挟まない複数回の観測でも、実際に書き込んだ前回の列を消す。
        try state.observe([10, 20])
        try state.observe([30, 40])
        values.withUnsafeMutableBufferPointer { state.write(to: $0.baseAddress!, rowStride: 512, columnStride: 2) }
        XCTAssertEqual(values[14], 0)
        XCTAssertEqual(values[512 + 16], 0)
        XCTAssertEqual(values[60], 1)
        XCTAssertEqual(values[512 + 80], 1)
        let repeated = values
        values.withUnsafeMutableBufferPointer { state.write(to: $0.baseAddress!, rowStride: 512, columnStride: 2) }
        XCTAssertEqual(values, repeated)
        // 更新対象以外の列に印を置き、全行のゼロクリアが走らないことを確かめる。
        values[200] = 0.5
        try state.observe([30, 40])
        values.withUnsafeMutableBufferPointer { state.write(to: $0.baseAddress!, rowStride: 512, columnStride: 2) }
        XCTAssertEqual(values[200], 0.5)
        XCTAssertEqual(values[60], 1)
        XCTAssertThrowsError(try state.observe([1]))
    }
}
