public import XCTest
@testable public import AccelLab

final class FrequencyQuantizerTests: XCTestCase {
    func testTotalMinimumAndDeterminism() {
        for seed in 1...32 {
            let probabilities = NeuralTestData.random(count: 256, seed: UInt64(seed)).map { Float($0) }
            let cumulative = FrequencyQuantizer.cumulativeFrequencies(for: probabilities)
            check(cumulative)
            XCTAssertEqual(cumulative, FrequencyQuantizer.cumulativeFrequencies(for: probabilities))
        }
    }

    func testZerosNaNsAndNonFiniteValues() {
        let uniform = (0...256).map { UInt32($0 * 256) }
        for value: Float in [0, .nan, .infinity, -.infinity, -1] {
            XCTAssertEqual(FrequencyQuantizer.cumulativeFrequencies(for: [Float](repeating: value, count: 256)), uniform)
        }
        var probabilities = [Float](repeating: .nan, count: 256)
        probabilities[42] = Float.greatestFiniteMagnitude
        let cumulative = FrequencyQuantizer.cumulativeFrequencies(for: probabilities)
        check(cumulative)
        for symbol in 0..<256 {
            XCTAssertEqual(cumulative[symbol + 1] - cumulative[symbol], symbol == 42 ? 65_281 : 1)
        }
    }

    func testOrderingForDistinctValuesAndExtremeScales() {
        for scale: Float in [Float.leastNormalMagnitude, 1, 1e30] {
            let values = (1...256).map { Float($0) * scale }
            let cumulative = FrequencyQuantizer.cumulativeFrequencies(for: values)
            check(cumulative)
            let frequencies = (0..<256).map { cumulative[$0 + 1] - cumulative[$0] }
            for symbol in 1..<256 { XCTAssertGreaterThanOrEqual(frequencies[symbol], frequencies[symbol - 1]) }
        }
        let tiny = [Float](repeating: Float.leastNonzeroMagnitude, count: 256)
        XCTAssertEqual(FrequencyQuantizer.cumulativeFrequencies(for: tiny), (0...256).map { UInt32($0 * 256) })
    }

    func testExcessIsRemovedFromLargestSymbolsWithoutReordering() {
        var values = (1...256).map { Float($0) * 1e-9 }
        values[254] = 1
        values[255] = 1.000001
        let cumulative = FrequencyQuantizer.cumulativeFrequencies(for: values)
        check(cumulative)
        let frequencies = (0..<256).map { cumulative[$0 + 1] - cumulative[$0] }
        for symbol in 1..<256 { XCTAssertGreaterThanOrEqual(frequencies[symbol], frequencies[symbol - 1]) }
    }

    func testReservedMinimumAndRemainderForPeakedRows() {
        var values = (0..<256).map { Float($0 + 1) * 1e-8 }
        values[17] = 0.9
        values[91] = 0.09
        values[233] = 0.01
        let sum = values.reduce(0.0) { $0 + Double($1) }
        var expected = values.map { UInt32(1) + UInt32(Double($0) / sum * 65_280) }
        let allocated = expected.reduce(0, +)
        XCTAssertLessThanOrEqual(allocated, 65_536)
        XCTAssertLessThan(65_536 - allocated, 257)
        expected[17] += 65_536 - allocated
        let cumulative = FrequencyQuantizer.cumulativeFrequencies(for: values)
        check(cumulative)
        let frequencies = (0..<256).map { cumulative[$0 + 1] - cumulative[$0] }
        XCTAssertEqual(frequencies, expected)
        let ordered = values.indices.sorted { values[$0] < values[$1] }
        for index in 1..<ordered.count {
            XCTAssertGreaterThanOrEqual(frequencies[ordered[index]], frequencies[ordered[index - 1]])
        }
    }

    private func check(_ cumulative: [UInt32], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(cumulative.count, 257, file: file, line: line)
        XCTAssertEqual(cumulative.first, 0, file: file, line: line)
        XCTAssertEqual(cumulative.last, 65_536, file: file, line: line)
        for symbol in 0..<256 { XCTAssertGreaterThan(cumulative[symbol + 1], cumulative[symbol], file: file, line: line) }
    }
}
