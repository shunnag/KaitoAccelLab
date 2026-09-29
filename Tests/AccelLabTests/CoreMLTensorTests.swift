internal import CoreML
public import Foundation
public import XCTest
@testable public import AccelLab

final class CoreMLTensorTests: XCTestCase {
    func testPointerCopiesRespectStridesAndPadding() throws {
        let width = FrequencyQuantizer.symbolCount
        for columnStride in [1, 3] {
            let source = try stridedArray(rowStride: width * columnStride + 7, columnStride: columnStride)
            let input = source.dataPointer.assumingMemoryBound(to: Float16.self)
            let rowStride = source.strides[0].intValue
            for block in 0..<2 {
                for symbol in 0..<width { input[block * rowStride + symbol * columnStride] = Float16(block * width + symbol) }
            }
            var expected = [Float](repeating: 0, count: 2 * width)
            CoreMLTensor.readProbabilities(source, into: &expected)
            XCTAssertEqual(expected, (0..<(2 * width)).map { Float($0) })
            for destinationColumnStride in [1, 2] {
                let destination = try stridedArray(rowStride: width * destinationColumnStride + 5,
                                                   columnStride: destinationColumnStride)
                CoreMLTensor.copyState(source, to: destination)
                var actual = [Float](repeating: 0, count: 2 * width)
                CoreMLTensor.readProbabilities(destination, into: &actual)
                XCTAssertEqual(actual, expected)
                let output = destination.dataPointer.assumingMemoryBound(to: Float16.self)
                XCTAssertEqual(output[width * destinationColumnStride], -1)
                if destinationColumnStride > 1 { XCTAssertEqual(output[1], -1) }
            }
            CoreMLTensor.copyState(source, to: source)
            var unchanged = [Float](repeating: 0, count: 2 * width)
            CoreMLTensor.readProbabilities(source, into: &unchanged)
            XCTAssertEqual(unchanged, expected)
        }
    }

    func testHalfPrecisionProbabilityRowsAndHiddenStateCopy() throws {
        let source = try CoreMLTensor.make(batch: 2, width: 256)
        source.withUnsafeBufferPointer(ofType: Float16.self) { values in
            XCTAssertTrue(values.allSatisfy { $0 == 0 })
        }
        source.withUnsafeMutableBytes { bytes, strides in
            let values = bytes.bindMemory(to: Float16.self)
            for block in 0..<2 {
                for symbol in 0..<256 {
                    values[block * strides[0] + symbol * strides[1]] = Float16(Float(block * 256 + symbol) / 512)
                }
            }
        }
        var probabilities = [Float](repeating: 0, count: 512)
        CoreMLTensor.readProbabilities(source, into: &probabilities)
        XCTAssertEqual(probabilities, (0..<512).map { Float(Float16(Float($0) / 512)) })
        let destination = try CoreMLTensor.make(batch: 2, width: 256)
        CoreMLTensor.copyState(source, to: destination)
        var copy = [Float](repeating: 0, count: 512)
        CoreMLTensor.readProbabilities(destination, into: &copy)
        XCTAssertEqual(copy, probabilities)
        XCTAssertThrowsError(try CoreMLTensor.validate(source, batch: 1, width: 256, name: "x"))
        XCTAssertThrowsError(try CoreMLTensor.validate(source, batch: 2, width: 64, name: "x"))
    }

    func testOutputValidationWithoutModelInference() throws {
        let array = try CoreMLTensor.make(batch: 2, width: 256)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["probabilities": MLFeatureValue(multiArray: array)])
        XCTAssertTrue(try CoreMLTensor.output(provider, name: "probabilities", batch: 2, width: 256) === array)
        XCTAssertThrowsError(try CoreMLTensor.output(provider, name: "missing", batch: 2, width: 256))
        XCTAssertThrowsError(try CoreMLTensor.output(provider, name: "probabilities", batch: 7, width: 256))
        let float32 = try MLMultiArray(shape: [2, 256], dataType: .float32)
        XCTAssertThrowsError(try CoreMLTensor.validate(float32, batch: 2, width: 256, name: "probabilities"))
    }

    private func stridedArray(rowStride: Int, columnStride: Int) throws -> MLMultiArray {
        let capacity = rowStride * 2
        let values = UnsafeMutablePointer<Float16>.allocate(capacity: capacity)
        values.initialize(repeating: -1, count: capacity)
        return try MLMultiArray(dataPointer: values, shape: [2, NSNumber(value: FrequencyQuantizer.symbolCount)], dataType: .float16,
                                strides: [NSNumber(value: rowStride), NSNumber(value: columnStride)]) { pointer in
            pointer.assumingMemoryBound(to: Float16.self).deinitialize(count: capacity)
            pointer.deallocate()
        }
    }
}
