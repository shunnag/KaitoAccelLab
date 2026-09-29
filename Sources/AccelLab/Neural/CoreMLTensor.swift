internal import CoreML
public import Foundation

internal enum CoreMLTensor {
    static func constraint(_ descriptions: [String: MLFeatureDescription], name: String,
                           width: Int? = nil) throws -> MLMultiArrayConstraint {
        guard let constraint = descriptions[name]?.multiArrayConstraint,
              constraint.dataType == .float16, constraint.shape.count == 2,
              constraint.shape[0].intValue > 0, constraint.shape[1].intValue > 0,
              width == nil || constraint.shape[1].intValue == width else {
            throw NeuralCodecError("Core ML feature \(name) must be a rank-2 fp16 array\(width.map { " with width \($0)" } ?? "")")
        }
        return constraint
    }

    static func validate(_ constraint: MLMultiArrayConstraint, batch: Int, width: Int, name: String) throws {
        _ = try PredictorValidation.elementCount(blockCount: batch, width: width)
        let shape = constraint.shapeConstraint
        let accepts: Bool
        switch shape.type {
        case .enumerated:
            accepts = shape.enumeratedShapes.contains { $0.count == 2 && $0[0].intValue == batch && $0[1].intValue == width }
        case .range:
            accepts = shape.sizeRangeForDimension.count == 2
                && NSLocationInRange(batch, shape.sizeRangeForDimension[0].rangeValue)
                && NSLocationInRange(width, shape.sizeRangeForDimension[1].rangeValue)
        default:
            accepts = constraint.shape[0].intValue == batch && constraint.shape[1].intValue == width
        }
        guard accepts else {
            throw NeuralCodecError("Core ML feature \(name) rejects blockCount \(batch); model shape is \(constraint.shape), requested [\(batch), \(width)]")
        }
    }

    static func make(batch: Int, width: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: width)], dataType: .float16)
        array.withUnsafeMutableBytes { bytes, _ in
            _ = bytes.initializeMemory(as: UInt8.self, repeating: 0)
        }
        return array
    }

    static func output(_ provider: any MLFeatureProvider, name: String, batch: Int, width: Int) throws -> MLMultiArray {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw NeuralCodecError("Core ML output is missing: \(name)")
        }
        try validate(array, batch: batch, width: width, name: name)
        return array
    }

    static func validate(_ array: MLMultiArray, batch: Int, width: Int, name: String) throws {
        guard array.dataType == .float16, array.shape.count == 2,
              array.shape[0].intValue == batch, array.shape[1].intValue == width else {
            throw NeuralCodecError("Core ML \(name) must have fp16 shape [\(batch), \(width)]")
        }
    }

    static func readProbabilities(_ array: MLMultiArray, into probabilities: inout [Float]) {
        let batch = array.shape[0].intValue
        let width = FrequencyQuantizer.symbolCount
        let count = batch * width
        precondition(probabilities.count == count)
        let source = array.dataPointer.assumingMemoryBound(to: Float16.self)
        let rowStride = array.strides[0].intValue
        let columnStride = array.strides[1].intValue
        defer { _fixLifetime(array) }
        probabilities.withUnsafeMutableBufferPointer { buffer in
            let destination = buffer.baseAddress!
            if columnStride == 1 && rowStride == width {
                for index in 0..<count { destination[index] = Float(source[index]) }
                return
            }
            for block in 0..<batch {
                let input = source.advanced(by: block * rowStride)
                let output = destination.advanced(by: block * width)
                for symbol in 0..<width {
                    output[symbol] = Float(input[symbol * columnStride])
                }
            }
        }
    }

    // h_in の配列と feature provider は固定し、h_out の値だけをコピーする。
    static func copyState(_ source: MLMultiArray, to destination: MLMultiArray) {
        let batch = source.shape[0].intValue
        let width = source.shape[1].intValue
        let input = source.dataPointer.assumingMemoryBound(to: Float16.self)
        let output = destination.dataPointer.assumingMemoryBound(to: Float16.self)
        let inputRowStride = source.strides[0].intValue
        let inputColumnStride = source.strides[1].intValue
        let outputRowStride = destination.strides[0].intValue
        let outputColumnStride = destination.strides[1].intValue
        defer { _fixLifetime(source); _fixLifetime(destination) }
        if input == output { return }
        if inputColumnStride == 1 && outputColumnStride == 1 {
            if inputRowStride == width && outputRowStride == width {
                UnsafeMutableRawPointer(output).copyMemory(from: input, byteCount: batch * width * MemoryLayout<Float16>.stride)
            } else {
                for block in 0..<batch {
                    UnsafeMutableRawPointer(output.advanced(by: block * outputRowStride))
                        .copyMemory(from: input.advanced(by: block * inputRowStride), byteCount: width * MemoryLayout<Float16>.stride)
                }
            }
            return
        }
        for block in 0..<batch {
            let inputRow = input.advanced(by: block * inputRowStride)
            let outputRow = output.advanced(by: block * outputRowStride)
            for column in 0..<width {
                outputRow[column * outputColumnStride] = inputRow[column * inputColumnStride]
            }
        }
    }
}
