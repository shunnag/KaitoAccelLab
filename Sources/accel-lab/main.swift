import Foundation
import AccelLab

let report = try MetalProbe.run()
print("device: \(report.deviceName)")
print("maxThreadsPerThreadgroup: \(report.maxThreadsPerThreadgroup), threadgroupMemory: \(report.threadgroupMemoryLength), unifiedMemory: \(report.hasUnifiedMemory)")
print("runtime MSL compile: \(report.compiledAtRuntime), sum-of-squares check: \(report.sumOfSquares != 0 ? "OK" : "MISMATCH")")
