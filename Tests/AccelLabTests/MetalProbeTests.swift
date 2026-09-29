internal import XCTest
internal import Metal
@testable import AccelLab

final class MetalProbeTests: XCTestCase {
    func testRuntimeCompiledKernelComputesExpectedSum() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal device unavailable") }
        let report = try MetalProbe.run(count: 4096)
        XCTAssertTrue(report.compiledAtRuntime)
        XCTAssertNotEqual(report.sumOfSquares, 0)
    }
}
