import XCTest
@testable import AccelLab

final class MetalProbeTests: XCTestCase {
    func testRuntimeCompiledKernelComputesExpectedSum() throws {
        let report = try MetalProbe.run(count: 4096)
        XCTAssertTrue(report.compiledAtRuntime)
        XCTAssertNotEqual(report.sumOfSquares, 0)
    }
}
