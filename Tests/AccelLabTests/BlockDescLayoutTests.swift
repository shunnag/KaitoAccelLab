internal import XCTest
@testable import AccelLab

final class BlockDescLayoutTests: XCTestCase {
    func testMetalDescriptorLayout() {
        XCTAssertEqual(MemoryLayout<BlockDesc>.size, 32)
        XCTAssertEqual(MemoryLayout<BlockDesc>.stride, 32)
        XCTAssertEqual(MemoryLayout<BlockDesc>.alignment, 4)
        XCTAssertEqual(MemoryLayout<BlockDesc>.offset(of: \.srcOffset), 0)
        XCTAssertEqual(MemoryLayout<BlockDesc>.offset(of: \.srcLength), 4)
        XCTAssertEqual(MemoryLayout<BlockDesc>.offset(of: \.dstOffset), 8)
        XCTAssertEqual(MemoryLayout<BlockDesc>.offset(of: \.dstLength), 12)
        XCTAssertEqual(MemoryLayout<BlockDesc>.offset(of: \.isStored), 16)
        XCTAssertEqual(MemoryLayout<BlockDesc>.offset(of: \.pad), 20)
    }
}
