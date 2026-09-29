internal import Foundation
private import Dispatch

struct LZ4Frame: Sendable {
    struct Block: Sendable, Equatable {
        let compressedOffset: Int
        let compressedLength: Int
        let isStored: Bool
    }

    // 各レーンは別々の要素だけを書き、全ポインタの寿命は同期 dispatch の終了まで保持する。
    private struct SizeScan: @unchecked Sendable {
        let frame: LZ4Frame
        let sizes: UnsafeMutableBufferPointer<Int>
        let failures: UnsafeMutableBufferPointer<(any Error)?>
        let lanes: Int

        func run(lane: Int) {
            let quotient = frame.blocks.count / lanes
            let remainder = frame.blocks.count % lanes
            let start = lane * quotient + min(lane, remainder)
            let end = start + quotient + (lane < remainder ? 1 : 0)
            for index in start..<end {
                do { sizes[index] = try frame.decodedLength(of: frame.blocks[index]) }
                catch { failures[index] = error }
            }
        }
    }

    static let magic: UInt32 = 0x184D_2204
    static let blockSizes = [65_536, 262_144, 1_048_576, 4_194_304]
    private static let storedMask: UInt32 = 0x8000_0000
    private static let scanLanes = 16

    let blockMaxSize: Int
    let contentSize: UInt64?
    let isIndependent: Bool
    let dictionaryID: UInt32?
    let headerChecksum: UInt8
    let hasBlockChecksum: Bool
    let hasContentChecksum: Bool
    let blocks: [Block]
    private let source: Data

    init(_ bytes: [UInt8]) throws { try self.init(Data(bytes)) }

    init(_ data: Data) throws {
        var cursor = 0
        let parsed = try data.withUnsafeBytes { bytes in
            func read(_ count: Int) throws -> UInt64 {
                guard count <= bytes.count - cursor else { throw LZ4Error.truncatedInput }
                var value: UInt64 = 0
                for index in 0..<count { value |= UInt64(bytes[cursor + index]) << (index * 8) }
                cursor += count
                return value
            }
            guard try read(4) == UInt64(Self.magic) else { throw LZ4Error.invalidMagic }
            let flags = try read(1)
            guard flags >> 6 == 1 else { throw LZ4Error.invalidVersion }
            guard flags & 0x02 == 0 else { throw LZ4Error.reservedBits }
            let descriptor = try read(1)
            guard descriptor & 0x8F == 0 else { throw LZ4Error.reservedBits }
            let sizeCode = Int(descriptor >> 4)
            guard (4...7).contains(sizeCode) else { throw LZ4Error.invalidBlockSize }
            let maximum = Self.blockSizes[sizeCode - 4]
            let size: UInt64? = flags & 0x08 != 0 ? try read(8) : nil
            let dictionary: UInt32? = flags & 0x01 != 0 ? UInt32(try read(4)) : nil
            let checksum = UInt8(try read(1))
            var blocks: [Block] = []
            while true {
                let encodedSize = UInt32(try read(4))
                if encodedSize == 0 { break }
                let length = Int(encodedSize & ~Self.storedMask)
                guard length <= maximum else { throw LZ4Error.invalidBlockSize }
                guard length <= bytes.count - cursor else { throw LZ4Error.truncatedInput }
                blocks.append(Block(compressedOffset: cursor, compressedLength: length,
                                    isStored: encodedSize & Self.storedMask != 0))
                cursor += length
                if flags & 0x10 != 0 { _ = try read(4) }
            }
            if flags & 0x04 != 0 { _ = try read(4) }
            guard cursor == bytes.count else { throw LZ4Error.trailingData }
            return (maximum, size, flags, dictionary, checksum, blocks)
        }
        blockMaxSize = parsed.0
        contentSize = parsed.1
        isIndependent = parsed.2 & 0x20 != 0
        hasBlockChecksum = parsed.2 & 0x10 != 0
        hasContentChecksum = parsed.2 & 0x04 != 0
        dictionaryID = parsed.3
        headerChecksum = parsed.4
        blocks = parsed.5
        source = data
    }

    func decodedLength(of block: Block) throws -> Int {
        try requireSupportedFrame()
        guard block.compressedOffset >= 0, block.compressedLength >= 0,
              block.compressedOffset <= source.count,
              block.compressedLength <= source.count - block.compressedOffset,
              block.compressedLength <= blockMaxSize else { throw LZ4Error.invalidBlockSize }
        if block.isStored { return block.compressedLength }
        return try source.withUnsafeBytes { bytes in
            try LZ4BlockDecoder.decodedLength(
                UnsafeRawBufferPointer(rebasing: bytes[block.compressedOffset..<(block.compressedOffset + block.compressedLength)]),
                limit: blockMaxSize)
        }
    }

    func expectedDecodedSizes() throws -> [Int] {
        try requireSupportedFrame()
        var sizes = [Int](repeating: 0, count: blocks.count)
        var failures = [(any Error)?](repeating: nil, count: blocks.count)
        if !blocks.isEmpty {
            sizes.withUnsafeMutableBufferPointer { sizes in
                failures.withUnsafeMutableBufferPointer { failures in
                    let scan = SizeScan(frame: self, sizes: sizes, failures: failures, lanes: Self.scanLanes)
                    DispatchQueue.concurrentPerform(iterations: Self.scanLanes) { scan.run(lane: $0) }
                }
            }
        }
        // 並列実行でも、入力順で最初の破損を報告する。
        for failure in failures { if let failure { throw failure } }
        try validateDecodedSizes(sizes)
        return sizes
    }

    func expectedDecodedSizesSerial() throws -> [Int] {
        try requireSupportedFrame()
        let sizes = try blocks.map { try decodedLength(of: $0) }
        try validateDecodedSizes(sizes)
        return sizes
    }

    // 内容サイズだけから配置を仮定し、圧縮列は走査しない。
    func assumedUniformDecodedSizes() throws -> [Int] {
        try requireSupportedFrame()
        guard let contentSize else { throw LZ4Error.invalidArgument("--assume-uniform requires contentSize") }
        guard !blocks.isEmpty else {
            guard contentSize == 0 else { throw LZ4Error.contentSizeMismatch }
            return []
        }
        let count = UInt64(blocks.count)
        let size = contentSize / count + (contentSize.isMultiple(of: count) ? 0 : 1)
        guard size <= UInt64(blockMaxSize) else { throw LZ4Error.invalidBlockSize }
        let prefix = size.multipliedReportingOverflow(by: count - 1)
        guard !prefix.overflow, prefix.partialValue <= contentSize else { throw LZ4Error.contentSizeMismatch }
        var sizes = [Int](repeating: Int(size), count: blocks.count)
        sizes[sizes.count - 1] = Int(contentSize - prefix.partialValue)
        try validateDecodedSizes(sizes)
        return sizes
    }

    private func validateDecodedSizes(_ sizes: [Int]) throws {
        var total = 0
        for size in sizes {
            let sum = total.addingReportingOverflow(size)
            guard !sum.overflow else { throw LZ4Error.sizeOverflow }
            total = sum.partialValue
        }
        if let contentSize, contentSize != UInt64(total) { throw LZ4Error.contentSizeMismatch }
    }

    func validate(source data: Data) throws {
        guard source == data else { throw LZ4Error.sourceMismatch }
        try requireSupportedFrame()
    }

    private func requireSupportedFrame() throws {
        guard isIndependent else { throw LZ4Error.unsupportedDependency }
        guard dictionaryID == nil else { throw LZ4Error.unsupportedDictionary }
    }
}

extension LZ4Frame {
    static func encodeFrame(_ input: [UInt8], blockSize: Int) throws -> Data {
        try LZ4FrameEncoder.encodeFrame(input, blockSize: blockSize)
    }
}
