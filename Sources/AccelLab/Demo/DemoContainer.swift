internal import Foundation

/// KADM のヘッダーに続けて、既存コーデックのバイト列をそのまま格納する。
internal struct DemoContainer {
    enum Kind: UInt8, Sendable { case neural = 1, lz4 = 2 }

    static let currentVersion: UInt8 = 1
    private static let magic = Array("KADM".utf8)
    private static let headerSize = magic.count + 2

    let kind: Kind
    let payload: Data

    func encode() -> Data {
        var bytes = Self.magic
        bytes.append(Self.currentVersion)
        bytes.append(kind.rawValue)
        bytes.append(contentsOf: payload)
        return Data(bytes)
    }

    static func decode(_ data: Data) throws -> DemoContainer {
        guard data.count >= headerSize else { throw DemoError("Truncated KADM header") }
        let header = Array(data.prefix(headerSize))
        guard header.starts(with: magic) else { throw DemoError("Invalid KADM magic") }
        let version = header[magic.count]
        guard version == currentVersion else { throw DemoError("Unsupported KADM version: \(version)") }
        guard let kind = Kind(rawValue: header[magic.count + 1]) else {
            throw DemoError("Unknown KADM payload kind: \(header[magic.count + 1])")
        }
        guard data.count > headerSize else { throw DemoError("Missing KADM payload") }
        return DemoContainer(kind: kind, payload: Data(data.dropFirst(headerSize)))
    }
}
