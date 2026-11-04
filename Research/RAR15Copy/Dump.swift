import Foundation
@testable import KaitoKit

let archive = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
let source = DataByteSource(data: archive)
let signature = try RARSignatureScanner.find(source: source)!
let parsed = try RAR4EntryPublisher.parse(
    source: source, sourceURL: nil, sourceDirectoryAnchor: nil,
    options: ReaderOptions(), signatureOffset: signature.offset,
    headerKeyCache: RAR3KeyCache(),
    passwordEncodings: RAR4Reader.PasswordEncodingSelection()
)
let recordIndex = CommandLine.arguments.count > 3 ? Int(CommandLine.arguments[3])! : 0
let record = parsed.records[recordIndex]
let segment = record.packedSegments.first!
let decoder = try RAR15Decoder(
    source: source, offset: segment.offset, compressedSize: record.packedSize,
    expectedSize: record.unpackedSize, unpackVersion: record.unpackVersion,
    method: record.method, dictionarySize: record.dictionarySize,
    isSolid: false, limits: ReadLimits()
)
var output = Data()
var buffer = [UInt8](repeating: 0, count: 97)
while output.count < record.unpackedSize {
    let count = try buffer.withUnsafeMutableBytes { try decoder.read(into: $0) }
    output.append(contentsOf: buffer.prefix(count))
}
try output.write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
print("wrote \(output.count) bytes")
