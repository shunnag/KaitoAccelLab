import Foundation
import CryptoKit
import KaitoKit

struct Entry: Codable { let index: Int; let count: Int; let digest: String; let error: String? }
struct Outcome: Codable { let path: String; let mutation: String; let buffer: Int; let entries: [Entry]; let error: String? }
let root = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent("Tests/Fixtures/rar20")
let output = URL(fileURLWithPath: CommandLine.arguments[2])
let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
FileManager.default.createFile(atPath: output.path, contents: nil)
let handle = try FileHandle(forWritingTo: output)
var cases = 0
for name in ["autorej.rar.b64", "audio.rar.b64", "biglz.rar.b64", "solid.rar.b64"] {
    let original = Data(base64Encoded: try Data(contentsOf: root.appendingPathComponent(name)),
                        options: .ignoreUnknownCharacters)!
    var variants = [("original", original)]
    for amount in Set([1, 7, max(1, original.count / 2), max(1, original.count - 32)]).sorted() {
        variants.append(("truncate-\(amount)", Data(original.dropLast(amount))))
    }
    for offset in Set([32, 33, 38, original.count / 4, original.count / 2, original.count - 1]).sorted() where offset < original.count {
        for bit: UInt8 in [1, 128] {
            var bytes = original; bytes[offset] ^= bit; variants.append(("flip-\(offset)-\(bit)", bytes))
        }
    }
    for (mutation, bytes) in variants {
        for size in mutation == "original" ? [97, 4099] : [1, 97, 4099, 262144] {
            var entries: [Entry] = []; var outer: String?
            do {
                let reader = try ArchiveReader.open(data: bytes, options: ReaderOptions(decodeThreads: 1))
                for (index, entry) in reader.entries.enumerated() {
                    var count = 0; var digest = SHA256(); var failure: String?
                    do {
                        let stream = try reader.stream(entry)
                        var buffer = [UInt8](repeating: 0, count: size)
                        while true {
                            let n = try buffer.withUnsafeMutableBytes { try stream.read(into: $0) }
                            if n == 0 { break }
                            count += n; digest.update(data: buffer.prefix(n))
                        }
                    } catch { failure = String(reflecting: error) }
                    entries.append(Entry(index: index, count: count,
                                         digest: digest.finalize().map { String(format: "%02x", $0) }.joined(),
                                         error: failure))
                }
            } catch { outer = String(reflecting: error) }
            try handle.write(contentsOf: encoder.encode(Outcome(path: name, mutation: mutation,
                                                                  buffer: size, entries: entries, error: outer)))
            try handle.write(contentsOf: Data([10])); cases += 1
        }
    }
}
try handle.close()
print("outcomes \(cases)")
