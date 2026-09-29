public import Foundation

internal enum NeuralTestData {
    static let corpusPath = "/private/tmp/claude-501/-Users-nagash-Github-KaitoFinder/404391ed-5c96-4f47-afa7-7911dab682d1/scratchpad/corpora/text256.txt"
    static let textLimit = 4 * 1_048_576

    static func random(count: Int, seed: UInt64 = 0x4E42_4331) -> [UInt8] {
        var state = seed
        return (0..<count).map { _ in
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return UInt8(truncatingIfNeeded: state)
        }
    }

    static func text() -> [UInt8] {
        if let file = try? FileHandle(forReadingFrom: URL(fileURLWithPath: corpusPath)) {
            defer { try? file.close() }
            if let data = try? file.read(upToCount: textLimit), !data.isEmpty { return Array(data) }
        }
        let sentence = Array("The quiet library holds stories about rivers and mountains. We measure, compare, and repeat each experiment.\n".utf8)
        return (0..<textLimit).map { sentence[$0 % sentence.count] }
    }
}
