@testable public import AccelLab

internal final class RecordingPredictor: BytePredictor {
    private let uniform = UniformPredictor()
    private(set) var predictions = 0
    private(set) var observations: [[UInt8]] = []

    func begin(blockCount: Int) throws {
        predictions = 0
        observations = []
        try uniform.begin(blockCount: blockCount)
    }

    func predictNext() throws -> [Float] {
        predictions += 1
        return try uniform.predictNext()
    }

    func observe(_ bytes: [UInt8]) throws {
        observations.append(bytes)
        try uniform.observe(bytes)
    }
}
