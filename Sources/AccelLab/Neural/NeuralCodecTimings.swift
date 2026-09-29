/// ロックステップ内の予測と CPU 符号化・復号の経過時間。初期化と observe は含まない。
public struct NeuralCodecTimings: Sendable {
    public internal(set) var steps = 0
    internal var predictorDuration: Duration = .zero
    internal var coderDuration: Duration = .zero

    public init() {}

    public var predictorSeconds: Double { Self.seconds(predictorDuration) }
    public var coderSeconds: Double { Self.seconds(coderDuration) }

    private static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
