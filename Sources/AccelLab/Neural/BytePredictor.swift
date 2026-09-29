public protocol BytePredictor: AnyObject {
    /// 予測するブロック数を告げ、内部状態を初期化する。
    func begin(blockCount: Int) throws
    /// 各ブロックの次のバイトの分布を返す。blockCount × 256 個の Float を行優先で並べる。
    func predictNext() throws -> [Float]
    /// 各ブロックの実際のバイト (blockCount 個) を通知する。
    func observe(_ bytes: [UInt8]) throws
}
