# KaitoAccelLab 計画（2026-09-29）

目的: M4 Max の GPU（Metal 4、40 core）と NPU（Apple Neural Engine、Core ML 経由）が書庫の圧縮・展開に使えるかを、
「動くか」→「どれだけ速いか」→「技術デモと解説」の順に徹底的に検証する。CPU より速くならなければ本体には入れない。

## 前提（確認済み）
- 現行の KaitoKit / GyoshukuKit は CPU のみ（自前 Swift、zlib、libbz2、Apple Compression、CommonCrypto）。Metal / Core ML / Accelerate は未使用。
- Metal の runtime compile（`MTLDevice.makeLibrary(source:)`）はこの機械で動く。offline の Metal toolchain は未導入で不要。
- ANE は Core ML の model 実行でしか使えない（汎用 compute は不可）。model は coremltools（Python venv）で作る。配置は `MLComputePlan` で確かめる。

## 段階
1. 研究（Fable subagent 並列）: GPU 圧縮の先行研究（nvCOMP 系の LZ4 / Snappy / Deflate、GPU Huffman 復号、GPU BWT、GPU ANS）、
   Metal の制約（threadgroup 32 KiB、SIMD 32、unified memory）、ANE の制約（fp16、op の対応、推論 latency、batch）、神経圧縮の先行研究
   （NNCP、DeepZip、TRACE、「Language Modeling Is Compression」）、本体 codec の並列化可能な段階の地図。
2. 可動性の検証（小さな kernel から）: GPU: CRC-32 の並列化、Huffman の histogram、LZ77 の match 探索、LZ4 block の並列復号、AES-CTR。
   NPU: byte 予測 model（小さな LSTM / Transformer）を Core ML で ANE に載せ、算術符号と組み合わせた lossless 圧縮の可動性。
3. 計測: CPU 単体（本体の実装 / zlib 等）と GPU / NPU 併用を同じ入力で比較。noise floor を取る。
4. 成果物: `Documentation/speed-report.md`（速度レポート）、`accel-lab` の技術デモ（GPU / NPU を CPU と併用して書庫を圧縮・展開）、
   `Documentation/technology.md`（解説）。本体への合流は速くなった場合だけ。

## 方針
- Fable の subagent で研究と設計を、実装は Codex か Fable subagent（GPU 実行が sandbox でできない場合）、判断は Fable advisor に相談。
- すべての計測は release build、同じ入力、alternating、中央値。
