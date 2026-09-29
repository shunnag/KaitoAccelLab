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

## 補正（advisor 2026-09-29）
- ANE: 1 byte ごとの推論は Core ML の呼び出し overhead（ms 級）で 1 KB/s 程度になり demo にならない。設計は最初から
  (1) 多数の独立 block を一つの推論に batch（入力 [blocks × context]）、(2) 再帰状態は `MLState` で保持、とする。
  ANE で動いた証明は `MLComputePlan` の op ごとの配置（ANE / GPU / CPU）を記録して行う。GPU に落ちた LSTM は NPU の結果ではない。
- GPU の第一候補は「既に独立 block である形式の block 並列復号」（bzip2 block、LZMA2 chunk、zstd frame、GyoshukuKit の圧縮 tar の chunk 切り）。
  Huffman / range coder の直列部分は block 内に留め、40 並列で走らせる。
- baseline は各 codec の現在の実装（CRC-32 は ARM の CRC 命令、AES は CommonCrypto の AES 命令、Deflate は zlib、多 core の並列を含む）で先に計測する。
  GPU の CRC-32 / AES は負けると予想される。計測はするが、レポートは先に「予想どおり負ける理由」を書く。
- 確認事項: 本体の codec ごとの既存の並列度（圧縮 tar の chunk pipeline、zstd frame、7z solid folder）; Apple Compression の LZFSE / LZ4 が M4 で
  CPU 以外の hardware を使うかどうか（文書か trace で確認。記憶で決めない）。

## 判定規則（計測の前に確定）
- 「動いた」= perf corpus で CPU 経路と byte 同一の出力。
- 「速い」= 同じ run の base-vs-base の noise floor を超えて、多 core の CPU 経路の wall 中央値に勝つ。
- それ以外は比と理由を添えて報告する。生の計測は `Results/` に file で残し、レポートは file を引く。
- Codex の sandbox は GPU / Core ML を使えない見込み。Codex には純 Swift（算術符号器、block 分割、harness、MSL の source 文字列）を、
  GPU / ANE の実行と計測は orchestrator か Fable subagent が行う。計測は他の agent や build が走っていない時に行う。

## 途中結果（2026-09-29 ANE probe、Results/ane-*.txt）
- 全 op が ANE 対応でも Core ML の cost model が CPU を選ぶことがある（小さな model）。hidden 1024・4 層・batch 1024 から ANE が選ばれる。
- 乱数重み MLP（context 64、hidden 1024、4 層、固定 batch 1024）: ANE 0.87 ms（117 万予測/s）、CPU 2.49 ms、GPU 1.68 ms。batch 4096: ANE 2.75 ms（149 万/s）。
- 学習済み MLP（text256、3,000 step、valid 4.0 bits/byte、可変 batch 1〜8192）: batch 1024 で ANE 1.22 ms（84 万/s）、CPU 2.71 ms、GPU 1.86 ms。
  可変形状は ANE で遅い（batch 4096 で 7.55 ms、固定形状の 2.75 ms より悪い）→ demo は列挙形状か固定形状を使う。
- 推論 1 回の overhead は 0.07〜0.1 ms（ms 級ではない）。
