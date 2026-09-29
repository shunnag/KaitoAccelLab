# KaitoAccelLab

M4 Max の GPU（Metal 4）と NPU（Apple Neural Engine、Core ML）が書庫の圧縮・展開に使えるかを検証した実験室です（2026-09-29）。
[KaitoKit](https://github.com/shunnag/KaitoKit) / [GyoshukuKit](https://github.com/shunnag/GyoshukuKit) の姉妹 repo で、製品には合流させていません。

## 結論

- 現在の macOS の圧縮・暗号 library（Apple Compression、CommonCrypto、MTLIO の展開）と KaitoKit / GyoshukuKit は CPU だけで動く。GPU / NPU は寄与していない。
- GPU で独立 block の LZ4 復号、histogram、CRC-32 は動く（byte 同一）。しかし LZ4 復号は CPU 16 lane の 0.15〜0.6 倍で、GPU 1 thread は CPU 1 core の 1/350。
- NPU で GRU の byte 予測器 + range coder の lossless codec が符号化・復号ともに動く（全 op が ANE）。学習 domain では PPMd を超える比を出すが、速度は 0.24〜0.65 MB/s で既存 codec より 2 桁遅く、符号化と復号の compute unit を揃える必要がある。
- どの経路も CPU より速くならなかったので本体には入れない。

詳細: [速度レポート](Documentation/speed-report.md)、[解説](Documentation/technology.md)、[計画](Documentation/plan.md)、[研究報告 JSON](Documentation/research/2026-09-29-research.json)、生の計測 `Results/`。

## 構成

- `Sources/AccelLab/LZ4`: LZ4 frame の解析、CPU 参照復号、Metal の block 並列復号 kernel（thread-per-block / SIMD-per-block）
- `Sources/AccelLab/Neural`: range coder、頻度量子化、byte 予測器 protocol、Core ML 予測器（MLP / GRU）、神経 block codec（NBC1）
- `Sources/AccelLab/Demo`: directory → tar（GyoshukuKit）→ 神経 codec か LZ4 → KADM container、展開は KaitoKit
- `Sources/AccelLab/Baseline.swift`, `MetalProbe.swift`: CPU 基準値、Metal の可動確認
- `Scripts/`: ANE probe、GPU byte kernel、MTLIO probe、liblz4 比較器、model の学習 / 書き出し（Python、coremltools）

## 使い方

`../KaitoKit` と `../GyoshukuKit` を同じ親 directory に checkout してから:

```
swift build -c release
swift test
.build/release/accel-lab probe
.build/release/accel-lab baseline --size 256 --rounds 5
.build/release/accel-lab lz4-make-frame <in> <out.lz4> --block-size 4096
.build/release/accel-lab lz4-gpu <out.lz4> --rounds 5
```

Core ML model（`Models/*.mlmodelc`）は git 管理外で、学習済み重み `Models/*.pt` から書き出します（Python 3.12、coremltools 9、torch）:

```
python3.12 -m venv .venv && source .venv/bin/activate && pip install coremltools torch numpy
python Scripts/train_gru_predictor.py <corpus> Models/gru-mixed-h1536 0 1536 256 1024
xcrun coremlcompiler compile Models/gru-mixed-h1536-b1024.mlpackage Models/
.build/release/accel-lab demo-pack <dir> out.kadm --neural --blocks 1024 --gru Models/gru-mixed-h1536-b1024.mlmodelc --units ane
.build/release/accel-lab demo-unpack out.kadm <outdir> --gru Models/gru-mixed-h1536-b1024.mlmodelc --units ane
```

計測機: Apple M4 Max（12P + 4E、GPU 40 core、ANE 16 core、128 GB）、macOS 27.2、Xcode 27.0。ライセンス: MIT。
