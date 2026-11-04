# KaitoAccelLab

M4 Max の GPU（Metal 4）と NPU（Apple Neural Engine、Core ML）が書庫の圧縮・展開に使えるかを検証した実験室です（2026-09-29、続編 2026-11-04〜05）。
[KaitoKit](https://github.com/shunnag/KaitoKit) / [GyoshukuKit](https://github.com/shunnag/GyoshukuKit) の姉妹 repo です。GPU / NPU の実験は製品から分離し、実測で有利だった CPU の改善だけを両 Kit の本流に採用しています。

## 結論

- 現在の macOS の圧縮・暗号 library（Apple Compression、CommonCrypto、MTLIO の展開）と KaitoKit / GyoshukuKit は CPU だけで動く。GPU / NPU は寄与していない。
- GPU で独立 block の LZ4 復号、histogram、CRC-32、LZMA BT4 の match 探索が動く。しかし既存 codec では両機を一貫して上回れず、LZ4 の GPU 1 thread は CPU 1 core の 1/350 だった。
- NPU で GRU の byte 予測器 + range coder の lossless codec が符号化・復号ともに動く（全 op が ANE）。学習 domain では PPMd を超える比を出すが、既存 codec より大幅に遅く、符号化と復号の compute unit を揃える必要がある。
- 2026-11 の続編では、GRU の `MLState` 保持で ANE の実符号化が約 1.33 倍、int8 重み量子化でさらに約 1.07 倍、batch 8192 で実測 0.681 MB/s まで改善した。それでも本体には採用しない。
- 副産物として見つけた CPU 側の伸びしろ（xz の block 並列復号、単一 stream bzip2 の block 単位並列）は KaitoKit 本流で実現した（KaitoKit PR #41 / #42、tar.xz staging 5〜6.7 倍、tar.bz2 staging 5.5〜6 倍）。
- 独立 LZ4 frame の検証済み block に Apple decoder を使い、小 block の集約と展開後サイズの並列スキャンを行うと、KaitoKit 0.21.0 の CLI が 2.2〜2.8 倍に改善した。これは GPU ではなく CPU 本体の改善として採用した。
- LArc・RAR・PPMd・LZX の復号と、LH5・LZW・bzip2 の圧縮も、両機で出力を照合した CPU 最適化を採用した。候補ごとの数値と不採用の理由は速度レポート §9〜10 に記録する。

詳細: [速度レポート](Documentation/speed-report.md)、[解説](Documentation/technology.md)、[計画](Documentation/plan.md)、[研究報告 JSON](Documentation/research/2026-09-29-research.json)、生の計測 `Results/`。

## 研究記録の配置

| 対象 | 実装・再現手順 | 結果 |
|---|---|---|
| GPU LZ4 / byte kernel | `Sources/AccelLab/LZ4/`、`Scripts/gpu-kernels.swift` | 速度レポート §4・§8.2・§8.5〜8.6 |
| NPU / Core ML の神経圧縮 | `Sources/AccelLab/Neural/`、`Scripts/train_gru_predictor.py`、`Scripts/quantize_gru_model.py` | §5・§8.1・§8.4・§8.7〜8.11 |
| GPU LZMA match 探索 | `Research/LZMABucketMetal/`（本番 target から分離した一時コピーで検査） | §9.1 |
| その他の CPU 候補 | `Scripts/benchmark_*.py`、`Research/RAR15Copy/`、`Research/RAR20Primary/` | §9〜10 |

生の全 round、エラーを含む出力、テスト log は `Results/` に残しています。初期の LArc 測定に使った方式名の誤りと、その結果を除外した理由は §10.14 に記録しています。両 Kit の本番依存には Metal / Core ML を追加していません。

## 構成

- `Sources/AccelLab/LZ4`: LZ4 frame の解析、CPU 参照復号、Metal の block 並列復号 kernel（thread-per-block / SIMD-per-block、実験用 16 byte copy variants）
- `Sources/AccelLab/Neural`: range coder、頻度量子化、byte 予測器 protocol、Core ML 予測器（MLP / GRU / MLState）、神経 block codec（NBC1）
- `Sources/AccelLab/Demo`: directory → tar（GyoshukuKit）→ 神経 codec か LZ4 → KADM container、展開は KaitoKit
- `Sources/AccelLab/Baseline.swift`, `MetalProbe.swift`: CPU 基準値、Metal の可動確認
- `Scripts/`: ANE probe、GPU byte kernel、MTLIO probe、liblz4 比較器、model の学習 / 書き出し / 量子化、corpus と CLI の測定（Python、coremltools）

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
python3.12 -m venv .venv && source .venv/bin/activate && pip install coremltools==9.0 torch numpy
python Scripts/train_gru_predictor.py <corpus> Models/gru-mixed-h1536 0 1536 256 1024
python Scripts/train_gru_predictor.py <corpus> Models/gru-mixed-h1536 0 1536 256 1024 --stateful
xcrun coremlcompiler compile Models/gru-mixed-h1536-b1024.mlpackage Models/
xcrun coremlcompiler compile Models/gru-mixed-h1536-b1024-stateful.mlpackage Models/
python Scripts/quantize_gru_model.py Models/gru-mixed-h1536-b1024-stateful.mlpackage Models/gru-mixed-h1536-b1024-stateful-int8.mlpackage --dtype int8
xcrun coremlcompiler compile Models/gru-mixed-h1536-b1024-stateful-int8.mlpackage Models/
.build/release/accel-lab neural-predictor-bench Models/gru-mixed-h1536-b1024.mlmodelc Models/gru-mixed-h1536-b1024-stateful.mlmodelc --units ane
.build/release/accel-lab demo-pack <dir> out.kadm --neural --blocks 1024 --gru Models/gru-mixed-h1536-b1024-stateful.mlmodelc --units ane
.build/release/accel-lab demo-unpack out.kadm <outdir> --gru Models/gru-mixed-h1536-b1024-stateful.mlmodelc --units ane
```

stateful model は `hidden_state` を Core ML の `MLState` に保持し、明示的な `h_in` / `h_out` の往復を省きます。
書き出し時に状態の初期化と更新を検査し、Swift のテストでは CPU / GPU / ANE の予測一致とリセットを確かめます。
`baseline` は `--text <path>` で入力を指定できます。省略時の生成テキストは反復が強いため、codec の比較には実データを指定してください。

計測機: Apple M4 Max（12P + 4E、GPU 40 core、ANE 16 core、128 GB）と Mac mini M4。macOS 27.2、Xcode 27.0（初回）/ 27.2（2026-11 の続編）。性能測定と全テストを同じ Mac で同時には実行していません。ライセンス: MIT。
