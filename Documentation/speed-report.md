# 速度レポート: M4 Max の GPU / NPU と書庫の圧縮・展開（2026-09-29）

計測機: Apple M4 Max（CPU 16 core = P 12 + E 4、GPU 40 core、ANE 16 core、128 GB）、macOS 27.2、Xcode 27.0、Swift 6.2。
すべて release build、同じ入力、中央値。生の数値は `Results/` の file を引く（file 名を各表に書く）。

## 1. 結論（先に）

（計測完了後に書く）

## 2. 現状: NPU / GPU は圧縮・展開に寄与しているか

寄与していない。根拠:
- KaitoKit / GyoshukuKit / KaitoFinder は Metal、Core ML、Accelerate を link しない（`otool -L`、import の全数確認）。使う codec は自前 Swift、zlib、libbz2、Apple Compression、CommonCrypto。
- Apple Compression（`libcompression.dylib`）は liblzma と libSystem 以外を link せず、LZFSE / LZ4 / zlib / LZMA の実行中に IOKit / Metal / ANE の image は読み込まれない。圧縮の IOService も存在しない（研究 JSON: apple-platform）。
- CommonCrypto の AES / SHA は CPU の暗号命令（ARMv8 AES / SHA2）で動く。zlib の CRC-32 は ARM の CRC32 命令で動く。
- ANE は Core ML の model 実行でしか触れない。汎用 compute（bit 操作、表引き、逐次の復号）は ANE には載らない。

## 3. CPU の基準値（Results/baseline-20260929-1539.tsv、256 MiB、5 round 中央値）

| 処理 | GB/s | 備考 |
|---|---|---|
| CRC-32（zlib、1 core） | 43.7 | ARM CRC32 命令 |
| CRC-32（zlib、16 lane + crc32_combine） | 224.7 | 本体 ZIP / GZIP の CRC はこの経路の 1 core 版 |
| CRC-32（8 bit 表、1 core） | 0.56 | 表引き版の参考値 |
| Adler-32（zlib、1 core） | 21.9 | |
| AES-256-CTR（CommonCrypto、1 core） | 14.2 | |
| AES-256-CBC 復号（1 core） | 16.8 | |
| SHA-256（1 core） | 3.43 | |
| deflate level 6 圧縮（zlib、1 core、text256） | 0.032 | 比 0.475 |
| inflate（zlib、1 core） | 0.856 | |
| bzip2 圧縮 / 展開（libbz2、1 core、900k） | 0.022 / 0.050 | 比 0.374 |
| LZFSE 圧縮 / 展開（1 core） | 0.077 / 1.41 | 比 0.473 |
| LZ4 圧縮 / 展開（1 core） | 0.61 / 3.86 | 比 0.777 |
| LZMA 圧縮 / 展開（1 core） | 0.0023 / 0.170 | 比 0.301 |
| memcpy（1 core） | 71.6 | |

## 4. GPU（Metal 4、40 core）

### 4.1 byte 単位の kernel: histogram、CRC-32、読み出し帯域（Scripts/gpu-kernels.swift、Results/gpu-kernels-*.tsv）

256 MiB、5 round 中央値。GPU の時間は command buffer の `gpuEndTime - gpuStartTime`（wall は +0.2〜0.4 ms）。

| 処理 | random256 | text256 | 備考 |
|---|---|---|---|
| GPU 読み出し（uint4 sum） | 277〜379 GB/s | 404〜483 GB/s | 帯域の上限の目安 |
| GPU histogram（threadgroup atomic） | 147〜149 GB/s | 67〜73 GB/s | text は偏った bin への atomic 競合で遅い |
| CPU histogram 16 lane | 44 GB/s | 28 GB/s | 私有 count の和 |
| GPU CRC-32 4 KiB chunk（slice-by-4）+ CPU 16 lane combine | 244 GB/s（kernel 355 + combine 0.34 ms） | 185 GB/s | 結果は zlib と一致 |
| GPU CRC-32 16 KiB chunk + combine | 105 GB/s（kernel 111） | 290 GB/s（kernel 334） | run 間のばらつきが大きい（±2 倍） |
| CPU CRC-32 zlib 16 lane + combine | 222〜257 GB/s | 222〜233 GB/s | 基準 |

読み: histogram は GPU が 3〜5 倍速い唯一の byte kernel だが、Deflate / LH5 が数えるのは LZ77 parse 後の記号なので生 byte の histogram
に用途がない（order-0 符号器にしか効かない）。CRC-32 は GPU kernel 単体なら CPU 16 lane と同程度〜上回るが、wall で見ると dispatch 往復と
combine を足して同程度で、CPU 1 core の 43.7 GB/s（CRC32 命令）で十分な用途に GPU を使う理由はない。

### 4.2 LZ4 block 並列復号（Sources/AccelLab/LZ4、accel-lab lz4-gpu）

（計測待ち）

### 4.3 見送った候補と理由

（研究の結果を書く）

## 5. NPU（Apple Neural Engine、Core ML）

### 5.1 ANE に載る条件（Results/ane-probe-scaling-*.txt）

### 5.2 神経 block codec（accel-lab neural-encode / neural-decode）

model: GRU 1 層 hidden 1024（Scripts/train_gru_predictor.py、words32 の先頭 90% で 4,000 step 学習、valid 2.50 bits/byte）、
固定 batch 1024 の fp16 mlprogram。入力は学習に使っていない words32 の末尾。

最初の版（Results/ane-roundtrip-v1-20260929-1636.txt、1 MiB、1024 block、`--units ane`、MLComputePlan: ANE 18 op / n.a. 14）:

| | bytes | 比 | bits/byte |
|---|---|---|---|
| 神経 block codec（ANE） | 332,788 | 0.317 | 2.54 |
| PPMd o8 mem 256m（7zz） | 350,387 | 0.334 | 2.67 |
| bzip2 -9 | 399,692 | 0.381 | 3.05 |
| xz -9 | 414,116 | 0.395 | 3.16 |
| LZMA2 mx9（7zz） | 414,268 | 0.395 | 3.16 |
| zstd -19 | 415,668 | 0.396 | 3.17 |

- 符号化 45.95 s（22.8 KB/s）、復号 42.58 s（24.6 KB/s）、`cmp` 一致。ANE で符号化した bit 列を ANE で復号して元に戻った。
- この版の時間はほぼ CPU 側（FrequencyQuantizer の余剰調整 loop）で、`sample` の main thread の 98% がそこ。ANE の推論は 1 step 1 ms 弱。
  修正版の数値は 5.3 に書く。
- 同じ bit 列を `--units cpu`（同じ model file、同じ batch、CPU 実行）で復号すると `Invalid range-coded payload` で失敗する
  （Results/ane-encoded-cpu-decode-mismatch-20260929.txt）。fp16 の計算結果が device で異なるため、符号化と復号は同じ compute unit で
  走らせなければならない。header の predictor tag はこの照合のためにある。

## 6. CPU 側で見つかった改善余地（NPU / GPU ではない）

## 7. 試さなかったこと

