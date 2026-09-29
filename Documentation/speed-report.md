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

### 4.1 byte 単位の kernel: histogram、CRC-32、読み出し帯域（Scripts/gpu-kernels.swift）

（計測待ち）

### 4.2 LZ4 block 並列復号（Sources/AccelLab/LZ4、accel-lab lz4-gpu）

（計測待ち）

### 4.3 見送った候補と理由

（研究の結果を書く）

## 5. NPU（Apple Neural Engine、Core ML）

### 5.1 ANE に載る条件（Results/ane-probe-scaling-*.txt）

### 5.2 神経 block codec（accel-lab neural-encode / neural-decode）

（計測待ち）

## 6. CPU 側で見つかった改善余地（NPU / GPU ではない）

## 7. 試さなかったこと

