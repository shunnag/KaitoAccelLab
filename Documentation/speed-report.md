# 速度レポート: M4 Max の GPU / NPU と書庫の圧縮・展開（2026-09-29）

計測機: Apple M4 Max（CPU 16 core = P 12 + E 4、GPU 40 core、ANE 16 core、128 GB）、macOS 27.2、Xcode 27.0、Swift 6.2。
すべて release build、同じ入力、中央値。他の作業が走っていない状態で取り直した値を「idle」と書き、生の数値は `Results/` の file を引く。
判定規則（plan.md）: 「動いた」= CPU 経路と byte 同一、「速い」= 多 core の CPU 経路の中央値に noise を超えて勝つ。

## 1. 結論（先に）

| 問い | 答え |
|---|---|
| 現在 NPU / GPU は圧縮・展開に寄与しているか | していない。本体も Apple の圧縮 / 暗号 library も CPU だけで動く（§2）。 |
| GPU で圧縮・展開の一部が動くか | 動く。独立 block の LZ4 復号を Metal kernel で書き、256 MiB を byte 同一に復号した（§4.2）。histogram、CRC-32 も動く（§4.1）。 |
| GPU は CPU より速いか | 速くない。LZ4 復号は CPU 16 lane の 0.15〜0.6 倍（block size による）。histogram だけ 2.5〜8 倍速いが用途がない。CRC-32 は同程度で往復の費用分だけ負ける。 |
| NPU で圧縮・展開の一部が動くか | 動く。GRU の byte 予測器を Core ML で ANE に載せ（MLComputePlan で全 18 op が ANE）、range coder と組み合わせた lossless codec で符号化・復号ともに ANE で往復した（§5）。 |
| NPU は CPU より速いか | 同じ model を Core ML の CPU で回すより 1.4〜1.6 倍速いが、同じ model を Core ML の GPU で回した方が ANE より 1.06〜1.23 倍速く、既存の CPU codec（xz 322 MB/s、PPMd 9〜14 MB/s）には 2 桁遅い（0.24〜0.53 MB/s）。ただし比は学習 domain（辞書の単語を無作為に並べた合成 text で、試験側の語彙はすべて学習で見ている）では PPMd を上回る（0.279 対 0.289）。domain 外では大きく負け、混合 corpus で学習し直しても PPMd 級か以下（§5.6）。 |
| 本体に合流させるか | しない。CPU より速い経路がない。条件が揃えば GPU が勝ちうる形（§4.4）と、NPU codec が意味を持つ条件（§5.5）を書き残す。 |

## 2. 現状: NPU / GPU は圧縮・展開に寄与しているか

寄与していない。根拠:
- KaitoKit / GyoshukuKit / KaitoFinder は Metal、Core ML、Accelerate を link しない。使う codec は自前 Swift、zlib、libbz2、Apple Compression、CommonCrypto。
- Apple Compression（`libcompression.dylib`）は liblzma と libSystem 以外を link せず、LZFSE / LZ4 / zlib / LZMA の実行中に IOKit / Metal / ANE の image は読み込まれない。圧縮の IOService も存在しない（Documentation/research/2026-09-29-research.json、apple-platform）。
- CommonCrypto の AES / SHA は CPU の暗号命令、zlib の CRC-32 は ARM の CRC32 命令で動く（1 core 43.7 GB/s、§3）。
- Metal 4 の MTLIO（`MTLIOCompressionContext` で圧縮した file を `MTLIOCommandQueue` で読み込む）は Apple が GPU 資産のために用意した唯一の「圧縮付き読み込み」だが、256 MiB の展開は LZ4 3.4 GB/s、LZBitmap 4.6 GB/s、LZFSE 1.25 GB/s、zlib 0.41 GB/s（Results/mtlio-idle-text256-20260929-1743.tsv）。libcompression の 1 core（LZ4 3.5〜4.7 GB/s、LZFSE 1.4 GB/s）と同じ水準で、16 lane の 41 GB/s には遠い。どの unit で展開しているかは計測していない（速度は CPU 1 core 相当）。
- ANE は Core ML の model 実行でしか触れない。bit 操作、表引き、逐次の復号は ANE には載らない。

## 3. CPU の基準値（Results/baseline-20260929-1539.tsv、256 MiB、5 round 中央値）

| 処理 | GB/s | 備考 |
|---|---|---|
| CRC-32（zlib、1 core） | 43.7 | ARM CRC32 命令 |
| CRC-32（zlib、16 lane + crc32_combine） | 224.7 | idle 再計測では 205〜246 |
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

### 4.1 byte 単位の kernel: histogram、CRC-32、読み出し帯域（Scripts/gpu-kernels.swift、Results/gpu-kernels-idle-*-20260929-1743.tsv）

256 MiB、5 round 中央値、idle。GPU の時間は command buffer の `gpuEndTime - gpuStartTime`（wall は +0.2〜0.4 ms）。

| 処理 | random256 | text256 | 備考 |
|---|---|---|---|
| GPU 読み出し（uint4 sum） | 404〜464 GB/s | 399〜426 GB/s | 帯域の上限の目安 |
| GPU histogram（threadgroup atomic） | 189〜338 GB/s | 69〜135 GB/s | text は偏った bin への atomic 競合で遅い |
| CPU histogram 16 lane | 43 GB/s | 28 GB/s | 私有 count の和 |
| GPU CRC-32 4 KiB chunk（slice-by-4）+ CPU 16 lane combine | 241 GB/s（kernel 350） | 190 GB/s（kernel 263） | zlib と一致 |
| GPU CRC-32 16 KiB chunk + combine | 178 GB/s | 211 GB/s | |
| CPU CRC-32 zlib 16 lane + combine | 205〜238 GB/s | 217〜246 GB/s | 基準 |

読み: histogram は GPU が 2.5〜8 倍速い唯一の byte kernel だが、Deflate / LH5 が数えるのは LZ77 parse 後の記号なので生 byte の histogram
に用途がない（order-0 符号器にしか効かない）。CRC-32 は GPU kernel 単体で CPU 16 lane と同程度、wall では dispatch 往復と combine を足して
同程度以下。CPU 1 core の 43.7 GB/s で足りる用途に GPU を使う理由はない。

### 4.2 LZ4 の独立 block 並列復号（Sources/AccelLab/LZ4、`accel-lab lz4-gpu`、Results/lz4-gpu-idle-*-20260929-1743.tsv、lz4-cpu-liblz4-idle-*）

入力: text256（256 MiB）を独立 block の LZ4 frame にしたもの三つ。64 KiB block は `lz4 -B4 -BI --content-size`（203,301,988 bytes、比 0.757）、
16 KiB（221,517,584、比 0.825）と 4 KiB（240,215,918、比 0.895）は `accel-lab lz4-make-frame`（`lz4 -d` で復号できることを確認）。
GPU は二つの kernel: thread-per-block（1 thread が 1 block を最後まで復号）と SIMD-per-block（32 lane が同じ sequence header を読み、
literal と match を 32 byte ずつ copy）。出力はすべて CPU の結果と byte 同一（match OK）。5 round 中央値、idle、単位 GB/s（展開後 bytes 基準）。

| 経路 | 64 KiB × 4,096 block | 16 KiB × 16,384 | 4 KiB × 65,536 | 時間に含むもの |
|---|---|---|---|---|
| CPU libcompression 1 lane | 3.54 | 4.30 | 4.71 | 復号のみ |
| CPU libcompression 16 lane 静的分割 | 34.4 | 37.8 | **41.4** | 復号 + dispatch |
| CPU libcompression 16 lane 動的（lock 付き counter） | **41.2** | **48.8** | 20.1 | 4 KiB では lock 競合で崩れる |
| CPU liblz4 `LZ4_decompress_safe` 12 thread | 36.6 | 40.4 | 49.5 | pthread 静的分割（Scripts/lz4-cpu-liblz4.c） |
| CPU liblz4 16 thread | 33.4 | 34.9 | 43.3 | |
| GPU thread-per-block（GPU 時間） | 6.41 | 25.4 | 23.9 | kernel のみ |
| GPU SIMD-per-block（GPU 時間） | 12.1 | 16.1 | 22.6 | kernel のみ |
| GPU 最良の wall | 11.7 | 23.6 | 22.6 | encode + commit + wait |
| GPU 出力配置のための scan（直列 / 16 lane） | 2.78 / 29.3 | 3.68 / 30.9 | 4.60 / 36.9 | GPU だけに必要（下記） |
| 入力の MTLBuffer への upload | 23.1 | 22.9 | 22.2 | 1 回 |
| 1 block だけ: GPU 1 thread | 0.0105 | 0.0111 | 0.0120 | 6.2 ms / 64 KiB |
| 1 block だけ: CPU 1 core libcompression | 3.63 | 4.37 | 6.55 | 18 µs / 64 KiB |

読み:
- GPU の 1 thread は CPU の 1 core の **1/350**（64 KiB block: 6.24 ms 対 18 µs）。分岐と byte 単位 copy の多い LZ 復号は GPU の 1 thread に最悪の形。
  thread 数で補うには数万 block が要り、64 KiB block × 4,096 では occupancy が足りず 6.4 GB/s に留まる。
- block を小さくすると GPU は 25 GB/s まで伸びるが CPU も伸び（block が小さいほど cache に収まる）、**どの block size でも CPU 16 lane が 1.7〜6 倍速い**。
  途中の計測（別の GPU 負荷が並走中）で GPU が 4 KiB block で 38 GB/s を出し CPU の動的分割 20 GB/s を上回ったように見えたが、CPU 側は lock 付き
  動的分割の崩れで、静的分割の libcompression（41 GB/s）と liblz4（50 GB/s）に対しては負ける。GPU 側の 38 と idle の 24 GB/s の差は GPU の clock 状態の
  違いと推定する（§4.5 で確かめる）。
- GPU 経路は block ごとの展開後 size を事前に知る必要がある。LZ4 frame にはその表がなく、sequence を舐める scan（直列 58〜97 ms、16 lane で 7〜9 ms）
  が復号本体（6〜12 ms）と同じ桁でかかる。CPU 経路にはこの費用がない。`--assume-uniform` で scan を省いても GPU 側の順位は変わらない。
- SIMD-per-block は 64 KiB block で thread-per-block の 1.9 倍だが、小さい block では barrier の費用が勝って逆転する。
- 技術デモ（§6）の 7.5 MB の tar では GPU 復号 22〜28 ms（Metal library の runtime compile 込み）、CPU 2 ms。小さい書庫では GPU の固定費だけで負ける。

### 4.3 見送った候補と理由（研究 JSON）

- Deflate / LZMA / bzip2 / PPMd / RAR / LHA の stream 内の復号: Huffman・range coder・BWT 逆変換は 1 stream の中で逐次。先行研究（nvCOMP、GDeflate、dietgpu）
  は format を GPU 向けに変えており、既存の書庫には当てはまらない。1 thread が 1/350 では stream 数が数万ないと成立しない。
- 圧縮側の match 探索: LH5 は 8 KiB window の古い format で、GPU に載せる価値がない。自前の LZMA / Zstandard encoder に対する CPU / GPU 候補は §9 で個別に評価する。zlib / libbz2 / Apple Compression は OS 側の実装に委譲している。
- AES / SHA / HMAC: CPU の暗号命令が 1 core で 14〜18 GB/s（AES）、書庫を縛るのは直列の HMAC-SHA1（3.1 GB/s）と CBC 暗号化（1.8 GB/s）で、いずれも
  chain 依存で並列化できない。
- zstd の block 並列: zstd の block は frame の window を共有し独立に復号できない。GyoshukuKit の 7z LZMA2 chunk は 16 MiB で数が少ない。

### 4.4 GPU が勝ちうる条件（本体には当てはまらない）

数万個の独立 block（4〜16 KiB）、container が block ごとの展開後 size を持つ、出力を GPU 側で使う（texture、GPU 上の後続処理）、入力が既に GPU memory にある。
これは nvCOMP / GDeflate が format を作り直した理由そのもので、ZIP / 7z / RAR / tar.xz の既存書庫には一つも当てはまらない。

### 4.5 GPU の clock 状態の確認（Results/lz4-gpu-clock-test-20260929-1826.tsv）

4 KiB frame、thread-per-block、20 round: GPU を数分休ませた直後 24.2 GB/s、gpu-kernels を走らせた直後 16.6 GB/s。「別の負荷で clock が上がっていた」という
推定は再現せず、途中の 38 GB/s の原因は未確定（その時は MPS の学習が数十分続いていた）。idle の値 24 GB/s を採る。いずれにせよ CPU 16 lane の 41〜50 GB/s に届かない。

## 5. NPU（Apple Neural Engine、Core ML）

### 5.1 ANE に載る条件（Results/ane-probe-scaling-20260929-1517.txt、ane-trained-mlp-20260929-1520.txt）

- 全 op が ANE 対応でも Core ML の cost model は小さな model に CPU を選ぶ。hidden 1024・4 層・batch 1024 から ANE が選ばれる。
- 可変形状（RangeDim）の入力は ANE で遅い（batch 4096 で固定形状 2.75 ms 対 可変 7.55 ms）。demo は固定 batch の model を batch ごとに書き出す。
- 推論 1 回の固定費は 0.07〜0.2 ms（ms 級ではない）。1 byte ごとに呼ぶと 10 KB/s にもならないので、数千 block を batch にして 1 推論で全 block を 1 byte 進める。
- native の `lstm` op は CPU に落ちる（研究）。線形 + sigmoid / tanh で書いた GRU cell は 18 op すべてが ANE（`MLComputePlan`、`preferred-device histogram: [ANE: 18]`）。

### 5.2 神経 block codec の仕組み（Sources/AccelLab/Neural、`accel-lab neural-encode / neural-decode`）

入力を N block に切り、位置 t の予測を全 block まとめて 1 推論（入力 one-hot [N, 256] と状態 h_in [N, H]、出力 softmax [N, 256] と h_out）。
256 個の確率を総和 2^16 の頻度に量子化し、block ごとの 32 bit range coder で符号化する。復号は同じ model を同じ順で回す。
range coder と量子化は CPU で 16 lane 並列。model は GRU 1 層（Scripts/train_gru_predictor.py、words32 の先頭 90% で学習、torch MPS → coremltools fp16 mlprogram）。

### 5.3 比（Results/neural-idle-words4m-20260929-1743.txt、cpu-codecs-words4m-20260929.txt）

入力: 学習に使っていない words32 の末尾 4 MiB（辞書の単語を空白区切りで並べた text）。

| codec | bytes | 比 | bits/byte |
|---|---|---|---|
| **GRU h1536、1024 block（ANE）** | **1,169,473** | **0.279** | **2.23** |
| GRU h1536、4096 block（ANE） | 1,198,470 | 0.286 | 2.29 |
| PPMd o8 mem 256m（7zz） | 1,213,978 | 0.289 | 2.32 |
| GRU h1024、1024 block（ANE） | 1,305,886 | 0.311 | 2.49 |
| GRU h1024、4096 block（ANE） | 1,334,207 | 0.318 | 2.54 |
| xz -9 | 1,460,024 | 0.348 | 2.78 |
| LZMA2 mx9（7zz） | 1,460,550 | 0.348 | 2.79 |
| zstd -19 | 1,463,092 | 0.349 | 2.79 |
| bzip2 -9 | 1,575,503 | 0.376 | 3.01 |

- 同じ入力の 1 MiB では h1024 が 0.317、PPMd 0.334、xz 0.395（Results/ane-roundtrip-v1-20260929-1636.txt）。PPMd は file が長いほど学習が進むので差は縮む
  （words32 全体 32 MiB では PPMd 0.243）。
- **domain 外では成立しない**（Results/neural-idle-offdomain-h1536-20260929-1743.txt、ane-neural-v2-offdomain-*）: War and Peace（英語の散文 3.36 MB）で
  h1536 は 0.635（xz 0.278、PPMd 0.220）、Swift の source 4 MiB では 1.087 と**膨らむ**（xz 0.143、PPMd 0.123）。静的な model は学習した分布しか知らない。
  words32 は辞書の約 23 万語を無作為に並べた合成 text で、文法も長距離の構造もなく、試験側の語彙はすべて学習で見ている。上の PPMd 超えは
  この語彙の記憶によるもので、一般の text に敷衍できる数字ではない。混合 corpus（単語 + 英語散文 + Swift）で学習した model の結果は §5.6。

### 5.4 速度（同 file、4 MiB、idle、`--units` で compute unit を切り替え、model と batch は同一）

| model | block 数 | ANE | Core ML CPU | Core ML GPU | 1 step の予測器 / 符号器 ms（ANE） |
|---|---|---|---|---|---|
| GRU h1024 | 1024 | 393 KB/s | 272 KB/s | 434 KB/s | 2.47 / 0.13 |
| GRU h1024 | 4096 | **531 KB/s** | 370 KB/s | **652 KB/s** | 7.35 / 0.37 |
| GRU h1536 | 1024 | 244 KB/s | 162 KB/s | 259 KB/s | 4.05 / 0.14 |
| GRU h1536 | 4096 | 301 KB/s | 190 KB/s | 350 KB/s | 13.2 / 0.38 |

- 復号は符号化と同じ速度（h1536 b1024 ANE: 240 KB/s、`cmp` 一致）。
- ANE は同じ model の Core ML CPU 実行の **1.4〜1.6 倍**。しかし Core ML の GPU 実行（40 core、fp16）がさらに 1.06〜1.23 倍速く、この model では **ANE は最速の unit ではない**。
  1 step あたり ANE 7.35 ms で 4096 × 6.82 M MAC（重み形状から: W_ih と W_hh が各 [3072, 1024]、embed [1024, 256]、out [256, 1024]）= 27.9 GMAC → 3.8 TMAC/s
  （7.6 TFLOPS 相当）。h1536 は 14.9 M MAC / 行、61.2 GMAC / step、13.2 ms → 4.6 TMAC/s。M4 の ANE の公称 38 TOPS（int8）の fp16 半分を上限とすれば 4〜5 割で、
  残りは状態テンソルの往復（[4096, H] fp16 の出入り）と Core ML の呼び出し。MLState で状態を ANE 側に置く版は試していない（§7）。
- compute plan の 32 op のうち 18 が ANE、残る 14 は定数・reshape で device を持たない（n/a）。
- 時間の 95% が予測器（Core ML 呼び出し）で、CPU の range coder は 16 lane 並列で 0.13〜0.38 ms / step。
- 既存の CPU codec との比較: xz -d 322 MB/s、KaitoKit の PPMd 9〜14 MB/s、bzip2 78 MB/s。神経 codec は最速でも 0.65 MB/s で、**2 桁遅い**。
- 最初の版は量子化の余剰調整 loop が 1 step 43 ms を使い 23 KB/s だった（Results/ane-roundtrip-v1）。修正後は Core ML が律速。

### 5.5 約束事: 符号化と復号は同じ unit で

ANE で符号化した bit 列を同じ model・同じ batch の Core ML CPU 実行で復号すると `Invalid range-coded payload` で失敗する
（Results/ane-encoded-cpu-decode-mismatch-20260929.txt）。fp16 の丸めが device で異なり、量子化後の頻度表が 1 でも違えば range coder は破綻する。
NBC1 の header に predictor tag（model 名と unit）を入れ、復号側で照合して警告する。同じ unit 同士で往復を確かめたもの: h1024 b1024 ANE（1 MiB）、
h1024 b4096 ANE（4 MiB）、h1024 b4096 Core ML GPU（4 MiB、Results/neural-roundtrip-b4096-ane-gpu-20260929.txt）、h1536 b1024 ANE（4 MiB）、demo（h1024 と h1536、ANE）。すべて `cmp` 一致。

### 5.6 混合 corpus の model（Results/neural-idle-mixed-h1536-20260929-1826.txt、cpu-codecs-swift-heldout-20260929.txt）

GRU h1536 を混合 corpus（words32 の先頭 16 MiB + Gutenberg の英語散文 10 冊 13 MB + swift-src の先頭 8 MiB、計 38 MB）で 4,000 step 学習。
試験入力はいずれも学習に含まれない: words32 の末尾 4 MiB、War and Peace（Gutenberg 2600、学習の 10 冊に含まない）、swift-src の 8 MiB 以降 3.5 MB。
ANE、1024 block、比（bytes / 元 bytes）:

| 入力 | 混合 GRU h1536 | 単語専用 GRU h1536 | PPMd o8 | xz -9 | zstd -19 |
|---|---|---|---|---|---|
| 単語 text 4 MiB | 0.290 | 0.279 | 0.289 | 0.348 | 0.349 |
| 英語散文 3.36 MB | **0.240** | 0.635 | 0.220 | 0.278 | 0.281 |
| Swift source 3.5 MB | 0.214 | 1.187 | 0.133 | 0.144 | 0.148 |

- 混合 model は三つの domain すべてで膨らまず、散文では xz / zstd / bzip2（0.264）を上回るが PPMd には負ける。source は LZ 系の長い一致（file 間の重複）が
  効く domain で、文脈 256 byte 以内の統計しか持たない GRU は xz にも届かない。
- 速度は 258〜262 KB/s（h1536 b1024 と同じ）。散文の復号は `cmp` 一致。
- 結論は変わらない: 静的 model の神経 codec は domain を跨ぐと PPMd 級か以下で、速度は 2 桁遅い。比で勝てるのは model が学習した分布に近い入力に限る。

## 6. 技術デモ（`accel-lab demo-pack / demo-unpack`、Results/demo-idle-20260929-1743.txt）

directory → tar（GyoshukuKit `ArchiveWriter`）→ 神経 block codec（ANE）か LZ4 frame → KADM container。展開は逆順で、tar は KaitoKit `ArchiveReader` が読む。
入力: 単語 text 512 KiB × 8（うち 4 つは sub/）+ War and Peace = 10 entries、tar 7,567,360 bytes。

| 経路 | 書庫 bytes | 比 | 圧縮 s | 展開 s | 木の一致 |
|---|---|---|---|---|---|
| 神経 codec GRU h1536、1024 block、ANE | 3,321,246 | 0.439 | 35.6（tar 0.005） | 35.6（extract 0.004） | `diff -r` 一致 |
| LZ4 4 KiB block、GPU thread-per-block | 6,285,981 | 0.831 | 0.011 | 0.082（Metal library の compile 込み）※ | 一致 |
| LZ4 4 KiB block、GPU SIMD-per-block | 同上 | | | 0.028 ※ | 一致 |
| LZ4 4 KiB block、CPU 16 lane | 同上 | | | 0.002 ※ | 一致 |

※ idle の再計測（Results/demo-idle-lz4-unpack-20260929-1826.txt）: thread 0.036 s、SIMD 0.023 s、CPU 0.0015 s、木はすべて一致。表の値は Results/demo-roundtrip-20260929-1725.txt。

## 7. 2026-09-29 時点で残っていた手

- MLState で GRU の状態を ANE 側に保持する版（状態テンソルの往復を省く）。batch 8192 の model。2 層 GRU / RWKV cell。
- 学習中の適応（online learning）。NNCP / cmix が比で勝つ理由はこれで、静的 model + 独立 block では domain 外に弱い。
- GPU LZ4 kernel の copy 幅の拡張（4 byte / 16 byte copy）。thread-per-block の 1 thread あたり 10 MB/s は byte 単位 copy の上限に近い。
- `powermetrics` による ANE / GPU の電力測定（sudo が要る）。効率（J/byte）は未計測。
- SME（CPU 側の行列命令、4.0 int8 TOPS / thread）で同じ GRU を回す比較。Core ML の CPU 経路（BNNS）がそれを既に使っているかは未確認。
- CPU 側の改善余地（NPU / GPU ではない）は同日に KaitoKit 本流で実現した: xz の block 並列復号（PR #41、tar.xz staging 5〜6.7 倍）と単一 stream bzip2 の block 単位並列（PR #42、5.5〜6 倍）。libcompression の呼び出し粒度は全 codec が既に 256 KiB で前提が誤りだった（KaitoKit Documentation/verification/2026-09-29-parallel-decoders.md）。

## 8. 追試: MLState と LZ4 copy kernel（2026-11-04〜05）

現行の M4 Max、macOS 27.2、Xcode 27.2、Swift 6.4.2 で再測定した。計測は release build、Mac mini M4 では全 61 test と別 GPU / ANE での出力一致を確認した。

### 8.1 GRU の状態を MLState に保持

`train_gru_predictor.py --stateful` は同じ `.pt` から `hidden_state` を持つ model を書き出す。`CoreMLRecurrentPredictor` は従来の `h_in` / `h_out` と stateful model を自動判別する。状態は zero 初期化と更新を書き出し時に検査し、Swift では CPU / GPU / ANE の予測一致と `begin` による reset を検査する。

重要な変換上の制約:
- `self.hidden_state = h_out` では `write_state` が生成されず、見かけ上 ANE で動いても状態がゼロのままだった。
- 全要素の slice 代入 `self.hidden_state[:, :] = h_out` では `write_state` が生成され、明示的な状態と全要素が一致した。
- `h + (h_new - h)` で代入を代用すると CPU の fp16 丸めが変わるため使わない。

4 MiB の固定英文を 1024 block に分け、GRU h1024、ANE で別 process の符号化を A/B/B/A/A/B の順に測った（`Results/neural-mlstate-codec-ane-20261104.txt`）。wall の中央値は従来 14.005870 s、MLState 10.494882 s で **1.334 倍**、0.299 から 0.400 MB/s。復号は 14.098763 s から 10.373353 s（1.359 倍）だった。range-coded payload は全 byte 一致し、両方とも元入力へ復元できる。file 全体の byte 数が 9 だけ違うのは predictor tag が `stateful` を含むためで、payload の差ではない。

予測器単体の別ラウンド交互測定（`Results/neural-mlstate-slice-20261104.txt`、`neural-mlstate-b4096-20261104.txt`）:

| batch | unit | 明示的な状態 | MLState | 倍率 |
|---|---|---:|---:|---:|
| 1024 | ANE | 2.906 ms/step | 1.996 ms/step | 1.456 |
| 1024 | GPU | 5.436 ms/step | 2.807 ms/step | 1.936 |
| 1024 | CPU | 3.277 ms/step | 2.966 ms/step | 1.105 |
| 4096 | ANE | 7.447 ms/step | 6.377 ms/step | 1.168 |
| 4096 | GPU | 7.759 ms/step | 5.829 ms/step | 1.331 |

GPU は A/A の変動が大きいため、上表だけで実 codec の倍率とみなさない。ANE については別 process の実符号化でも改善を確認した。それでも現行の KaitoKit の PPMd（9〜14 MB/s）より大幅に遅く、本体へは取り込まない。状態保持の知見は今後の model の基礎として使う。

### 8.2 LZ4 の 16 byte copy は既存 kernel に及ばない

`thread16` / `simd16` を追加し、既存の byte copy と比較した。alignment を仮定する pointer cast は使わず、4 つの `uchar4` で 16 byte を構成する。match copy は offset 以下の chunk に制限し、重なりの前後関係を壊さない。offset 15 / 16 / 17、511 / 512 / 513 を含む全 variant の出力を CPU と比較した。

64 MiB の Swift source corpus（SHA-256 `97d3f4c1fb66073ec68e76b205598a5b494c6470f44a51d0fa3cd0bdebe2d7db`）では:

| block size | 既存 thread | thread16 | 既存 SIMD | SIMD16 |
|---|---:|---:|---:|---:|
| 4 KiB | 13.14 GB/s | 10.38 GB/s | 10.12 GB/s | 6.44 GB/s |
| 16 KiB | 6.16 GB/s | 未測定 | 20.95 GB/s | 11.27 GB/s |

`Results/lz4-simd16-source64-*`、`lz4-simd16-vector-source64-*`、`lz4-thread16-vector-source64-4096-20261104.tsv` に生の値を残す。最初の scalar 展開より vector 候補は改善したが、既存方式には届かない。copy 幅を広げれば速くなるという仮説は棄却し、これらは研究用の選択肢に留める。

### 8.3 再現性と測定条件

旧 baseline が持っていた他 agent の一時 directory への絶対 path を廃止し、`baseline --text <path>` を追加した。`Results/baseline-20261104-mini.tsv` は旧 corpus がない環境の生成英文なので旧結果と直接比較しない。`Results/baseline-source64-mini.tsv` は上記の Swift source corpus を指定した Mac mini M4 の値で、全 codec が実際に往復した。

以後の corpus は `Scripts/make_source_corpus.py <out> <MiB> <repo...>` で生成する。例: `python3 Scripts/make_source_corpus.py /tmp/source64.bin 64 ../KaitoKit ../GyoshukuKit . ../KaitoFinder`。編集で source が変わるため、入力の hash も記録する。

full-slice copy を伴わない状態更新、batch 8192、小さな LZ4 block のジョブ集約の結果は §8.6〜8.10 に記録する。電力効果と SME は未測定であり、GPU / NPU の消費電力や効率が優れるとは主張しない。性能測定と全テストを同じ Mac で同時に実行しない。

### 8.4 重みの int8 / int4 量子化

`Scripts/quantize_gru_model.py` で stateful h1024 model を線形対称量子化した。計算は fp16 のままで、重みの保存形式だけを変える。`Results/neural-quantized-roundtrip-20261104.txt` のとおり、どちらも ANE の全 21 op に配置され、実符号化・復号が元入力と一致した。

| model | package bytes | 圧縮率（小さいほど良い） | 評価 |
|---|---:|---:|---|
| fp16 | 13,654,802 | 0.511809 | MLState の基準 |
| int8 | 6,854,472 | 0.512067 | 約 0.05% の比の悪化で実用候補 |
| int4 | 3,446,597 | 1.054183 | 元入力より膨らむため不採用 |

int8 と fp16 の実符号化を別 process で交互に 3 回ずつ測った wall は、fp16 10.560678 s、int8 9.842644 s（中央値、**1.073 倍**）。0.397 から 0.426 MB/s へ改善した（`Results/neural-quantized-ab-20261104.txt`）。int8 は CPU / GPU / ANE で 4 step の符号化・復号テストも通る。既存 codec には依然届かないが、model の容量削減と ANE 高速化を両立できた。

### 8.5 現行 KaitoKit への Apple decoder 適用

現行 KaitoKit（0.21.0、`sha --sink --threads auto`）は、checksum のない独立 LZ4 block を Swift で復号していた。並列計画は全 block の文法と展開後サイズを検証済みなので、その経路では checksum がなくても `LZ4AppleBlockDecoder.decodeValidated` を使える。従来の Swift fallback も残す。

同じ 64 MiB source corpus で現行 binary と変更後 binary を交互に測った（`Scripts/benchmark_lz4_cli.py`、`Results/lz4-kaito-apple-nocheck-source64-20261104.txt`）。

| block size | 現行 | Apple 適用後 | 改善 |
|---|---:|---:|---:|
| 4 KiB | 0.105124 s | 0.097289 s | 1.080 倍 |
| 16 KiB | 0.050313 s | 0.047519 s | 1.059 倍 |
| 64 KiB | 0.037022 s | 0.033762 s | 1.097 倍 |

A/A のばらつきを超えて、3 種類とも改善した。checksum なしを含む破損・切断・失敗時 buffer の比較を追加し、LZ4 の 33 test が通る。読み出しを 256 KiB に合体する案は、無条件では失敗時 buffer の互換を壊し、checksum なしに限定しても速度差が誤差範囲だったため取り込まない。

Mac mini M4 でも同じ入力で比較した。GPU は 4 KiB の最良で 9.18 GB/s、16 KiB は 6.16 GB/s、64 KiB は 6.30 GB/s で、CPU 16 lane の 25.04 / 24.07 / 23.64 GB/s に届かない（`Results/lz4-kaito-current-mini-20261104.txt`）。実 CLI の wall と raw kernel は区別する。次の本体側の候補は、小さな block を一つのジョブに集約して、submit・待機・結果管理の費用を減らすことである。

Apple 化だけの版は Mac mini の全 2,096 test（64 skip）でも failure 0。実 CLI は 4 KiB 0.110558 → 0.096743 s、16 KiB 0.057632 → 0.050397 s、64 KiB 0.043613 → 0.034974 s で、それぞれ 1.143 / 1.144 / 1.247 倍だった（`Results/lz4-kaito-apple-mini-20261104.txt`）。

### 8.6 小 block の集約と展開後サイズの並列スキャン

256 KiB の出力を一つの復号ジョブとし、内部では各 block の read 境界を保つ `LZ4FrameBlockBatch` を挟む。worker は batch 全体を処理し、consumer で失敗した場合は batch の各 block を順に直列再実行するため、部分出力・失敗時 buffer・エラー順を変えない。1 MiB への増加は誤差範囲だったため、メモリーと再試行範囲が小さい 256 KiB とした。小さな frame が一つの batch になる場合は従来の block 単位を使う。

さらに、block table の読み取りと展開後サイズの計算を分離し、最大 `workers` 個の静的 lane で `validatedSize` を評価する。結果は入力順に `get()` し、最初の失敗を従来どおり直列 reader へ戻す。並列に求めたサイズの prefix sum が上限を超える場合も、従来の推測的計画と同じく serial fallback にする。

現行 0.21.0 と最終候補の `sha --sink --threads auto`、同じ source corpus、別 binary の交互測定:

| 入力 | block size | 現行 | 集約＋並列スキャン | 改善 |
|---|---|---:|---:|---:|
| 64 MiB | 4 KiB | 0.105119 s | 0.046040 s | 2.28 倍 |
| 64 MiB | 16 KiB | 0.049523 s | 0.022293 s | 2.22 倍 |
| 64 MiB | 64 KiB | 0.037343 s | 0.016607 s | 2.25 倍 |
| 256 MiB | 4 KiB | 0.397496 s | 0.155642 s | 2.55 倍 |
| 256 MiB | 16 KiB | 0.175224 s | 0.067703 s | 2.59 倍 |
| 256 MiB | 64 KiB | 0.129713 s | 0.045844 s | 2.83 倍 |

256 MiB の最終候補は 1.72 / 3.96 / 5.86 GB/s。生の値は `Results/lz4-kaito-parallel-scan-source64-20261104.txt`、`lz4-kaito-parallel-scan-source256-20261104.txt` に残す。34 の LZ4 test に加え、集約内の後続 block が失敗する注入 test、後続 block の文法エラー、乱数・ゼロ列・外部 `lz4` の独立 block＋両 checksum、256 MiB の SHA-256 一致を確認した。これは研究用 GPU kernel ではなく、現行 CPU 本体の改善として取り込む候補である。

### 8.7 MLState の形状変更と直接状態更新

`--stateful-select` は状態を `[1,B,H]` とし、`--stateful-direct` は全要素を置き換える `slice_update` を graph pass で直接 `h_out` への状態更新へ置き換える。select の内部にも `slice_update` は残るため、copy を完全に削除した結果ではない。direct の MIL には `slice_update` がない。

| 経路 | 既存 MLState | 候補 | 判定 |
|---|---:|---:|---|
| select / ANE | 2.031 ms/step | 1.994 ms/step | 約 2% で誤差に近い |
| select / GPU | 3.478 ms/step | 2.986 ms/step | 約 14% 改善 |
| direct / GPU | 2.991 ms/step | 2.471 ms/step | 約 17% 改善 |
| direct / ANE | 2.031 ms/step | 実行計画 error -14 | 不採用 |

select と direct とも CPU / GPU では既存 model と全予測が一致し、begin による reset も確認した（`Results/neural-state-select-20261104.txt`、`neural-state-direct-20261104.txt`）。ANE では既存の MLState を使い、direct は CPU / GPU の研究経路に限定する。テストは三つの状態形式と int8 を対象に追加した。

### 8.8 ASan / UBSan の変異入力

`Scripts/fuzz/run-mutants.sh --count 200 --timeout 5` を、乱数・ゼロ列・Swift source・外部 `lz4` の独立 block＋両 checksum の 4 seed に対して実行した。総 200 件で crash 0、hang 0、sanitizer finding 0。`--count` は seed ごとの数ではなく、4 seed 合計の変異数である。

### 8.9 Mac mini での最終確認

集約＋並列スキャンの KaitoKit は、Mac mini M4 の全 2,097 test（64 skip）で failure 0、compat の 34 test でも failure 0 だった。実 CLI は次のとおり（`Results/lz4-kaito-parallel-mini-20261104.txt`）。

| block size | 現行 | 最終候補 | 改善 |
|---|---:|---:|---:|
| 4 KiB | 0.111276 s | 0.043320 s | 2.57 倍 |
| 16 KiB | 0.056996 s | 0.022517 s | 2.53 倍 |
| 64 KiB | 0.043240 s | 0.017926 s | 2.41 倍 |

KaitoKit の変更は `LZ4FrameDecompressor.swift` とその回帰テストに限定し、Swift の serial fallback と ABI を維持している。研究用の GPU kernel は本体には入れず、測定で裏付けた CPU 経路の改善を本体用の差分とした。

最新の AccelLab も Mac mini で全 62 test が failure 0、release build も成功した（`Results/accel-lab-latest-mini-20261104.txt`）。三つの状態形式・int8・Metal variants を含む。

### 8.10 batch 8192

固定 batch を 8192 に拡大し、明示的な状態・MLState・int8・直接状態更新を比較した（`Results/neural-b8192-20261104.txt`）。予測値は明示的な状態と MLState / direct で一致した。

| unit / 経路 | 既存方式 | MLState / direct | 倍率 |
|---|---:|---:|---:|
| ANE / 明示的な状態 → MLState | 14.273 ms/step | 12.085 ms/step | 1.181 |
| GPU / 明示的な状態 → MLState | 11.564 ms/step | 10.250 ms/step | 1.128 |
| GPU / MLState → direct | 10.437 ms/step | 10.210 ms/step | 1.022（誤差範囲） |

実符号化は fp16 の MLState が中央値 6.567283 s、int8 が 6.157422 s で、**1.067 倍**（0.639 → 0.681 MB/s）。int8 の ANE / GPU 往復は元入力と一致した（`Results/neural-b8192-int8-ab-20261104.txt`、`neural-b8192-codec-20261104.txt`）。int8 GPU は fp16 GPU より遅く、unit ごとの有利・不利がある。

batch 1024 の圧縮率 0.511809 に対し、8192 は fp16 0.526516、int8 0.526764 となり、速度の代わりに独立 block 初期化の費用で約 2.9% 大きくなる。したがって比を優先すれば小さな batch、速度を優先すれば 8192 が良い。MLState ANE の予測器 throughput は 4096 の 0.642 MB/s から 8192 の 0.678 MB/s への増加に留まり、無制限に拡大しても比例して速くはならない。既存 codec には依然届かないため、神経 codec は本体には取り込まない。

## 9. GyoshukuKit の圧縮側

### 9.1 LZMA BT4 の bucket 並列 Metal 探索

既存の `LZMAMatchFinder` と同じ HASH4 bucket ごとに位置を集め、一つの Metal thread が一つの木を位置順に処理する研究実装を試した。H2/H3 の候補は事前に保存し、BT の候補順・距離・短い距離 record を維持する。dictionary の周期をまたぐ並列化では、未来の位置が別 bucket の使用中の node を上書きするため、周期ごとの wave と現行・前周期の 2 面の tree buffer が必要だった。

64 KiB text、20,000 byte・dictionary 4,096、短入力、600 prefix、途中の tree 状態を CPU と比較した。release benchmark でも 4 MiB text の全候補・hash・最終 tree が一致した。4 MiB text、preset 6、best-of-3 の結果は次のとおり（`Results/gyoshuku-lzma-bucket-metal-20261104.txt`）。

| machine | CPU finder | GPU 全処理 | bucket 作成 | GPU kernel | finder 比 |
|---|---:|---:|---:|---:|---:|
| M4 Max | 0.829685 s | 0.741794 s | 0.109419 s | 0.456629 s | 1.118 倍 |
| Mac mini M4 | 0.554095 s | 0.845353 s | 0.104902 s | 0.457024 s | 0.655 倍 |

GPU 全処理は bucket 作成・buffer・結果の Swift array への転送を含み、Metal 初期化と parser / range encoder を含まない。したがって圧縮全体の倍率ではない。Mac mini の full CPU encode は 0.850818 s で、parser 前の GPU 処理だけでほぼ同じ時間を使った。M4 Max の限定的な改善をもって採用せず、Metal link と研究 file は GyoshukuKit から外した。再現用の code は `Research/LZMABucketMetal/` に保存する。

`Research/LZMABucketMetal/run-tests.sh` は、本番 checkout を変更せず一時 copy に Metal link を足して CPU 参照との比較を実行する。本番への Metal 依存追加や、M4 Max だけの結果を根拠にした採用はしない。

### 9.2 Zstandard doubleHash の repeat 先読み

三つの repeat について、どれかを長く延長する前に 8 byte の差分をすべて求める CPU 候補を試した。重複 repeat の除外・比較順・最長一致の選び方は変えていない。既存の 21 test（3 skip）が成功し、text / binary / mixed、level 1 / 3 / 9 / 19 の 12 frame は変更前と byte 一致した。

| level 3 | 変更前 | 変更後 | 倍率 |
|---|---:|---:|---:|
| text | 298.797 MB/s | 310.925 MB/s | 1.041 倍 |
| binary | 212.074 MB/s | 210.795 MB/s | 0.994 倍 |
| mixed | 414.474 MB/s | 406.831 MB/s | 0.982 倍 |

text だけは改善したが、変更していない level にも数 % の変動があり、一貫した end-to-end 改善とは判断できない。採用せず、実験差分だけを戻した。前後の全結果は `Results/gyoshuku-zstd-repeat-preload-before-mini-20261104.txt`、`Results/gyoshuku-zstd-repeat-preload-after-mini-20261104.txt` に残す。

### 9.3 Zstandard の short / long hash 候補の先読み

short hash の候補を延長する前に、long hash の候補に対する 8 byte の履歴比較も計算する候補を試した。niceLength に達した short 候補では長い候補の load が不要になる場合も、今回は潜伏時間を隠すために投機的に読む。既存の 21 test（3 skip）が成功した。

数 % の環境変動を避けるため、変更前の xctest bundle を保存し、level 3 の text / binary / mixed を変更前後で交互に 5 回ずつ実行した。15 組の frame はすべて byte 一致した。中央値は次のとおり（`Results/gyoshuku-zstd-hash-preload-ab-mini-20261104.txt`）。

| corpus | 変更前 | 変更後 | 倍率 |
|---|---:|---:|---:|
| text | 313.614 MB/s | 311.879 MB/s | 0.994 倍 |
| binary | 213.672 MB/s | 214.516 MB/s | 1.004 倍 |
| mixed | 416.792 MB/s | 411.865 MB/s | 0.988 倍 |

short / long の両方を読む負担が効果を相殺し、一貫した改善は得られなかった。不採用として parser と測定用の一時 filter を戻した。

### 9.4 Zstandard の辞書挿入 2 位置化

fast / doubleHash が match 内を辞書へ登録する loop で、2 位置の入力と hash を先に求める候補を試した。hash 衝突時の最終値を保つため store 順は変更せず、末尾は scalar で処理する。既存の 21 test（3 skip）が成功し、level 1 / 3 の 30 組の frame は byte 一致した。保存した baseline bundle と交互に 5 回測った中央値は次のとおり（`Results/gyoshuku-zstd-pair-insert-ab-mini-20261104.txt`）。

| corpus | level 1 倍率 | level 3 倍率 |
|---|---:|---:|
| text | 1.001 倍 | 0.998 倍 |
| binary | 0.997 倍 | 0.996 倍 |
| mixed | 0.996 倍 | 1.005 倍 |

全条件が約 ±0.5% に収まり、loop 展開による一貫した改善は確認できなかった。不採用として parser と一時的な測定 filter を戻した。

### 9.5 LH5 辞書の pointer 化（採用）

LH5 の `heads` / `previous` を `write` 全体で有効な `UnsafeMutablePointer<Int>` として確保し、`findMatch` には pointer で渡す。chain・byte 比較・候補の採用規則を変えず、配列の境界検査と管理費用を除いた。

Mac mini で 3 corpus・3 level を変更前後の bundle で交互に 5 回測り、全 45 組の出力が byte 一致した。中央値は次のとおり（`Results/gyoshuku-lh5-pointer-20261104.txt`）。

| corpus | level | 変更前 | 変更後 | 倍率 |
|---|---:|---:|---:|---:|
| text | 1 | 132.942 MB/s | 150.081 MB/s | 1.129 倍 |
| text | 6 | 79.996 MB/s | 93.016 MB/s | 1.163 倍 |
| text | 9 | 44.512 MB/s | 50.983 MB/s | 1.145 倍 |
| binary | 1 | 113.785 MB/s | 119.763 MB/s | 1.053 倍 |
| binary | 6 | 90.459 MB/s | 99.369 MB/s | 1.098 倍 |
| binary | 9 | 36.289 MB/s | 43.236 MB/s | 1.191 倍 |
| random | 1 | 83.634 MB/s | 84.617 MB/s | 1.012 倍 |
| random | 6 | 83.749 MB/s | 85.732 MB/s | 1.024 倍 |
| random | 9 | 84.776 MB/s | 85.296 MB/s | 1.006 倍 |

M4 Max でも 9 条件すべて改善し、出力は byte 一致だった。独立 decoder 3 種類を使う 7 test、最大距離・block 跨ぎの ASan 2 test に加え、Mac mini の LHA method / stream splice / parallel writer を含む関連 23 test も成功した。最後は GyoshukuKit 全 944 test（51 skip）が成功した。codec 単体の再現用 benchmark を `LH5EncoderBenchmarkTests` に追加した。

### 9.6 LH5 適応型の 8 byte 一致長比較（採用）

pointer 化した `findMatch` の一致延長を、未整列 UInt64 の XOR と trailing-zero count に置き換えた。読み出しは入力 limit を越えず、末尾は従来どおり scalar で処理する。M4 Max で pointer 版と交互に 3 回測り、27 組の出力はすべて byte 一致した（`Results/gyoshuku-lh5-match8-max-20261104.txt`）。

| corpus | level | pointer | 8 byte | 倍率 |
|---|---:|---:|---:|---:|
| text | 1 | 143.072 MB/s | 166.380 MB/s | 1.163 倍 |
| text | 6 | 88.130 MB/s | 97.971 MB/s | 1.112 倍 |
| text | 9 | 48.125 MB/s | 55.803 MB/s | 1.160 倍 |
| binary | 1 | 114.571 MB/s | 119.408 MB/s | 1.042 倍 |
| binary | 6 | 94.138 MB/s | 97.427 MB/s | 1.035 倍 |
| binary | 9 | 41.136 MB/s | 42.527 MB/s | 1.034 倍 |
| random | 1 | 83.716 MB/s | 83.178 MB/s | 0.994 倍 |
| random | 6 | 84.124 MB/s | 83.150 MB/s | 0.988 倍 |
| random | 9 | 83.926 MB/s | 83.212 MB/s | 0.991 倍 |

最初の 8 byte を scalar にする guarded 版は random を改善する一方、text の利益を失ったため不採用とした。unguarded 版は Mac mini の random level 1・6 で約 4〜5% 低下したため、そのままでは採用しない。長さ 3 を先に返す early-exit も text・binary を下げるため不採用とした。

そこで、長さ 8 以上の一致が 2 回見つかった後だけ 8 byte 比較を有効にし、random では scalar を維持する適応型を試した。Mac mini の 3 条件ずつの交互測定では、text / binary が 5〜15% 向上し、random level 1・6 も pointer を上回った。random level 9 は 0.8% 低下した。M4 Max では 9 条件すべて改善した。81＋9 組の出力は byte 一致で、ASan と LHA 関連 23 test も成功したため、この適応型を採用した（`Results/gyoshuku-lh5-adaptive-mini-20261104.txt`）。

### 9.7 PPMd state 検索の 2 件化（不採用）

`PPMdEncodingModel.find` を 2 state ずつ処理する候補と、3 state 以下は従来 loop にするしきい値版を試した。PPMdEncoderTests の 8 件（2 skip）が成功し、全 12 条件の出力サイズと独立 oracle が一致した。

Mac mini の unroll は binary を 1.3〜3.3% 向上したが、text は最大 0.5% 低下した。M4 Max でも binary は 1.8〜2.8% 向上し、text は約 ±0.5% だった。しきい値版も binary を 0.9〜2.2% 向上した一方、text H6 は 1.3% 低下した。どちらも全入力で有利とは確認できず、元の state 検索へ戻した。全値は `Results/gyoshuku-ppmd-unroll-20261104.txt` に記録する。

### 9.8 LZW の強制 inline（不採用）

LZW の `put` と `flushGroup` に `@inline(__always)` を付けた。LZW の differential・実 compress 互換テストは通ったが、Mac mini の 10 回の交互測定では 81.55 → 75.95 MB/s と約 6.9% 低速だった。コード肥大による不利を確認し、変更を戻した（`Results/gyoshuku-lzw-inline-20261104.txt`）。

### 9.9 LZ4 出力 buffer の統合（不採用）

Apple の圧縮結果を frame 出力 buffer の offset 4 に直接書き、payload の再コピーを省く候補を試した。stored block と payload-only checksum は従来どおりとし、全 20 組の frame が byte 一致、ASan も成功した。

Mac mini の text は 956.967 → 967.979 MB/s（1.011 倍）だったが、M4 Max では 869.241 → 867.830 MB/s（0.998 倍）だった。M4 Max の random は 20 回の内側反復でも 1.0% 低下し、Mac mini の小さな text 改善が両機で再現しなかった。stored block を悪化させないため、元の組立へ戻した。測定用の `LZ4FrameEncoderBenchmarkTests` は再現用に残す（`Results/gyoshuku-lz4-buffer-20261104.txt`）。

### 9.10 LZW dictionary の下位 32 bit key（採用）

線形探索 dictionary の entry を `key << 16 | code` から `code << 48 | key` に変え、probe の比較を entry の shift 付き 64 bit から下位 32 bit の直接比較にした。hash・probe・code 割り当て順は同じで、生成する圧縮 byte は変えない。自分の binary では `cmp x22, x8, lsr #16` が `cmp w22, w8` になり、write 関数は 3,276 から 3,260 byte に縮んだ。

新しい `LZWStreamEncoderBenchmarkTests` で 4 MiB text・Mach-O・random、maxbits 12 / 16 を測る。独立した参照 encoder と全出力が一致した。各 run の内側 10 回の最良時間を、両者 3 run ずつの中央値で比較した時間比（小さいほど良い）。

| corpus | maxbits | Mac mini | M4 Max |
|---|---:|---:|---:|
| text | 12 | 0.963 | 0.988 |
| text | 16 | 0.987 | 1.008 |
| binary | 12 | 0.972 | 0.985 |
| binary | 16 | 0.996 | 1.001 |
| random | 12 | 0.955 | 0.995 |
| random | 16 | 0.961 | 0.997 |

Mac mini は全条件が約 0.4〜4.5% 改善した。M4 Max は ±1% 前後の混在で、一貫した改善とは見なさない。別の交互測定では周波数変動が大きかったため、best-of-inner-loop を使い、全生値を `Results/gyoshuku-lzw-key32-20261104.txt` と `gyoshuku-lzw-key32-ab10-{mini,max}-20261104.txt` に残す。

差分・実 `compress` 互換を含む 10 test（1 skip）と ASan は成功した。最後に Mac mini で全 946 test（53 skip、failure 0、3,259.073 秒）を確認し、採用した（`Results/gyoshuku-lzw-key32-full-mini-20261105.txt`）。

## 10. その他の decoder

### 10.1 CAB LZX の 12 bit Huffman 高速表（採用）

`LZXHuffmanTable.fastBits` を 10 から 12 に増やし、canonical tree への fallback を減らした。検証と出力は変えない。公式 7-Zip 25.01 MSI の `product.cab`（LZX:21、106 file）を使い、cabextract と baseline / candidate の SHA-256 が一致することを確認した。

| machine | baseline | 12 bit 表 | 時間比 |
|---|---:|---:|---:|
| Mac mini M4 | 30.762 ms | 28.757 ms | 0.935 |
| M4 Max | 30.642 ms | 28.673 ms | 0.936 |

LZX の 17 test（1 skip）と ASan が成功した。最後に Mac mini で全 2,097 test（64 skip、failure 0、1,419.091 秒）と Compat の 34 test（failure 0、2.166 秒）を確認した（`Results/kaito-lzx-fast12-full-mini-20261104.txt`）。

### 10.2 Deflate64 primary 11 bit（不採用）

Deflate64 の primary lookup を 9 から 11 bit に増やしたが、M4 Max の 8 回の交互測定では 3787.815 → 3773.945 MiB/s と 0.37% 低速だった。LZX のような fallback 減少の利益はなく、9 bit を維持する（`Results/kaito-deflate64-primary11-20261104.txt`）。

### 10.3 PPMd H/I の文脈・state 検査の再利用（採用）

`PPMd7Model` と `PPMdVarIModel` で、同じ 12 byte context や 6 byte state を symbol ごとに繰り返し検査していた経路を集約した。最初の検証で証明した state 数・block 参照を、arena 割り当てや state 移動が起きない範囲に限って再利用する。suffix・successor・拡大後の state 配列・range coder の検査は残し、確率計算と model 更新順は変えていない。

`Scripts/make_ppmd_corpus.py` で作った 16 MiB text、2048 平方 BMP 3 枚、JPEG 3 枚を、7zz の黒箱 writer で PPMd H（7z）と I（ZIP）にした。`kaito sha --sink --threads 1`、Mac mini は 5 round、M4 Max は 3 round の交互測定（ms）である。段階ごとの A/B 値と生の全 round は `Results/kaito-ppmd-context-20261104.txt` に記録する。

| archive | mini 変更前 | mini 最終 | Max 変更前 | Max 最終 |
|---|---:|---:|---:|---:|
| text.7z | 184.931 | 170.333 | 185.504 | 170.098 |
| text.zip | 232.875 | 195.796 | 235.889 | 199.152 |
| color.7z | 5873.238 | 5569.464 | 6817.878 | 6530.736 |
| color.zip | 5285.860 | 4941.207 | 6130.307 | 5736.085 |
| jpeg.7z | 974.711 | 888.130 | 1176.791 | 1082.952 |
| jpeg.zip | 870.799 | 840.493 | 1060.755 | 1024.351 |

凍結した旧実装との差分、38 生成書庫、破損入力と batch size の比較に加え、独立に link した公開 API harness の 496 結果が、byte 数・SHA-256・返却済み prefix・error まで一致した。PPMd の ASan 40 test（1 skip）も failure / finding 0。最後に Mac mini で全 2,098 test（64 skip）と Compat の 34 test が failure 0 だった（`Results/kaito-codec-final-full-mini-20261104.txt`）。

### 10.4 LHA primary 12 bit（不採用）

LHA の primary table を 11 から 12 bit に拡大した。lh5 / lh6 / lh7 の 16 MiB text と 3 枚の BMP（37,748,898 byte）について、`sha --sink --threads 1` を 8 回交互に測った（ms）。

| archive | mini 11 bit | mini 12 bit | Max 11 bit | Max 12 bit |
|---|---:|---:|---:|---:|
| text-lh5 | 37.370 | 36.858 | 43.197 | 42.940 |
| text-lh6 | 31.882 | 32.056 | 37.464 | 38.124 |
| text-lh7 | 30.770 | 30.592 | 35.993 | 35.807 |
| color-lh5 | 235.027 | 235.751 | 248.236 | 251.144 |
| color-lh6 | 218.541 | 221.140 | 235.348 | 235.330 |
| color-lh7 | 216.754 | 214.681 | 230.739 | 230.610 |

21 の decoder test は成功し、独立した `lha-unix t` でも全 entry の CRC が一致した。しかし lh6 の色画像が両機で後退し、text / lh7 の改善も小さく、表構築と cache の費用を上回らなかった。11 bit に戻した（`Results/kaito-lha-primary12-{mini,max}-20261104.txt`）。

### 10.5 RAR2 の short-code lookup と raw pointer 表（採用）

RAR2（RAR4 unpack version 20 / 26）は、全 symbol を 1 bit ずつ最大 15 回読んでいた。1...10 bit の符号を 1,024 entry の UInt32 表で即時復号し、長い符号と不完全な tree は従来の canonical loop に残す。raw pointer 表は table の寿命で確保・解放し、canonical code の検証済み範囲だけを埋める。`peekPadded` は look-ahead の padding では `overrun` を立てず、実際に消費する symbol 幅では従来の切断検査を行う。

自作 RAR 2.50 fixture 4 件を `Scripts/benchmark_kaito_inprocess.py` で測った。`kaito bench --threads 1`、200 回の実 entry 読み出しを 1 計測とし、変更前後を 4 回ずつ交互に実行した中央値（ms）。

| fixture | mini 変更前 | mini 最終 | Max 変更前 | Max 最終 |
|---|---:|---:|---:|---:|
| autorej | 0.026 | 0.027 | 0.026 | 0.027 |
| audio | 0.5775 | 0.5400 | 0.6135 | 0.5690 |
| biglz | 2.8005 | 1.6495 | 3.0260 | 1.7160 |
| solid | 0.2915 | 0.2915 | 0.3080 | 0.3010 |

biglz は両機で約 1.7 倍、audio は約 7% 改善した。solid は mini で同じ、Max で約 2% 改善、最小の autorej は 0.001 ms の差である。これは 4 fixture の結果であり、全 RAR2 入力への一律の倍率ではない。

RAR20 の 6 test、公開 API の 264 結果の完全一致（出力・prefix・error）、ASan の 8 testを確認した。最後に Mac mini で全 2,098 test（64 skip）と Compat の 34 test が failure 0 だった（`Results/kaito-codec-final-full-mini-20261104.txt`）。

### 10.6 RAR3／RAR5 の primary 12 bit（不採用）

RAR2 の成功をそのまま他形式へ拡大せず、RAR3 / RAR5 も研究用の複製で primary table を 12 bit にして比較した。RAR 6.24 の黒箱 writer で作った 16 MiB text と 3 枚の BMP を、`kaito bench --threads 1`、5 回の実 entry 読み出しを 1 計測として 4 回ずつ交互に測った（M4 Max、ms）。

| archive | 10 bit | 12 bit | 判定 |
|---|---:|---:|---|
| rar29-text | 30.525 | 31.370 | 約 2.8% 低速 |
| rar29-color | 220.540 | 218.038 | 約 1.1% 改善 |
| rar5-text | 20.5215 | 21.4635 | 約 4.6% 低速 |
| rar5-color | 198.6795 | 200.198 | 約 0.8% 低速 |

RAR4ReaderTests、RAR5HuffmanTableTests、RAR5DecoderTests は成功したが、RAR5 は両方、RAR3 は text が後退した。RAR3 の color だけの改善を優先せず、両形式とも現行 10 bit を維持する。本番 tree には入れず、測定と複製の説明を `Results/kaito-rar-primary12-20261104.txt` に残す。

### 10.7 RAR5 long-code fallback の展開（不採用）

表を広げずに長い符号だけを改善するため、canonical loop の 11〜15 bit を 5 個の明示的な条件分岐へ展開した。全 32,768 prefix、切断・非整列入力、table 再構築を含む 23 test は成功したが、性能は大きく後退した。

| archive | 現行 loop | 展開 | 時間比 |
|---|---:|---:|---:|
| rar5-text | 20.591 ms | 24.1065 ms | 1.171 |
| rar5-color | 198.391 ms | 223.1105 ms | 1.125 |

コードサイズと分岐が増え、既存ループの方が明確に速い。本番には入れず、研究用複製と `Results/kaito-rar5-long-unroll-20261104.txt` に結果を残す。

### 10.8 RAR15 の consume-only bit 更新（採用）

`RAR15Decoder.addBits` は `bits.read(count)` の返り値を捨てていた。`bits.consume(count)` に変え、不要な peek・返り値生成・bit 数の二重検査を省いた。packed-bit budget、実 bit 位置、alignment、切断時の overrun は変えない。

`Scripts/benchmark_kaito_inprocess.py` で、既存の RAR 1.54 文書・音声書庫を 50 回ずつ展開する計測を 4 回交互に行った。

| archive | M4 Max 変更前 | consume | 短縮 |
|---|---:|---:|---:|
| 17 file の文書 | 6.0985 ms | 5.6735 ms | 約 7.0% |
| 音声 | 1.0415 ms | 0.9260 ms | 約 11.1% |

Mac mini の独立 baseline / candidate でも、文書は 6.0840 → 5.6260 ms（約 7.5%）、音声は 1.0165 → 0.8755 ms（約 13.9%）短縮した（`Results/kaito-rar15-consume-mini-20261105.txt`）。両者の 9 test は文書・音声の全 byte、solid、multi-volume、切断入力まで一致した。M4 Max と Mac mini の ASan 9 test、Mac mini の全 2,098 test（60 skip）と Compat の 34 test も成功した（`Results/kaito-rar15-full-mini-20261105.txt`）。結果は `Results/kaito-rar15-consume-20261104.txt` に記録する。

### 10.9 RAR3 filter global-data の一括 consume（採用）

RAR3 filter の global data は、検査済みの長さに対して 1 byte ずつ `read(8)` を呼び捨てていた。同じ切断エラーを返す `RAR3MemoryBitCursor.consume` を追加し、残 bit 数の検査後に `globalLength * 8` を一度だけ進める。上限 0x1FC0 のため乗算は安全で、payload grammar や filter 出力は変えない。

global-data が 16 filter × 128 byte の実 RAR29 stream を 1,000 回展開する計測を、保存した XCTest bundle と交互に行った。baseline は中央値 24.070 ms、一括 consume は 18.620 ms で、約 **22.6%** 短縮した。global-data が多い入力に限った結果である。

長さ 0・1・17・128・上限 0x1FC0 の実 stream が global-data なしと同じ出力になること、1・9・33 byte の切断を拒否することを確認した。16 bit payload-length の経路も含む。M4 Max と Mac mini の RAR29R1Tests・ASan は各 8 test、0 failure / finding である（`Results/kaito-rar3-global-20261105.txt`）。

### 10.10 RAR3 memory cursor の直接読み出し（不採用）

`RAR3MemoryBitCursor.read` を 64 bit の未整列 load にする版と、8 bit 読みだけを 16 bit load にする版を試した。4 種類の実 filter 書庫と global-data 上限を含む 13 test は成功したが、complete decode への効果は小さかった。

| fixture | 現行（ms） | 64 bit 版 | 16 bit 版 |
|---|---:|---:|---:|
| rgb | 0.3365 | 0.3390 | 0.3380 |
| delta | 0.1300 | 0.1300 | 0.1285 |
| e8 | 0.1045 | 0.1030 | 0.1035 |
| audio | 0.2820 | 0.2830 | 0.2825 |

e8 だけがわずかに改善したが変動範囲で、RGB は改善せず、delta / audio は同じだった。parser のコストが実展開の大部分ではないため、直接読み出しは本番に入れず、現行の scalar read を維持する（`Results/kaito-rar3-cursor-read-20261105.txt`）。

### 10.11 RAR15 decodeNum の index 表（不採用）

threshold の線形探索を `num >> 5` の 2,048-entry 表に変える候補を試した。全 dec threshold が 32 単位であることを使い、`dec[0] << 4 | dec.count` を key にして、同じ 8 要素の `decHf1` と `decHf2` を区別した。実書庫の 9 test は成功したが、速度は文書 5.7225 → 6.1500 ms、音声 0.9345 → 1.0700 ms と、それぞれ約 7.5%・14.5% 低速だった。

Dictionary の参照と table 選択の費用が、短い線形探索を省く利益を上回った。本番には入れず、元の探索を維持する（`Results/kaito-rar15-decode-indices-20261105.txt`）。

### 10.12 RAR15 decodeNum・addBits の inline（採用）

`decodeNum` は outline のままだったため、この小さな関数と `addBits` に `@inline(__always)` を付けた。threshold 探索・bit 数・packed-bit budget・エラーは変更していない。自分の binary では、対象の outline symbol が消えた。

| machine / stage | 文書 | 音声 |
|---|---:|---:|
| M4 Max / 元の実装 | 5.7425 ms | 0.9390 ms |
| M4 Max / decodeNum inline | 5.0890 ms | 0.8080 ms |
| M4 Max / 両 inline | 4.8740 ms | 0.7470 ms |
| Mac mini / 元の実装 | 6.0515 ms | 1.0155 ms |
| Mac mini / 両 inline | 4.7730 ms | 0.7305 ms |

元の実装との累積差は、M4 Max で文書 15.1%・音声 20.4%、Mac mini で文書 21.1%・音声 28.1% の短縮。`decodeNum` 単独から `addBits` を足した段階でも、文書 2.7%・音声 4.8% 改善した。全 9 test と両機の ASan が成功し、Mac mini の全 2,099 test（60 skip）と Compat の 34 test も成功した（`Results/kaito-rar15-inline-full-mini-20261105.txt`）。

### 10.13 RAR15 の非重複 match 一括コピー（不採用）

非重複 match を history ring と caller buffer の両方へ一括コピーする候補を試した。source と destination をそれぞれ独立に ring 境界で分割し、history には `memmove`、caller には `memcpy` を使った。全 9 test と ASan は成功したが、16 byte 以上に限定しても文書 4.8865 → 4.9690 ms、音声 0.7440 → 0.7590 ms と約 1.7%・2.0% 低速だった。

短い match が多く、関数呼び出しと二つの出力先へのコピー費用が loop 削減を上回った。本番には入れず、scalar copy と両 inline を維持する（`Results/kaito-rar15-copy-20261105.txt`）。

### 10.14 LArc benchmark corpus の誤り（再測定）

`LArcDecoder.emit` の `@inline(__always)` 化と、直前の `produced < expectedSize` でオーバーフローしないことが保証される加算の `&+=` 化を試した。生成コードで `emit` symbol が消えることと、両機の既存 7 test は確認した。しかし測定用に使った lha-unix の `-o1` / `-o2` は圧縮方式ではなく header level の指定で、生成された 4 書庫の `kaito list` はいずれも `-lh1-` だった。

したがって、これらの書庫を使った emit と LZS 専用 bit cursor の速度結果は、LArc の採用・不採用を判断する証拠から除外する。ファイル名に lzs / lz5 と付いた旧測定ログも `-lh1-` の control として扱い、正しい方式を自作の test encoder で生成して測り直す。旧ログは `Results/kaito-lh1-emit-control-{max,mini}-20261105.txt` と `Results/kaito-lh1-bit-cursor-control-{max,mini}-20261105.txt` に残す。実装元が `-lh1-` であったことの確認には `kaito list` と独立 lha-unix を使った。

### 10.15 LZS 専用 bit cursor（採用）

`Scripts/make_larc_corpus.py` は、実入力から hash table で一致を探索し、lzs / lz5 の token と level-0 LHA header、CRC16-ARC を生成する。text は 4 MiB、color は 3 枚の BMP の先頭 4 MiB ずつ（合計 12,582,912 byte）を使った。`kaito list` で実方式が `-lzs-` / `-lz5-` であること、KaitoKit の SHA-256、独立 lha-unix の全 entry CRC を確認した。生成と検証は `Results/kaito-larc-corpus-20261105.txt` に記録した。

lzs 専用の `LArcLZSBitCursor` は、owner が保証する raw pointer、左詰めの UInt64 reservoir、利用可能 bit 数だけを保持する。1 / 4 / 8 / 11 bit の読み出しで、共有 `MSBFirstBitReader` の Array fallback、bit 数の動的検査、peek と consume の outline 呼び出しを省く。実入力を超える場合は従来どおり overrun とし、呼び出し側が `KaitoError.truncated` を返す。共有 reader と lz5 の bit 解釈は変更していない。

| archive | mini 変更前 | mini cursor | Max 変更前 | Max cursor |
|---|---:|---:|---:|---:|
| text-lzs | 23.3630 ms | 18.3435 ms | 24.5905 ms | 18.8905 ms |
| color-lzs | 165.1490 ms | 90.3975 ms | 169.5880 ms | 90.3320 ms |

text は 100 展開、color は 10 展開を 1 sample とし、4 回ずつ交互に実行した中央値。文書は mini 21.5%、Max 23.2%、画像は mini 45.3%、Max 46.7% の時間短縮。画像の速度比はそれぞれ約 1.83 倍、1.88 倍である（`Results/kaito-larc-candidates-{mini,max}-20261105.txt`）。

10,000 token の決定的な長尺 stream、辞書 wrap、距離 1・2,048、buffer size 1 / 7 / 4,096 / 65,536、途中までの読み出し、先頭・等間隔・末尾を含む切断を追加した。この test は元の実装でも成功を確認してから candidate に使い、両機の 8 test が成功した。raw log は `Results/kaito-larc-long-vector-baseline-{mini,max}-20261105.txt` と `Results/kaito-lh1-bit-cursor-control-{mini,max}-20261105.txt`（test は有効、誤った `-lh1-` benchmark は除外）にある。統合後の全 test は後述の最終検証で確認する。

### 10.16 LArc emit の形式別適用と beginMatch（lz5 限定のみ採用）

正しい corpus で共通 `emit` の inline + wrapping add を再測定した。lzs は text が mini 0.6%、Max 2.3% 低速、color は mini 4.3%、Max 3.7% 低速だった。一方 lz5 は text がほぼ同じで、color は mini 84.3640 → 78.6680 ms、Max 87.6770 → 81.4915 ms と約 7% 改善した。したがって lzs には元の outline emit を残し、lz5 の呼び出しだけを別の `emitLZ5` にした。guard で `produced < expectedSize` が保証されるため、この加算だけは `&+=` でオーバーフローしない。

lz5 限定版の再測定でも、color は mini 85.0075 → 78.6620 ms、Max 87.6605 → 81.4960 ms と約 7% 改善した。text は mini 16.7665 → 16.6795 ms、Max 17.3270 → 17.3805 ms で、差は小さい。本番の LZS cursor との統合後も M4 Max で text-lzs 24.6050 → 18.8675 ms、color-lzs 167.8565 → 90.4005 ms、color-lz5 87.2795 → 80.8800 ms を確認した（`Results/kaito-larc-candidates-{mini,max}-20261105.txt`、`Results/kaito-larc-specialized-{mini,max}-20261105.txt`、`Results/kaito-larc-final-max-20261105.txt`）。

`beginMatch` の inline は単独では text-lzs / text-lz5 を約 2〜3% 改善したが、color-lz5 は両機で後退した。さらに LZS cursor との組み合わせは、text-lzs が両機でほぼ同じ、color-lzs が mini 89.4915 → 109.3225 ms、Max 90.6830 → 114.1600 ms と明確に後退したため採用しない。inline の単独結果を組み合わせても効果が直和にならないことを確認した（`Results/kaito-larc-match-{mini,max}-20261105.txt`）。

統合した本番 source の 8 test は M4 Max、ASan の 8 test は Mac mini で成功した。Mac mini の全 2,100 test（68 skip、0 failure、1,404.471 秒）と Compat の 34 test（0 failure）も成功した（`Results/kaito-larc-final-full-mini-20261105.txt`）。初回の検証用コピーに CHANGELOG / Documentation / Scripts / fixtures がなく環境起因の failure が出たため、完全な tracked source に現行の Sources / Tests を重ねて再実行した。初回の記録は `Results/kaito-larc-incomplete-tree-full-mini-20261105.txt` に残し、最終結果とは区別する。

### 10.17 Bzip2 block scanner の適応型 run 走査（採用）

`ParallelBzip2StreamEncoder` が libbz2 へ渡す block を決める `Bzip2BlockScanner` を対象にした。無条件に 8 byte 比較を挿入した版は、4 MiB の一定長 run で run 長 8 が約 29%、9 が約 14% 低速だった。32 bit の事前照合でも短い run の後退を十分に解消できなかったため、これらは採用していない（`Results/gyoshuku-bzip-lengths8-max-20261105.txt`、`Results/gyoshuku-bzip-lengths-preflight-max-20261105.txt`）。

最終版は、各 scan の先頭 64 byte 以内に 16 byte 以上の run があるときだけ専用ループを選ぶ。通常入力は元の scalar ループ、長い run では 8 byte の未整列 load と trailing-zero count で同じ文字を進める。`length` は 255 で必ず flush し、count・block 境界・pending run・途中で返す位置は変更しない。scalar / wide / detector を分離し、通常のループの肥大化を避けた。短い run の独立測定では、無条件版の後退を避けられた（`Results/gyoshuku-bzip-lengths-adaptive-max-20261105.txt`）。

スキャン単体では run 長 127 が約 34%、同じ byte の連続が約 30% 短縮された。text の scanner は約 2.6% 増えたが、圧縮全体では次のとおり text がほぼ同じで、run が多い入力だけを短縮した。M4 Max で 4 回ずつ交互に実行した中央値（ms）。

| input / level | 変更前 | 適応型 |
|---|---:|---:|
| repeated / 1 | 25.572 | 24.807 |
| repeated / 9 | 25.492 | 24.784 |
| runs127 / 1 | 31.416 | 30.541 |
| runs127 / 9 | 33.812 | 32.884 |
| text / 1 | 179.257 | 179.490 |
| text / 9 | 194.408 | 194.317 |

出力は逐次 `Bzip2StreamEncoder`（system libbz2）と全 byte 一致。`GYOSHUKU_LARGE_ENCODER_TESTS=1` を含む 15 test（0 skip、41.869 秒）が研究版と本番の両方で成功した。測定は `Results/gyoshuku-bzip-adaptive-max-20261105.txt`、本番 test は `Results/gyoshuku-bzip-production-max-20261105.txt` に記録した。

Mac mini でも、repeated は level 1 が 25.288 → 24.272 ms、level 9 が 25.344 → 24.089 ms、runs127 は level 1 が 30.987 → 29.796 ms、level 9 が 33.466 → 32.008 ms と、約 3.8〜5.0% 短縮した。text の差は +0.1% 前後で、両機の全 15 test も成功した（`Results/gyoshuku-bzip-candidate-mini-20261105.txt`）。M4 Max の ASan でも全 15 test、0 finding（`Results/gyoshuku-bzip-asan-max-20261105.txt`）。Mac mini の GyoshukuKit 全 948 test（55 skip、0 failure、3,285.640 秒）も成功した（`Results/gyoshuku-bzip-final-full-mini-20261105.txt`）。

### 10.18 LZMA BT の子の先読み削除（不採用）

実 PID を使う 10 秒の CPU sampling で、`parseNormal` と `treeMatchesPair` が主な時間だった（`Results/gyoshuku-lzma-sample-max-20261105.txt`）。pair walk の各候補について、子の距離と循環 index を計算し、両方の子を先読みする部分を削除した。hash の更新順、木の更新、候補列、既存の hash 先読みは変更していない。

paired-finder の 4 test と、出力が 694,823 / 4,614,086 byte で一致する encoder benchmark は成功した。しかし M4 Max の 4 回ずつの交互測定では、text が 1.053608 → 1.077889 秒（約 2.3% 低速）、Mach-O が 3.481163 → 3.476764 秒（約 0.1% 高速）で、text の後退を正当化できなかった。子の先読みを維持し、本番には入れない（`Results/gyoshuku-lzma-no-child-prefetch-max-20261105.txt`）。

### 10.19 Zstandard row hash の再利用（不採用）

工程別の `ZSTD-PROFILE` では、level 9 の match+parse が text 0.124038 秒、Mach-O 0.639532 秒、mixed 0.585638 秒で、他工程より支配的だった。level 19 でも同様に match 探索が大部分を占める（`Results/gyoshuku-zstd-profile-max-20261105.txt`）。

`ZstdMatchFinder.hash` は既に row 用の 32 bit 値を計算しているが、`shortRowMatch` と `insertShortRow` が同じ load・乗算をやり直していた。hash の戻り値にその UInt32 を加え、検索と挿入へ渡す候補を実装した。hash の値、row / tag / cursor の選択、候補の順序、辞書更新は変えない。fast・doubleHash では返すだけで使用しないため、これらの経路への影響も測定する。

初回は乗算後の hash ではなく生の 32 bit 入力を渡し、mixed-9 の圧縮結果が変わったため A/B を中止した。`v &* 0x9E3779B1` を渡すように修正し、関連 39 test（7 skip、0 failure）を再確認した。最終候補は 3 corpus・4 level の交互測定と、生成した 12 個の `.zst` の byte 比較を行った。

乗算後の値を返り値に加えた版は、mixed-9 が 0.585977 → 0.570318 秒と約 2.7% 改善した一方、text-9 が 0.112857 → 0.114681 秒、text-19 が 1.223225 → 1.239165 秒と後退した（`Results/gyoshuku-zstd-hash-reuse-fixed-max-20261105.txt`）。そこで、返り値を増やさず row 内で一度だけ乗算する版に変更した。この版は mixed-9 が 0.578410 → 0.574969 秒（約 0.6%）にとどまり、binary-9 が 0.643803 → 0.646920 秒（約 0.5%）、text-3 が約 1.2% 後退した。text-9 は改善したが、元版の 2 sample が 0.114144 / 0.121828 秒と変動し、他の条件を一貫して上回るものではなかった（`Results/gyoshuku-zstd-hash-reuse-local-max-20261105.txt`）。

両版とも関連 39 test、外部 zstd / KaitoKit の復号、12 個の `.zst` の byte 一致を確認したが、速度の一貫した優位を示せなかったため、本番の `ZstdMatchFinder` は変更していない。

## 11. 最終検証と記録の保存（2026-11-05）

- Mac mini M4 で最新 Lab の `swift test -j 4` は **62 tests、2 skipped、0 failures**（40.021 秒）、`swift build -c release -j 4` は 98.59 秒で成功した（`Results/accel-lab-release-mini-20261105.txt`）。2 skip は、再現用 copy に生成 model を持ち込まなかった stateful / int8 の検査。同じ model 経路を含む 62 tests（0 skip）の成功は `Results/accel-lab-latest-mini-20261104.txt` にあり、CPU / GPU / ANE の予測一致と実往復の log も保存している。
- `Research/LZMABucketMetal/run-tests.sh` を Mac mini で実行し、**7 tests、1 skipped、0 failures**（0.327 秒）を確認した。1 skip は opt-in benchmark。本番 checkout は変更せず、Metal link 付きの一時 copy で CPU 参照との比較を再現できる（`Results/lzma-bucket-metal-release-mini-20261105.txt`）。
- 初回の composite log は、Lab の全 test と release build の完了後、後段の研究 build で実行期限に達した。これは失敗した codec の結果ではない。研究側は再現 script で完結した別 log に記録する。
- KaitoKit の最終全 test は **2,100 tests、68 skipped、0 failures**、Compat は **34 tests、0 failures**（§10.16）。GyoshukuKit は **948 tests、55 skipped、0 failures**（§10.17）。
- 採用したのは両機で正しさと速度を確認した CPU 経路だけ。GPU / NPU 経路、研究用 Metal code、却下した CPU 候補は本番へ持ち込まず、この repo の `Sources/`・`Research/`・`Scripts/`・`Results/` に保存する。未測定の電力効果と SME を実測済みと見なさない。
