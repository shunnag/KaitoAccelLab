# 速度レポート: M4 Max の GPU / NPU と書庫の圧縮・展開（2026-09-29）

計測機: Apple M4 Max（CPU 16 core = P 12 + E 4、GPU 40 core、ANE 16 core、128 GB）、macOS 27.2、Xcode 27.0、Swift 6.2。
すべて release build、同じ入力、中央値。他の作業が走っていない状態で取り直した値を「idle」と書き、生の数値は `Results/` の file を引く。
判定規則（plan.md）: 「動いた」= CPU 経路と byte 同一、「速い」= 多 core の CPU 経路の中央値に noise を超えて勝つ。

## 1. 結論（先に）

| 問い | 答え |
|---|---|
| 現在 NPU / GPU は圧縮・展開に寄与しているか | していない。本体も Apple の圧縮 / 暗号 library も CPU だけで動く（§2）。 |
| GPU で圧縮・展開の一部が動くか | 動く。独立 block の LZ4 復号を Metal kernel で書き、256 MiB を byte 同一に復号した（§4.2）。histogram、CRC-32 も動く（§4.1）。 |
| GPU は CPU より速いか | 速くない。LZ4 復号は最良でも CPU 16 lane の 0.5〜0.6 倍。histogram だけ 3〜8 倍速いが用途がない。CRC-32 は同程度で往復の費用分だけ負ける。 |
| NPU で圧縮・展開の一部が動くか | 動く。GRU の byte 予測器を Core ML で ANE に載せ（MLComputePlan で全 18 op が ANE）、range coder と組み合わせた lossless codec で符号化・復号ともに ANE で往復した（§5）。 |
| NPU は CPU より速いか | 同じ model を CPU で回すより 1.4〜1.6 倍速いが、既存の CPU codec（xz 322 MB/s、PPMd 9〜14 MB/s）には 2 桁遅い（0.24〜0.53 MB/s）。ただし比は学習 domain では PPMd を上回る（0.279 対 0.289）。 |
| 本体に合流させるか | しない。CPU より速い経路がない。条件が揃えば GPU が勝ちうる形（§4.4）と、NPU codec が意味を持つ条件（§5.5）を書き残す。 |

## 2. 現状: NPU / GPU は圧縮・展開に寄与しているか

寄与していない。根拠:
- KaitoKit / GyoshukuKit / KaitoFinder は Metal、Core ML、Accelerate を link しない。使う codec は自前 Swift、zlib、libbz2、Apple Compression、CommonCrypto。
- Apple Compression（`libcompression.dylib`）は liblzma と libSystem 以外を link せず、LZFSE / LZ4 / zlib / LZMA の実行中に IOKit / Metal / ANE の image は読み込まれない。圧縮の IOService も存在しない（Documentation/research/2026-09-29-research.json、apple-platform）。
- CommonCrypto の AES / SHA は CPU の暗号命令、zlib の CRC-32 は ARM の CRC32 命令で動く（1 core 43.7 GB/s、§3）。
- Metal 4 の MTLIO（`MTLIOCompressionContext` で圧縮した file を `MTLIOCommandQueue` で読み込む）は Apple が GPU 資産のために用意した唯一の「圧縮付き読み込み」だが、256 MiB の展開は LZ4 3.4 GB/s、LZBitmap 4.6 GB/s、LZFSE 1.25 GB/s、zlib 0.41 GB/s（Results/mtlio-idle-text256-20260929-1743.tsv）。libcompression の 1 core（LZ4 3.5〜4.7 GB/s、LZFSE 1.4 GB/s）と同じ水準で、16 lane の 41 GB/s には遠い。GPU が展開している痕跡はない。
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

読み: histogram は GPU が 3〜8 倍速い唯一の byte kernel だが、Deflate / LH5 が数えるのは LZ77 parse 後の記号なので生 byte の histogram
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
  途中の計測で GPU が 4 KiB block で CPU を上回ったように見えたのは、CPU 側の lock 付き動的分割の崩れと、GPU の clock が別の負荷で上がっていた
  ためで、静的分割の libcompression と liblz4 に対しては負ける。GPU の値は clock 状態で 24〜38 GB/s の幅があった（idle は低い側）。
- GPU 経路は block ごとの展開後 size を事前に知る必要がある。LZ4 frame にはその表がなく、sequence を舐める scan（直列 58〜97 ms、16 lane で 7〜9 ms）
  が復号本体（6〜12 ms）と同じ桁でかかる。CPU 経路にはこの費用がない。`--assume-uniform` で scan を省いても GPU 側の順位は変わらない。
- SIMD-per-block は 64 KiB block で thread-per-block の 1.9 倍だが、小さい block では barrier の費用が勝って逆転する。
- 技術デモ（§6）の 7.5 MB の tar では GPU 復号 22〜28 ms（Metal library の runtime compile 込み）、CPU 2 ms。小さい書庫では GPU の固定費だけで負ける。

### 4.3 見送った候補と理由（研究 JSON）

- Deflate / LZMA / bzip2 / PPMd / RAR / LHA の stream 内の復号: Huffman・range coder・BWT 逆変換は 1 stream の中で逐次。先行研究（nvCOMP、GDeflate、dietgpu）
  は format を GPU 向けに変えており、既存の書庫には当てはまらない。1 thread が 1/350 では stream 数が数万ないと成立しない。
- 圧縮側の match 探索: 本体の自前 encoder は LH5 だけで、zlib / libbz2 / Apple Compression / liblzma は黒箱。LH5 は 8 KiB window の古い format で、
  GPU に載せる価値がない。
- AES / SHA / HMAC: CPU の暗号命令が 1 core で 14〜18 GB/s（AES）、書庫を縛るのは直列の HMAC-SHA1（3.1 GB/s）と CBC 暗号化（1.8 GB/s）で、いずれも
  chain 依存で並列化できない。
- zstd の block 並列: zstd の block は frame の window を共有し独立に復号できない。GyoshukuKit の 7z LZMA2 chunk は 16 MiB で数が少ない。

### 4.4 GPU が勝ちうる条件（本体には当てはまらない）

数万個の独立 block（4〜16 KiB）、container が block ごとの展開後 size を持つ、出力を GPU 側で使う（texture、GPU 上の後続処理）、入力が既に GPU memory にある。
これは nvCOMP / GDeflate が format を作り直した理由そのもので、ZIP / 7z / RAR / tar.xz の既存書庫には一つも当てはまらない。

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
  混合 corpus（単語 + 英語散文 + Swift）で学習した model の結果は §5.6。

### 5.4 速度（同 file、4 MiB、idle、`--units` で compute unit を切り替え、model と batch は同一）

| model | block 数 | ANE | Core ML CPU | Core ML GPU | 1 step の予測器 / 符号器 ms（ANE） |
|---|---|---|---|---|---|
| GRU h1024 | 1024 | 393 KB/s | 272 KB/s | 434 KB/s | 2.47 / 0.13 |
| GRU h1024 | 4096 | **531 KB/s** | 370 KB/s | **652 KB/s** | 7.35 / 0.37 |
| GRU h1536 | 1024 | 244 KB/s | 162 KB/s | 259 KB/s | 4.05 / 0.14 |
| GRU h1536 | 4096 | 301 KB/s | 190 KB/s | 350 KB/s | 13.2 / 0.38 |

- 復号は符号化と同じ速度（h1536 b1024 ANE: 240 KB/s、`cmp` 一致）。
- ANE は同じ model の Core ML CPU 実行の **1.4〜1.6 倍**。しかし Core ML の GPU 実行（40 core、fp16）がさらに 1.1〜1.2 倍速く、この model では **ANE は最速の unit ではない**。
  1 step あたり ANE 7.35 ms で 4096 × 4.6 M MAC = 19 GMAC → 2.6 TMAC/s。ANE の公称値の 1/6 程度で、状態テンソルの往復（[4096, 1536] fp16 の出入り）と
  Core ML の呼び出しが効いている。MLState で状態を ANE 側に置く版は試していない（§7）。
- 時間の 95% が予測器（Core ML 呼び出し）で、CPU の range coder は 16 lane 並列で 0.13〜0.38 ms / step。
- 既存の CPU codec との比較: xz -d 322 MB/s、KaitoKit の PPMd 9〜14 MB/s、bzip2 78 MB/s。神経 codec は最速でも 0.65 MB/s で、**2 桁遅い**。
- 最初の版は量子化の余剰調整 loop が 1 step 43 ms を使い 23 KB/s だった（Results/ane-roundtrip-v1）。修正後は Core ML が律速。

### 5.5 約束事: 符号化と復号は同じ unit で

ANE で符号化した bit 列を同じ model・同じ batch の Core ML CPU 実行で復号すると `Invalid range-coded payload` で失敗する
（Results/ane-encoded-cpu-decode-mismatch-20260929.txt）。fp16 の丸めが device で異なり、量子化後の頻度表が 1 でも違えば range coder は破綻する。
NBC1 の header に predictor tag（model 名と unit）を入れ、復号側で照合して警告する。同じ unit 同士なら 4 MiB × 複数 model で全て一致した。

### 5.6 混合 corpus の model

（学習完了後に追記）

## 6. 技術デモ（`accel-lab demo-pack / demo-unpack`、Results/demo-idle-20260929-1743.txt）

directory → tar（GyoshukuKit `ArchiveWriter`）→ 神経 block codec（ANE）か LZ4 frame → KADM container。展開は逆順で、tar は KaitoKit `ArchiveReader` が読む。
入力: 単語 text 512 KiB × 8（うち 4 つは sub/）+ War and Peace = 10 entries、tar 7,567,360 bytes。

| 経路 | 書庫 bytes | 比 | 圧縮 s | 展開 s | 木の一致 |
|---|---|---|---|---|---|
| 神経 codec GRU h1536、1024 block、ANE | 3,321,246 | 0.439 | 35.6（tar 0.005） | 35.6（extract 0.004） | `diff -r` 一致 |
| LZ4 4 KiB block、GPU thread-per-block | 6,285,981 | 0.831 | 0.011 | 0.082（初回の Metal compile 込み） | 一致 |
| LZ4 4 KiB block、GPU SIMD-per-block | 同上 | | | 0.028 | 一致 |
| LZ4 4 KiB block、CPU 16 lane | 同上 | | | 0.002 | 一致 |

## 7. 試さなかったこと・残る手

- MLState で GRU の状態を ANE 側に保持する版（状態テンソルの往復を省く）。batch 8192 の model。2 層 GRU / RWKV cell。
- 学習中の適応（online learning）。NNCP / cmix が比で勝つ理由はこれで、静的 model + 独立 block では domain 外に弱い。
- GPU LZ4 kernel の copy 幅の拡張（4 byte / 16 byte copy）。thread-per-block の 1 thread あたり 10 MB/s は byte 単位 copy の上限に近い。
- `powermetrics` による ANE / GPU の電力測定（sudo が要る）。効率（J/byte）は未計測。
- SME（CPU 側の行列命令、4.0 int8 TOPS / thread）で同じ GRU を回す比較。Core ML の CPU 経路（BNNS）がそれを既に使っているかは未確認。
- CPU 側の改善余地（NPU / GPU ではない）: xz の block 並列復号（`xz -lvv` で perf corpus の text.tar.xz は 11 block）、libcompression の呼び出し単位を 16 KiB 以上に保つこと。
