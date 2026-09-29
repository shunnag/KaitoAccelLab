# 解説: GPU / NPU を書庫の圧縮・展開に使うということ（2026-09-29）

この文書は KaitoAccelLab で試した技術の解説である。数値は `speed-report.md` に置き、ここでは「なぜそうなるか」を書く。

## 1. 圧縮・展開は何をしている処理か

書庫の codec は大きく三つの段階からなる。

1. **モデル化**: 次に来る byte（または記号）の確率を見積もる。LZ 系は「過去に同じ並びがあったか」を辞書で探し（match 探索）、
   PPMd や神経圧縮は文脈から確率分布を直接出す。
2. **符号化**: 見積もった確率で記号を bit 列にする。Huffman、range coder、ANS。
3. **周辺**: CRC-32 / SHA、暗号（AES）、container の読み書き、memcpy。

展開はこの逆で、符号化の逆変換（復号）は原理的に **逐次** である。次の記号を復号するには前の記号までの状態（bit 位置、
range coder の区間、LZ の window）が必要で、1 本の stream の中では並列にできない。並列にできるのは
「独立した stream（block）」の単位だけである。これが GPU / NPU を使うときの最初の壁になる。

## 2. M4 Max の三つの計算機

| | CPU（P core） | GPU（40 core） | ANE（16 core） |
|---|---|---|---|
| 得意 | 分岐の多い逐次処理、表引き、bit 操作 | 数万 thread の同じ処理、帯域 | fp16 の行列積（静的な graph） |
| 逐次の 1 thread の速さ | 速い（LZ4 復号 3.9 GB/s） | 遅い（同じ処理で 1/60 程度） | 逐次処理は書けない |
| 呼び出しの費用 | なし | 0.13〜0.17 ms（dispatch 往復） | 0.07〜0.2 ms（推論 1 回） |
| 触り方 | Swift / C | Metal（MSL を runtime compile 可） | Core ML の model 実行のみ |
| 整数 / bit 操作 | 全部 | 32 bit atomic、bit 操作あり | なし（fp16 の演算 graph だけ） |

CPU の基準値が高いことが重要である。CRC-32 は ARM の CRC32 命令で 1 core 43 GB/s、AES は暗号命令で 14〜18 GB/s、
16 core を使えば CRC-32 は 224 GB/s に達する。GPU が 256 MiB を読むだけでも 300 GB/s 級で、しかも dispatch の往復が
0.15 ms かかる。byte 単位の「軽い」処理を GPU に送っても、勝てる余地はほとんどない。

## 3. GPU で試したこと

### 3.1 独立 block の並列復号（LZ4）

LZ4 frame を `-BI`（block independent）で作ると、各 64 KiB block は他の block を参照しない。block ごとに 1 thread
（あるいは 1 SIMD group）を割り当てれば、復号は「多数の逐次処理を同時に走らせる」形になる。GPU の 1 thread は
CPU の 1 core より遥かに遅いが、thread 数で補えるかどうかが問題になる。試した二つの kernel:

- thread-per-block: 1 thread が block を最後まで復号する。分岐と byte 単位の copy が多く、GPU の苦手な形。
- SIMD-per-block: 32 lane が同じ sequence header を読んで揃って進み、literal と match の copy を 32 byte ずつ行う。
  offset < 32 の重なった match は `min(offset, 32)` byte ずつ段階的に copy する。

結果（speed-report §4.2）: どちらも 256 MiB を byte 同一に復号した。GPU の 1 thread は CPU の 1 core の 1/350 で、64 KiB block × 4,096 では
thread が足りず 6 GB/s、4 KiB block × 65,536 で 24 GB/s。CPU は 16 lane で 41〜50 GB/s。さらに GPU だけが block ごとの展開後 size を事前に
必要とし、LZ4 frame にはその表がないので sequence を舐める scan がもう一度かかる。「独立 block を GPU で並列復号」は正しく動くが、
既存の書庫 format の block 数と粒度では CPU に届かない。

### 3.2 byte 単位の kernel（histogram、CRC-32）

- histogram: threadgroup memory の 256 個の 32 bit atomic に数え、最後に device atomic へ足す。GPU の帯域が生きる数少ない形。
  ただし Deflate / LH5 が数えるのは LZ77 parse 後の記号で、生 byte ではない。parse が GPU にない限り使い道がない。
- CRC-32: 4 KiB chunk ごとに slice-by-4 の表引きで計算し、CPU の `crc32_combine` で結合する。CPU の CRC32 命令
  （1 core 43 GB/s）に対し、GPU の表引きは帯域があっても命令数で負ける。

### 3.3 見送ったもの

- Deflate / LZMA / bzip2 の block 内の復号: Huffman・range coder・BWT 逆変換は 1 stream の中で逐次。先行研究（nvCOMP、GDeflate）
  は format 自体を GPU 向けに変えており、既存の書庫には当てはまらない。
- match 探索（圧縮側）: 本体で自前の encoder は LH5 だけで、zlib / libbz2 / Apple Compression / liblzma は黒箱。
- AES / SHA: CPU の暗号命令が 1 core で 14〜18 GB/s。GPU に送る帯域と往復の費用で負ける。

## 4. NPU で試したこと: 神経 block codec

### 4.1 仕組み

「予測 + 算術符号」は最も比の良い lossless 圧縮の形である（NNCP、cmix、"Language Modeling Is Compression"）。
model が次の byte の分布 p を出し、range coder が実際の byte x を −log2 p(x) bit で書く。復号側は同じ model を
同じ順に走らせ、同じ p を得て x を復元する。**符号化と復号で p が bit 単位で一致すること** が前提になる。

ANE で走らせる model は 1 回の推論に 0.1 ms 前後の固定費があるので、1 byte ごとに呼ぶと 10 KB/s にもならない。
そこで入力を N 個の独立 block に切り、位置 t の予測を全 block まとめて 1 回の推論にする（batch = N）。
range coder は block ごとに CPU で走る。推論 1 回あたり N byte 進むので、N = 1024〜4096 で MB/s 級になる。

### 4.2 model

- 再帰型（GRU 1 層、hidden 1024）。状態 h を MLState ではなく入出力のテンソルとして往復させる。native の `lstm` op は
  CPU に落ちるが、線形 + 活性で書いた cell は ANE に載る。batch 形状は固定（1024 と 4096 の二つの model を同じ重みから書き出す）。
- 入力は直前の byte の one-hot [N, 256]（embedding は one-hot と重みの行列積で行う。gather op の ANE 対応は試していない）、出力は softmax [N, 256] と h_out。

### 4.3 何が起きたか（speed-report §5）

- 学習 domain（辞書の単語を無作為に並べた合成 text、試験側の語彙は学習で全て見ている）では GRU h1536 が 2.23 bits/byte で PPMd（2.32）と xz（2.78）を上回った。model が「次に来る文字」の分布を
  文脈から直接出せる text では、辞書 match に頼る LZ 系より比が良い。これは神経圧縮の先行研究が示してきたことの再現である。
- domain 外（英語の散文、Swift の source）では PPMd / xz に大きく負け、source では膨らんだ。静的な model は学習した分布しか知らない。
  単語・散文・source を混ぜて学習し直すと膨らみは消え、散文では xz を上回るが PPMd には負け、source では xz にも届かない（speed-report §5.6）。
  NNCP や cmix が比で勝つのは符号化しながら model を更新する（online learning）からで、独立 block を batch にして NPU に載せる設計とは
  相性が悪い（block 間で学習を共有できない）。
- 速度は 0.24〜0.65 MB/s。時間の 95% が Core ML の呼び出しで、ANE は同じ model の CPU 実行より 1.4〜1.6 倍速いが、Core ML の GPU 実行が
  さらに 1.1〜1.2 倍速かった。この大きさの GRU では ANE は最速の unit ではなく、状態テンソルの往復と呼び出しの固定費が効いている。
- 1 step の予測に必要な演算は 4096 block × 6.8 M MAC（h1024）= 28 GMAC。既存の CPU codec（xz 322 MB/s、PPMd 9〜14 MB/s）が 1 byte に使う演算の
  数千倍で、「NPU が速い」以前に「予測に使う演算量が違う」。比の 4〜15% の改善に速度の 2 桁を払う取引になる。

### 4.4 約束事（demo の container header に書く）

fp16 の演算は device ごとに結果が僅かに違う（ANE と CPU で最大 3e-3 程度）。符号化と復号は **同じ model file、同じ compute unit、
同じ batch 形状** で走らせなければならない。header に predictor の tag（model 名と unit）を書き、復号側で照合する。

## 5. 結論

- **今日の M4 Max で、書庫の圧縮・展開に GPU / NPU は使われていない。** Apple の圧縮 library、暗号 library、MTLIO の展開まで含めて CPU で動く。
  CPU 側の基準値が高い（CRC-32 命令、AES 命令、16 core）ので、byte 単位の軽い処理を送る余地はない。
- **GPU で動くもの**: 独立 block の LZ 復号、histogram、CRC-32 の chunk 計算。正しく動くが、既存の format の block 粒度では CPU 16 lane に負ける。
  GPU が勝つには format 側の設計（数万の小 block、size 表、GPU 上で使う出力）が要る。これは既存の書庫を読む library の仕事ではない。
- **NPU で動くもの**: 予測 + 算術符号の予測部分。ANE で符号化・復号ともに往復し、学習 domain では PPMd を超える比を出した。
  速度は 2 桁遅く、domain 外に弱く、符号化と復号の unit を揃える約束が要る。「書庫の圧縮に NPU を使う」なら、対象 domain が決まっていて
  比が最優先で速度を捨てられる用途（特定 domain の log や text の長期保存）に限られる。
- **本体（KaitoKit / GyoshukuKit）には入れない。** どの経路も CPU より速くならなかった。lab の code と計測は KaitoAccelLab に残す。
