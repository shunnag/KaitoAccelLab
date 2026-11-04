# LZMA BT4 の bucket 並列 Metal 研究

GyoshukuKit の `LZMAMatchFinder` と同じ HASH4 bucket ごとに位置を集め、一つの Metal thread が一つの BT4 tree を位置順に処理する実験です。dictionary の周期ごとに wave を分け、現行・前周期の 2 面の tree buffer を使います。

結果は M4 Max の finder 比 1.118 倍、Mac mini M4 で 0.655 倍でした。M4 Max の限定的な改善を採用せず、Metal link と研究 file は GyoshukuKit から外しています。bucket 作成・転送を含み、parser / range encoder を含まない finder の比較なので、圧縮全体の倍率ではありません。詳細は `Documentation/speed-report.md` §9.1、生の値は `Results/gyoshuku-lzma-bucket-metal-20261104.txt` を参照してください。

## 再現

`KaitoKit`、`GyoshukuKit`、`KaitoAccelLab` を同じ親 directory に置きます。研究 code は当時の GyoshukuKit の internal API を使うため、本番 checkout へ直接コピーせず、次の script で一時コピーを作って検査します。

```console
Research/LZMABucketMetal/run-tests.sh
```

script は `Package.swift` に一時的な Metal link だけを足し、CPU 参照実装との全位置比較・dictionary wrap・途中の tree 状態・短入力を検査します。opt-in の大入力 benchmark は skip されます。作業 copy と build 物は終了時に削除します。

```console
GYOSHUKU_LZMA_GPU_BENCHMARK=1 swift test -c release --filter LZMABucketMetalTests
```

上記は script が作った一時 copy 内で benchmark を有効にする形式です。研究の評価には M4 Max と Mac mini M4 の交互測定を使い、GPU 単独の時間だけでなく転送込みの wall を比較してください。
