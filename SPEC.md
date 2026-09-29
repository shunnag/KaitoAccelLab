# accel-lab baseline: CPU bars for the acceleration study

Repository: KaitoAccelLab (SwiftPM, macOS 26+, Swift 6.2), a research lab that depends on the sibling packages ../KaitoKit and ../GyoshukuKit (do not modify them). Only pure-Swift code; no Metal or Core ML in this task.

## OBJECTIVE
Add a `baseline` subcommand to the `accel-lab` executable that measures, on the CPU, the throughput of the primitives that a GPU or NPU implementation would have to beat, and writes the results as a TSV file the speed report can cite.

## SCOPE
1. `Sources/AccelLab/Baseline.swift`: a `public enum Baseline` with `static func run(sizeMiB: Int = 256, rounds: Int = 5) throws -> [Measurement]` where `Measurement` has `name`, `bytes`, `medianSeconds`, `throughputGBps` (bytes / seconds / 1e9), `note`. Use `ContinuousClock` and the median of `rounds` runs after one warm-up; input is a deterministic pseudo-random buffer (xorshift seed) of `sizeMiB` and, for compression cases, a second input made from a repeated text corpus so ratios are meaningful (use `/private/tmp/claude-501/-Users-nagash-Github-KaitoFinder/404391ed-5c96-4f47-afa7-7911dab682d1/scratchpad/corpora/text256.txt` if it exists, else a generated English-like text).
   Measurements (single thread unless noted):
   - `crc32-zlib`: zlib `crc32()` over the random buffer (link zlib: `import zlib` works on macOS via the system module map; if not, declare the C function with `@_silgen_name`).
   - `crc32-table`: a plain Swift 8-bit-table CRC-32 (write it) to show the software bar.
   - `adler32-zlib`.
   - `aes-ctr-commoncrypto` and `aes-cbc-decrypt-commoncrypto`: CommonCrypto `CCCrypt` with a 256-bit key over the random buffer.
   - `sha256-commoncrypto`.
   - `deflate6-zlib-compress` and `inflate-zlib` on the text buffer (level 6), reporting ratio too.
   - `bzip2-compress` / `bzip2-decompress` via libbz2 (`import bz2` module or `@_silgen_name`), block size 900k.
   - `lzfse-compress/decompress`, `lz4-compress/decompress`, `lzma-compress/decompress` via Apple `Compression` framework `compression_encode_buffer` / `compression_decode_buffer`.
   - `parallel-crc32-zlib-16`: the random buffer split into 16 equal chunks, each `crc32()` on a `DispatchQueue.concurrentPerform` lane, then combined with `crc32_combine` (from zlib) — the multi-core CPU bar for CRC.
   - `memcpy` over the buffer as the memory-bandwidth reference.
2. `Sources/accel-lab/main.swift`: dispatch on `CommandLine.arguments[1]`: `probe` (existing Metal probe) and `baseline [--size MiB] [--rounds N] [--out path]`. `baseline` prints a table and writes `Results/baseline-<yyyyMMdd-HHmm>.tsv` (columns: name, bytes, median_s, GB_per_s, note) under the repository root unless `--out` is given.
3. `Tests/AccelLabTests/BaselineTests.swift`: one test that runs `Baseline.run(sizeMiB: 4, rounds: 1)` and asserts every measurement has positive throughput and that the round-trips (deflate/inflate, bzip2, lzfse, lz4, lzma) restore the input bytes.

## CONSTRAINTS
- Japanese comments, one main type per file, Swift 6 strict concurrency (use `nonisolated(unsafe)` only if unavoidable and say why).
- No allocation inside the timed region except what the library does.
- Do not commit. Do not edit Documentation/.

## ACCEPTANCE
- `swift build -c release` clean; `swift test` passes; `swift run -c release accel-lab baseline --size 64 --rounds 3` prints the table and writes the TSV.

## VERIFICATION
    swift build -c release
    swift test
    swift run -c release accel-lab baseline --size 64 --rounds 3
