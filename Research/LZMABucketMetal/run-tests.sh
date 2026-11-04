#!/bin/bash
set -euo pipefail

research=$(cd -- "$(dirname -- "$0")" && pwd)
parent=$(cd -- "$research/../../.." && pwd)
gyoshuku="$parent/GyoshukuKit"
kaito="$parent/KaitoKit"
test -f "$gyoshuku/Package.swift"
test -f "$kaito/Package.swift"

work=$(mktemp -d "${TMPDIR:-/tmp}/kaito-lzma-metal.XXXXXX")
trap 'rm -rf "$work"' EXIT
mkdir "$work/GyoshukuKit"
cp "$gyoshuku/Package.swift" "$work/GyoshukuKit/Package.swift"
cp -R "$gyoshuku/Sources" "$work/GyoshukuKit/Sources"
cp -R "$gyoshuku/Tests" "$work/GyoshukuKit/Tests"
cp "$research/LZMABucketMetalEncoder.swift" "$research/LZMABucketMetalKernels.swift" \
    "$work/GyoshukuKit/Sources/GyoshukuKit/Compression/"
cp "$research/LZMABucketMetalTests.swift" "$work/GyoshukuKit/Tests/GyoshukuKitTests/Probes/"
ln -s "$kaito" "$work/KaitoKit"

python3 - "$work/GyoshukuKit/Package.swift" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
old = 'dependencies: ["CGyoshukuBzip2", .product(name: "KaitoKit", package: "KaitoKit")], linkerSettings: [.linkedLibrary("z")]'
new = 'dependencies: ["CGyoshukuBzip2", .product(name: "KaitoKit", package: "KaitoKit")], linkerSettings: [.linkedLibrary("z"), .linkedFramework("Metal")]'
assert source.count(old) == 1, "GyoshukuKit package layout differs from this research snapshot"
path.write_text(source.replace(old, new))
PY

cd "$work/GyoshukuKit"
swift test -c release -j 4 --filter LZMABucketMetalTests
