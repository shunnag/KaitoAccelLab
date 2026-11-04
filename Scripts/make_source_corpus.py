#!/usr/bin/env python3
"""tracked な Swift source を連結し、指定した MiB まで繰り返す。"""

import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("size_mib", type=int)
parser.add_argument("roots", nargs="+", type=Path)
args = parser.parse_args()
if args.size_mib <= 0:
    parser.error("size_mib must be positive")

parts = []
files = 0
for root in args.roots:
    names = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.swift"]
    ).split(b"\0")
    for name in names:
        if name:
            parts.append((root / name.decode()).read_bytes())
            files += 1
if not parts:
    parser.error("no tracked Swift source files")

source = b"\n".join(parts)
count = args.size_mib * 1_048_576
payload = (source * ((count + len(source) - 1) // len(source)))[:count]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_bytes(payload)
print(f"source_bytes: {len(source)}, source_files: {files}, output_bytes: {len(payload)}")
