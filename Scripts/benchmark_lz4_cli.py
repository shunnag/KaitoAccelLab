#!/usr/bin/env python3
"""二つの kaito CLI を同じ frame で交互に測定する。"""

import argparse
from pathlib import Path
import statistics
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("frame", type=Path)
parser.add_argument("--bytes", type=int, required=True)
parser.add_argument("--rounds", type=int, default=6)
parser.add_argument("--threads", default="auto")
args = parser.parse_args()
if args.bytes <= 0 or args.rounds <= 0:
    parser.error("--bytes and --rounds must be positive")

executables = {"baseline": args.baseline, "candidate": args.candidate}
for name, executable in executables.items():
    result = subprocess.run(
        [str(executable), "sha", "--sink", "--threads", args.threads, str(args.frame)],
        stdout=subprocess.DEVNULL, check=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed")

order = ["baseline", "candidate", "candidate", "baseline"]
times = {name: [] for name in executables}
for index in range(args.rounds):
    name = order[index % len(order)]
    start = time.perf_counter()
    subprocess.run(
        [str(executables[name]), "sha", "--sink", "--threads", args.threads, str(args.frame)],
        stdout=subprocess.DEVNULL, check=True,
    )
    times[name].append(time.perf_counter() - start)

print("name\tmedian_s\tGB_per_s\trounds\tseconds")
for name, samples in times.items():
    median = statistics.median(samples)
    raw = ",".join(f"{sample:.9f}" for sample in samples)
    print(f"{name}\t{median:.9f}\t{args.bytes / median / 1e9:.6f}\t{len(samples)}\t{raw}")
