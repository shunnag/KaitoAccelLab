#!/usr/bin/env python3
"""Compare in-process kaito extraction using alternating executables."""
import argparse
import statistics
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline")
parser.add_argument("candidate")
parser.add_argument("archive")
parser.add_argument("--reps", type=int, default=100)
parser.add_argument("--rounds", type=int, default=8)
parser.add_argument("--threads", default="1")
args = parser.parse_args()
if args.reps <= 0 or args.rounds <= 0:
    parser.error("--reps and --rounds must be positive")
executables = {"baseline": args.baseline, "candidate": args.candidate}

def extract(name):
    result = subprocess.run([executables[name], "bench", "--threads", args.threads,
                             args.archive, str(args.reps)], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)
    values = dict(line.split("\t") for line in result.stdout.splitlines())
    return float(values["extract-median-ms"]), int(values["bytes"])

expected = {}
for name in executables:
    _, expected[name] = extract(name)
if len(set(expected.values())) != 1:
    raise RuntimeError(f"decoded byte counts differ: {expected}")
times = {name: [] for name in executables}
order = ["baseline", "candidate", "candidate", "baseline"]
for index in range(args.rounds):
    name = order[index % len(order)]
    duration, count = extract(name)
    if count != expected[name]:
        raise RuntimeError(f"decoded byte count changed: {name}: {count}")
    times[name].append(duration)
print("archive\tname\tmedian_ms\treps\tbytes\trounds\tsamples_ms")
for name, samples in times.items():
    raw = ",".join(f"{sample:.6f}" for sample in samples)
    print(f"{args.archive}\t{name}\t{statistics.median(samples):.6f}\t{args.reps}\t{expected[name]}\t{len(samples)}\t{raw}")
