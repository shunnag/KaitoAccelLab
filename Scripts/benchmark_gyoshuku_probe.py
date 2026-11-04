#!/usr/bin/env python3
"""Compare tabular GyoshukuKit probes using alternating bundles or executables."""

import argparse
import os
import statistics
import subprocess


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline")
parser.add_argument("candidate")
parser.add_argument("test")
parser.add_argument("gate")
parser.add_argument("marker")
parser.add_argument("--rounds", type=int, default=8)
parser.add_argument("--direct", action="store_true", help="run executables directly; test is ignored")
parser.add_argument("--time-columns", type=int, choices=(2, 4), default=4)
args = parser.parse_args()
if args.rounds <= 0:
    parser.error("--rounds must be positive")
executables = {"baseline": args.baseline, "candidate": args.candidate}


def measure(name):
    environment = os.environ.copy()
    environment[args.gate] = "1"
    command = [executables[name]] if args.direct else [
        "xcrun", "xctest", "-XCTest", args.test, executables[name]
    ]
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(result.stdout)
    values = {}
    for line in result.stdout.splitlines():
        columns = line.split("\t")
        if len(columns) < args.time_columns + 2 or columns[0] != args.marker:
            continue
        if columns[-1] in ("median_MBps", "seconds"):
            continue
        key = tuple(columns[:-args.time_columns])
        if args.time_columns == 2:
            seconds = float(columns[-1])
            values[key] = (seconds, seconds)
        else:
            values[key] = (float(columns[-4]), float(columns[-2]))
    if not values:
        raise RuntimeError(f"no {args.marker} rows:\n{result.stdout}")
    return values


times = {}
conditions = None
order = ["baseline", "candidate", "candidate", "baseline"]
for index in range(args.rounds):
    name = order[index % len(order)]
    result = measure(name)
    if conditions is None:
        conditions = set(result)
    if set(result) != conditions:
        raise RuntimeError(f"probe conditions differ: {name}")
    for key, value in result.items():
        times.setdefault((name, key), []).append(value)

print("condition\tname\tbest_s\tmedian_s\trounds\tbest_samples_s\tmedian_samples_s")
for key in sorted(conditions):
    for name in ("baseline", "candidate"):
        samples = times[name, key]
        best = statistics.median(value[0] for value in samples)
        median = statistics.median(value[1] for value in samples)
        raw_best = ",".join(f"{value[0]:.9f}" for value in samples)
        raw_median = ",".join(f"{value[1]:.9f}" for value in samples)
        condition = "\t".join(key)
        print(f"{condition}\t{name}\t{best:.9f}\t{median:.9f}\t{len(samples)}\t{raw_best}\t{raw_median}")
