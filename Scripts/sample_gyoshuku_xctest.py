#!/usr/bin/env python3
"""Sample the actual process running a GyoshukuKit XCTest benchmark."""

import argparse
from pathlib import Path
import subprocess
import time


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bundle")
parser.add_argument("test")
parser.add_argument("marker")
parser.add_argument("seconds")
parser.add_argument("log")
parser.add_argument("sample_output")
args = parser.parse_args()
log = Path(args.log)
with log.open("w") as output:
    process = subprocess.Popen(
        ["xcrun", "xctest", "-XCTest", args.test, args.bundle],
        stdout=output,
        stderr=subprocess.STDOUT,
    )
    pid = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and process.poll() is None:
        for line in log.read_text().splitlines():
            if args.marker in line:
                pid = int(line.split(args.marker, 1)[1].split(None, 1)[0])
                break
        if pid is not None:
            break
        time.sleep(0.05)
    if pid is None:
        status = process.wait()
        raise RuntimeError(f"benchmark PID was not reported (status {status}); see {log}")
    sampled = subprocess.run(["sample", str(pid), args.seconds, "-file", args.sample_output])
    status = process.wait()
    if sampled.returncode:
        raise RuntimeError(f"sample failed with status {sampled.returncode}")
    if not Path(args.sample_output).is_file():
        raise RuntimeError(f"sample did not create {args.sample_output}")
    if status:
        raise RuntimeError(f"benchmark failed with status {status}; see {log}")
