#!/usr/bin/env python3
"""GRU model の重みを int8 / int4 の線形対称量子化で圧縮する。"""

import argparse
from pathlib import Path
import coremltools as ct
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--dtype", choices=["int8", "int4"], default="int8")
args = parser.parse_args()
if args.input == args.output:
    parser.error("input and output must differ")

model = ct.models.MLModel(str(args.input), compute_units=ct.ComputeUnit.CPU_AND_NE)
config = ct.optimize.coreml.OptimizationConfig(
    global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype=np.int8 if args.dtype == "int8" else "int4",
    )
)
compressed = ct.optimize.coreml.linear_quantize_weights(model, config)
compressed.save(str(args.output))


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

print(f"input_bytes: {directory_size(args.input)}")
print(f"output_bytes: {directory_size(args.output)}")
print(f"saved: {args.output}")
