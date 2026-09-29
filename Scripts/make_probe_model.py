"""ANE 配置の probe 用に、小さな byte 予測 model を coremltools の MIL builder で作る（PyTorch 不要）。
入力: [batch, context] の byte 値（fp16 に正規化済み）。出力: 256 通りの確率（softmax）。
層: embedding 相当の行列積 → GELU → 行列積 → softmax。重みは乱数（配置と latency の確認だけが目的）。
"""
import sys, numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
context = int(sys.argv[2]) if len(sys.argv) > 2 else 16
hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 256
layers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
out = sys.argv[5] if len(sys.argv) > 5 else f"Models/probe-b{batch}-c{context}-h{hidden}-l{layers}.mlpackage"
rng = np.random.default_rng(1)
w1 = rng.standard_normal((context, hidden)).astype(np.float16) * 0.05
ws = [rng.standard_normal((hidden, hidden)).astype(np.float16) * 0.05 for _ in range(layers)]
w3 = rng.standard_normal((hidden, 256)).astype(np.float16) * 0.05

@mb.program(input_specs=[mb.TensorSpec(shape=(batch, context), dtype=types.fp16)], opset_version=ct.target.macOS15)
def prog(x):
    h = mb.matmul(x=x, y=w1)
    h = mb.gelu(x=h)
    for w in ws:
        h = mb.matmul(x=h, y=w)
        h = mb.gelu(x=h)
    logits = mb.matmul(x=h, y=w3)
    return mb.softmax(x=logits, axis=-1, name="probabilities")

model = ct.convert(prog, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_AND_NE,
                   minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16)
model.save(out)
print("saved", out)
