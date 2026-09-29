"""byte 予測 model（MLP、context 64 byte → 次の byte の分布）を torch(MPS) で学習し、Core ML(mlprogram, fp16) に変換する。
入力は [batch, context] の byte 値を 1/255 に正規化した fp16、出力は [batch, 256] の確率。batch は可変(1〜8192)。
使い方: train_byte_predictor.py <corpus> <out.mlpackage> [steps] [hidden] [layers] [context]
"""
import sys, time, numpy as np, torch, torch.nn as nn
import coremltools as ct

corpus_path, out = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
hidden = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
layers = int(sys.argv[5]) if len(sys.argv) > 5 else 4
context = int(sys.argv[6]) if len(sys.argv) > 6 else 64
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data = np.frombuffer(open(corpus_path, "rb").read(), dtype=np.uint8)
split = int(len(data) * 0.9)
train, valid = data[:split], data[split:]

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 位置ごとに byte を 16 次元へ埋め込む（正規化した値と one-hot 風の特徴の混合を線形で作る）
        self.inp = nn.Linear(context, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.GELU()) for _ in range(layers)])
        self.out = nn.Linear(hidden, 256)
    def forward(self, x):  # x: [batch, context] in [0,1]
        h = torch.nn.functional.gelu(self.inp(x))
        for b in self.blocks:
            h = h + b(h)
        return torch.softmax(self.out(h), dim=-1)

def batch_from(arr, size, rng):
    idx = rng.integers(0, len(arr) - context - 1, size)
    xs = np.stack([arr[i:i + context] for i in idx]).astype(np.float32) / 255.0
    ys = arr[idx + context].astype(np.int64)
    return torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)

model = Predictor().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
rng = np.random.default_rng(7)
t0 = time.time()
for step in range(steps):
    x, y = batch_from(train, 1024, rng)
    p = model(x)
    loss = torch.nn.functional.nll_loss(torch.log(p + 1e-9), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == steps - 1:
        with torch.no_grad():
            vx, vy = batch_from(valid, 4096, rng)
            vp = model(vx)
            bits = -torch.log2(vp[torch.arange(len(vy)), vy] + 1e-9).mean().item()
        print(f"step {step} loss {loss.item():.3f} valid bits/byte {bits:.3f} ({time.time() - t0:.0f}s)", flush=True)

model.eval().cpu()
example = torch.zeros(1024, context)
traced = torch.jit.trace(model, example)
mlmodel = ct.convert(traced, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_AND_NE,
                     minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                     inputs=[ct.TensorType(name="x", shape=ct.Shape(shape=(ct.RangeDim(1, 8192, default=1024), context)), dtype=np.float16)],
                     outputs=[ct.TensorType(name="probabilities", dtype=np.float16)])
mlmodel.save(out)
torch.save(model.state_dict(), out.replace(".mlpackage", ".pt"))
print("saved", out)
