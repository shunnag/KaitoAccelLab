"""再帰型 byte 予測 model（GRU 1 層）を torch(MPS) で学習し、Core ML(mlprogram, fp16, 固定 batch) に変換する。
通常版の入出力: x_onehot [B,256] fp16、h_in [B,H] fp16 → probabilities [B,256] fp16、h_out [B,H] fp16。
--stateful 版は同じ重みで x_onehot → probabilities とし、h は Core ML の MLState に保持する。
--stateful-select 版は h を [1,B,H] にして一行だけを選択更新する。--stateful-direct は graph pass で全要素の
slice_update を直接の状態更新に置き換える。direct は現行 ANE では実行できず、CPU / GPU 向けの実験である。
使い方: train_gru_predictor.py <corpus> <out-prefix> [steps] [hidden] [seq] [batch-shapes 例 1024,4096] [--stateful | --stateful-select | --stateful-direct]
steps に 0 を渡すと学習せず <out-prefix>.pt を読んで Core ML への書き出しだけを行う。
"""
import sys, time, numpy as np, torch, torch.nn as nn
import coremltools as ct
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

corpus_path, prefix = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
hidden = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
seq = int(sys.argv[5]) if len(sys.argv) > 5 else 256
shapes = [int(s) for s in (sys.argv[6] if len(sys.argv) > 6 else "1024,4096").split(",")]
train_batch = 128
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data = np.frombuffer(open(corpus_path, "rb").read(), dtype=np.uint8)
split = int(len(data) * 0.9)
train, valid = data[:split], data[split:]

class Trainer(nn.Module):
    """学習用: 埋め込み → GRU → 線形。入力列の先頭は「開始」(全 0 埋め込み) から始める。"""
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(256, hidden, bias=False)  # one-hot @ W と同じ意味
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 256)
    def forward(self, onehot):  # onehot: [B, T, 256]（位置 0 は全 0 = 開始）
        e = self.embed(onehot)
        h, _ = self.gru(e)
        return self.out(h)  # logits [B, T, 256]

class Cell(nn.Module):
    """書き出し用: 1 step 分。torch の GRU と同じ式（ゲート順 r, z, n）を線形と活性で書く。"""
    def __init__(self, t: Trainer):
        super().__init__()
        self.embed = t.embed
        self.w_ih = nn.Parameter(t.gru.weight_ih_l0.detach().clone())
        self.w_hh = nn.Parameter(t.gru.weight_hh_l0.detach().clone())
        self.b_ih = nn.Parameter(t.gru.bias_ih_l0.detach().clone())
        self.b_hh = nn.Parameter(t.gru.bias_hh_l0.detach().clone())
        self.out = t.out
    def forward(self, x_onehot, h_in):
        e = self.embed(x_onehot)
        gi = torch.nn.functional.linear(e, self.w_ih, self.b_ih)
        gh = torch.nn.functional.linear(h_in, self.w_hh, self.b_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_out = (1.0 - z) * n + z * h_in
        return torch.softmax(self.out(h_out), dim=-1), h_out

@register_pass(namespace="common")
class direct_state_update(AbstractGraphPass):
    """この exporter の全要素 slice_update を直接の状態更新へ置き換える。"""
    def apply(self, prog):
        replacements = 0
        for function in prog.functions.values():
            for op in list(function.operations):
                if op.op_type != "coreml_update_state" or op.value.op is None:
                    continue
                update = op.value.op
                if update.op_type != "slice_update" or update.inputs["x"].op.op_type != "read_state":
                    continue
                value = update.inputs["update"]
                if value.sym_type != op.value.sym_type:
                    continue
                if function.try_replace_uses_of_var_after_op(
                    anchor_op=update, old_var=update.outputs[0], new_var=value,
                ):
                    function.remove_ops([update])
                    replacements += 1
        if replacements != 1:
            raise RuntimeError(f"expected one full state slice update, got {replacements}")

class StatefulCell(nn.Module):
    """書き出し用: GRU の hidden state を Core ML の state として登録する。"""
    def __init__(self, cell: Cell, batch: int, width: int, select: bool = False):
        super().__init__()
        self.cell = cell
        self.select = select
        shape = (1, batch, width) if select else (batch, width)
        self.register_buffer("hidden_state", torch.zeros(shape))
    def forward(self, x_onehot):
        if self.select:
            probabilities, h_out = self.cell(x_onehot, self.hidden_state[0])
            self.hidden_state[0] = h_out
        else:
            probabilities, h_out = self.cell(x_onehot, self.hidden_state)
            self.hidden_state[:, :] = h_out
        return probabilities

if len(sys.argv) > 8 or (len(sys.argv) == 8 and sys.argv[7] not in ("--stateful", "--stateful-select", "--stateful-direct")):
    raise SystemExit("unknown arguments; use --stateful, --stateful-select, or --stateful-direct")
stateful = len(sys.argv) == 8
select_state = stateful and sys.argv[7] == "--stateful-select"
direct_state = stateful and sys.argv[7] == "--stateful-direct"

def batch_from(arr, size, rng):
    idx = rng.integers(0, len(arr) - seq - 1, size)
    ys = np.stack([arr[i:i + seq] for i in idx]).astype(np.int64)      # 予測対象 [B, T]
    y = torch.from_numpy(ys).to(device)
    prev = torch.cat([torch.full((size, 1), 256, device=device), y[:, :-1]], dim=1)  # 直前 byte、先頭は 256=開始
    onehot = torch.nn.functional.one_hot(prev, 257)[..., :256].to(torch.float32)  # 開始は全 0
    return onehot, y

model = Trainer().to(device)
if steps == 0:
    model.load_state_dict(torch.load(prefix + ".pt", map_location=device))
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
rng = np.random.default_rng(7)
t0 = time.time()
for step in range(steps):
    x, y = batch_from(train, train_batch, rng)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 250 == 0 or step == steps - 1:
        with torch.no_grad():
            vx, vy = batch_from(valid, 256, rng)
            vl = torch.nn.functional.cross_entropy(model(vx).reshape(-1, 256), vy.reshape(-1))
        print(f"step {step} loss {loss.item():.3f} valid bits/byte {vl.item() / np.log(2):.3f} ({time.time() - t0:.0f}s)", flush=True)

model.eval().cpu()
if steps > 0:
    torch.save(model.state_dict(), prefix + ".pt")
cell = Cell(model).eval()
for b in shapes:
    if stateful:
        traced = torch.jit.trace(StatefulCell(cell, b, hidden, select=select_state).eval(),
                                 torch.zeros(b, 256), check_trace=False)
        state_shape = (1, b, hidden) if select_state else (b, hidden)
        pipeline = ct.PassPipeline.DEFAULT
        if direct_state:
            pipeline.append_pass("common::direct_state_update")
        mlmodel = ct.convert(traced, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_AND_NE,
                             minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                             inputs=[ct.TensorType(name="x_onehot", shape=(b, 256), dtype=np.float16)],
                             outputs=[ct.TensorType(name="probabilities", dtype=np.float16)],
                             states=[ct.StateType(wrapped_type=ct.TensorType(shape=state_shape, dtype=np.float16),
                                                  name="hidden_state")], pass_pipeline=pipeline)
    else:
        traced = torch.jit.trace(cell, (torch.zeros(b, 256), torch.zeros(b, hidden)))
        mlmodel = ct.convert(traced, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_AND_NE,
                             minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                             inputs=[ct.TensorType(name="x_onehot", shape=(b, 256), dtype=np.float16),
                                     ct.TensorType(name="h_in", shape=(b, hidden), dtype=np.float16)],
                             outputs=[ct.TensorType(name="probabilities", dtype=np.float16),
                                      ct.TensorType(name="h_out", dtype=np.float16)])
    suffix = "-stateful-direct" if direct_state else "-stateful-select" if select_state else "-stateful" if stateful else ""
    out = f"{prefix}-b{b}{suffix}.mlpackage"
    mlmodel.save(out)
    if stateful:
        probe = ct.models.MLModel(out, compute_units=ct.ComputeUnit.CPU_ONLY)
        state = probe.make_state()
        if np.any(state.read_state("hidden_state")):
            raise RuntimeError("new MLState is not initialized to zero")
        probe.predict({"x_onehot": np.zeros((b, 256), dtype=np.float16)}, state=state)
        if not np.any(state.read_state("hidden_state")):
            raise RuntimeError("stateful export did not update hidden_state")
        print("state update: OK", flush=True)
    print("saved", out, flush=True)
