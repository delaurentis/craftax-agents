"""priml's Craftax policy (rekursiv.ai, github.com/rekursiv-ai/priml, Apache-2.0, read at commit 0e22e8a1) in NumPy, one step, fp32:
MinGRUPolicy with exp002's BoardEncoder and injections (priml/baselines/craftax/model.py, policies/encoder.py, experiments.py
exp002-exp004; priml/model/min_gru.py). Mirrors their PyTorch forward op for op; priml runs it with bf16 weights and activations
rounded at each GEMM, this keeps every activation in fp32 (the weights are the bf16 ones, exactly widened).

    net = PrimlNet.load("weights/priml-boss/priml-boss.safetensors")
    logits, value, h = net.step(obs [B, 843], prev [B], h [4, B, 1024])     # prev: the action taken last step, 43 at an episode start

    cells     the 99 tiles' 8 ids, each looked up in its slice of one 154 × 16 table and summed (puffer_net.py's embedding)
    status    the 51 scalars + 2 residual MLPs 83 → 192 → 51 (SiLU), each reading [status, 32-d embedding of the previous action]
    board     the cells as a 16-channel 9 × 11 board: 2 depthwise residual blocks, a 128-wide FiLM branch on the refined status,
              a multiscale mix (cell, 3 × 3 average, whole-board mean) → 1,584
    trunk     proj_in [board, status] 1,635 → 1,024 = e; 4 MinGRU layers (PufferLib's gate + highway), layer l reading x + inj_l(e),
              inj_l = 1,024 → 128 → 1,024 (SiLU); decoder 1,024 → 44 = 43 logits + value
No biases, no norms. priml's `auxiliary` head (a 43 × 1,024 feasibility loss) is training-only and is not loaded.
The observation is PufferLib's packed 843 (priml game/observation.py), the engine's cx_obs; the two differ only where the engine
copies JAX Craftax (priml docs/differences.md D9: mob channels written as JAX scatters them; D11: a reciprocal in the `/ 10` scalars).
"""
import json, struct
import numpy as np

from craftax_agents.puffer_net import OFFSETS, OBS_ROWS, OBS_COLS, CHANNELS, EMB, VOCAB, N_SCALARS, N_ACTIONS

IDS = OBS_ROWS * OBS_COLS * CHANNELS                                               # 792 tile ids, then 51 scalars
NO_ACTION = N_ACTIONS                                                              # priml game/state.py NO_ACTION = ATN_DIM: an episode start


def read_safetensors(path):
    """{name: float32 array} from a safetensors file, bf16 widened exactly (bits << 16); no torch needed."""
    b = open(path, "rb").read(); n = struct.unpack("<Q", b[:8])[0]; head = json.loads(b[8:8 + n]); out = {}
    for k, v in head.items():
        if k == "__metadata__": continue
        lo, hi = v["data_offsets"]; raw = b[8 + n + lo:8 + n + hi]
        if v["dtype"] == "BF16": a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif v["dtype"] == "F32": a = np.frombuffer(raw, np.float32)
        else: raise ValueError(f"{k}: dtype {v['dtype']}")
        out[k] = a.reshape(v["shape"]).copy()
    return out


def silu(x): return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))                  # functional.silu
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def conv1(x, w):                                                                   # 1 × 1 Conv2d, no bias: x [B, C, R, K], w [O, C, 1, 1]
    return np.einsum("oc,bcrk->bork", w[:, :, 0, 0], x)


def depthwise(x, w):                                                               # 3 × 3 Conv2d, groups = channels, padding "same" (zeros)
    B, C, R, K = x.shape; p = np.zeros((B, C, R + 2, K + 2), x.dtype); p[:, :, 1:-1, 1:-1] = x
    out = np.zeros_like(x)
    for i in range(3):
        for j in range(3): out += w[None, :, 0, i, j, None, None] * p[:, :, i:i + R, j:j + K]
    return out


def avg_pool3(x):                                                                  # avg_pool2d(3, stride 1, padding 1): count_include_pad, / 9
    B, C, R, K = x.shape; p = np.zeros((B, C, R + 2, K + 2), x.dtype); p[:, :, 1:-1, 1:-1] = x
    return sum(p[:, :, i:i + R, j:j + K] for i in range(3) for j in range(3)) / np.float32(9)


class PrimlNet:
    def __init__(self, w, layers=4):
        self.w = {k: v for k, v in w.items() if not k.startswith("auxiliary.")}  # model.py FeasibilityLoss: training only
        self.layers = layers; self.hidden = self.w["proj_in.weight"].shape[0]

    @classmethod
    def load(cls, path):
        return cls(read_safetensors(path))

    def zero_state(self, batch=1):
        return np.zeros((self.layers, batch, self.hidden), np.float32)

    def encode(self, x, prev):
        """encoder.py BoardEncoder.forward: packed obs [B, 843] + previous action [B] → features [B, 1,635]."""
        w = lambda k: self.w["embedding." + k + ".weight"]
        B = x.shape[0]
        ids = x[:, :IDS].reshape(B, OBS_ROWS * OBS_COLS, CHANNELS).astype(np.int64)
        rows = w("cells")[np.clip(ids + OFFSETS, OFFSETS, VOCAB - 1)]                    # embedding.py forward_torch: ids + offsets (in range: no clamp there)
        cells = rows[:, :, 0]
        for f in range(1, CHANNELS): cells = cells + rows[:, :, f]                 # summed in field order, as forward_torch
        status = x[:, IDS:IDS + N_SCALARS]
        # ActionConditionedStatus.forward (encoder.py:982-987): status += proj_out(silu(proj_in(cat(status, emb[action]))))
        cond = w("status.embedding")[np.clip(np.asarray(prev).reshape(B).astype(np.int64), 0, NO_ACTION)]
        for l in range(2):
            h = silu(np.concatenate([status, cond], 1) @ w(f"status.blocks.{l}.proj_in").T)
            status = status + h @ w(f"status.blocks.{l}.proj_out").T
        # BoardCNN.forward (encoder.py:890-896): cells row-major [9, 11, 16] → [B, 16, 9, 11]
        board = cells.reshape(B, OBS_ROWS, OBS_COLS, EMB).transpose(0, 3, 1, 2)
        for l in range(2):                                                         # DepthwiseResidual.forward (encoder.py:249)
            board = board + conv1(silu(depthwise(board, w(f"board.blocks.{l}.depthwise"))), w(f"board.blocks.{l}.pointwise"))
        # FilmBranch.forward (encoder.py:643-647): scale, shift = chunk(2) in that order; spatial * (1 + scale) + shift
        sp = silu(conv1(board, w("board.spatial.proj_in")))
        scale, shift = np.split(status @ w("board.spatial.proj_condition").T, 2, axis=1)
        sp = sp * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        board = board + conv1(silu(depthwise(sp, w("board.spatial.depthwise"))), w("board.spatial.proj_out"))
        # MultiscaleMix.forward (encoder.py:777-785): cat(board, 3 × 3 average, whole-board mean) on channels → 32 → 16
        whole = np.broadcast_to(board.mean(axis=(2, 3), keepdims=True), board.shape)
        mixed = conv1(np.concatenate([board, avg_pool3(board), whole], 1), w("board.multiscale.proj_in"))
        board = board + conv1(silu(mixed), w("board.multiscale.proj_out"))
        return np.concatenate([board.transpose(0, 2, 3, 1).reshape(B, -1), status], 1)   # BoardEncoder.forward: cat(board, status)

    def step(self, x, prev, h):
        """x [B, 843], prev [B] (0-42, 43 = none), h [layers, B, hidden] → logits [B, 43], value [B], new h. model.py forward_fused."""
        x = np.asarray(x, np.float32); w = lambda k: self.w[k + ".weight"]
        e = v = self.encode(x, prev) @ w("proj_in").T                              # _encode: proj_in, no proj_feature
        h = np.array(h, np.float32)
        for l in range(self.layers):
            # _inject (model.py:743-747): the block reads v + MLP_l(e), MLP = proj_out(silu(proj_in)) 1,024 → 128 → 1,024
            u = v + silu(e @ w(f"injections.{l}.proj_in").T) @ w(f"injections.{l}.proj_out").T
            # MinGRUBlock.step → TorchScan.step (min_gru.py:885-905, 490-516): hidden, gate, highway = chunk(3);
            # h' = lerp(h, candidate(hidden), sigmoid(gate)), candidate = hidden + 0.5 if hidden >= 0 else sigmoid(hidden) (_candidate);
            # out = s * h' + (1 - s) * u, s = sigmoid(highway)
            hid, gate, hw = np.split(u @ w(f"blocks.{l}.proj_gates").T, 3, axis=1)
            upd = h[l] + sigmoid(gate) * (np.where(hid >= 0, hid + 0.5, sigmoid(hid)) - h[l])
            s = sigmoid(hw)
            h[l] = upd; v = s * upd + (1 - s) * u
        d = v @ w("proj_out").T                                                    # _decode: one fused 44-row decoder, no bias
        return d[:, :N_ACTIONS], d[:, N_ACTIONS], h
