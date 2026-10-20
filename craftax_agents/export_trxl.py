"""Gautier Hamon's TransformerXL-PPO Craftax policy (github.com/Reytuag/transformerXL_PPO_JAX, MIT) as an ONNX graph for the browser.

    scripts/fetch_trxl.sh                                         # the checkpoint + their model code → weights/trxl/
    python -m craftax_agents.export_trxl                          # → out/trxl.onnx (18.8 MB, float32)
    python -m craftax_agents.export_trxl --check 400              # + 400 native-engine steps, ONNX vs their JAX code (a reset at 200)
    python -m craftax_agents.export_trxl --play 3                 # + 3 greedy episodes: cx_score, steps/s
    python -m craftax_agents.export_trxl --seed 4 --out out/trxl-seed4.onnx --check 400   # another seed (fetch_trxl.sh --all)

The checkpoint: results_craftax/base_A100_1e9 seed 2 (1e9 steps, 2 layers of 256, 8 heads, memory 128). Of the repo's ten Craftax
checkpoints it ends level on return with seed 4 (41.9) and goes deepest: gnomish mines 9.4 % of episodes, the only seed in the sewers.
It was trained with the code at 518de77, BEFORE the MLP fix (db3e52a) and the 2nd-gating fix (d914d55), so this graph is that code:
the position-wise MLP is dense2 alone (dense1 and ln2 run but nothing reads them), and gate2 takes the MLP output as its residual.
The reference is their own Transformer.forward_eval, imported unmodified from weights/trxl/src (scripts/fetch_trxl.sh), batch 2 because its
x.squeeze() drops a batch of 1; the actor/critic heads (three Dense + ReLU each) are restated here, distrax not needed.

One call = one agent step, as their rollout (_env_step in trainer_PPO_trXL.py) does it:
    obs [1, 843]  (the engine's packed observation; symbolic.py unpacks it to Craftax-Symbolic's 8,268 inside the graph)
    kv [4, 1, 128, 256]  per layer the key and value of the last 128 steps (k0, v0, k1, v1); zeros at an episode start
    fill [1]  how many of those slots are this episode's: 0 at an episode start, +1 a step, stops at 128
  → logits [1, 43], value [1], kv_out, fill_out, layers [3, 1, 256] (the encoded obs, block 1's output, block 2's output: the brain view).
Their cache holds each layer's input embedding and projects all 129 keys/values again every step; the graph keeps the projections
instead (LayerNorm and Dense act row by row, so it is the same numbers) and projects only the new row. Their mask lets a step see
min(steps since reset, 128) past slots plus itself; fill is that count, so zeros at reset reproduce it, stale memory never leaks.
The action mask stays outside the graph (the engine's cx_mask); the model was trained without one.
"""
import argparse, os, sys, time
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from craftax_agents import WEIGHTS, OUT as OUT_DIR
from craftax_agents.symbolic import unpack, unpack_nodes

RUN = os.path.join(WEIGHTS, "trxl", "base_A100_1e9")
SEED = 2
SRC = os.path.join(WEIGHTS, "trxl", "src")
OUT = os.path.join(OUT_DIR, "trxl.onnx")
N_ACTIONS = 43


def load(run=RUN, seed=SEED):
    cfg = np.load(os.path.join(run, f"{seed}_config.npy"), allow_pickle=True).item()
    p = np.load(os.path.join(run, f"{seed}_params.npy"), allow_pickle=True).item()["params"]
    return cfg, p


def pos_embed(cfg):
    """Their PositionalEmbedding on arange(W+1, -1, -1)[:W+1]: slot j of the W+1 keys sits at distance W+1-j (the current step at 1)."""
    E, W = cfg["EMBED_SIZE"], cfg["WINDOW_MEM"]
    inv = 1 / (10000 ** (np.arange(0.0, E, 2.0) / E))
    s = np.outer(np.arange(W + 1, -1, -1)[:W + 1], inv)
    return np.concatenate([np.sin(s), np.cos(s)], -1).astype(np.float32)          # [W+1, E]


def build(cfg, p, seed=SEED):
    E, H, L, W = cfg["EMBED_SIZE"], cfg["num_heads"], cfg["num_layers"], cfg["WINDOW_MEM"]
    D = cfg["qkv_features"] // H
    assert cfg["gating"] and cfg["qkv_features"] == E and cfg["ACTIVATION"] == "relu"
    inits, nodes = [], []
    def const(name, a):
        inits.append(numpy_helper.from_array(np.array(a, order="C"), name)); return name
    def node(op, ins, outs=None, **kw):
        outs = outs or [f"{op.lower()}_{len(nodes)}"]
        nodes.append(helper.make_node(op, ins, outs, **kw)); return outs[0] if len(outs) == 1 else outs
    f32 = lambda a: np.asarray(a, np.float32)
    i64 = lambda *v: np.array(v, np.int64)
    dense = lambda x, name, d: node("Add", [node("MatMul", [x, const(f"{name}_w", f32(d["kernel"]))]), const(f"{name}_b", f32(d["bias"]).reshape(-1))])
    const("one", f32(1.0)); const("big_neg", f32(np.finfo(np.float32).min)); const("sqrt_d", f32(np.sqrt(D)))
    const("Wf", f32(W)); const("slots", np.arange(W + 1, dtype=np.float32)); const("ax0", i64(0)); const("ax1", i64(1))
    const("s1", i64(1)); const("sW1", i64(W + 1)); const("heads", i64(H, 1, D))
    const("flat", i64(1, E)); const("kv_row", i64(1, 1, E)); const("kvt", i64(W + 1, H, D)); const("split_q", i64(E, 2 * E))

    sym = unpack_nodes(node, const, "obs")                                         # [1, 8268] Craftax-Symbolic
    x = dense(sym, "enc", p["transformer"]["encoder"])                             # [1, E]
    see = node("GreaterOrEqual", [node("Add", ["slots", "fill"]), "Wf"])            # slot j visible iff j >= W - fill (j = W: this step)
    R = pos_embed(cfg)
    embs, kv_outs = [x], []
    for l in range(L):
        t = p["transformer"][f"tf_layers_{l}"]; a = t["attention1"]
        ln = lambda v, k=f"l{l}_ln1": node("LayerNormalization", [v, const(f"{k}_s", f32(t["ln1"]["scale"])), const(f"{k}_b", f32(t["ln1"]["bias"]))], axis=-1, epsilon=1e-6)
        wqkv = np.concatenate([f32(a[k]["kernel"]).reshape(E, E) for k in ("query", "key", "value")], 1)
        bqkv = np.concatenate([f32(a[k]["bias"]).reshape(E) for k in ("query", "key", "value")])
        qkv = node("Add", [node("MatMul", [ln(x), const(f"l{l}_wqkv", wqkv)]), const(f"l{l}_bqkv", bqkv)])
        q, kn = node("Split", [qkv, "split_q"], [f"l{l}_q", f"l{l}_kvnew"], axis=1)
        kn, vn = node("Split", [kn], [f"l{l}_kn", f"l{l}_vn"], axis=1)
        const(f"l{l}_ik", np.array(2 * l, np.int64)); const(f"l{l}_iv", np.array(2 * l + 1, np.int64))   # scalars: Gather drops the axis
        K = node("Concat", [node("Gather", ["kv", f"l{l}_ik"], axis=0), node("Reshape", [kn, "kv_row"])], axis=1)   # [1, W+1, E]
        V = node("Concat", [node("Gather", ["kv", f"l{l}_iv"], axis=0), node("Reshape", [vn, "kv_row"])], axis=1)
        kv_outs += [node("Unsqueeze", [node("Slice", [K, "s1", "sW1", "ax1"]), "ax0"]), node("Unsqueeze", [node("Slice", [V, "s1", "sW1", "ax1"]), "ax0"])]
        Kt = node("Transpose", [node("Reshape", [K, "kvt"])], perm=[1, 2, 0])          # [H, D, W+1]
        Vh = node("Transpose", [node("Reshape", [V, "kvt"])], perm=[1, 0, 2])          # [H, W+1, D]
        r = np.einsum("ke,ehd->khd", R, f32(a["pos_embed_mat"]["kernel"])).transpose(1, 2, 0)   # pos_embed_mat(R): [H, D, W+1], fixed
        qw = node("Reshape", [node("Add", [q, const(f"l{l}_rw", f32(a["r_w_bias"]).reshape(E))]), "heads"])        # [H, 1, D]
        qr = node("Reshape", [node("Add", [q, const(f"l{l}_rr", f32(a["r_r_bias"]).reshape(E))]), "heads"])
        s = node("Div", [node("Add", [node("MatMul", [qw, Kt]), node("MatMul", [qr, const(f"l{l}_R", r)])]), "sqrt_d"])  # [H, 1, W+1]
        w = node("Softmax", [node("Where", [see, s, "big_neg"])], axis=-1)           # their jnp.where(mask, w, finfo.min)
        o = node("Reshape", [node("MatMul", [w, Vh]), "flat"])                         # [1, H*D], head-major as their out kernel
        att = node("Add", [node("MatMul", [o, const(f"l{l}_wo", f32(a["out"]["kernel"]).reshape(E, E))]), const(f"l{l}_bo", f32(a["out"]["bias"]))])
        def gate(g, xr, y, k):                                                     # their Gating(x, y), Dense_0..5 in call order
            G = [f32(g[f"Dense_{i}"]["kernel"]) for i in range(6)]
            ay, by, cy = node("Split", [node("MatMul", [y, const(f"{k}_wy", np.concatenate([G[0], G[2], G[4]], 1))])], axis=1, outs=[f"{k}_ay", f"{k}_by", f"{k}_cy"])
            ax_, bx = node("Split", [node("MatMul", [xr, const(f"{k}_wx", np.concatenate([G[1], G[3]], 1))])], axis=1, outs=[f"{k}_ax", f"{k}_bx"])
            rr = node("Sigmoid", [node("Add", [ay, ax_])])
            z = node("Sigmoid", [node("Sub", [node("Add", [by, bx]), const(f"{k}_bg", f32(g["gating_bias"]))])])
            h = node("Tanh", [node("Add", [cy, node("MatMul", [node("Mul", [rr, xr]), const(f"{k}_w5", G[5])])])])
            return node("Add", [node("Mul", [node("Sub", ["one", z]), xr]), node("Mul", [z, h])])
        out_att = gate(t["gate1"], x, node("Relu", [att]), f"l{l}_g1")             # gate1(queries, relu(attention))
        mlp = dense(out_att, f"l{l}_d2", t["dense2"])                              # 518de77: out = self.dense2(out_attention)
        x = gate(t["gate2"], mlp, node("Relu", [out_att]), f"l{l}_g2")             # gate2(out, relu(out_attention))
        embs.append(x)
    node("Concat", kv_outs, ["kv_out"], axis=0)
    node("Min", [node("Add", ["fill", "one"]), "Wf"], ["fill_out"])
    node("Concat", [node("Unsqueeze", [e, "ax0"]) for e in embs], ["layers"], axis=0)
    head = lambda v, k: dense(node("Relu", [dense(node("Relu", [dense(v, f"{k}1", p[f"{k}_ln1"])]), f"{k}2", p[f"{k}_ln2"])]), f"{k}o", p[f"{k}_out"])
    node("Identity", [head(x, "actor")], ["logits"])
    node("Reshape", [head(x, "critic"), "s1"], ["value"])

    f = TensorProto.FLOAT
    g = helper.make_graph(nodes, "trxl",
                          [helper.make_tensor_value_info("obs", f, [1, 843]), helper.make_tensor_value_info("kv", f, [2 * L, 1, W, E]),
                           helper.make_tensor_value_info("fill", f, [1])],
                          [helper.make_tensor_value_info("logits", f, [1, N_ACTIONS]), helper.make_tensor_value_info("value", f, [1]),
                           helper.make_tensor_value_info("kv_out", f, [2 * L, 1, W, E]), helper.make_tensor_value_info("fill_out", f, [1]),
                           helper.make_tensor_value_info("layers", f, [L + 1, 1, E])], inits)
    # producer_name and doc_string are stored in the .onnx: changing them changes its sha256 (models.json), so they keep the
    # wording of the file the site serves.
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="craftax-agents export_trxl.py")
    m.ir_version = 8                                                               # onnxruntime-web reads IR 8 at opset 17
    m.doc_string = ("TransformerXL-PPO Craftax policy (MIT, Gautier Hamon, github.com/Reytuag/transformerXL_PPO_JAX @518de77, "
                    f"results_craftax/base_A100_1e9 seed {seed}): {L} gated TrXL layers of {E}, {H} heads, memory {W}. obs = craftax-wasm "
                    f"cx_obs (843); kv [{2 * L}, 1, {W}, {E}] and fill [1] zeros at an episode start, fed back from kv_out / fill_out.")
    onnx.checker.check_model(m)
    return m


class Reference:
    """Their Transformer.forward_eval (weights/trxl/src, unmodified) + the heads, with _env_step's memory, mask and roll bookkeeping."""
    def __init__(self, cfg, p):
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        import jax, jax.numpy as jnp
        sys.path.insert(0, SRC)
        from transformerXL import Transformer
        self.cfg, self.jnp = cfg, jnp
        net = Transformer(encoder_size=cfg["EMBED_SIZE"], num_heads=cfg["num_heads"], qkv_features=cfg["qkv_features"],
                          num_layers=cfg["num_layers"], gating=cfg["gating"], gating_bias=cfg["gating_bias"])
        def fwd(prm, mem, obs, mask):
            x, mem_out = net.apply({"params": prm["transformer"]}, mem, obs, mask, method=Transformer.forward_eval)
            head = lambda k: prm[f"{k}_out"]["bias"] + jax.nn.relu(prm[f"{k}_ln2"]["bias"] + jax.nn.relu(x @ prm[f"{k}_ln1"]["kernel"] + prm[f"{k}_ln1"]["bias"]) @ prm[f"{k}_ln2"]["kernel"]) @ prm[f"{k}_out"]["kernel"]
            return head("actor"), head("critic")[:, 0], mem_out
        self.fwd, self.p = jax.jit(fwd), jax.tree_util.tree_map(jnp.asarray, p)
        W, L, E, H = cfg["WINDOW_MEM"], cfg["num_layers"], cfg["EMBED_SIZE"], cfg["num_heads"]
        self.mem = np.zeros((2, W, L, E), np.float32); self.mask = np.zeros((2, H, 1, W + 1), bool)
        self.idx = np.full(2, W + 1); self.done = False                              # their initial memories_mask_idx = WINDOW_MEM + 1
    def step(self, sym):
        W = self.cfg["WINDOW_MEM"]
        self.idx = np.full(2, W) if self.done else np.clip(self.idx - 1, 0, W)
        if self.done: self.mask[:] = False                                         # memories NOT cleared, only masked: as theirs
        self.mask |= (np.arange(W + 1) == self.idx[:, None])[:, None, None, :]
        obs = np.repeat(np.asarray(sym, np.float32).reshape(1, -1), 2, 0)          # batch 2: forward_eval's x.squeeze() eats a batch of 1
        lg, v, mo = self.fwd(self.p, self.mem, obs, self.mask)
        self.mem = np.concatenate([self.mem[:, 1:], np.asarray(mo)[:, None]], 1)   # jnp.roll(memories, -1, axis=1).at[:, -1].set(out)
        self.done = False
        return np.asarray(lg[:1]), np.asarray(v[:1])


def zero_state(cfg):
    L, W, E = cfg["num_layers"], cfg["WINDOW_MEM"], cfg["EMBED_SIZE"]
    return {"kv": np.zeros((2 * L, 1, W, E), np.float32), "fill": np.zeros(1, np.float32)}


def check(cfg, p, path, steps, flags):
    """Greedy play on the native engine, the ONNX graph choosing, their JAX code alongside; a new episode at steps // 2 (or on death)."""
    import onnxruntime as ort
    from craftax_agents.native import Game
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    g, ref, st = Game(flags), Reference(cfg, p), zero_state(cfg)
    g.reset(7); worst, wv, agree, resets, since, longest = 0.0, 0.0, 0, 0, 0, 0
    for t in range(steps):
        obs = g.obs()[None]
        lo, vo, kv, fill, _ = sess.run(None, {"obs": obs, **st})
        ln, vn = ref.step(unpack(obs))
        worst, wv = max(worst, float(np.abs(lo - ln).max())), max(wv, float(np.abs(vo - vn).max()))
        m = g.mask()
        a = int(np.argmax(np.where(m, lo[0], -np.inf)))
        agree += a == int(np.argmax(np.where(m, ln[0], -np.inf)))
        st = {"kv": kv, "fill": fill}; since += 1; longest = max(longest, since)
        if g.step(a) or t + 1 == steps // 2:
            g.reset(8 + resets); resets += 1; since = 0; st = zero_state(cfg); ref.done = True
    print(f"{steps} steps ({resets} new episodes, longest run {longest} steps vs memory {cfg['WINDOW_MEM']}): greedy action agrees "
          f"{agree}/{steps}, max |ONNX − JAX| logits {worst:.2e}, value {wv:.2e}")
    return worst, agree == steps


def play(cfg, path, episodes, flags, max_steps=20000):
    """Greedy (masked argmax) episodes on the native engine: Craftax's score (cx_score) and ONNX steps/s on CPU onnxruntime."""
    import onnxruntime as ort
    from craftax_agents.native import Game
    o = ort.SessionOptions(); o.intra_op_num_threads = 1                            # one thread, nearer the browser's wasm
    sess = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
    g, run, n, scores = Game(flags), 0.0, 0, []
    for e in range(episodes):
        g.reset(100 + e); st = zero_state(cfg)
        for t in range(max_steps):
            t0 = time.perf_counter()
            lo, _, kv, fill, _ = sess.run(None, {"obs": g.obs()[None], **st})
            run += time.perf_counter() - t0; n += 1
            st = {"kv": kv, "fill": fill}
            if g.step(int(np.argmax(np.where(g.mask(), lo[0], -np.inf)))):
                break
        scores.append(score := g.score())
        print(f"episode {e} (seed {100 + e}): {t + 1} steps, score {score:.0f}{'' if t + 1 < max_steps else ' (cut)'}")
    print(f"scores {[round(s) for s in scores]}, mean {np.mean(scores):.1f}: a few local greedy episodes, not an evaluation")
    print(f"ONNX {n / run:.0f} steps/s (CPU onnxruntime, 1 thread)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=SEED, help="which base_A100_1e9 seed (0-4; the others need fetch_trxl.sh --all)")
    ap.add_argument("--check", type=int, default=0, help="steps of native-engine play to compare ONNX with their JAX code (0 = skip)")
    ap.add_argument("--play", type=int, default=0, help="greedy episodes to score on the native engine (0 = skip)")
    ap.add_argument("--single-step", action="store_true", help="engine plays sleeps step by step as JAX Craftax did in training (default: fast-forwarded, as the browser)")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(RUN, f"{a.seed}_params.npy")):
        sys.exit("weights missing: run scripts/fetch_trxl.sh")
    cfg, p = load(seed=a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    onnx.save(build(cfg, p, a.seed), a.out)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB")
    flags = 4 | 2 | (1 if a.single_step else 0)                                    # CX_ACTION_MASK + CX_KEEP_DEAD (score readable at the end) (+ CX_SINGLE_STEP)
    if a.check:
        worst, same = check(cfg, p, a.out, a.check, flags)
        if worst > 1e-3 or not same:
            sys.exit("ONNX and JAX disagree")
    if a.play:
        play(cfg, a.out, a.play, flags)
