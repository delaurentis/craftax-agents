"""priml's Craftax policy (rekursiv.ai, Apache-2.0; priml_net.py) as an ONNX graph in the playable format, with the optional
prev_action input (FORMAT.md): the policy reads the action it took last step beside the observation.

    scripts/fetch_priml_boss.sh                                 # the weights → weights/priml-boss/priml-boss.safetensors (31 MB, bf16)
    python -m craftax_agents.export_priml                        # → out/priml-boss.onnx (62 MB, float32)
    python -m craftax_agents.export_priml --check 300            # also: play 300 steps on the native engine, ONNX vs NumPy

One call = one agent step:  obs [1, 843] (cx_obs), prev_action [1] (0-42, 43 at an episode start), h [4, 1, 1024]  →  logits [1, 43],
value [1], h_out [4, 1, 1024]. priml evaluates by sampling the masked softmax (temperature 1); the action mask stays outside the graph.
Needs `pip install onnx onnxruntime`.
"""
import argparse, os, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from craftax_agents import WEIGHTS as WEIGHTS_DIR, OUT as OUT_DIR
from craftax_agents.priml_net import PrimlNet, IDS, NO_ACTION, OFFSETS, OBS_ROWS, OBS_COLS, CHANNELS, EMB, VOCAB, N_SCALARS, N_ACTIONS

WEIGHTS = os.path.join(WEIGHTS_DIR, "priml-boss", "priml-boss.safetensors")
OUT = os.path.join(OUT_DIR, "priml-boss.onnx")


def build(net):
    H, L, W = net.hidden, net.layers, net.w
    inits, nodes = [], []
    def const(name, a):
        inits.append(numpy_helper.from_array(np.array(a, order="C"), name)); return name
    def node(op, ins, outs=None, **kw):
        outs = outs or [f"{op.lower()}_{len(nodes)}"]
        nodes.append(helper.make_node(op, ins, outs, **kw)); return outs[0] if len(outs) == 1 else outs
    def wt(k, t=False):                                                            # a weight by priml's name; t: transposed for MatMul
        a = W[k + ".weight"]; return const(k, a.T if t else a)
    silu = lambda x: node("Mul", [x, node("Sigmoid", [x])])
    mm = lambda x, k: node("MatMul", [x, wt(k, True)])
    conv = lambda x, k, **kw: node("Conv", [x, wt(k)], **kw)
    dw = lambda x, k: conv(x, k, group=W[k + ".weight"].shape[0], kernel_shape=[3, 3], pads=[1, 1, 1, 1])   # padding "same", zeros

    i64 = lambda *v: np.array(v, np.int64)
    const("offsets", OFFSETS.astype(np.int64)); const("vmax", i64(VOCAB - 1)); const("amax", i64(NO_ACTION)); const("amin", i64(0))
    const("s0", i64(0)); const("s_ids", i64(IDS)); const("s_end", i64(IDS + N_SCALARS)); const("ax1", i64(1)); const("ax0", i64(0)); const("ax2", i64(2))
    const("shape_ids", i64(1, OBS_ROWS * OBS_COLS, CHANNELS)); const("shape_board", i64(1, OBS_ROWS, OBS_COLS, EMB))
    const("shape_cells", i64(1, OBS_ROWS * OBS_COLS * EMB)); const("half", np.array(0.5, np.float32))
    const("zero", np.array(0.0, np.float32)); const("one", np.array(1.0, np.float32))
    const("split2", i64(128, 128)); const("split3", i64(H, H, H)); const("a0", i64(0)); const("a43", i64(N_ACTIONS)); const("a44", i64(N_ACTIONS + 1))

    # cells: one table, 8 ids a tile, summed (embedding.py forward_torch; puffer_net.py)
    ids = node("Reshape", [node("Cast", [node("Slice", ["obs", "s0", "s_ids", "ax1"])], to=TensorProto.INT64), "shape_ids"])
    rows = node("Min", [node("Max", [node("Add", [ids, "offsets"]), "offsets"]), "vmax"])
    cells = node("ReduceSum", [node("Gather", [wt("embedding.cells"), rows], axis=0), "ax2"], keepdims=0)          # [1, 99, 16]
    # status: 2 residual MLPs on [status, emb[prev action]] (encoder.py:982-987)
    status = node("Slice", ["obs", "s_ids", "s_end", "ax1"])
    act = node("Min", [node("Max", [node("Cast", ["prev_action"], to=TensorProto.INT64), "amin"]), "amax"])
    cond = node("Gather", [wt("embedding.status.embedding"), act], axis=0)          # [1, 32]
    for l in range(2):
        hid = silu(mm(node("Concat", [status, cond], axis=1), f"embedding.status.blocks.{l}.proj_in"))
        status = node("Add", [status, mm(hid, f"embedding.status.blocks.{l}.proj_out")])
    # board [1, 16, 9, 11] (encoder.py:890-896)
    board = node("Transpose", [node("Reshape", [cells, "shape_board"])], perm=[0, 3, 1, 2])
    for l in range(2):                                                             # DepthwiseResidual (encoder.py:249)
        p = f"embedding.board.blocks.{l}."
        board = node("Add", [board, conv(silu(dw(board, p + "depthwise")), p + "pointwise")])
    p = "embedding.board.spatial."                                                 # FilmBranch (encoder.py:643-647)
    sp = silu(conv(board, p + "proj_in"))
    scale, shift = node("Split", [mm(status, p + "proj_condition"), "split2"], ["film_scale", "film_shift"], axis=1)
    const("film_hw", i64(1, 128, 1, 1))
    sp = node("Add", [node("Mul", [sp, node("Add", ["one", node("Reshape", [scale, "film_hw"])])]), node("Reshape", [shift, "film_hw"])])
    board = node("Add", [board, conv(silu(dw(sp, p + "depthwise")), p + "proj_out")])
    p = "embedding.board.multiscale."                                              # MultiscaleMix (encoder.py:777-785)
    local = node("AveragePool", [board], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], count_include_pad=1)
    whole = node("Expand", [node("ReduceMean", [board], axes=[2, 3], keepdims=1), node("Shape", [board])])
    mixed = conv(node("Concat", [board, local, whole], axis=1), p + "proj_in")
    board = node("Add", [board, conv(silu(mixed), p + "proj_out")])
    flat = node("Reshape", [node("Transpose", [board], perm=[0, 2, 3, 1]), "shape_cells"])
    e = mm(node("Concat", [flat, status], axis=1), "proj_in")                      # [1, 1024]
    z, outs = e, []
    for l in range(L):                                                             # injection, then MinGRUBlock.step (priml_net.py)
        const(f"l{l}", np.array(l, np.int64))
        hl = node("Gather", ["h", f"l{l}"], axis=0)
        u = node("Add", [z, mm(silu(mm(e, f"injections.{l}.proj_in")), f"injections.{l}.proj_out")])
        hid, gate, hw = node("Split", [mm(u, f"blocks.{l}.proj_gates"), "split3"], [f"hid{l}", f"gate{l}", f"hw{l}"], axis=1)
        ht = node("Where", [node("GreaterOrEqual", [hid, "zero"]), node("Add", [hid, "half"]), node("Sigmoid", [hid])])
        out = node("Add", [hl, node("Mul", [node("Sigmoid", [gate]), node("Sub", [ht, hl])])])
        s = node("Sigmoid", [hw])
        z = node("Add", [node("Mul", [s, out]), node("Mul", [node("Sub", ["one", s]), u])])
        outs.append(node("Unsqueeze", [out, "ax0"]))
    node("Concat", outs, ["h_out"], axis=0)
    d = mm(z, "proj_out")
    node("Slice", [d, "a0", "a43", "ax1"], ["logits"])
    node("Squeeze", [node("Slice", [d, "a43", "a44", "ax1"]), "ax1"], ["value"])

    f = TensorProto.FLOAT
    g = helper.make_graph(nodes, "priml-boss",
                          [helper.make_tensor_value_info("obs", f, [1, IDS + N_SCALARS]), helper.make_tensor_value_info("prev_action", f, [1]),
                           helper.make_tensor_value_info("h", f, [L, 1, H])],
                          [helper.make_tensor_value_info("logits", f, [1, N_ACTIONS]), helper.make_tensor_value_info("value", f, [1]),
                           helper.make_tensor_value_info("h_out", f, [L, 1, H])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="craftax-agents export_priml.py")
    m.ir_version = 8                                                               # onnxruntime-web reads IR 8 at opset 17
    m.doc_string = ("priml Craftax policy, exp004 boss fine-tune (recipe: rekursiv.ai priml, Apache-2.0, github.com/rekursiv-ai/priml), "
                    "board encoder + 4 MinGRU layers of 1024 with injections. obs = craftax-wasm cx_obs (843), prev_action = [1] the "
                    "action taken last step (43 at an episode start), h = [4, 1, 1024] zeros at an episode start.")
    onnx.checker.check_model(m)
    return m


def sample(logits, mask, rng, temp=1.0):
    """priml's evaluation: a draw from the masked softmax (temp 0: the masked argmax)."""
    z = np.where(mask, logits, -np.inf)
    if temp <= 0: return int(np.argmax(z))
    p = np.exp((z - z.max()) / temp); return int(rng.choice(N_ACTIONS, p=p / p.sum()))


def check(net, path, steps, seed=7, temp=1.0):
    """Play on the native engine (sleeps fast-forwarded as in training), the ONNX graph choosing by priml's sampling; NumPy alongside."""
    import onnxruntime as ort
    from craftax_agents.native import Game
    g = Game(4 | 2); g.reset(seed); rng = np.random.default_rng(seed)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    h, prev, worst, wh, agree = net.zero_state(1), NO_ACTION, 0.0, 0.0, 0
    for t in range(steps):
        obs, mask = g.obs()[None], g.mask()
        pa = np.array([prev], np.float32)
        lo, vo, ho = sess.run(None, {"obs": obs, "prev_action": pa, "h": h})
        ln, vn, hn = net.step(obs, pa, h)
        worst = max(worst, float(np.abs(lo - ln).max())); wh = max(wh, float(np.abs(ho - hn).max()))
        agree += int(np.argmax(np.where(mask, lo[0], -np.inf))) == int(np.argmax(np.where(mask, ln[0], -np.inf)))
        a = sample(lo[0], mask, rng, temp); h = ho; prev = a                       # both on the ONNX trajectory
        if g.step(a):
            break
    print(f"{t + 1} steps: greedy action agrees {agree}/{t + 1}, max |ONNX − NumPy| logits {worst:.2e}, h {wh:.2e}, score {g.score():.0f}")


def reference(entry=None, weights=None):
    """For check_model.py --ref craftax_agents.export_priml:reference: the NumPy net as step(obs, {"h", "prev_action"}) → logits [1, 43].
    The weights: $PRIML_WEIGHTS, else weights/priml-boss/priml-boss.safetensors."""
    net = PrimlNet.load(weights or os.environ.get("PRIML_WEIGHTS", WEIGHTS))
    return lambda obs, st: net.step(obs, st["prev_action"], st["h"])[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS, help="priml's MinGRUPolicy state_dict as safetensors (ours: scripts/fetch_priml_boss.sh)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", type=int, default=0, help="steps of native-engine play to compare ONNX with NumPy (0 = skip)")
    a = ap.parse_args()
    if not os.path.exists(a.weights):
        sys.exit(f"{a.weights} missing: run scripts/fetch_priml_boss.sh")
    net = PrimlNet.load(a.weights)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    onnx.save(build(net), a.out)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB")
    if a.check:
        check(net, a.out, a.check)
