"""PufferLib 5.0's Craftax policy (puffer_net.py) as an ONNX graph in the playable format, so the browser can run it (onnxruntime-web).

    scripts/fetch_puffer162.sh                                  # the weights → weights/craftax_1024x4.bin (57 MB)
    python -m craftax_agents.export_onnx                         # → out/puffer162.onnx (57 MB, float32)
    python -m craftax_agents.export_onnx --check 300             # also: play 300 steps on the native engine, ONNX vs NumPy

One call = one agent step:  obs [B, 843] (the engine's packed observation, cx_obs), h [4, B, 1024]  →  logits [B, 43], value [B], h_out.
The action mask stays outside the graph (the engine's cx_mask, CX_ACTION_MASK); pick argmax of the masked logits as PufferLib's demo.
Needs `pip install onnx onnxruntime`.
"""
import argparse, os, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from craftax_agents import WEIGHTS as WEIGHTS_DIR, OUT as OUT_DIR
from craftax_agents.puffer_net import PufferNet, OFFSETS, OBS_ROWS, OBS_COLS, CHANNELS, EMB, VOCAB, N_ACTIONS

WEIGHTS = os.path.join(WEIGHTS_DIR, "craftax_1024x4.bin")
OUT = os.path.join(OUT_DIR, "puffer162.onnx")
IDS = OBS_ROWS * OBS_COLS * CHANNELS                                               # 792 tile ids, then 51 scalars


def build(net):
    H, L = net.hidden, net.layers
    inits, nodes = [], []
    def const(name, a):
        inits.append(numpy_helper.from_array(np.array(a, order="C"), name)); return name
    def node(op, ins, outs=None, **kw):
        outs = outs or [f"{op.lower()}_{len(nodes)}"]
        nodes.append(helper.make_node(op, ins, outs, **kw)); return outs[0] if len(outs) == 1 else outs

    i64 = lambda *v: np.array(v, np.int64)
    const("embed", net.embed); const("proj_t", net.proj.T); const("dec_t", net.decoder.T)
    for l, W in enumerate(net.gru):
        const(f"gru{l}_t", W.T)
    const("offsets", OFFSETS.astype(np.int64)); const("vmax", i64(VOCAB - 1))
    const("s0", i64(0)); const("s_ids", i64(IDS)); const("s_end", i64(IDS + 51)); const("ax1", i64(1)); const("ax0", i64(0))
    const("shape_ids", i64(-1, OBS_ROWS * OBS_COLS, CHANNELS)); const("shape_emb", i64(-1, OBS_ROWS * OBS_COLS * EMB))
    const("ax2", i64(2)); const("half", np.array(0.5, np.float32)); const("zero", np.array(0.0, np.float32))
    const("one", np.array(1.0, np.float32)); const("split3", i64(H, H, H))
    const("a0", i64(0)); const("a43", i64(N_ACTIONS)); const("a44", i64(N_ACTIONS + 1))

    ids = node("Reshape", [node("Cast", [node("Slice", ["obs", "s0", "s_ids", "ax1"])], to=TensorProto.INT64), "shape_ids"])
    rows = node("Min", [node("Max", [node("Add", [ids, "offsets"]), "offsets"]), "vmax"])          # craftax_encode's clamps
    e = node("ReduceSum", [node("Gather", ["embed", rows], axis=0), "ax2"], keepdims=0)          # [B, 99, 16]
    x = node("Concat", [node("Reshape", [e, "shape_emb"]), node("Slice", ["obs", "s_ids", "s_end", "ax1"])], axis=1)
    z = node("MatMul", [x, "proj_t"])
    outs = []
    for l in range(L):                                                             # puffercpu.c mingru(): gate + highway
        const(f"l{l}", np.array(l, np.int64))                                       # scalar: Gather drops the axis
        hl = node("Gather", ["h", f"l{l}"], axis=0)                               # [B, H]
        hid, gate, hw = node("Split", [node("MatMul", [z, f"gru{l}_t"]), "split3"], [f"hid{l}", f"gate{l}", f"hw{l}"], axis=1)
        ht = node("Where", [node("GreaterOrEqual", [hid, "zero"]), node("Add", [hid, "half"]), node("Sigmoid", [hid])])
        out = node("Add", [hl, node("Mul", [node("Sigmoid", [gate]), node("Sub", [ht, hl])])])
        s = node("Sigmoid", [hw])
        z = node("Add", [node("Mul", [s, out]), node("Mul", [node("Sub", ["one", s]), z])])
        outs.append(node("Unsqueeze", [out, "ax0"]))
    node("Concat", outs, ["h_out"], axis=0)
    d = node("MatMul", [z, "dec_t"])
    node("Slice", [d, "a0", "a43", "ax1"], ["logits"])
    node("Squeeze", [node("Slice", [d, "a43", "a44", "ax1"]), "ax1"], ["value"])

    f = TensorProto.FLOAT
    g = helper.make_graph(nodes, "puffer162",
                          [helper.make_tensor_value_info("obs", f, ["B", IDS + 51]), helper.make_tensor_value_info("h", f, [L, "B", H])],
                          [helper.make_tensor_value_info("logits", f, ["B", N_ACTIONS]), helper.make_tensor_value_info("value", f, ["B"]),
                           helper.make_tensor_value_info("h_out", f, [L, "B", H])], inits)
    # producer_name and doc_string are stored in the .onnx: changing them changes its sha256 (models.json), so they keep the
    # wording of the file the site serves.
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="craftax-agents export_onnx.py")
    m.ir_version = 8                                                               # onnxruntime-web reads IR 8 at opset 17
    m.doc_string = ("PufferLib 5.0 Craftax policy craftax_1024x4.bin (MIT, github.com/PufferAI/PufferLib release 5.0-experiments), "
                    "hidden 1024, 4 MinGRU layers. obs = craftax-wasm cx_obs (843), h = [4, B, 1024] zeros at an episode start.")
    onnx.checker.check_model(m)
    return m


def check(net, path, steps):
    """Greedy play on the native engine (engine/build/libcraftax.dylib|.so), the ONNX graph choosing; NumPy run alongside."""
    import onnxruntime as ort
    from craftax_agents.native import Game
    g = Game(4 | 2)                                                                # CX_ACTION_MASK + CX_KEEP_DEAD; sleeps fast-forwarded as in training
    g.reset(7)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    h_o, h_n, worst, agree = net.zero_state(1), net.zero_state(1), 0.0, 0
    for t in range(steps):
        obs, mask = g.obs()[None], g.mask()
        lo, vo, h_o = sess.run(None, {"obs": obs, "h": h_o})
        ln, vn, h_n = net.step(obs, h_n)
        worst = max(worst, float(np.abs(lo - ln).max()), float(np.abs(h_o - h_n).max()))
        a = int(np.argmax(np.where(mask, lo[0], -np.inf)))
        agree += a == int(np.argmax(np.where(mask, ln[0], -np.inf)))
        h_n = h_o.copy()                                                           # keep both on the same trajectory
        if g.step(a):
            break
    print(f"{t + 1} steps: greedy action agrees {agree}/{t + 1}, max |ONNX − NumPy| {worst:.2e}, score {g.score():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS, help="PufferLib's craftax_1024x4.bin (scripts/fetch_puffer162.sh)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", type=int, default=0, help="steps of native-engine play to compare ONNX with NumPy (0 = skip)")
    a = ap.parse_args()
    if not os.path.exists(a.weights):
        sys.exit(f"{a.weights} missing: run scripts/fetch_puffer162.sh")
    net = PufferNet.load(a.weights)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    onnx.save(build(net), a.out)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB")
    if a.check:
        check(net, a.out, a.check)
