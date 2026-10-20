"""Our reproduction of the Craftax paper's PPO-RNN (Matthews et al. 2024, trained 1e9 steps on Craftax-Symbolic) as an ONNX graph
for the browser; also the worked example of converting a flax checkpoint by hand.

    python -m craftax_agents.export_ppo_rnn --weights path/to/params.msgpack              # → out/ppo-rnn.onnx (~27 MB, float32)
    python -m craftax_agents.export_ppo_rnn --weights path/to/params.msgpack --check 500  # also: 500 native-engine steps, ONNX vs flax

One call = one agent step (the FORMAT.md contract):  obs [1, 843] (cx_obs), h [1, 1, 512]  →  logits [1, 43], value [1], h_out [1, 1, 512].
Inside: symbolic.py unpack_nodes (843 → Craftax-Symbolic's 8,268) → Dense 512 + relu → flax GRUCell 512 → actor 2 × (Dense 512 + relu)
→ 43 logits; critic the same → value. h is [layers = 1, 1, 512] so the brain view can sample it. Trained without an action mask; the
browser masks anyway (argmax of the masked logits).
Weights: a flax serialization (msgpack) of the params dict, {"params": {Dense_0..6, GRU_0: {GRUCell_0}}}; the network is ppo_rnn_net.py.
Ours will be on Hugging Face as neuralcrystal/craftax-ppo-rnn; until then pass the file with --weights (default weights/ppo-rnn/params.msgpack).
The GRU width is read from a config.json beside the weights ({"width": 512}) when there is one, else --width.
"""
import argparse, json, os, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from craftax_agents import WEIGHTS as WEIGHTS_DIR, OUT as OUT_DIR
from craftax_agents.symbolic import unpack, unpack_nodes

WEIGHTS = os.path.join(WEIGHTS_DIR, "ppo-rnn", "params.msgpack")
OUT = os.path.join(OUT_DIR, "ppo-rnn.onnx")
N_ACTIONS = 43


def load(weights=WEIGHTS, width=512):
    """params, width from a flax msgpack (and the config.json beside it, if any)."""
    from flax import serialization
    cfg = os.path.join(os.path.dirname(os.path.abspath(weights)), "config.json")
    if os.path.exists(cfg):
        width = json.load(open(cfg)).get("width", width)
    return serialization.msgpack_restore(open(weights, "rb").read()), width


def build(params, width):
    p = params["params"]; c = p["GRU_0"]["GRUCell_0"]; W = width
    inits, nodes = [], []
    def const(name, a):
        inits.append(numpy_helper.from_array(np.array(a, np.float32 if np.asarray(a).dtype.kind == "f" else None, order="C"), name)); return name
    def node(op, ins, outs=None, **kw):
        outs = outs or [f"{op.lower()}_{len(nodes)}"]
        nodes.append(helper.make_node(op, ins, outs, **kw)); return outs[0] if len(outs) == 1 else outs
    dense = lambda x, k, b: node("Add", [node("MatMul", [x, k]), b])
    i64 = lambda *v: np.array(v, np.int64)

    for i in range(7):
        const(f"d{i}_k", p[f"Dense_{i}"]["kernel"]); const(f"d{i}_b", p[f"Dense_{i}"]["bias"])
    # flax 0.12 GRUCell: r = σ(ir x + hr h), z = σ(iz x + hz h), n = tanh(in x + r ⊙ (hn h + b_hn)), h' = (1 − z) n + z h;
    # the input Denses (ir, iz, in) carry biases, of the hidden ones only hn. One MatMul each side, split three ways.
    const("gi_k", np.concatenate([c[k]["kernel"] for k in ("ir", "iz", "in")], 1)); const("gi_b", np.concatenate([c[k]["bias"] for k in ("ir", "iz", "in")]))
    const("gh_k", np.concatenate([c[k]["kernel"] for k in ("hr", "hz", "hn")], 1)); const("gh_b", np.concatenate([np.zeros(2 * W, np.float32), c["hn"]["bias"]]))
    const("split3", i64(W, W, W)); const("flat", i64(-1, W)); const("state", i64(1, 1, W)); const("ax1", i64(1))

    x = unpack_nodes(node, const, "obs")                                           # [1, 8268]
    e = node("Relu", [dense(x, "d0_k", "d0_b")])
    h = node("Reshape", ["h", "flat"])                                             # [1, 512]
    ir, iz, inn = node("Split", [dense(e, "gi_k", "gi_b"), "split3"], ["g_ir", "g_iz", "g_in"], axis=1)
    hr, hz, hn = node("Split", [dense(h, "gh_k", "gh_b"), "split3"], ["g_hr", "g_hz", "g_hn"], axis=1)
    r, z = node("Sigmoid", [node("Add", [ir, hr])]), node("Sigmoid", [node("Add", [iz, hz])])
    n = node("Tanh", [node("Add", [inn, node("Mul", [r, hn])])])
    y = node("Add", [n, node("Mul", [z, node("Sub", [h, n])])])                    # (1 − z) n + z h
    node("Reshape", [y, "state"], ["h_out"])
    a = node("Relu", [dense(node("Relu", [dense(y, "d1_k", "d1_b")]), "d2_k", "d2_b")])
    node("Add", [node("MatMul", [a, "d3_k"]), "d3_b"], ["logits"])
    v = node("Relu", [dense(node("Relu", [dense(y, "d4_k", "d4_b")]), "d5_k", "d5_b")])
    node("Squeeze", [dense(v, "d6_k", "d6_b"), "ax1"], ["value"])

    f = TensorProto.FLOAT
    g = helper.make_graph(nodes, "ppo-rnn",
                          [helper.make_tensor_value_info("obs", f, [1, 843]), helper.make_tensor_value_info("h", f, [1, 1, W])],
                          [helper.make_tensor_value_info("logits", f, [1, N_ACTIONS]), helper.make_tensor_value_info("value", f, [1]),
                           helper.make_tensor_value_info("h_out", f, [1, 1, W])], inits)
    # producer_name and doc_string are stored in the .onnx: changing them changes its sha256 (models.json), so they keep the
    # wording of the file the site serves.
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="craftax-agents export_ppo_rnn.py")
    m.ir_version = 8                                                               # onnxruntime-web reads IR 8 at opset 17
    m.doc_string = (f"PPO-RNN on Craftax-Symbolic (the Craftax paper's baseline, Matthews et al. 2024, retrained for "
                    f"1e9 steps), GRU {W}. obs = craftax-wasm cx_obs (843), h = [1, 1, {W}] zeros at an episode start.")
    onnx.checker.check_model(m)
    return m


def check(params, width, path, steps):
    """Greedy play on the native engine, the ONNX graph choosing; the original flax ActorCritic (T = 1) fed the same observations."""
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    import jax, jax.numpy as jnp, onnxruntime as ort
    from craftax_agents.ppo_rnn_net import ActorCritic                             # the training network, restated
    from craftax_agents.native import Game
    net = ActorCritic(N_ACTIONS, width)
    ref = jax.jit(lambda h, o: net.apply(params, h, (o[None], jnp.zeros((1, 1), bool))))
    g = Game(4 | 2); g.reset(7)                                                    # CX_ACTION_MASK + CX_KEEP_DEAD
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    h_o, h_f, worst, worst_v, agree, ep = np.zeros((1, 1, width), np.float32), jnp.zeros((1, width)), 0.0, 0.0, 0, 0
    for t in range(steps):
        obs, mask = g.obs()[None], g.mask()
        lo, vo, h_o = sess.run(None, {"obs": obs, "h": h_o})
        h_f, lf, vf = ref(h_f, jnp.asarray(unpack(obs)))
        lf = np.asarray(lf)[0]
        worst = max(worst, float(np.abs(lo - lf).max()), float(np.abs(h_o[0] - np.asarray(h_f)).max())); worst_v = max(worst_v, float(np.abs(vo - np.asarray(vf)[0]).max()))
        a = int(np.argmax(np.where(mask, lo[0], -np.inf)))
        agree += a == int(np.argmax(np.where(mask, lf[0], -np.inf)))
        h_f = jnp.asarray(h_o[0])                                                  # keep both on the same trajectory
        if g.step(a):
            print(f"  episode {ep}: score {g.score():.0f} after step {t + 1}"); ep += 1
            g.reset(7 + ep); h_o = np.zeros_like(h_o); h_f = jnp.zeros_like(h_f)
    print(f"{t + 1} steps: greedy action agrees {agree}/{t + 1}, max |ONNX − flax| logits/h {worst:.2e}, value {worst_v:.2e}, score {g.score():.0f}")
    assert worst < 1e-4 and agree == t + 1, "ONNX and flax disagree"


def reference(entry=None, weights=None):
    """For check_model.py --ref craftax_agents.export_ppo_rnn:reference: the flax module as step(obs [1, 843], {"h": [1, 1, W]}) → logits [1, 43].
    The weights: $PPO_RNN_WEIGHTS, else weights/ppo-rnn/params.msgpack."""
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    import jax, jax.numpy as jnp
    from craftax_agents.ppo_rnn_net import ActorCritic
    params, width = load(weights or os.environ.get("PPO_RNN_WEIGHTS", WEIGHTS)); net = ActorCritic(N_ACTIONS, width)
    f = jax.jit(lambda h, o: net.apply(params, h, (o[None], jnp.zeros((1, 1), bool)))[1][0])
    return lambda obs, st: np.asarray(f(jnp.asarray(st["h"][0]), jnp.asarray(unpack(obs))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS, help="the params as a flax msgpack (neuralcrystal/craftax-ppo-rnn on Hugging Face)")
    ap.add_argument("--width", type=int, default=512, help="GRU width, when no config.json sits beside the weights")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", type=int, default=0, help="steps of native-engine play to compare ONNX with the flax module (0 = skip)")
    a = ap.parse_args()
    if not os.path.exists(a.weights):
        sys.exit(f"{a.weights} missing: pass the PPO-RNN params with --weights")
    params, width = load(a.weights, a.width)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    onnx.save(build(params, width), a.out)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB")
    if a.check:
        check(params, width, a.out, a.check)
