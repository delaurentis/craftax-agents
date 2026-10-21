"""Bit levels and formats of one playable model (FORMAT.md): smaller downloads, and safetensors for PyTorch / JAX users.

    python -m craftax_agents.variants out/puffer162.onnx [--steps 2000] [--episodes 2]

From <id>.onnx (fp32) it writes, beside it:
    <id>-16.onnx         weights stored as float16, cast back to float32 when the session loads (half the download, same maths)
    <id>-8.onnx          weights stored as int8, a float16 scale per 32 inputs, dequantized when the session loads (~ a third)
    <id>.safetensors     the fp32 weights by their graph names, for PyTorch / JAX users (the ONNX graph says how they connect)
Only weights are stored lower: every model still computes in float32, so the browser runs all three the same way. Then it plays
on the native engine with the fp32 model choosing and the others run alongside: how often each one's greedy action agrees, and
the largest logit difference; and a few episodes with each one choosing, their scores (a few local episodes, not an evaluation).
State: every input but "obs" is recurrent, fed back from the output named <in>_out (h → h_out), zeros at an episode start; but an
input "prev_action" [1] gets the action taken last step (43 at an episode start), FORMAT.md.
"""
import argparse, hashlib, os, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN = 4096                                                                         # tensors smaller than this stay fp32 (biases, offsets)
GROUP = 32                                                                         # 8-bit: one scale per 32 inputs (a scale per whole column
                                                                                   # let PufferLib's growing MinGRU memory drift: 65 % agreement)


def lower(m, bits):
    """A copy of model m with its large float weights stored at 16 or 8 bits, restored to float32 by nodes at the graph's start."""
    m = onnx.ModelProto.FromString(m.SerializeToString())
    g, keep, front = m.graph, [], []
    for t in g.initializer:
        a = numpy_helper.to_array(t)
        if t.data_type != TensorProto.FLOAT or a.ndim < 2 or a.size < MIN:
            keep.append(t); continue
        if bits == 16:
            keep.append(numpy_helper.from_array(a.astype(np.float16), t.name + "_f16"))
            front.append(helper.make_node("Cast", [t.name + "_f16"], [t.name], to=TensorProto.FLOAT))
        else:                                                                      # symmetric, a float16 scale per GROUP inputs of each output
            n, rest = a.shape[0], a.shape[1:]
            pad = -n % GROUP
            b = np.concatenate([a, np.zeros((pad,) + rest, a.dtype)]).reshape((-1, GROUP) + rest)
            s = (np.abs(b).max(1, keepdims=True) / 127.0).astype(np.float16)
            s = np.where(s > 0, s, np.float16(1))
            q = np.clip(np.rint(b / s.astype(np.float32)), -127, 127).astype(np.int8)
            nm = t.name
            keep += [numpy_helper.from_array(q, nm + "_q"), numpy_helper.from_array(s, nm + "_s"),
                     numpy_helper.from_array(np.array((n + pad,) + rest, np.int64), nm + "_shape"),
                     numpy_helper.from_array(np.array([0], np.int64), nm + "_0"), numpy_helper.from_array(np.array([n], np.int64), nm + "_n")]
            w = helper.make_node("Mul", [nm + "_qf", nm + "_sf"], [nm + "_g"])
            front += [helper.make_node("Cast", [nm + "_q"], [nm + "_qf"], to=TensorProto.FLOAT),
                      helper.make_node("Cast", [nm + "_s"], [nm + "_sf"], to=TensorProto.FLOAT), w,
                      helper.make_node("Reshape", [nm + "_g", nm + "_shape"], [nm + "_p"]),
                      helper.make_node("Slice", [nm + "_p", nm + "_0", nm + "_n", nm + "_0"], [nm])]
    del g.initializer[:]; g.initializer.extend(keep)
    nodes = front + list(g.node)
    del g.node[:]; g.node.extend(nodes)
    m.doc_string += f" Weights stored at {bits} bits (craftax_agents.variants), computed in float32."
    onnx.checker.check_model(m)
    return m


def safetensors(m, path):
    from safetensors.numpy import save_file
    save_file({t.name: numpy_helper.to_array(t) for t in m.graph.initializer if t.data_type == TensorProto.FLOAT},
              path, metadata={"format": "craftax-agents playable model (FORMAT.md)", "doc": m.doc_string[:1000]})


class Player:
    def __init__(self, path):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.ins = [i for i in self.s.get_inputs() if i.name not in ("obs", "prev_action")]
        self.pa = any(i.name == "prev_action" for i in self.s.get_inputs())
        outs = [o.name for o in self.s.get_outputs()]
        self.outs = ["logits"] + [i.name + "_out" for i in self.ins]
        missing = [o for o in self.outs if o not in outs]
        if missing: sys.exit(f"{path}: no output {missing} (state input X must come back as X_out)")
        self.reset()
    def reset(self):
        self.h = {i.name: np.zeros([1 if not isinstance(d, int) else d for d in i.shape], np.float32) for i in self.ins}; self.prev = 43
    def took(self, a): self.prev = a                                               # the action played: the next step's prev_action
    def logits(self, obs):
        r = self.s.run(self.outs, {"obs": obs[None], **self.h, **({"prev_action": np.array([self.prev], np.float32)} if self.pa else {})})
        self.h = {i.name: v for i, v in zip(self.ins, r[1:])}
        return r[0][0]


def play(paths, steps, episodes, seed=7):
    from craftax_agents.native import Game
    greedy = lambda lo, mask: int(np.argmax(np.where(mask, lo, -np.inf)))
    ps = [Player(p) for p in paths]
    g = Game(4 | 2); g.reset(seed)
    agree, worst = np.zeros(len(ps), int), np.zeros(len(ps))
    for t in range(steps):                                                         # the fp32 model chooses; the others follow its trajectory
        obs, mask = g.obs(), g.mask()
        lo = [p.logits(obs) for p in ps]
        a = greedy(lo[0], mask)
        for k in range(len(ps)):
            agree[k] += greedy(lo[k], mask) == a; worst[k] = max(worst[k], float(np.abs(lo[k] - lo[0]).max())); ps[k].took(a)
        if g.step(a):
            g.reset(seed + t + 1); [p.reset() for p in ps]
    scores = []
    for p in ps:                                                                   # each one choosing, its own episodes
        sc = []
        for e in range(episodes):
            g = Game(4 | 2); g.reset(1000 + e); p.reset()
            for _ in range(100000):
                o = g.obs(); act = greedy(p.logits(o), g.mask()); p.took(act)
                if g.step(act): break
            sc.append(g.score())
        scores.append(sc)
    return agree / steps, worst, scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx"); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--episodes", type=int, default=2)
    a = ap.parse_args()
    base = a.onnx[:-5]
    m = onnx.load(a.onnx)
    out = {32: a.onnx}
    for bits in (16, 8):
        onnx.save(lower(m, bits), out.setdefault(bits, f"{base}-{bits}.onnx"))
    safetensors(m, base + ".safetensors")
    for f in list(out.values()) + [base + ".safetensors"]:
        print(f"{os.path.basename(f):28s} {os.path.getsize(f) / 1e6:6.1f} MB  sha256 {hashlib.sha256(open(f, 'rb').read()).hexdigest()}")
    agree, worst, scores = play([out[32], out[16], out[8]], a.steps, a.episodes)
    for bits, ag, w, sc in zip((32, 16, 8), agree, worst, scores):
        print(f"{bits:2d} bit: greedy agrees with fp32 {ag:.1%}, max |Δlogit| {w:.2e}, scores {' '.join(f'{s:.0f}' for s in sc)}")
