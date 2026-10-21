"""Check a Craftax model's ONNX file against its registry entry (FORMAT.md), then play it on the native engine as the browser does.

    python -m craftax_agents.check_model out/ppo-rnn.onnx --id ppo-rnn                    # the entry from models.json
    python -m craftax_agents.check_model my.onnx --entry entry.json --steps 2000 --episodes 3   # an entry of your own (one JSON object)
    python -m craftax_agents.check_model my.onnx                                          # no entry: states guessed as X → X_out, an entry printed
    python -m craftax_agents.check_model out/ppo-rnn.onnx --id ppo-rnn --ref craftax_agents.export_ppo_rnn:reference   # also vs a reference step
    python -m craftax_agents.check_model out/ppo-rnn.onnx --id ppo-rnn --write-registry   # fill that entry's bytes + sha256 in models.json

Checks: inputs = "obs" float32 [1, 843] + the entry's state inputs (+ its "prev_action" input [1], when it declares one), outputs = "logits" [1, 43] + the state outputs (+ optional "value",
the brain output [layers, 1, units]), every shape and dtype; then plays --episodes episodes (at most --steps steps in all) with the engine's
action mask and argmax (or sampling at --temperature), states zeros at each episode start and fed back from their outputs, prev_action
the action picked last step (43 at an episode start). Fails on
NaN / Inf, on a state that never changes (not fed back?), on shapes that drift. Reports steps/s, each episode's score (cx_score,
Craftax's achievements) and how often each action was picked. --ref module:fn: fn(entry) → step(obs, states) → logits [1, 43], run on
the same trajectory (the ONNX states), max |diff| and argmax agreement reported. Needs the native engine (native.py says where it looks).
"""
import argparse, hashlib, importlib, json, os, re, sys, time
import numpy as np

from craftax_agents import REGISTRY
OBS, N_ACTIONS = 843, 43
try:
    from craftax.craftax.constants import Action
    NAMES = [a.name.lower() for a in Action]
except Exception:                                                                  # no craftax installed: numbers only
    NAMES = [str(i) for i in range(N_ACTIONS)]

problems = []
def bad(msg): problems.append(msg); print("  FAIL", msg)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return os.path.getsize(path), h.hexdigest()


def registry_entries(text):
    d = json.loads(text)
    return d["models"] if isinstance(d, dict) else d


def check_io(sess, entry):
    """Names, shapes, dtypes of the graph's inputs and outputs against the entry."""
    ins = {i.name: i for i in sess.get_inputs()}; outs = {o.name: o for o in sess.get_outputs()}
    def want(io, name, shape, what):
        if name not in io: return bad(f"{what} {name!r} missing (graph has {sorted(io)})")
        t = io[name]
        if t.type != "tensor(float)": bad(f"{what} {name!r} is {t.type}, want tensor(float)")
        if shape is not None and len(t.shape) != len(shape): return bad(f"{what} {name!r} shape {t.shape}, want {shape}")
        for d, w in zip(t.shape, shape or []):
            if isinstance(d, str) or d is None: print(f"  note {what} {name!r} has a symbolic dim {d!r} (fine: the browser feeds {shape})")
            elif d != w: bad(f"{what} {name!r} shape {t.shape}, want {shape}")
    if entry.get("obs", "packed843") != "packed843": bad(f"obs {entry.get('obs')!r}: the browser only gives packed843")
    want(ins, "obs", [1, OBS], "input"); want(outs, "logits", [1, N_ACTIONS], "output")
    for s in entry.get("state", []):
        want(ins, s["in"], s["shape"], "state input"); want(outs, s["out"], s["shape"], "state output")
    if entry.get("prev_action"): want(ins, entry["prev_action"], [1], "prev_action input")
    extra = set(ins) - {"obs", entry.get("prev_action")} - {s["in"] for s in entry.get("state", [])}
    if extra: bad(f"inputs {sorted(extra)} not in the entry's state list (or its prev_action): the browser would not feed them")
    if "value" in outs: want(outs, "value", None, "output")
    b = entry.get("brain")
    if b:
        o = outs.get(b["output"])
        if o is None: bad(f"brain output {b['output']!r} missing")
        elif len(o.shape) != 3 or o.shape[0] != b.get("layers", o.shape[0]):
            bad(f"brain output {b['output']!r} shape {o.shape}, want [layers={b.get('layers')}, 1, units]")
        elif o.shape[1] != 1: print(f"  note brain output {b['output']!r} is {o.shape}: the brain view reads [layers, 1, units]")


def guess_entry(sess, path):
    """No entry: an input named prev_action is the previous action (FORMAT.md); every other input but obs is a state, its output
    named X_out (or the one output left with its shape)."""
    outs = {o.name: o for o in sess.get_outputs()}; st = []; pa = {}
    for i in sess.get_inputs():
        if i.name == "prev_action": pa = {"prev_action": i.name}
        if i.name in ("obs", "prev_action"): continue
        o = i.name + "_out" if i.name + "_out" in outs else next((n for n, t in outs.items() if t.shape == i.shape and n not in ("logits", "value")), None)
        st.append({"in": i.name, "out": o, "shape": [d if isinstance(d, int) else 1 for d in i.shape]})
    return {"id": os.path.splitext(os.path.basename(path))[0], "obs": "packed843", "state": st, **pa}


def play(sess, entry, a, ref):
    from craftax_agents.native import Game
    st = entry.get("state", []); names = [o.name for o in sess.get_outputs()]; pa = entry.get("prev_action")
    zeros = lambda: {s["in"]: np.zeros(s["shape"], np.float32) for s in st}
    g = Game(4 | 2)                                                                # CX_ACTION_MASK + CX_KEEP_DEAD, sleeps fast-forwarded (the browser's engine)
    rng = np.random.default_rng(a.seed); counts = np.zeros(N_ACTIONS, np.int64); scores, lengths = [], []
    moved = {s["in"]: False for s in st}; t_run = 0.0; steps = 0; worst = 0.0; agree = 0; refd = 0
    for ep in range(a.episodes):
        g.reset(a.seed + ep); feeds = zeros(); n = 0; prev = N_ACTIONS                # 43: no previous action at an episode start
        while steps < a.steps:
            obs, mask = g.obs()[None], g.mask()
            if pa: feeds[pa] = np.array([prev], np.float32)
            t0 = time.perf_counter(); res = dict(zip(names, sess.run(None, dict(feeds, obs=obs)))); t_run += time.perf_counter() - t0
            for k, v in res.items():
                if not np.all(np.isfinite(v)): bad(f"output {k!r} not finite at episode {ep} step {n}"); return
            lo = res["logits"].reshape(-1)
            if lo.shape != (N_ACTIONS,): bad(f"logits shape {res['logits'].shape} at step {n}"); return
            m = np.where(mask, lo, -np.inf)
            if a.temperature > 0:
                p = np.exp((m - m.max()) / a.temperature); act = int(rng.choice(N_ACTIONS, p=p / p.sum()))
            else:
                act = int(np.argmax(m))
            if ref is not None:
                lr = np.asarray(ref(obs, feeds), np.float32).reshape(-1); worst = max(worst, float(np.abs(lr - lo).max()))
                agree += int(np.argmax(np.where(mask, lr, -np.inf))) == int(np.argmax(m)); refd += 1
            nxt = {}
            for s in st:
                v = res[s["out"]]
                if list(v.shape) != list(s["shape"]): bad(f"state {s['out']!r} came back {list(v.shape)}, want {s['shape']}"); return
                moved[s["in"]] |= not np.array_equal(v, feeds[s["in"]]); nxt[s["in"]] = v
            feeds = nxt; prev = act; counts[act] += 1; steps += 1; n += 1
            if g.step(act): break
        scores.append(g.score()); lengths.append(n)
        print(f"  episode {ep} (seed {a.seed + ep}): score {g.score():.0f}, {n} steps{'' if g.cx.cx_done(g.g) else ' (cut: --steps reached)'}")
        if steps >= a.steps: break
    for k, v in moved.items():
        if not v: bad(f"state {k!r} never changed: is its output fed back?")
    print(f"  {steps} steps, {steps / t_run:.0f} steps/s ONNX ({t_run / max(steps, 1) * 1e3:.2f} ms a step, CPU onnxruntime, batch 1)")
    print(f"  scores {[round(s) for s in scores]}, mean {np.mean(scores):.1f} over {len(scores)} episode(s) (a few local episodes: not an evaluation)")
    order = np.argsort(-counts); print("  actions: " + ", ".join(f"{NAMES[i]} {100 * counts[i] / steps:.1f}%" for i in order if counts[i]))
    if refd:
        print(f"  vs --ref: max |logits diff| {worst:.2e}, argmax agrees {agree}/{refd}")
        if agree != refd: bad("argmax differs from the reference")
    return scores


def write_registry(mid, size, sha):
    """Only this entry's "bytes" and "sha256", as a text edit of the file as it is now (another tool owns the rest)."""
    if not os.path.exists(REGISTRY): print(f"  {REGISTRY} not there: \"bytes\": {size}, \"sha256\": \"{sha}\""); return
    text = open(REGISTRY).read(); before = registry_entries(text)
    m = re.search(r'"id"\s*:\s*"%s"' % re.escape(mid), text)
    if not m: sys.exit(f"no entry {mid!r} in {REGISTRY}")
    lo = text.rindex("{", 0, m.start()); depth, i, q = 0, lo, False                # the entry's object: braces outside strings
    while True:
        c = text[i]
        if q: q = not (c == '"' and text[i - 1] != "\\")
        elif c == '"': q = True
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if not depth: break
        i += 1
    body = text[lo:i + 1]
    for k, v in (("bytes", str(size)), ("sha256", f'"{sha}"')):
        pat = r'("%s"\s*:\s*)("[^"]*"|-?\d+)' % k
        if re.search(pat, body): body = re.sub(pat, lambda mm: mm.group(1) + v, body, count=1)
        else: body = re.sub(r'("onnx"\s*:\s*"[^"]*")', lambda mm: f'{mm.group(1)}, "{k}": {v}', body, count=1)
    new = text[:lo] + body + text[i + 1:]
    after = registry_entries(new)                                                   # nothing else may differ
    strip = lambda es: [{k: v for k, v in e.items() if not (e.get("id") == mid and k in ("bytes", "sha256"))} for e in es]
    assert strip(before) == strip(after), "registry edit touched more than bytes/sha256"
    e = next(e for e in after if e.get("id") == mid); assert e["bytes"] == size and e["sha256"] == sha
    tmp = REGISTRY + ".tmp"; open(tmp, "w").write(new); os.replace(tmp, REGISTRY)
    print(f"  {REGISTRY}: {mid} bytes {size}, sha256 {sha}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--entry", help="a JSON file holding the model's registry entry")
    ap.add_argument("--id", help="read the entry from models.json")
    ap.add_argument("--steps", type=int, default=2000, help="most steps to play in all")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = masked argmax; > 0 samples softmax(logits / T) over the allowed actions")
    ap.add_argument("--ref", help="module:fn, fn(entry) → step(obs [1, 843], states {in: array}, + prev_action) → logits [1, 43]")
    ap.add_argument("--write-registry", action="store_true", help="fill this entry's bytes + sha256 in models.json")
    a = ap.parse_args()
    import onnxruntime as ort
    if a.entry: entry = json.load(open(a.entry))
    elif a.id: entry = next((e for e in registry_entries(open(REGISTRY).read()) if e.get("id") == a.id), None) or sys.exit(f"no {a.id!r} in {REGISTRY}")
    else: entry = None
    size, sha = file_hash(a.onnx)
    print(f"{a.onnx}: {size} bytes, sha256 {sha}")
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    if entry is None:
        entry = guess_entry(sess, a.onnx); print("  no entry given; guessed:", json.dumps(entry))
    print("interface"); check_io(sess, entry)
    for k, v in (("bytes", size), ("sha256", sha)):
        if entry.get(k) and entry[k] != v: print(f"  note entry {k} {entry[k]} != the file's {v}" + ("" if a.write_registry else " (--write-registry fixes it)"))
    if not problems:
        ref = None
        if a.ref:
            mod, _, fn = a.ref.partition(":"); ref = getattr(importlib.import_module(mod), fn)(entry)
        print("play (native engine)"); play(sess, entry, a, ref)
    if problems:
        sys.exit(f"{len(problems)} problem(s)")
    print("ok")
    if a.write_registry:
        write_registry(entry.get("id") or sys.exit("--write-registry needs an id"), size, sha)
    elif not a.id:
        print("registry fields:", json.dumps(dict(entry, bytes=size, sha256=sha)))
