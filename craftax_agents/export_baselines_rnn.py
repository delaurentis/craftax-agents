"""A Craftax_Baselines PPO-RNN checkpoint (Orbax) as an ONNX graph for the browser: the ReMDM paper's Craftax-Symbolic-v1 expert.

    scripts/fetch_remdm.sh                                           # checkpoint + config + their model code → weights/remdm/
    python -m craftax_agents.export_baselines_rnn                    # → out/remdm-ppo-rnn.onnx (~27.5 MB)
    python -m craftax_agents.export_baselines_rnn --check 400        # + 400 native-engine steps, ONNX vs their flax module

The weights: huggingface.co/mathisweil/remdm-craftax-checkpoints @5b66c77 (MIT), checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M,
step 1e9: the PPO-RNN expert of "Return-Weighted ELBO Fine-Tuning Degrades Masked Diffusion Planners" (github.com/mathisweil/craftax-ReMDM-planner),
trained with their Craftax_Baselines fork (github.com/mathisweil/Craftax_Baselines @3aa6716, ppo_rnn.py; MIT, Michael Matthews).
Their ActorCriticRNN is the Craftax paper's: Dense 512 + relu → ScannedRNN (flax GRUCell 512) → actor 2 × (Dense 512 + relu) → 43,
critic the same → 1. config.yaml says ACTIVATION: tanh, but ppo_rnn.py never reads it: the layers are relu (nn.relu hard-coded).
That is exactly export_ppo_rnn.py's graph, so this file only renames the params (ScannedRNN_0/GRUCell_1 → GRU_0/GRUCell_0; the
_1 because ScannedRNN.initialize_carry makes a GRUCell first) and reuses its build(). Contract (FORMAT.md): obs [1, 843], h [1, 1, 512]
→ logits [1, 43], value [1], h_out [1, 1, 512]; h zeros at an episode start (their ScannedRNN zeroes it on done). The reference below is
their ScannedRNN + ActorCriticRNN restated line for line (their module imports wandb / distrax / their wrappers; the Categorical is
dropped, its logits returned), fed done = True at each episode start and its own hidden state, never the ONNX one.
"""
import argparse, functools, os, sys
import numpy as np
import onnx

from craftax_agents import WEIGHTS, OUT as OUT_DIR, export_ppo_rnn
from craftax_agents.symbolic import unpack

CKPT = os.path.join(WEIGHTS, "remdm", "Craftax-Symbolic-v1-PPO_RNN-1000M")
OUT = os.path.join(OUT_DIR, "remdm-ppo-rnn.onnx")
N_ACTIONS, WIDTH = 43, 512


def load(ckpt=CKPT):
    """Orbax restore of the step-1e9 checkpoint's params only (the Adam moments stay on disk): {"params": {Dense_0..6, ScannedRNN_0}}."""
    import jax, orbax.checkpoint as ocp
    path = os.path.join(ckpt, "1000000000", "default")
    meta = ocp.StandardCheckpointer().metadata(path)
    tree = meta.item_metadata.tree if hasattr(meta, "item_metadata") else meta
    tm = jax.tree_util.tree_map
    target = {"params": tm(lambda m: np.zeros(m.shape, m.dtype), tree["params"])}
    with ocp.PyTreeCheckpointer() as c:                                            # as NumPy arrays: no device sharding needed
        r = c.restore(path, args=ocp.args.PyTreeRestore(item=target, restore_args=tm(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), target), partial_restore=True))
    return tm(lambda a: np.asarray(a, np.float32), r["params"])


def as_ours(params):
    """Their param names → export_ppo_rnn.build's (same arrays)."""
    p = dict(params["params"]); p["GRU_0"] = {"GRUCell_0": p.pop("ScannedRNN_0")["GRUCell_1"]}
    return {"params": p}


def build(params):
    m = export_ppo_rnn.build(as_ours(params), WIDTH)
    # stored in the .onnx: changing producer_name or doc_string changes the file's sha256
    m.graph.name = "remdm-ppo-rnn"; m.producer_name = "craftax-agents export_baselines_rnn.py"
    m.doc_string = ("PPO-RNN expert of the ReMDM paper (Return-Weighted ELBO Fine-Tuning Degrades Masked Diffusion Planners), MIT: "
                    "huggingface.co/mathisweil/remdm-craftax-checkpoints @5b66c77 checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M, "
                    "Craftax_Baselines ppo_rnn.py (GRU 512, relu MLPs, 1e9 steps). obs = craftax-wasm cx_obs (843), h = [1, 1, 512] zeros at an episode start.")
    onnx.checker.check_model(m)
    return m


def their_module():
    """Craftax_Baselines ppo_rnn.py @3aa6716 (MIT, Copyright (c) 2024 Michael Matthews; adapted from purejaxrl, Chris Lu): ScannedRNN +
    ActorCriticRNN as written there, the initialisers dropped (restored params only) and distrax.Categorical(logits=·) → the logits."""
    import jax, jax.numpy as jnp, flax.linen as nn

    class ScannedRNN(nn.Module):
        @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
        @nn.compact
        def __call__(self, carry, x):
            rnn_state = carry
            ins, resets = x
            rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(ins.shape[0], ins.shape[1]), rnn_state)
            new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
            return new_rnn_state, y

        @staticmethod
        def initialize_carry(batch_size, hidden_size):
            cell = nn.GRUCell(features=hidden_size)
            return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

    class ActorCriticRNN(nn.Module):
        action_dim: int
        config: dict

        @nn.compact
        def __call__(self, hidden, x):
            obs, dones = x
            embedding = nn.relu(nn.Dense(self.config["LAYER_SIZE"])(obs))
            hidden, embedding = ScannedRNN()(hidden, (embedding, dones))
            actor_mean = nn.relu(nn.Dense(self.config["LAYER_SIZE"])(embedding))
            actor_mean = nn.relu(nn.Dense(self.config["LAYER_SIZE"])(actor_mean))
            actor_mean = nn.Dense(self.action_dim)(actor_mean)
            critic = nn.relu(nn.Dense(self.config["LAYER_SIZE"])(embedding))
            critic = nn.relu(nn.Dense(self.config["LAYER_SIZE"])(critic))
            critic = nn.Dense(1)(critic)
            return hidden, actor_mean, jnp.squeeze(critic, axis=-1)

    return ActorCriticRNN(N_ACTIONS, {"LAYER_SIZE": WIDTH})


def reference_step(params):
    """jit'd step(hidden [1, 512], sym [1, 8268], done bool) → hidden, logits [1, 43], value [1]; asserts the param tree matches theirs."""
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    import jax, jax.numpy as jnp
    net = their_module()
    shapes = jax.eval_shape(lambda: net.init(jax.random.PRNGKey(0), jnp.zeros((1, WIDTH)), (jnp.zeros((1, 1, 8268)), jnp.zeros((1, 1), bool))))
    want = jax.tree_util.tree_map(lambda s: s.shape, shapes); got = jax.tree_util.tree_map(np.shape, params)
    assert want == got, f"param tree differs from their module: {want} vs {got}"
    f = jax.jit(lambda h, o, d: net.apply(params, h, (o[None], jnp.full((1, 1), d))))
    return lambda h, sym, done: (lambda r: (r[0], np.asarray(r[1])[0], np.asarray(r[2])[0]))(f(h, jnp.asarray(sym), done))


def check(params, path, steps):
    """Greedy play on the native engine, the ONNX graph choosing; their module fed the same observations, keeping its own hidden state."""
    import onnxruntime as ort
    from craftax_agents.native import Game
    ref = reference_step(params)
    g = Game(4 | 2); g.reset(7)                                                    # CX_ACTION_MASK + CX_KEEP_DEAD
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    h_o, h_f, start, worst, worst_h, worst_v, agree, ep = np.zeros((1, 1, WIDTH), np.float32), np.zeros((1, WIDTH), np.float32), True, 0.0, 0.0, 0.0, 0, 0
    for t in range(steps):
        obs, mask = g.obs()[None], g.mask()
        lo, vo, h_o = sess.run(None, {"obs": obs, "h": h_o})
        h_f, lf, vf = ref(h_f, unpack(obs), start); start = False
        worst = max(worst, float(np.abs(lo - lf).max())); worst_h = max(worst_h, float(np.abs(h_o[0] - np.asarray(h_f)).max()))
        worst_v = max(worst_v, float(np.abs(vo - vf).max()))
        a = int(np.argmax(np.where(mask, lo[0], -np.inf)))
        agree += a == int(np.argmax(np.where(mask, lf[0], -np.inf)))
        if g.step(a) or t + 1 == steps // 2:                                       # a new episode halfway (or on death): their done path
            print(f"  episode {ep}: score {g.score():.0f} after step {t + 1}"); ep += 1
            g.reset(7 + ep); h_o = np.zeros_like(h_o); start = True                # their hidden NOT zeroed here: done = True must do it
    print(f"{steps} steps ({ep} new episodes): greedy action agrees {agree}/{steps}, max |ONNX − flax| logits {worst:.2e}, h {worst_h:.2e}, value {worst_v:.2e}")
    return worst, agree == steps


def reference(entry=None):
    """For check_model.py --ref craftax_agents.export_baselines_rnn:reference: their module as step(obs [1, 843], {"h": [1, 1, 512]}) → logits [1, 43]."""
    f = reference_step(load())
    return lambda obs, st: f(np.asarray(st["h"][0]), unpack(obs), False)[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT, help="a Craftax_Baselines PPO-RNN run dir: <step>/default (Orbax) + config.yaml")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", type=int, default=0, help="steps of native-engine play to compare ONNX with their flax module (0 = skip)")
    a = ap.parse_args()
    if not os.path.isdir(os.path.join(a.ckpt, "1000000000", "default")):
        sys.exit("checkpoint missing: run scripts/fetch_remdm.sh")
    params = load(a.ckpt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    onnx.save(build(params), a.out)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB")
    if a.check:
        worst, same = check(params, a.out, a.check)
        if worst > 1e-4 or not same:
            sys.exit("ONNX and flax disagree")
