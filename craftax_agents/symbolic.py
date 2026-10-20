"""The engine's packed observation (cx_obs, 843 floats) back to Craftax-Symbolic's 8,268: the inverse of puffer_net.py pack(),
exact for every observation the engine makes. Models trained on JAX Craftax-Symbolic (the paper's PPO-RNN, TransformerXL...) read
the 8,268; the browser hands every model the 843, so their ONNX graphs start with unpack_nodes().

    sym = unpack(packed)                                   # NumPy, [..., 843] → [..., 8268]
    x = unpack_nodes(node, const, "obs")                   # the same as ONNX nodes inside a graph builder (export_onnx.py style)
    python -m craftax_agents.symbolic --check 64           # pack(unpack(p)) == p and unpack(pack(o)) == o on JAX Craftax worlds
"""
import numpy as np

from craftax_agents.puffer_net import OBS_ROWS, OBS_COLS, CHANNELS, N_BLOCKS, N_ITEMS, N_MOB_CLASSES, N_MOB_TYPES, N_SCALARS, TILE, pack

CELLS = OBS_ROWS * OBS_COLS
IDS = CELLS * CHANNELS                                                             # 792 tile ids, then the 51 scalars


def unpack(packed):
    """[..., 843] → [..., 8268]: per tile one-hot block, item, mob type per class (all masked by light, as the renderer does), light."""
    p = np.asarray(packed, np.float32)
    lead = p.shape[:-1]
    c = np.rint(p[..., :IDS]).astype(np.int64).reshape(lead + (CELLS, CHANNELS))
    vis = c[..., 2:3].astype(np.float32)
    blocks = (c[..., 0:1] == np.arange(N_BLOCKS)) * vis
    items = c[..., 1:2] - 1 == np.arange(N_ITEMS)                                  # 0 = dark: no item at all
    mobs = (c[..., 3:, None] - 1 == np.arange(N_MOB_TYPES)).reshape(lead + (CELLS, N_MOB_CLASSES * N_MOB_TYPES))
    t = np.concatenate([blocks, items, mobs, vis], -1).astype(np.float32)
    return np.concatenate([t.reshape(lead + (CELLS * TILE,)), p[..., IDS:]], -1)


def unpack_nodes(node, const, obs):
    """ONNX nodes for unpack(): obs [B, 843] → [B, 8268]. node(op, ins, **attrs) → output name; const(name, array) → name."""
    i64 = lambda *v: np.array(v, np.int64)
    const("u_s0", i64(0)); const("u_ids", i64(IDS)); const("u_end", i64(IDS + N_SCALARS)); const("u_ax1", i64(1)); const("u_one", i64(1))
    const("u_shape", i64(-1, CELLS, CHANNELS)); const("u_flat", i64(-1, CELLS * TILE)); const("u_mobs", i64(-1, CELLS, N_MOB_CLASSES * N_MOB_TYPES))
    const("u_blocks", np.arange(N_BLOCKS, dtype=np.int64)); const("u_items", np.arange(N_ITEMS, dtype=np.int64))
    const("u_types", np.arange(N_MOB_TYPES, dtype=np.int64))
    for k in (0, 1, 2, 3, CHANNELS):
        const(f"u_c{k}", i64(k))
    ids = node("Reshape", [node("Slice", [obs, "u_s0", "u_ids", "u_ax1"]), "u_shape"])
    c = node("Cast", [node("Round", [ids])], to=7)                                 # int64
    const("u_axm1", i64(-1))
    ch = lambda a, b: node("Slice", [c, f"u_c{a}", f"u_c{b}", "u_axm1"])           # channels a..b-1 of each tile
    vis = node("Cast", [ch(2, 3)], to=1)
    hot = lambda x, ar: node("Cast", [node("Equal", [x, ar])], to=1)
    blocks = node("Mul", [hot(ch(0, 1), "u_blocks"), vis])
    items = hot(node("Sub", [ch(1, 2), "u_one"]), "u_items")
    mobs = node("Reshape", [hot(node("Unsqueeze", [node("Sub", [ch(3, 8), "u_one"]), "u_axm1"]), "u_types"), "u_mobs"])
    t = node("Concat", [blocks, items, mobs, vis], axis=-1)
    return node("Concat", [node("Reshape", [t, "u_flat"]), node("Slice", [obs, "u_ids", "u_end", "u_ax1"])], axis=1)


def _check(n):
    import os
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    import jax, onnx, onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    step, reset = jax.jit(env.step), jax.jit(env.reset)
    rng = jax.random.PRNGKey(0)
    obs, st = reset(rng, env.default_params)
    seen = []
    for i in range(n):                                                              # random play: dark tiles, mobs, items, floors
        rng, k1, k2 = jax.random.split(rng, 3)
        obs, st, *_ = step(k1, st, jax.random.randint(k2, (), 0, 43), env.default_params)
        seen.append(np.asarray(obs))
    o = np.stack(seen)
    p = pack(o)
    assert np.array_equal(unpack(p), o), "unpack(pack(o)) != o"
    assert np.array_equal(pack(unpack(p)), p), "pack(unpack(p)) != p"
    inits, nodes = [], []
    def const(name, a): inits.append(numpy_helper.from_array(np.array(a, order="C"), name)); return name
    def node(op, ins, outs=None, **kw):
        outs = outs or [f"{op.lower()}_{len(nodes)}"]; nodes.append(helper.make_node(op, ins, outs, **kw)); return outs[0]
    y = unpack_nodes(node, const, "obs")
    nodes.append(helper.make_node("Identity", [y], ["sym"]))
    g = helper.make_graph(nodes, "unpack", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["B", IDS + N_SCALARS])],
                          [helper.make_tensor_value_info("sym", TensorProto.FLOAT, ["B", CELLS * TILE + N_SCALARS])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
    onnx.checker.check_model(m)
    got = ort.InferenceSession(m.SerializeToString()).run(None, {"obs": p})[0]
    assert np.array_equal(got, o), "ONNX unpack != JAX obs"
    print(f"unpack ok: {n} JAX observations, NumPy and ONNX exact; {int((o[:, :CELLS * TILE].reshape(n, CELLS, TILE)[..., -1] == 0).sum())} dark tiles")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--check", type=int, default=64); _check(ap.parse_args().check)
