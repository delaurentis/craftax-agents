"""The Craftax paper's PPO-RNN network in flax (Matthews et al. 2024; Craftax_Baselines ppo_rnn.py, MIT): the module the
neuralcrystal/craftax-ppo-rnn weights were trained with, here only to check export_ppo_rnn.py's ONNX graph against it.

    Dense 512 + relu → GRU (flax GRUCell 512, reset to zeros where an episode begins) → actor 2 × (Dense 512 + relu) → 43 logits,
    critic the same → value.

    net = ActorCritic(43, 512)
    h, logits, value = net.apply(params, h [B, 512], (obs [T, B, 8268], resets [T, B] bool))

The parameter names (Dense_0..6, GRU_0/GRUCell_0) are the checkpoint's, so the layers must be created in this order. The
initializers of the training code are left out: this module only runs restored params.
"""
from functools import partial

import jax.numpy as jnp
import flax.linen as nn


class GRU(nn.Module):                                                              # scanned over time; the carry resets where an episode began
    @partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, h, x):
        ins, resets = x
        h = jnp.where(resets[:, None], jnp.zeros_like(h), h)
        h, y = nn.GRUCell(features=h.shape[-1])(h, ins)
        return h, y


class ActorCritic(nn.Module):
    n_actions: int
    width: int = 512

    @nn.compact
    def __call__(self, h, x):                                                      # x = (obs [T, B, D], resets [T, B])
        obs, resets = x
        e = nn.relu(nn.Dense(self.width)(obs))
        h, y = GRU()(h, (e, resets))
        a = nn.relu(nn.Dense(self.width)(y))
        a = nn.relu(nn.Dense(self.width)(a))
        logits = nn.Dense(self.n_actions)(a)
        c = nn.relu(nn.Dense(self.width)(y))
        c = nn.relu(nn.Dense(self.width)(c))
        v = nn.Dense(1)(c)
        return h, logits, v[..., 0]
