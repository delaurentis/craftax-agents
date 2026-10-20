# The playable format

[playcraftax.com](https://playcraftax.com) plays agents as ONNX models (onnxruntime-web, in a worker). If you trained a Craftax agent, in PyTorch, JAX or anything else, you can turn its weights into one ONNX file and add it to the game's agent list. This page is the contract; [README.md](README.md) has the quick start and [how to list your agent](README.md#add-your-agent).

## What the browser gives your model, and what it wants back

One ONNX file per model, **batch 1**, float32. One call = one agent step.

| | name | shape | what |
|---|---|---|---|
| input | `obs` | `[1, 843]` | the engine's packed observation (`cx_obs`): the 9 × 11 view, 8 ids a tile (block, item + 1, visible, then type + 1 of melee / passive / ranged mob / mob projectile / player projectile, 0 = none), then Craftax's 51 inventory and stat floats as they are. Exact layout: `craftax_agents/puffer_net.py` (docstring and `pack()`). |
| input | your state(s) | any | recurrent state: as many float32 inputs as you like, each **fed back from a named output** at the next step, **all zeros** at an episode start. |
| output | `logits` | `[1, 43]` | Craftax's 43 actions, in Craftax's order. |
| output | `value` | `[1]` | optional. |
| output | your state output(s) | same shape as the input | the next step's state. |

The browser masks `logits` with the engine's action mask (PufferLib's: actions that cannot do anything now are off) and takes the argmax, or samples at a temperature. A model trained without a mask works fine.

**Trained on JAX Craftax-Symbolic (8,268 floats)?** Start your graph with `craftax_agents/symbolic.py` `unpack_nodes()`: it turns the 843 back into the exact 8,268 floats JAX gave you (checked bit-exact against JAX Craftax; `python -m craftax_agents.symbolic --check 64`). In NumPy the same thing is `unpack()`. Pixel observations are not supported.

**Sleep.** The engine plays a whole sleep / rest as ONE agent step (as PufferLib trains; craftax-wasm's `tests/parity.py` explains), where JAX Craftax asks the agent every tick while asleep. A recurrent model trained on JAX simply sees fewer steps while asleep. That is fine in practice: those steps carry no choice (every action is a no-op while asleep).

## The registry entry

Each model is one entry in `models.json`:

```json
{
  "id": "ppo-rnn", "name": "PPO-RNN (Craftax paper)",
  "about": "one line for the picker", "by": "who trained it", "license": "MIT",
  "source": "where the weights / code come from (URL)",
  "score": "a score line, saying where it comes from",
  "onnx": "agent/ppo-rnn.onnx", "bytes": 27535422, "sha256": "a7e57b78...",
  "obs": "packed843",
  "state": [{ "in": "h", "out": "h_out", "shape": [1, 1, 512] }],
  "brain": { "output": "h_out", "layers": 1 }
}
```

- `onnx`: an https URL (the agents the game builds itself use paths relative to its page, like `agent/ppo-rnn.onnx`) (it must answer with CORS headers so the page can fetch it). `bytes` and `sha256` are the file's (`check_model.py` prints them).
- `state`: one item per recurrent input; `[]` for a feed-forward model.
- `brain` (optional): an output shaped `[layers, 1, units]` that the brain view samples and draws (a GRU's `h_out` is ideal).

## From PyTorch

Wrap the policy so `forward(obs, h)` returns `(logits, value, h_out)` for one step, then:

```python
torch.onnx.export(Step(policy).eval(), (torch.zeros(1, 843), torch.zeros(1, 1, 512)), "my-model.onnx",
                  input_names=["obs", "h"], output_names=["logits", "value", "h_out"], opset_version=17, dynamo=False)
```

- No dynamic axes needed: batch is 1. An `nn.GRU` / `nn.LSTM` state `[layers, 1, hidden]` already has the brain view's shape (an LSTM has two state inputs, `h` and `c`: two `state` items).
- If your network ate the 8,268 symbolic floats, do the unpacking inside `forward` (port `craftax_agents/symbolic.py unpack()` to torch: one-hots of the 843's integer ids), or build the ONNX with `unpack_nodes()` in front, as below.
- One file: keep the weights inside the .onnx (under 2 GB; don't let the exporter split them into `.onnx.data`).

## From JAX / Flax

There is no first-party JAX → ONNX exporter. What works:

1. **Build the graph by hand** (what we do; recommended for MLPs, GRUs, small transformers). The ops are a handful (`MatMul`, `Add`, `Relu`, `Sigmoid`, `Tanh`, `Split`, `Reshape`), the weights come straight from your params pytree. Copy `craftax_agents/export_ppo_rnn.py`: it reads a flax `msgpack`, builds `unpack_nodes → Dense → GRUCell → heads` with the `node()` / `const()` helpers in ~40 lines, and checks the result against the original flax module on the native engine. Mind the exact cell equations (flax's GRUCell puts its hidden-side bias only on `hn`, inside `r ⊙ (·)`; PyTorch's differs).
2. **jax2onnx** (third-party, converts jaxprs and flax modules): worth a try for standard layers; check the result with `check_model.py --ref`. We have not used it.
3. **jax2tf → tf2onnx**: possible for plain ops but needs TensorFlow and leaves a noisy graph; a last resort.

Whatever the route: opset 17 and `model.ir_version = 8` (what our files use; onnxruntime-web 1.30 runs them).

## Check it

```
python -m craftax_agents.check_model my-model.onnx --entry my-entry.json --episodes 3          # or --id <id> for an entry already in models.json
python -m craftax_agents.check_model my-model.onnx                                             # no entry yet: guesses one (state X → X_out) and prints it
python -m craftax_agents.check_model my-model.onnx --entry my-entry.json --ref mymod:reference # also compare with your framework, step by step
```

It checks the inputs and outputs against the entry (names, shapes, dtypes), then plays on the native engine as the browser does (masked argmax, state zeros at an episode start and fed back): fails on NaN / Inf or a state that never changes, and reports steps a second, each episode's score (`cx_score`: Craftax's achievement points, out of 226) and how often each action was picked. `--ref module:fn`: `fn(entry)` returns `step(obs [1, 843], states {name: array}) → logits [1, 43]` in your own framework; the checker runs it on the same trajectory and reports the largest logit difference and argmax agreement (aim for ~1e-4 and all agree). It prints `bytes` and `sha256` for the entry. Needs the native engine (see the README's [setup](README.md#setup); Python: `numpy onnx onnxruntime`).

## Add it to the game

Test it in your browser, then open a pull request adding its entry to `models.json`: [README.md, Add your agent](README.md#add-your-agent).
