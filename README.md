# craftax-agents

Bring your [Craftax](https://github.com/MichaelTMatthews/Craftax) agent to the browser, and let users play with it on [playcraftax.com](https://playcraftax.com) 

Trained an agent for Craftax in PyTorch, JAX or anything else? Turn it into an ONNX file, and anyone can watch it play or use it for assistance on [playcraftax.com](https://playcraftax.com).

This kit holds everything that takes:

- **The playable format** ([FORMAT.md](FORMAT.md)): one ONNX file, the engine's 843-float observation in, Craftax's 43 action logits out, any memory fed back from step to step.
- **Converters** for the agents the game already offers: PufferLib's 162 policy, the Craftax paper's PPO-RNN, and Gautier Hamon's Transformer-XL PPO. Each one is checked step by step against the original code. Copy one to convert yours.
- **The checker** (`check_model.py`): it plays your file on [craftax-wasm](https://github.com/delaurentis/craftax-wasm) the way the browser does, and prints its registry entry.
- **The registry** ([`models.json`](models.json)): the agents playcraftax.com lists. To add yours, open a pull request here.

Craftax is by Michael Matthews et al. (ICML 2024). The engine the agents play on is a fork of PufferLib's C port of Craftax. The agents belong to their authors. See [Credits](#credits).

## Setup

```sh
git clone --recursive https://github.com/delaurentis/craftax-agents
cd craftax-agents
make -C engine native              # the native engine, engine/build/libcraftax.{dylib,so} (a C compiler, nothing else)
pip install -e .                   # numpy, onnx, onnxruntime
pip install -e ".[jax]"            # optional: jax, flax and craftax, for the JAX converters' checks
```

`engine/` is [craftax-wasm](https://github.com/delaurentis/craftax-wasm) as a git submodule (in a clone made without `--recursive`, run `git submodule update --init`). The kit looks for the native library in this order:

1. `$CRAFTAX_ENGINE_LIB`: the library file itself, or the folder that holds it
2. `engine/build/`: the submodule, built with `make -C engine native`
3. `../engine/build/`: a craftax-wasm checkout next to this one

Run the commands below from the kit's folder. Downloaded checkpoints go to `weights/` and exported models to `out/`. Neither is in git.

## Quick start

Pull a sample model (PufferLib's, one of the best Craftax agents), convert it, check it and play it:

```sh
scripts/fetch_puffer162.sh                                        # 57 MB of weights → weights/craftax_1024x4.bin
python -m craftax_agents.export_onnx --check 200                  # → out/puffer162.onnx, then 200 steps comparing ONNX with NumPy
python -m craftax_agents.check_model out/puffer162.onnx --id puffer162 --episodes 1
```

## The agents

| id | Agent | By |
|---|---|---|
| `puffer162` | PPO + 4 MinGRU layers of 1024 (14M parameters). 162 / 226 in PufferLib's own evaluation; reaches the fire realm. | PufferAI ([PufferLib 5.0](https://github.com/PufferAI/PufferLib)) |
| `ppo-rnn` | PPO with one GRU of 512 (6.9M parameters), trained for 1e9 steps | Reproduction of the PPO-RNN baseline from Matthews et al.'s Craftax paper ([Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines)) |
| `trxl` | PPO with a gated Transformer-XL (2 layers of 256, memory 128, 4.7M parameters) | Gautier Hamon ([transformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX)) |

`variants.py` writes any playable model at 16 and 8 bits (smaller downloads, the same float32 maths) and as safetensors.

## Convert your agent

The full contract is in [FORMAT.md](FORMAT.md). In short:

| | name | shape |
|---|---|---|
| input | `obs` | `[1, 843]` the engine's packed observation |
| input | your memory, any name `X` | any shape, all zeros at an episode start |
| output | `logits` | `[1, 43]` Craftax's actions, in Craftax's order |
| output | `X_out` | the next step's `X` |
| output | `value` | `[1]`, optional |

Opset 17, IR version 8, float32, batch 1, and one file under 2 GB. If your agent was trained on JAX Craftax-Symbolic (8,268 floats), start the graph with `symbolic.py`'s `unpack_nodes()`. It turns the 843 back into those exact 8,268 floats.

- **PyTorch:** wrap the policy so that `forward(obs, h)` returns `(logits, value, h_out)` for one step, then call `torch.onnx.export`. [FORMAT.md](FORMAT.md#from-pytorch) has the call.
- **JAX / Flax:** there is no first-party exporter. Build the graph by hand from your params. It takes a handful of ONNX ops, and `export_ppo_rnn.py` does it in about 40 lines for a Dense → GRU → heads network. `export_trxl.py` does the same for a Transformer-XL. [FORMAT.md](FORMAT.md#from-jax--flax) lists the other routes.

## Check it

```sh
python -m craftax_agents.check_model your.onnx --episodes 3
```

The checker plays your file on the native engine exactly as the browser does, and fails on wrong inputs or outputs, NaN, or memory that never changes. It reports each episode's score (out of 226) and prints a registry entry with your file's `bytes` and `sha256`. Add `--ref module:fn` to compare its logits with your own framework, step by step.

## Test it on playcraftax.com

Host the `.onnx` where a browser can fetch it. Hugging Face works, as does a GitHub release asset or any https URL that sends CORS headers. On [playcraftax.com](https://playcraftax.com), open the bot window, choose **Test your agent** in its model list and paste the link. The agent then plays in your own browser, and the link stays there: nothing is uploaded.

## Add your agent

To list your agent for everyone, open a pull request to this repository that adds one entry to the `"models"` array in [`models.json`](models.json). Put it after the entries already there:

```json
{
  "id": "my-agent",
  "name": "My Agent",
  "about": "6.9M parameters · one GRU, trained 1e9 steps",
  "by": "you or your team",
  "license": "MIT",
  "source": "https://github.com/you/my-agent",
  "score": "41 / 226 (3 local episodes)",
  "onnx": "https://huggingface.co/you/my-agent/resolve/main/my-agent.onnx",
  "bytes": 27535422,
  "sha256": "…",
  "obs": "packed843",
  "state": [{ "in": "h", "out": "h_out", "shape": [1, 1, 512] }],
  "brain": { "output": "h_out", "layers": 1 }
}
```

| Field | What to put |
|---|---|
| `id` | short, lowercase, unique in the file |
| `name` | the name players see in the bot panel |
| `about` | one short line: parameter count and a few words on the network |
| `by` | who trained it: you, your team, or the paper you reproduced |
| `license` | the weights' license (MIT, Apache-2.0, CC-BY-4.0…); it must let the site serve them to players |
| `source` | where people can read more: your code, paper or model page |
| `score` | the score and where it comes from: your own evaluation, or "N local episodes" from the checker |
| `onnx` | the https link from the test step |
| `bytes`, `sha256` | exactly what `check_model.py` printed for that file; if the file changes, these change too |
| `obs` | always `packed843` for now |
| `state` | one item per memory input, with its output and shape (`[]` for a network without memory) |
| `brain` | optional: an output shaped `[layers, 1, units]` that the game's brain view draws |

In the PR description, paste the output of `python -m craftax_agents.check_model your.onnx --entry your-entry.json --episodes 3`.

## What's here

| Path | What it does |
|---|---|
| `FORMAT.md` | the playable format: inputs, outputs, the registry entry, the PyTorch and JAX routes |
| `models.json` | the registry of agents playcraftax.com offers |
| `craftax_agents/check_model.py` | the checker |
| `craftax_agents/native.py` | the native engine through ctypes |
| `craftax_agents/symbolic.py` | the 843-float observation back to Craftax-Symbolic's 8,268, in NumPy and as ONNX nodes (`--check` compares with JAX Craftax) |
| `craftax_agents/puffer_net.py` | PufferLib 5.0's Craftax network, observation packing and action mask, in NumPy |
| `craftax_agents/export_onnx.py` | PufferLib's 162 policy → ONNX |
| `craftax_agents/export_ppo_rnn.py`, `ppo_rnn_net.py` | the Craftax paper's PPO-RNN (flax msgpack) → ONNX |
| `craftax_agents/export_baselines_rnn.py` | a Craftax_Baselines PPO-RNN (Orbax checkpoint) → ONNX |
| `craftax_agents/export_trxl.py` | transformerXL_PPO_JAX → ONNX |
| `craftax_agents/variants.py` | 16-bit, 8-bit and safetensors copies of a playable model |
| `scripts/fetch_*.sh` | the upstream weights (and, for the checks, upstream model code), pinned and sha256-checked |

## Credits

- **Craftax**: Michael Matthews, Michael Beukman, Benjamin Ellis, Mikayel Samvelyan, Matthew Jackson, Samuel Coward and Jakob Foerster. *Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning*, ICML 2024 ([paper](https://arxiv.org/abs/2402.16801), [code](https://github.com/MichaelTMatthews/Craftax), MIT). The game, its symbolic observation and its rules.
- **PufferLib**: Joseph Suarez and PufferAI ([code](https://github.com/PufferAI/PufferLib), MIT). The C port of Craftax that craftax-wasm is forked from, and the 162 policy (Daphne Cornelisse, *Lessons from reaching level 6 in Craftax through tabula rasa on-policy RL*, 2026). `puffer_net.py` is a NumPy port of their network, observation packing and action mask.
- **Craftax_Baselines**: Michael Matthews ([code](https://github.com/MichaelTMatthews/Craftax_Baselines), MIT). The Craftax paper's PPO-RNN, adapted from PureJaxRL (Chris Lu). `ppo_rnn_net.py` and `export_baselines_rnn.py` restate its network. The `ppo-rnn` agent is our retraining of it.
- **transformerXL_PPO_JAX**: Gautier Hamon ([code](https://github.com/Reytuag/transformerXL_PPO_JAX), MIT). The `trxl` agent and its weights. `export_trxl.py` checks against their code, downloaded unmodified.
- **ReMDM checkpoints**: the authors of *Return-Weighted ELBO Fine-Tuning Degrades Masked Diffusion Planners* ([weights](https://huggingface.co/mathisweil/remdm-craftax-checkpoints), MIT). The PPO-RNN that `fetch_remdm.sh` downloads.
- **ONNX Runtime** (Microsoft, MIT) runs the agents, in Python here and as onnxruntime-web on playcraftax.com.

The third-party license texts are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT (see [LICENSE](LICENSE)). © 2026 TextJam, Inc. Weights are not part of this repository: each agent's weights are under its authors' license, listed in `models.json`.
