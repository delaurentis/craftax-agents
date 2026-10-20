#!/usr/bin/env bash
# Fetch our PPO-RNN (the Craftax paper's baseline, retrained for 1e9 steps; MIT) into weights/ppo-rnn/: its flax params (27.5 MB) and
# config.json, from huggingface.co/delaurentis/craftax-agents (ppo-rnn/), pinned to one commit and sha256-checked.
set -euo pipefail
cd "$(dirname "$0")/.."
R=https://huggingface.co/delaurentis/craftax-agents/resolve/f64e0278bb427e540852db397401e5d7a8f5607e/ppo-rnn
mkdir -p weights/ppo-rnn
for f in "params.msgpack 430698455214691ac842dc94d00faf530034282538b188e34c169b5be08855fc" "config.json 6bf2b43f95e5527d18e9a55838f1f7dac805bccd591ab892df0d9423c89398a7"; do
  set -- $f
  if [ -f "weights/ppo-rnn/$1" ] && echo "$2  weights/ppo-rnn/$1" | shasum -a 256 -c - >/dev/null 2>&1; then echo "weights/ppo-rnn/$1 ok"; continue; fi
  curl -fL --progress-bar "$R/$1" -o "weights/ppo-rnn/$1"
  echo "$2  weights/ppo-rnn/$1" | shasum -a 256 -c -
done
