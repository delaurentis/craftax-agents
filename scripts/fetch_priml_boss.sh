#!/usr/bin/env bash
# Fetch priml-boss (our reproduction of rekursiv.ai's priml recipe, exp004: the boss fine-tune; MIT) into weights/priml-boss/: its
# MinGRUPolicy state_dict in priml's tensor names (bf16 safetensors, 31 MB), from huggingface.co/delaurentis/craftax-agents (priml-boss/),
# pinned to one commit and sha256-checked. Then: python -m craftax_agents.export_priml
set -euo pipefail
cd "$(dirname "$0")/.."
R=https://huggingface.co/delaurentis/craftax-agents/resolve/b0361f5e8a8c72bb73306f9716606e91ac37da0e/priml-boss
mkdir -p weights/priml-boss
for f in "priml-boss.safetensors b4165fa6b76631dca816d32e4fff3083f52b15cb05dd8fc8c002cc5f65150a53"; do
  set -- $f
  if [ -f "weights/priml-boss/$1" ] && echo "$2  weights/priml-boss/$1" | shasum -a 256 -c - >/dev/null 2>&1; then echo "weights/priml-boss/$1 ok"; continue; fi
  curl -fL --progress-bar "$R/$1" -o "weights/priml-boss/$1"
  echo "$2  weights/priml-boss/$1" | shasum -a 256 -c -
done
