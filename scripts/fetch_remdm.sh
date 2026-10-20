#!/usr/bin/env bash
# Fetch the ReMDM paper's Craftax-Symbolic-v1 PPO-RNN expert (MIT, huggingface.co/mathisweil/remdm-craftax-checkpoints) into
# weights/remdm/: the Orbax OCDBT checkpoint dir (step 1e9, 51.9 MB) + config.yaml + wandb-summary.json + the HF LICENSE,
# and the model code it was trained with (Craftax_Baselines ppo_rnn.py + LICENSE, the craftax-ReMDM-planner submodule at 3aa6716).
#   scripts/fetch_remdm.sh       then: python -m craftax_agents.export_baselines_rnn --check 400
# HF pinned to revision 5b66c77 (2026-10-13); planner repo github.com/mathisweil/craftax-ReMDM-planner d2a66f4 pins the submodule. Every file sha256-checked.
set -euo pipefail
cd "$(dirname "$0")/.."
H=https://huggingface.co/mathisweil/remdm-craftax-checkpoints/resolve/5b66c77b497777e4ffdcbf20c3563402e404daf8
G=https://raw.githubusercontent.com/mathisweil/Craftax_Baselines/3aa6716aa3dec3f662f70dff36fc487469a86071
P=checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M; S=1000000000/default
FILES="
$H/LICENSE LICENSE d65ceebccae7d251265a5033ee630984a3c8875e46fe3866f6b34297f06aa81b
$G/ppo_rnn.py src/ppo_rnn.py 67cc7a9f715778293d0ba7b303e9e9653f898b42c49c52e00dd3a694d7cc1a3d
$G/LICENSE src/LICENSE ff84c271770b3b266e06a1bfc50b81e05dfb9811b55e9cfb546853c85190beef
$H/$P/config.yaml $P/config.yaml 6210603fbdb6d427214f116f15baf8e3e93a45f6ad9ba03a309ebbfb9344491e
$H/$P/wandb-summary.json $P/wandb-summary.json cb2505591b3e82839f9e02e3795fef486af318af9677002b8976e49ec4fd9ae6
$H/$P/1000000000/_CHECKPOINT_METADATA $P/1000000000/_CHECKPOINT_METADATA b37959cde771cff0acdd702fc5874ef451f6ec57f343f67bfd2612e784546971
$H/$P/$S/_METADATA $P/$S/_METADATA f872e88b1afa213823d924e90403d6376a7b7a597e3d07d9a1569f5679231752
$H/$P/$S/_sharding $P/$S/_sharding 3197bd1bfeedd14abc4d5168e8ec57e8ebf2115db3df185a0693857194551d33
$H/$P/$S/array_metadatas/process_0 $P/$S/array_metadatas/process_0 ee8ac4503713e04d8cbc47d74b7b50f3b6b472db12c20db6534c978240f3e97b
$H/$P/$S/d/4c4db8700ec36cc1416c034bfbe0f71a $P/$S/d/4c4db8700ec36cc1416c034bfbe0f71a 42bf3beeb1952e0e964b6db2bd27e7641418ed51b251caaa9f9c6f9d0a542326
$H/$P/$S/manifest.ocdbt $P/$S/manifest.ocdbt 6b33e00a877ae19528544736d5f19753468c00e3f9305197c01dbd59d481483c
$H/$P/$S/ocdbt.process_0/d/0f5cd6bf71d63c0de8e8764ba0de8349 $P/$S/ocdbt.process_0/d/0f5cd6bf71d63c0de8e8764ba0de8349 753e7f4f0973944369849fee75e563fa8da5931378c6bee1935b472a5423ef10
$H/$P/$S/ocdbt.process_0/d/3258293b820e7cc97e3af7b3d69255b6 $P/$S/ocdbt.process_0/d/3258293b820e7cc97e3af7b3d69255b6 2fe1d8d099363313b871b84c85d79c733ff1a4db1a7d16ad5a9cfbb41855b0b0
$H/$P/$S/ocdbt.process_0/d/3566d5252a486c26c607b2637742f32f $P/$S/ocdbt.process_0/d/3566d5252a486c26c607b2637742f32f bc7c33d9abb3f64541923336fa0416c15358acba1570430d46462baf3d11dab9
$H/$P/$S/ocdbt.process_0/d/e5a4020f50167115120fe5dac41c20fb $P/$S/ocdbt.process_0/d/e5a4020f50167115120fe5dac41c20fb fba36a05ef8ebbfda38b7f622137b5e9c9bd56cf6d6a534c4acb34e1897082f1
$H/$P/$S/ocdbt.process_0/manifest.ocdbt $P/$S/ocdbt.process_0/manifest.ocdbt 1afa8cb3dfd89f0c8e0a6a62e26a865c77033d8270d713ad55f979fbcf4040f8"
while read -r url path sha; do
  [ -n "$url" ] || continue
  out=weights/remdm/${path#checkpoints/ppo_agents/}
  if [ -f "$out" ] && echo "$sha  $out" | shasum -a 256 -c - >/dev/null 2>&1; then continue; fi
  mkdir -p "$(dirname "$out")"
  curl -fsSL "$url" -o "$out.part" && mv "$out.part" "$out"
  echo "$sha  $out" | shasum -a 256 -c - >/dev/null || { echo "sha256 mismatch: $out" >&2; rm -f "$out"; exit 1; }
  echo "$out"
done <<< "$FILES"
echo "weights/remdm ok"
