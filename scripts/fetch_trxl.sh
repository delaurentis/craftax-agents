#!/usr/bin/env bash
# Fetch Gautier Hamon's TransformerXL-PPO Craftax checkpoints (MIT, github.com/Reytuag/transformerXL_PPO_JAX) into weights/trxl/.
#   scripts/fetch_trxl.sh          the one export_trxl.py plays: base_A100_1e9 seed 2 (params 19.6 MB + config + metrics) and the model code
#   scripts/fetch_trxl.sh --all    every Craftax checkpoint in the repo: base_A100_1e9 and pmap2V100_1e9steps_memory128_layer2_gamma999, seeds 0-4 (218 MB)
# Pinned to 518de77 (2024-05-15), the commit that added results_craftax/: the code in src/ is the code those params were trained with
# (before db3e52a's MLP fix, 2024-11, and d914d55's 2nd-gating fix, 2026-02). LICENSE was added later, so it comes from d914d55. Every file sha256-checked.
set -euo pipefail
cd "$(dirname "$0")/.."
R=https://raw.githubusercontent.com/Reytuag/transformerXL_PPO_JAX
C=518de77e8e4c8a324e48478fcbba2a1f012448f1; HEAD=d914d55b8e972c481845529d8b61cf3db88bd379
B=results_craftax/base_A100_1e9; P=results_craftax/pmap2V100_1e9steps_memory128_layer2_gamma999
FILES="
$C src/transformerXL.py 96b206a00fc8b8384671d4b6b9c599bff92fce00b9230c8dc581e932fce97d5c
$C src/rel_multi_head.py d548569095b4fc405938e227f2fbecfd247fd8813acc825db08ee8c8d0b7649a
$C src/trainer_PPO_trXL.py 87aacb569e6f9dc8ef99e4d7cb40fffed807dfc42830e2de4e8d1cc366e439e7
$HEAD src/LICENSE c72fdb6281dcaa4dcf7c979fff30435c8c896c9e82b26c9b0d2967c6a33d47b4
$C $B/2_params.npy 33aa9b8658bddce12c6ed8a359654536e026b9a4dce783ed8174043029594964
$C $B/2_config.npy 2d0b7e3a0434811dfe7d667765414d8e3ef1deed4f48c33b525df895c08fe745
$C $B/2_metrics.npy 3262cd93a19ff7fbd13bb40b19ec1d9620797344e3a133b57e6bcad3472bea3c"
ALL="
$B/0 034ef94a932811f089c9a7ced1b93f717fe76fec32888380ac0bae39756430d8 2d0b7e3a0434811dfe7d667765414d8e3ef1deed4f48c33b525df895c08fe745 8f14dc211800066f0a60725b4f14adc43e75cc7b4977dd968957a81b9de0e831
$B/1 86aec2b6245c40ac28139353dd08e9ae04ebce5289d728dcc0286466f69c478f 2d0b7e3a0434811dfe7d667765414d8e3ef1deed4f48c33b525df895c08fe745 737321e7addcb1eb29c8898f41b443d3a3f5e474f80cd48fb0ebb8c85c7a65ed
$B/3 0df705b9d131655bec5b5863291757cbd8b1dd8a65963ca7e85cfa29d48da578 2d0b7e3a0434811dfe7d667765414d8e3ef1deed4f48c33b525df895c08fe745 57bbbab0e46e58bdf87e4c077b8f169f336151cc7e60d0056714ba63957319a1
$B/4 c808d30ae54b9e50c6177e149e23c211c2e128dd79494ee07d226b779f53a669 2d0b7e3a0434811dfe7d667765414d8e3ef1deed4f48c33b525df895c08fe745 53e1b7b0eaf51279e4a9e8de43a3d53b27e0d8b61eafc20b083772066fdb9679
$P/0 f0299a02d96edd31c88a22e86159f1612b82393bd9b5196713c17d095dfbdaf6 40055945fc50a213ae3449e88b432d0e600c914f80fb861e1d4554f03bc9c401 67d2bf93c447a547cd928b0f1362da6eb8845434e2ce4c64b69dbf93e12b13c8
$P/1 21475358a45bd5a709a0aa4c679505bcac15b1424d91740ea862908de133f7a9 40055945fc50a213ae3449e88b432d0e600c914f80fb861e1d4554f03bc9c401 b5cb7675dfdcd6932c7dc29c91b5f385369273a842b9b1e3745b7310679ceb29
$P/2 34b114e6170fa929816aa4e7c281323423d6c9ae16e248566d76e494efc4caf5 40055945fc50a213ae3449e88b432d0e600c914f80fb861e1d4554f03bc9c401 7cde8757dc3cb9a3a91c7ae2852eaf548aa3f084b1ede1f08e5919b1239c11f8
$P/3 cbd2050fc8139e09dd250e080660c5a8de325ed43e987a66833f8254c8d8120b 40055945fc50a213ae3449e88b432d0e600c914f80fb861e1d4554f03bc9c401 e6d61f92cbc036519d7620486fbb655295f47813aca11865522e7f67028fd9ce
$P/4 0ad650cf43d92c852c436a7fd44c7998c666007929224a336b59e38bb3bc3b87 40055945fc50a213ae3449e88b432d0e600c914f80fb861e1d4554f03bc9c401 c4c807cca41559dd8eccfb1fbbaf5e429974a8ae08ecbbb050d5487120371b5b"
if [ "${1:-}" = "--all" ]; then
  while read -r s p c m; do [ -n "$s" ] && FILES="$FILES
$C ${s}_params.npy $p
$C ${s}_config.npy $c
$C ${s}_metrics.npy $m"; done <<< "$ALL"
fi
while read -r ref path sha; do
  [ -n "$ref" ] || continue
  out=weights/trxl/${path#results_craftax/}
  if [ -f "$out" ] && echo "$sha  $out" | shasum -a 256 -c - >/dev/null 2>&1; then continue; fi
  mkdir -p "$(dirname "$out")"
  curl -fsSL "$R/$ref/$path" -o "$out.part" && mv "$out.part" "$out"
  echo "$sha  $out" | shasum -a 256 -c - >/dev/null || { echo "sha256 mismatch: $out" >&2; rm -f "$out"; exit 1; }
  echo "$out"
done <<< "$FILES"
echo "weights/trxl ok"
