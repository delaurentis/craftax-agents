#!/usr/bin/env bash
# Fetch PufferLib 5.0's released Craftax policy (MIT) into weights/: craftax_1024x4.bin, 57 MB, checked by sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=weights/craftax_1024x4.bin
SHA=53adc0d29f317d0efb1dababfbf6ef52b43ab65da2bdb65410b177aa6558e234
mkdir -p weights
if [ -f "$OUT" ] && echo "$SHA  $OUT" | shasum -a 256 -c - >/dev/null 2>&1; then echo "$OUT ok"; exit 0; fi
TMP=$(mktemp -d)
curl -fL --progress-bar https://github.com/PufferAI/PufferLib/releases/download/5.0-experiments/puffer5_models.zip -o "$TMP/m.zip"
unzip -o -q "$TMP/m.zip" puffer5_models/craftax_1024x4.bin -d "$TMP"
mv "$TMP/puffer5_models/craftax_1024x4.bin" "$OUT"; rm -rf "$TMP"
echo "$SHA  $OUT" | shasum -a 256 -c -
