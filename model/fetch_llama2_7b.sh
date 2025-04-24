#!/usr/bin/env bash
# Download Meta‑Llama‑2‑7B weights to the shared project cache.
set -euo pipefail

export HF_HOME=${HF_HOME:-/gpfs/wolf2/olcf/trn040/world-shared/vjodo/hf_cache}
export MODEL_DIR=/gpfs/wolf2/olcf/trn040/world-shared/vjodo/models/Llama-2-7b-hf

mkdir -p "$MODEL_DIR"
# Requires an HF token that has accepted the Llama 2 license.
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir "$MODEL_DIR" --resume-download --quiet