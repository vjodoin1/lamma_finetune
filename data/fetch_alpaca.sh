#!/usr/bin/env bash
# Download the Alpaca JSON locally so compute nodes can run fully offline.
set -euo pipefail

export HF_HOME=${HF_HOME:-/gpfs/wolf2/olcf/trn040/world-shared/vjodo/hf_cache}
export DATA_DIR=/gpfs/wolf2/olcf/trn040/world-shared/vjodo/data/alpaca
mkdir -p "$DATA_DIR"
huggingface-cli download tetsu-lab/alpaca --repo-type dataset --local-dir "$DATA_DIR" --quiet