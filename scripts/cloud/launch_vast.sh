#!/usr/bin/env bash
# Phase 3 (§3.2): launch a Vast.ai GPU instance for Unsloth fine-tuning.
#
# Reads VASTAI_API_KEY from the environment or ./.env, searches for the cheapest
# reliable GPU offer matching the constraints, and creates an on-demand instance.
#
# Usage:
#   ./scripts/cloud/launch_vast.sh                 # dry run: search + print top offers
#   ./scripts/cloud/launch_vast.sh --create        # actually create the instance ($$)
#
# Env overrides:
#   GPU_NAME   (default RTX_4090)   VAST_IMAGE (default pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel)
#   DISK_GB    (default 60)         MIN_RELIABILITY (default 0.95)   MIN_GPU_RAM (default 23)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- credentials -------------------------------------------------------------
if [[ -z "${VASTAI_API_KEY:-}" && -f "$REPO_ROOT/.env" ]]; then
    VASTAI_API_KEY="$(grep -E '^VASTAI_API_KEY=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2-)"
fi
if [[ -z "${VASTAI_API_KEY:-}" ]]; then
    echo "ERROR: VASTAI_API_KEY not set (env or .env)." >&2
    exit 1
fi

VASTAI="${VASTAI:-vastai}"
"$VASTAI" set api-key "$VASTAI_API_KEY" >/dev/null

# --- search constraints ------------------------------------------------------
GPU_NAME="${GPU_NAME:-RTX_4090}"
DISK_GB="${DISK_GB:-60}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.95}"
MIN_GPU_RAM="${MIN_GPU_RAM:-23}"  # GiB; 8B 4-bit + activations fits in 24 GB
VAST_IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel}"

QUERY="gpu_name=${GPU_NAME} reliability>${MIN_RELIABILITY} gpu_ram>${MIN_GPU_RAM} \
disk_space>${DISK_GB} rentable=true cuda_vers>=12.1 inet_down>200"

echo "Searching offers: $QUERY"
"$VASTAI" search offers "$QUERY" --order 'dph_total asc' | head -8

OFFER_ID="$("$VASTAI" search offers "$QUERY" --order 'dph_total asc' --raw \
    | python3 -c 'import sys,json; o=json.load(sys.stdin); print(o[0]["id"] if o else "")')"

if [[ -z "$OFFER_ID" ]]; then
    echo "No matching offers. Loosen constraints (GPU_NAME / MIN_RELIABILITY) or try RunPod." >&2
    exit 1
fi
echo "Cheapest matching offer: $OFFER_ID"

if [[ "${1:-}" != "--create" ]]; then
    echo
    echo "Dry run only. Re-run with --create to launch offer $OFFER_ID (incurs charges)."
    exit 0
fi

# --- onstart: install unsloth stack on top of the base PyTorch image ---------
ONSTART='pip install -q --no-cache-dir unsloth "transformers>=4.46" trl peft datasets accelerate bitsandbytes pyyaml && echo UNSLOTH_READY'

echo "Creating instance from offer $OFFER_ID ..."
"$VASTAI" create instance "$OFFER_ID" \
    --image "$VAST_IMAGE" \
    --disk "$DISK_GB" \
    --ssh --direct \
    --onstart-cmd "$ONSTART"

echo
echo "Instance requested. Track with:  vastai show instances"
echo "Get SSH once running:            vastai ssh-url <instance_id>"
echo "Then upload data:                ./scripts/cloud/upload_data.sh <instance_id>"
