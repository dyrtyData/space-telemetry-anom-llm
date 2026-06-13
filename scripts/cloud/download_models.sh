#!/usr/bin/env bash
# Phase 3->4 bridge (§4.2): pull trained LoRA + GGUF off the Vast.ai instance.
#
# STORAGE RULE: the internal disk is nearly full. Models land on DUAL DRIVE
# (STAR_MODEL_DIR), never the repo. An 8B GGUF is multi-GB.
#
# Usage: ./scripts/cloud/download_models.sh <instance_id>
set -euo pipefail

INSTANCE_ID="${1:-}"
if [[ -z "$INSTANCE_ID" ]]; then
    echo "Usage: ./scripts/cloud/download_models.sh <instance_id>" >&2
    exit 1
fi

VASTAI="${VASTAI:-vastai}"
REMOTE_ROOT="/workspace/star-pipeline"
STAR_MODEL_DIR="${STAR_MODEL_DIR:-/Volumes/DUAL DRIVE/star-pipeline/models}"

SSH_URL="$("$VASTAI" ssh-url "$INSTANCE_ID")"
SSH_USER_HOST="${SSH_URL#ssh://}"
SSH_PORT="${SSH_USER_HOST##*:}"
SSH_USER_HOST="${SSH_USER_HOST%:*}"
RSYNC_SSH="ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}"

mkdir -p "$STAR_MODEL_DIR/gguf" "$STAR_MODEL_DIR/lora"
echo "Downloading models -> $STAR_MODEL_DIR"

rsync -avz --progress -e "$RSYNC_SSH" \
    "${SSH_USER_HOST}:${REMOTE_ROOT}/models/gguf/" "$STAR_MODEL_DIR/gguf/" || true
rsync -avz --progress -e "$RSYNC_SSH" \
    "${SSH_USER_HOST}:${REMOTE_ROOT}/models/lora/" "$STAR_MODEL_DIR/lora/"

echo "Download complete. Models in $STAR_MODEL_DIR"
echo "REMEMBER to terminate the instance: vastai destroy instance $INSTANCE_ID"
