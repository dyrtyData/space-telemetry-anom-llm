#!/usr/bin/env bash
# Phase 3 (§3.3): upload training data + code to a running Vast.ai instance.
#
# Uploads the ChatML training data, the raw/enriched splits (for later eval), the
# advice labels, the PNG plots (for the §3.7 VL model), and the code/config needed
# to run training on the instance.
#
# Usage: ./scripts/cloud/upload_data.sh <instance_id>
#
# Note: only DERIVED artifacts under the repo's data/ tree are sent (small). The
# multi-GB RAW ESA-AD on DUAL DRIVE is NOT needed on the instance.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

INSTANCE_ID="${1:-}"
if [[ -z "$INSTANCE_ID" ]]; then
    echo "Usage: ./scripts/cloud/upload_data.sh <instance_id>" >&2
    exit 1
fi

VASTAI="${VASTAI:-vastai}"
REMOTE_ROOT="/workspace/star-pipeline"

# vastai ssh-url -> ssh://root@host:port ; parse into host/port/user for rsync -e ssh.
SSH_URL="$("$VASTAI" ssh-url "$INSTANCE_ID")"
SSH_USER_HOST="${SSH_URL#ssh://}"          # root@host:port
SSH_PORT="${SSH_USER_HOST##*:}"
SSH_USER_HOST="${SSH_USER_HOST%:*}"        # root@host
SSH_OPTS="-o StrictHostKeyChecking=no -p ${SSH_PORT}"
RSYNC_SSH="ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}"

echo "Target: ${SSH_USER_HOST}:${SSH_PORT}  ->  ${REMOTE_ROOT}"

# shellcheck disable=SC2086
ssh $SSH_OPTS "$SSH_USER_HOST" "mkdir -p ${REMOTE_ROOT}/data ${REMOTE_ROOT}/models"

# Data (formatted ChatML is what training reads; splits/labels/plots for eval + VL).
rsync -avz --progress -e "$RSYNC_SSH" \
    --relative \
    data/formatted/ \
    data/splits/ \
    data/labels/ \
    data/processed/plots/ \
    "${SSH_USER_HOST}:${REMOTE_ROOT}/"

# Code + config.
rsync -avz --progress -e "$RSYNC_SSH" \
    --relative \
    src/ config/ pyproject.toml \
    "${SSH_USER_HOST}:${REMOTE_ROOT}/"

echo "Upload complete."
echo "On the instance:  cd ${REMOTE_ROOT} && python src/training/train_advice.py --config config/unsloth-train.yaml"
