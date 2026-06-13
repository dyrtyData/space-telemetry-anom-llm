#!/bin/bash
# Parallel GGUF download using 8 SSH dd streams to bypass per-stream TCP congestion limits.
# Each stream downloads a 600MB chunk independently; chunks are concatenated after all complete.

set -e

SSH_OPTS="-i ~/.ssh/vast_star -p 60642 -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o Compression=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
REMOTE="root@81.183.231.113"
REMOTE_FILE="/workspace/star-pipeline/models/gguf/star-pipeline-advice_gguf/qwen3-8b.Q4_K_M.gguf"
LOCAL_BASE="/Users/laptop/Developer/fdl_technicalInterview/models/gguf/star-pipeline-advice_gguf"
EXPECTED_BYTES=5027784160
NUM_PARTS=8
CHUNK_MB=600

mkdir -p "$LOCAL_BASE"

echo "=== Parallel GGUF download: ${NUM_PARTS} streams × ${CHUNK_MB}MB ==="
echo "Remote: $REMOTE_FILE"
echo "Local:  $LOCAL_BASE/qwen3-8b.Q4_K_M.gguf"
echo ""

# Launch all streams in parallel
declare -a PIDS
for i in $(seq 0 $((NUM_PARTS - 1))); do
    SKIP=$((i * CHUNK_MB))
    OUTFILE="$LOCAL_BASE/part_${i}.tmp"
    ssh $SSH_OPTS "$REMOTE" \
        "dd if=$REMOTE_FILE bs=1M skip=$SKIP count=$CHUNK_MB status=none 2>/dev/null" > "$OUTFILE" &
    PIDS[$i]=$!
    echo "[start] Part $i (skip=${SKIP}MB, count=${CHUNK_MB}MB) → PID ${PIDS[$i]}"
done

echo ""
echo "Waiting for all $NUM_PARTS parts..."
ALL_OK=1
for i in $(seq 0 $((NUM_PARTS - 1))); do
    if wait "${PIDS[$i]}"; then
        SZ=$(wc -c < "$LOCAL_BASE/part_${i}.tmp" 2>/dev/null || echo 0)
        echo "[done] Part $i: $SZ bytes"
    else
        echo "[FAIL] Part $i failed (exit $?)"
        ALL_OK=0
    fi
done

if [ "$ALL_OK" -ne 1 ]; then
    echo "ERROR: one or more parts failed, aborting concatenation."
    exit 1
fi

echo ""
echo "Concatenating parts..."
cat "$LOCAL_BASE"/part_0.tmp \
    "$LOCAL_BASE"/part_1.tmp \
    "$LOCAL_BASE"/part_2.tmp \
    "$LOCAL_BASE"/part_3.tmp \
    "$LOCAL_BASE"/part_4.tmp \
    "$LOCAL_BASE"/part_5.tmp \
    "$LOCAL_BASE"/part_6.tmp \
    "$LOCAL_BASE"/part_7.tmp \
    > "$LOCAL_BASE/qwen3-8b.Q4_K_M.gguf"

ACTUAL=$(wc -c < "$LOCAL_BASE/qwen3-8b.Q4_K_M.gguf")
echo ""
echo "=== Verification ==="
echo "Expected: $EXPECTED_BYTES bytes"
echo "Got:      $ACTUAL bytes"
if [ "$ACTUAL" -eq "$EXPECTED_BYTES" ]; then
    echo "MATCH: download complete and verified!"
    rm -f "$LOCAL_BASE"/part_*.tmp
    echo "Temp parts cleaned up."
else
    echo "MISMATCH: file corrupted or incomplete."
    exit 1
fi
