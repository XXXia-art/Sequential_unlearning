#!/bin/bash
set -euo pipefail

# =============================================================================
# Unified Sampling Script for Concept Erasure
# =============================================================================
# Usage:
#   bash scripts/sample.sh <method> <dataset> <split> <mode> <gpus>
#
# Parameters:
#   method   : alphaedit | speed | alpha_delta | alpha_delta_v2
#   dataset  : instance | style | celebrity | coco
#   split    : target | retain | all | coco
#   mode     : edit | original
#   gpus     : comma-separated GPU ids, e.g. 0,1,2,3
#
# Examples:
#   bash scripts/sample.sh alpha_delta instance target edit 0,1,2,3
#   bash scripts/sample.sh speed style retain edit 0,1,2,3
#   bash scripts/sample.sh alpha_delta_v2 celebrity all edit 0,1,2,3
#   bash scripts/sample.sh speed coco coco edit 0
#   bash scripts/sample.sh alphaedit instance target original 0,1
#
# Advanced:
#   # Use a specific step
#   STEP=50 bash scripts/sample.sh speed style retain edit 0
#
#   # Resume: skip already-sampled concepts
#   RESUME=true bash scripts/sample.sh alpha_delta instance target edit 0,1
#
#   # Specify custom concepts (semicolon-separated)
#   CONCEPTS="SpongeBob" bash scripts/sample.sh alpha_delta instance target edit 0
#   CONCEPTS="SpongeBob;Mickey Mouse" bash scripts/sample.sh speed instance target edit 0,1
# =============================================================================

METHOD="${1:?Usage: $0 <method> <dataset> <split> <mode> <gpus>}"
DATASET="${2:?Usage: $0 <method> <dataset> <split> <mode> <gpus>}"
SPLIT="${3:?Usage: $0 <method> <dataset> <split> <mode> <gpus>}"
MODE="${4:?Usage: $0 <method> <dataset> <split> <mode> <gpus>}"
GPUS="${5:?Usage: $0 <method> <dataset> <split> <mode> <gpus>}"

PYTHON_BIN="${PYTHON_BIN:-python}"
STEP="${STEP:-100}"
step_name=$(printf "step_%03d" "$STEP")
RESUME="${RESUME:-false}"

# ---------------------------------------------------------------------------
# 1. Validate inputs
# ---------------------------------------------------------------------------
case "$METHOD" in
  alphaedit|speed|alpha_delta|alpha_delta_v2) ;;
  *) echo "[ERROR] Unknown method: $METHOD"; exit 1 ;;
esac

case "$DATASET" in
  instance|style|celebrity|coco) ;;
  *) echo "[ERROR] Unknown dataset: $DATASET"; exit 1 ;;
esac

case "$SPLIT" in
  target|retain|all|coco) ;;
  *) echo "[ERROR] Unknown split: $SPLIT"; exit 1 ;;
esac

case "$MODE" in
  edit|original) ;;
  *) echo "[ERROR] Unknown mode: $MODE"; exit 1 ;;
esac

if [[ "$SPLIT" == "coco" && "$DATASET" != "coco" ]]; then
  echo "[ERROR] split=coco only valid with dataset=coco"
  exit 1
fi

if [[ "$DATASET" == "coco" && "$SPLIT" != "coco" ]]; then
  echo "[ERROR] dataset=coco requires split=coco"
  exit 1
fi

if [[ "$MODE" == "edit" && "$DATASET" == "coco" ]]; then
  : # coco edit is valid
fi

# ---------------------------------------------------------------------------
# 2. Dataset configuration
# ---------------------------------------------------------------------------
case "$DATASET" in
  instance)
    ERASE_TYPE="instance"
    CSV="data/instance.csv"
    ;;
  style)
    ERASE_TYPE="style"
    CSV="data/style.csv"
    ;;
  celebrity)
    ERASE_TYPE="celebrity"
    CSV="data/celebrity.csv"
    ;;
  coco)
    ERASE_TYPE="style"
    ;;
esac

# ---------------------------------------------------------------------------
# 3. Method → save_root + checkpoint
# ---------------------------------------------------------------------------
case "$METHOD" in
  alphaedit)      SAVE_ROOT="logs/alphaedit/$DATASET" ;;
  speed)          SAVE_ROOT="logs/speed/$DATASET" ;;
  alpha_delta)    SAVE_ROOT="logs/alpha_delta/$DATASET" ;;
  alpha_delta_v2) SAVE_ROOT="logs/alpha_delta_v2/$DATASET" ;;
esac

CKPT="${SAVE_ROOT}/${step_name}/weight.pt"

if [ "$MODE" == "original" ]; then
  SAVE_ROOT="data/pretrain/$DATASET"
  CKPT=""
fi

# ---------------------------------------------------------------------------
# 4. Sampler & parameters by scenario
# ---------------------------------------------------------------------------
if [ "$DATASET" == "coco" ]; then
  SAMPLER="sample2.py"
  NUM_SAMPLES=1
  BATCH_SIZE=40
else
  SAMPLER="sample_fast.py"
  NUM_SAMPLES=5
  BATCH_SIZE=5
fi

# ---------------------------------------------------------------------------
# 5. Build contents list
# ---------------------------------------------------------------------------
CONCEPTS="${CONCEPTS:-}"

build_contents() {
  if [ -n "$CONCEPTS" ]; then
    # User-specified concepts override CSV
    echo "$CONCEPTS"
    return
  fi
  case "$SPLIT" in
    target)
      "$PYTHON_BIN" - <<PY
import pandas as pd
df = pd.read_csv("$CSV")
df = df.sort_values("id")
concepts = df[df["id"] <= 200]["concept"].dropna().astype(str).tolist()
print(";".join(concepts))
PY
      ;;
    retain)
      "$PYTHON_BIN" - <<PY
import pandas as pd
df = pd.read_csv("$CSV")
df = df.sort_values("id")
concepts = df[(df["id"] >= 201) & (df["id"] <= 600)]["concept"].dropna().astype(str).tolist()
print(";".join(concepts))
PY
      ;;
    all)
      "$PYTHON_BIN" - <<PY
import pandas as pd
df = pd.read_csv("$CSV")
df = df.sort_values("id")
concepts = df["concept"].dropna().astype(str).tolist()
print(";".join(concepts))
PY
      ;;
    coco)
      echo "coco"
      ;;
  esac
}

CONTENTS=$(build_contents)
if [ -z "$CONTENTS" ]; then
  echo "[ERROR] No contents found for split=$SPLIT"
  exit 1
fi

# Count concepts
if [ "$DATASET" == "coco" ]; then
  NUM_CONCEPTS=1000  # approx, for display only
else
  NUM_CONCEPTS=$(echo "$CONTENTS" | tr ';' '\n' | grep -c '^.' || true)
fi

# ---------------------------------------------------------------------------
# 6. GPU setup
# ---------------------------------------------------------------------------
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

# ---------------------------------------------------------------------------
# 7. Print config
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo "[CONFIG] Method=$METHOD | Dataset=$DATASET | Split=$SPLIT | Mode=$MODE"
echo "[CONFIG] Concepts=$NUM_CONCEPTS | Step=$step_name"
echo "[CONFIG] Sampler=$SAMPLER | NumSamples=$NUM_SAMPLES | BatchSize=$BATCH_SIZE"
echo "[CONFIG] GPUs=${GPU_LIST[*]} (count=$NUM_GPUS)"
echo "[CONFIG] SaveRoot=$SAVE_ROOT"
if [ -n "$CKPT" ]; then
  echo "[CONFIG] Checkpoint=$CKPT"
fi
if [ "$RESUME" == "true" ]; then
  echo "[CONFIG] RESUME=true"
fi
echo "=============================================================================="

# ---------------------------------------------------------------------------
# 8. Multi-GPU concept sharding (all scenarios except coco use single-GPU)
# ---------------------------------------------------------------------------
mkdir -p logs

if [ "$DATASET" == "coco" ]; then
  # -------------------------------------------------------------------------
  # COCO: single GPU (sample2.py handles batching internally)
  # -------------------------------------------------------------------------
  gpu="${GPU_LIST[0]}"
  echo "[LAUNCH] COCO sampling on GPU $gpu"

  local edit_args=()
  if [ "$MODE" == "edit" ] && [ -n "$CKPT" ]; then
    edit_args+=(--edit_ckpt "$CKPT")
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$SAMPLER" \
    --erase_type "$ERASE_TYPE" \
    --contents "coco" \
    --mode "$MODE" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --save_root "${SAVE_ROOT}/${step_name}" \
    --gpu 0 \
    "${edit_args[@]}" \
    > "logs/sample_${METHOD}_${DATASET}_${SPLIT}_gpu${gpu}.log" 2>&1

else
  # -------------------------------------------------------------------------
  # Instance / Style / Celebrity: shard concepts across GPUs
  # -------------------------------------------------------------------------

  # Write concepts to temp file and split
  tmp_base="/tmp/sample_${METHOD}_${DATASET}_${SPLIT}_${MODE}"
  echo "$CONTENTS" > "${tmp_base}.txt"

  "$PYTHON_BIN" - <<PY
with open("${tmp_base}.txt") as f:
    concepts = f.read().strip().split(";")
n = len(concepts)
num_gpus = $NUM_GPUS
chunk_size = (n + num_gpus - 1) // num_gpus
for i in range(num_gpus):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, n)
    chunk = concepts[start:end]
    if chunk:
        with open(f"${tmp_base}_{i}.txt", "w") as f:
            f.write(";".join(chunk))
PY

  PIDS=()
  for i in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$i]}"
    chunk_file="${tmp_base}_${i}.txt"

    if [ ! -f "$chunk_file" ]; then
      echo "[SKIP] GPU $gpu has no chunk"
      continue
    fi

    chunk=$(cat "$chunk_file")
    chunk_n=$(echo "$chunk" | tr ';' '\n' | grep -c '^.' || true)
    if [ "$chunk_n" -eq 0 ]; then
      echo "[SKIP] GPU $gpu has empty chunk"
      continue
    fi

    echo "[LAUNCH] GPU $gpu → $chunk_n concepts"

    local edit_args=()
    if [ "$MODE" == "edit" ] && [ -n "$CKPT" ]; then
      edit_args+=(--edit_ckpt "$CKPT")
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$SAMPLER" \
      --erase_type "$ERASE_TYPE" \
      --target_concept "$step_name" \
      --contents "$chunk" \
      --mode "$MODE" \
      --num_samples "$NUM_SAMPLES" \
      --batch_size "$BATCH_SIZE" \
      --save_root "$SAVE_ROOT" \
      --gpu 0 \
      "${edit_args[@]}" \
      > "logs/sample_${METHOD}_${DATASET}_${SPLIT}_gpu${gpu}.log" 2>&1 &

    PIDS+=($!)
  done

  if [ ${#PIDS[@]} -gt 0 ]; then
    echo ""
    echo "[WAIT] Waiting for ${#PIDS[@]} GPU jobs..."
    wait "${PIDS[@]}"
  fi
fi

echo ""
echo "=============================================================================="
echo "✅ Sampling complete: $METHOD $DATASET $SPLIT $MODE"
echo "=============================================================================="
