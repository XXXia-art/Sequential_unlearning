#!/bin/bash
set -euo pipefail

# =============================================================================
# Unified Evaluation Script for Concept Erasure
# =============================================================================
# Computes CLIP Score + FID using src/clip_score_cal.py
#
# Usage:
#   bash scripts/eval.sh <method> <dataset> <split> <gpu>
#
# Parameters:
#   method   : alphaedit | speed | alpha_delta | alpha_delta_v2 | all
#   dataset  : instance | style | coco
#   split    : retain | target | coco
#   gpu      : GPU id, e.g. 0
#
# Examples:
#   # Instance retain (id 201-600)
#   bash scripts/eval.sh alpha_delta instance retain 0
#
#   # Instance target (id 1-200)
#   bash scripts/eval.sh speed instance target 0
#
#   # Style retain
#   bash scripts/eval.sh alpha_delta_v2 style retain 1
#
#   # COCO
#   bash scripts/eval.sh speed coco coco 0
#
#   # Evaluate all methods sequentially
#   bash scripts/eval.sh all instance retain 0
#
# Advanced:
#   # Use a specific step
#   STEP=50 bash scripts/eval.sh alpha_delta style retain 0
#
#   # Evaluate specific concept(s)
#   CONCEPTS="SpongeBob" bash scripts/eval.sh alpha_delta instance retain 0
#   CONCEPTS="SpongeBob;Mickey Mouse" bash scripts/eval.sh speed instance target 0
#
#   # Override root path directly
#   ROOT_PATH=logs/MyExp/instance/step_100 bash scripts/eval.sh alpha_delta instance retain 0
# =============================================================================

METHOD="${1:?Usage: $0 <method> <dataset> <split> <gpu>}"
DATASET="${2:?Usage: $0 <method> <dataset> <split> <gpu>}"
SPLIT="${3:?Usage: $0 <method> <dataset> <split> <gpu>}"
GPU="${4:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
STEP="${STEP:-100}"
step_name=$(printf "step_%03d" "$STEP")

# ---------------------------------------------------------------------------
# 1. Validate inputs
# ---------------------------------------------------------------------------
case "$METHOD" in
  alphaedit|speed|alpha_delta|alpha_delta_v2|all) ;;
  *) echo "[ERROR] Unknown method: $METHOD"; exit 1 ;;
esac

case "$DATASET" in
  instance|style|coco) ;;
  *) echo "[ERROR] Unknown dataset: $DATASET"; exit 1 ;;
esac

case "$SPLIT" in
  retain|target|coco) ;;
  *) echo "[ERROR] Unknown split: $SPLIT"; exit 1 ;;
esac

if [[ "$SPLIT" == "coco" && "$DATASET" != "coco" ]]; then
  echo "[ERROR] split=coco only valid with dataset=coco"
  exit 1
fi

if [[ "$DATASET" == "coco" && "$SPLIT" != "coco" ]]; then
  echo "[ERROR] dataset=coco requires split=coco"
  exit 1
fi

if [[ "$METHOD" == "all" && "$SPLIT" == "coco" && "$DATASET" != "coco" ]]; then
  : # all + coco is valid only for dataset=coco
fi

# ---------------------------------------------------------------------------
# 2. Dataset configuration
# ---------------------------------------------------------------------------
case "$DATASET" in
  instance)
    CSV="data/instance.csv"
    PRETRAINED="data/pretrain/instance"
    ;;
  style)
    CSV="data/style.csv"
    PRETRAINED="data/pretrain/style"
    ;;
  coco)
    PRETRAINED=""
    ;;
esac

# ---------------------------------------------------------------------------
# 3. Build contents list
# ---------------------------------------------------------------------------
CONCEPTS="${CONCEPTS:-}"

build_contents() {
  if [ -n "$CONCEPTS" ]; then
    echo "$CONCEPTS"
    return
  fi

  case "$SPLIT" in
    retain)
      "$PYTHON_BIN" - <<PY
import pandas as pd
df = pd.read_csv("$CSV")
df = df.sort_values("id")
concepts = df[(df["id"] >= 201) & (df["id"] <= 600)]["concept"].dropna().astype(str).tolist()
print(";".join(concepts))
PY
      ;;
    target)
      "$PYTHON_BIN" - <<PY
import pandas as pd
df = pd.read_csv("$CSV")
df = df.sort_values("id")
concepts = df[df["id"] <= 200]["concept"].dropna().astype(str).tolist()
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

NUM_CONCEPTS=$(echo "$CONTENTS" | tr ';' '\n' | grep -c '^.' || true)

# ---------------------------------------------------------------------------
# 4. Method → root_path
# ---------------------------------------------------------------------------
method_to_dir() {
  case "$1" in
    alphaedit) echo "logs/alphaedit/$DATASET/$step_name" ;;
    speed) echo "logs/speed/$DATASET/$step_name" ;;
    alpha_delta) echo "logs/alpha_delta/$DATASET/$step_name" ;;
    alpha_delta_v2) echo "logs/alpha_delta_v2/$DATASET/$step_name" ;;
  esac
}

# ---------------------------------------------------------------------------
# 5. Run evaluation for one method
# ---------------------------------------------------------------------------
run_eval() {
  local method="$1"
  local root_path="$2"

  echo "======================================"
  echo "[EVAL] method: $method"
  echo "[EVAL] dataset: $DATASET"
  echo "[EVAL] split: $SPLIT"
  echo "[EVAL] root_path: $root_path"
  echo "[EVAL] concepts: $NUM_CONCEPTS"

  local extra_args=()
  if [ "$SPLIT" != "coco" ]; then
    extra_args+=(--pretrained_path "$PRETRAINED")
  fi

  if [ -n "${ROOT_PATH:-}" ]; then
    root_path="$ROOT_PATH"
  fi

  if [ ! -d "$root_path" ]; then
    echo "[SKIP] directory not found: $root_path"
    return
  fi

  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" src/clip_score_cal.py \
    --contents "$CONTENTS" \
    --root_path "$root_path" \
    "${extra_args[@]}"
}

# ---------------------------------------------------------------------------
# 6. Main evaluation loop
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo "[CONFIG] Method=$METHOD | Dataset=$DATASET | Split=$SPLIT | Step=$step_name"
echo "[CONFIG] Concepts=$NUM_CONCEPTS | GPU=$GPU"
echo "=============================================================================="

if [ "$METHOD" == "all" ]; then
  METHOD_LIST=(alphaedit speed alpha_delta alpha_delta_v2)
else
  METHOD_LIST=("$METHOD")
fi

for m in "${METHOD_LIST[@]}"; do
  root_path=$(method_to_dir "$m")
  run_eval "$m" "$root_path"
done

echo ""
echo "=============================================================================="
echo "✅ Evaluation complete: $METHOD $DATASET $SPLIT"
echo "=============================================================================="
