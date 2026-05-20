#!/bin/bash
set -euo pipefail

# =============================================================================
# Unified Training Script for Concept Erasure
# =============================================================================
# Supports: alphaedit | speed | alpha_delta | alpha_delta_v2
# Datasets: instance | style | celebrity
#
# Usage:
#   bash scripts/train.sh <method> <dataset> <group_size> <gpu>
#
# Examples:
#   bash scripts/train.sh alphaedit instance 1 0
#   bash scripts/train.sh speed style 1 0
#   bash scripts/train.sh alpha_delta celebrity 2 0
#   bash scripts/train.sh alpha_delta_v2 instance 2 1
#
# Advanced:
#   # Resume from step 50
#   START_STEP=50 bash scripts/train.sh speed instance 1 0
#
#   # Keep only step 10/20/.../100 checkpoints
#   KEEP_STEPS="10 20 30 40 50 60 70 80 90 100" bash scripts/train.sh alpha_delta instance 4 0
#
#   # Override any parameter
#   SAVE_ROOT=logs/MyExp DELTA_COEF=0.95 bash scripts/train.sh alpha_delta style 2 0
# =============================================================================

METHOD="${1:?Usage: $0 <method> <dataset> <group_size> <gpu>}"
DATASET="${2:?Usage: $0 <method> <dataset> <group_size> <gpu>}"
GROUP_SIZE="${3:?Usage: $0 <method> <dataset> <group_size> <gpu>}"
GPU="${4:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------------------------------------------------------------------------
# 1. Dataset configuration
# ---------------------------------------------------------------------------
case "$DATASET" in
  instance)
    CSV="data/instance.csv"
    ANCHOR=" "
    ;;
  style)
    CSV="data/style.csv"
    ANCHOR="art"
    ;;
  celebrity)
    CSV="data/celebrity.csv"
    ANCHOR="person"
    ;;
  *)
    echo "[ERROR] Unknown dataset: $DATASET. Expected: instance | style | celebrity"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# 2. Method configuration
# ---------------------------------------------------------------------------
case "$METHOD" in
  alphaedit)
    PY_SCRIPT="Alphaedit.py"
    SAVE_ROOT="${SAVE_ROOT:-logs/alphaedit/$DATASET}"
    ;;
  speed)
    PY_SCRIPT="speed.py"
    SAVE_ROOT="${SAVE_ROOT:-logs/speed/$DATASET}"
    ;;
  alpha_delta)
    PY_SCRIPT="Alpha_delta.py"
    SAVE_ROOT="${SAVE_ROOT:-logs/alpha_delta/$DATASET}"
    ;;
  alpha_delta_v2)
    PY_SCRIPT="Alpha_delta_v2.py"
    SAVE_ROOT="${SAVE_ROOT:-logs/alpha_delta_v2/$DATASET}"
    ;;
  *)
    echo "[ERROR] Unknown method: $METHOD. Expected: alphaedit | speed | alpha_delta | alpha_delta_v2"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# 3. Hyperparameters (overridable via environment variables)
# ---------------------------------------------------------------------------
PARAMS="${PARAMS:-V}"
AUG_NUM="${AUG_NUM:-10}"
THRESHOLD="${THRESHOLD:-1e-1}"
RETAIN_SCALE="${RETAIN_SCALE:-1.0}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-float32}"
START_STEP="${START_STEP:-1}"
KEEP_STEPS="${KEEP_STEPS:-}"
DELTA_COEF="${DELTA_COEF:-0.90}"
ETA="${ETA:-1}"

# ---------------------------------------------------------------------------
# 4. Method-specific extra arguments
# ---------------------------------------------------------------------------
EXTRA_PY_ARGS=()
if [[ "$METHOD" == "alpha_delta" || "$METHOD" == "alpha_delta_v2" ]]; then
    EXTRA_PY_ARGS+=(--delta_coef "$DELTA_COEF" --eta "$ETA")
fi

# ---------------------------------------------------------------------------
# 5. Read target concepts (id 1-200) from CSV
# ---------------------------------------------------------------------------
mapfile -t concepts < <(
  "$PYTHON_BIN" - <<PY
import pandas as pd

df = pd.read_csv("$CSV")
if "id" not in df.columns or "concept" not in df.columns:
    raise ValueError("CSV must contain 'id' and 'concept' columns")

df = df.sort_values("id").copy()
df["concept"] = df["concept"].fillna("").astype(str).str.strip()
df = df[df["concept"] != ""]

targets = df.loc[df["id"] <= 200, "concept"].tolist()
if len(targets) < 200:
    raise ValueError(f"Only {len(targets)} target concepts found, need 200")

for x in targets[:200]:
    print(x)
PY
)

TOTAL=${#concepts[@]}
NEED_TOTAL=$((GROUP_SIZE * 100))
if [ "$TOTAL" -lt "$NEED_TOTAL" ]; then
    echo "[ERROR] target concept count ($TOTAL) is less than required ($NEED_TOTAL)"
    exit 1
fi

# ---------------------------------------------------------------------------
# 6. Prepare log directory
# ---------------------------------------------------------------------------
mkdir -p "$SAVE_ROOT"
CONFIG_FILE="${SAVE_ROOT}/config.txt"
{
  echo "=============================="
  echo "method: $METHOD"
  echo "dataset: $DATASET"
  echo "csv: $CSV"
  echo "anchor: '$ANCHOR'"
  echo "group_size: $GROUP_SIZE"
  echo "num_steps: 100"
  echo "start_step: $START_STEP"
  echo "params: $PARAMS"
  echo "aug_num: $AUG_NUM"
  echo "threshold: $THRESHOLD"
  echo "retain_scale: $RETAIN_SCALE"
  echo "seed: $SEED"
  echo "dtype: $DTYPE"
  echo "gpu: $GPU"
  echo "save_root: $SAVE_ROOT"
  if [[ "$METHOD" == "alpha_delta" || "$METHOD" == "alpha_delta_v2" ]]; then
      echo "delta_coef: $DELTA_COEF"
      echo "eta: $ETA"
  fi
  if [ -n "$KEEP_STEPS" ]; then
      echo "keep_steps: $KEEP_STEPS"
  fi
  echo "run_time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=============================="
} >> "$CONFIG_FILE"

echo "=============================================================================="
echo "[CONFIG] Method=$METHOD | Dataset=$DATASET | GroupSize=$GROUP_SIZE"
echo "[CONFIG] Anchor='$ANCHOR' | CSV=$CSV"
echo "[CONFIG] SaveRoot=$SAVE_ROOT | GPU=$GPU"
echo "[CONFIG] StartStep=$START_STEP"
if [ -n "$KEEP_STEPS" ]; then
    echo "[CONFIG] KeepSteps=$KEEP_STEPS (will delete intermediate checkpoints)"
fi
echo "=============================================================================="

# ---------------------------------------------------------------------------
# 7. Sequential training loop
# ---------------------------------------------------------------------------
for ((step=START_STEP; step<=100; step++)); do
    start=$(( (step - 1) * GROUP_SIZE ))
    current_group=("${concepts[@]:start:GROUP_SIZE}")

    if [ "${#current_group[@]}" -eq 0 ]; then
        echo "[INFO] empty group at step ${step}, stopping."
        break
    fi

    current_targets=$(printf "%s, " "${current_group[@]}")
    current_targets=${current_targets%, }
    step_name=$(printf "step_%03d" "$step")

    echo ""
    echo "[TRAIN] ${step_name}"
    echo "        targets = ${current_targets}"

    # Build edit_ckpt argument (resume from previous step)
    EXTRA_ARGS=()
    if [ "$step" -gt "$START_STEP" ]; then
        prev_step=$(printf "step_%03d" "$((step - 1))")
        prev_ckpt_dir="${SAVE_ROOT}/${prev_step}"
        if [ -f "${prev_ckpt_dir}/weight.pt" ]; then
            EXTRA_ARGS+=(--edit_ckpt "${prev_ckpt_dir}/weight.pt")
        elif [ -d "$prev_ckpt_dir" ]; then
            EXTRA_ARGS+=(--edit_ckpt "$prev_ckpt_dir")
        else
            echo "[WARN] previous checkpoint not found: $prev_ckpt_dir, starting from scratch"
        fi
    fi

    # Optional SD checkpoint override
    SD_CKPT_ARG=()
    if [ -n "${SD_CKPT:-}" ]; then
        SD_CKPT_ARG=(--sd_ckpt "$SD_CKPT")
    fi

    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" "$PY_SCRIPT" \
        --target_concepts "$current_targets" \
        --anchor_concepts "$ANCHOR" \
        --retain_path "$CSV" \
        --heads "concept" \
        --save_path "${SAVE_ROOT}/${step_name}" \
        --params "$PARAMS" \
        --aug_num "$AUG_NUM" \
        --threshold "$THRESHOLD" \
        --retain_scale "$RETAIN_SCALE" \
        --seed "$SEED" \
        --dtype "$DTYPE" \
        "${EXTRA_PY_ARGS[@]}" \
        "${SD_CKPT_ARG[@]}" \
        "${EXTRA_ARGS[@]}"

    # -----------------------------------------------------------------------
    # 8. Optional: clean up non-essential checkpoints
    # -----------------------------------------------------------------------
    if [ -n "$KEEP_STEPS" ] && [ "$step" -gt "$START_STEP" ]; then
        prev_step=$(printf "step_%03d" "$((step - 1))")
        prev_ckpt="${SAVE_ROOT}/${prev_step}/weight.pt"

        should_keep=false
        for k in $KEEP_STEPS; do
            keep_step_name=$(printf "step_%03d" "$k")
            if [ "$prev_step" == "$keep_step_name" ]; then
                should_keep=true
                break
            fi
        done

        if [ "$should_keep" == false ] && [ -f "$prev_ckpt" ]; then
            rm -f "$prev_ckpt"
            echo "[CLEAN] removed intermediate checkpoint: $prev_ckpt"
        fi
    fi
done

echo ""
echo "=============================================================================="
echo "✅ Training complete: $METHOD on $DATASET"
echo "   Checkpoints saved under: $SAVE_ROOT"
echo "=============================================================================="
