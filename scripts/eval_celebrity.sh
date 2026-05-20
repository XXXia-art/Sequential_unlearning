#!/bin/bash
set -euo pipefail

# =============================================================================
# Unified GCD Celebrity Evaluation Script
# =============================================================================
# Usage:
#   bash scripts/eval_celebrity.sh <method> <step> <gpus>
#
# Parameters:
#   method : alphaedit | speed | alpha_delta | alpha_delta_v2 | all
#   step   : step directory name, e.g., step_100 (default: step_100)
#   gpus   : comma-separated GPU ids, e.g., 0,1,2,3
#
# Examples:
#   # Single method on GPU 0
#   bash scripts/eval_celebrity.sh alpha_delta step_100 0
#
#   # All methods in parallel (one per GPU)
#   bash scripts/eval_celebrity.sh all step_100 0,1,2,3
#
# Advanced:
#   # Override default method list when using 'all'
#   GCD_METHODS="alpha_delta speed" bash scripts/eval_celebrity.sh all step_100 0,1
#
#   # Use a specific Python binary
#   GCD_PYTHON=/opt/conda/envs/gcd_tf1/bin/python bash scripts/eval_celebrity.sh ...
# =============================================================================

METHOD="${1:?Usage: $0 <method> <step> <gpus>}"
STEP="${2:-step_100}"
GPUS="${3:?Usage: $0 <method> <step> <gpus>}"

# ---------------------------------------------------------------------------
# 1. Resolve Python in gcd_tf1 environment
# ---------------------------------------------------------------------------
PYTHON="${GCD_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if command -v conda &> /dev/null; then
        PYTHON="$(conda run -n gcd_tf1 which python 2>/dev/null || true)"
    fi
    if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
        PYTHON="$(which python3 2>/dev/null || which python)"
    fi
fi

echo "[CONFIG] Python: $PYTHON"
$PYTHON --version

# ---------------------------------------------------------------------------
# 2. Build method list
# ---------------------------------------------------------------------------
ALL_METHODS=(alphaedit speed alpha_delta alpha_delta_v2)

if [ "$METHOD" == "all" ]; then
    if [ -n "${GCD_METHODS:-}" ]; then
        IFS=' ' read -ra METHODS <<< "$GCD_METHODS"
    else
        METHODS=("${ALL_METHODS[@]}")
    fi
else
    METHODS=("$METHOD")
fi

# ---------------------------------------------------------------------------
# 3. Validate GPUs
# ---------------------------------------------------------------------------
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}
NUM_METHODS=${#METHODS[@]}

if [ "$NUM_METHODS" -gt "$NUM_GPUS" ]; then
    echo "[ERROR] $NUM_METHODS methods but only $NUM_GPUS GPUs provided."
    echo "        Provide more GPUs or set GCD_METHODS to a subset."
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Launch evaluation jobs
# ---------------------------------------------------------------------------
EVAL_SCRIPT="eval_gcd_original.py"
OUTPUT_DIR="eval_results/gcd_original"
ERASE_TYPE="celebrity"
CSV="data/celebrity.csv"

mkdir -p "$OUTPUT_DIR"

echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GCD Evaluation Started"
echo "   Methods: ${METHODS[*]}"
echo "   Step:    $STEP"
echo "   GPUs:    ${GPU_LIST[*]} (${NUM_GPUS} total)"
echo "   Output:  $OUTPUT_DIR"
echo "=============================================================================="

PIDS=()
for i in "${!METHODS[@]}"; do
    m="${METHODS[$i]}"
    gpu="${GPU_LIST[$i]}"
    logfile="$OUTPUT_DIR/eval_${m}_${STEP}_gpu${gpu}.log"

    echo "[LAUNCH] $m -> GPU $gpu  (log: $logfile)"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON $EVAL_SCRIPT \
        --method "$m" \
        --erase_type "$ERASE_TYPE" \
        --step_name "$STEP" \
        --data_csv "$CSV" \
        --output_dir "$OUTPUT_DIR" \
        --gpu 0 \
        > "$logfile" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${#PIDS[@]} job(s) launched. Waiting..."
wait "${PIDS[@]}"

echo ""
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL EVALUATION JOBS COMPLETE"
echo "=============================================================================="

# ---------------------------------------------------------------------------
# 5. Merge summaries
# ---------------------------------------------------------------------------
echo ""
echo "Merging summaries..."
$PYTHON << 'PYEOF'
import os, glob, pandas as pd

outdir = "eval_results/gcd_original"
methods = ["alphaedit", "speed", "alpha_delta", "alpha_delta_v2"]

all_rows = []
for method in methods:
    csv = os.path.join(outdir, f"{method}_summary.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            all_rows.append({
                "method": method,
                "top_n": row.get("top_n", ""),
                "split": row.get("split", ""),
                "acc_e": row.get("acc_e", ""),
                "acc_r": row.get("acc_r", ""),
                "h_o": row.get("h_o", ""),
                "total_images": row.get("total_images", ""),
                "correct_images": row.get("correct_images", ""),
            })

if all_rows:
    merged = pd.DataFrame(all_rows)
    merged_path = os.path.join(outdir, "merged_summary.csv")
    merged.to_csv(merged_path, index=False)
    print(f"\n✅ Merged summary saved: {merged_path}")
    print("\n" + "="*70)
    print("FINAL RESULTS (Top-5)")
    print("="*70)
    for _, row in merged[(merged['split']=='overall') & (merged['top_n']==5)].iterrows():
        print(f"\n[{row['method'].upper()}]")
        print(f"  Acc_e : {row['acc_e']}%")
        print(f"  Acc_r : {row['acc_r']}%")
        print(f"  H_o   : {row['h_o']}")
    print("\n" + "="*70)
    print("FINAL RESULTS (Top-1)")
    print("="*70)
    for _, row in merged[(merged['split']=='overall') & (merged['top_n']==1)].iterrows():
        print(f"\n[{row['method'].upper()}]")
        print(f"  Acc_e : {row['acc_e']}%")
        print(f"  Acc_r : {row['acc_r']}%")
        print(f"  H_o   : {row['h_o']}")
else:
    print("[ERROR] No summaries found!")
PYEOF

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done!"
