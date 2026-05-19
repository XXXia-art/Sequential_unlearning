#!/bin/bash
set -euo pipefail

# =============================================================================
# Parallel GCD Original Evaluation for Celebrity_v2
# =============================================================================
# Uses the ORIGINAL celeb-detection-oss codebase:
#   - TF 1.x MTCNN (CPU, because RTX 4090 doesn't support CUDA 10.0)
#   - PyTorch ResNet50 + GMM clustering (GPU)
#
# CRITICAL: Run this script inside the project root directory.
# Each method gets its own GPU for PyTorch; TF MTCNN shares CPU cores.
#
# Usage:
#   bash run_gcd_original_eval.sh
# =============================================================================

PROJECT_DIR="/data/coding/Instance_log"
cd "$PROJECT_DIR"

PYTHON="/data/miniconda/envs/gcd_tf1/bin/python"
EVAL_SCRIPT="eval_gcd_original.py"
OUTPUT_DIR="eval_results/gcd_original"
STEP="step_100"
ERASE_TYPE="celebrity"
CSV="data/celebrity.csv"

METHODS=("alpha_delta" "speed")
GPUS=(0 1 2 3)

mkdir -p "$OUTPUT_DIR"

echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GCD Original Evaluation Started"
echo "   Methods: ${METHODS[*]}"
echo "   GPUs:    ${GPUS[*]} (PyTorch ResNet)"
echo "   TF MTCNN: CPU only (RTX 4090 incompatible with CUDA 10.0)"
echo "   Output:  $OUTPUT_DIR"
echo "=============================================================================="

PIDS=()
for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    gpu="${GPUS[$i]}"
    logfile="$OUTPUT_DIR/eval_${method}_gpu${gpu}.log"

    echo "[LAUNCH] $method -> GPU $gpu + CPU TF  (log: $logfile)"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON $EVAL_SCRIPT \
        --method "$method" \
        --erase_type "$ERASE_TYPE" \
        --step_name "$STEP" \
        --data_csv "$CSV" \
        --output_dir "$OUTPUT_DIR" \
        --gpu 0 \
        > "$logfile" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All ${#PIDS[@]} jobs launched."
echo "  PIDs: ${PIDS[*]}"
echo "  Waiting for completion..."
echo ""

wait "${PIDS[@]}"

echo ""
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL EVALUATION JOBS COMPLETE"
echo "=============================================================================="

# -----------------------------------------------------------------------------
# Merge summaries
# -----------------------------------------------------------------------------
echo ""
echo "Merging summaries..."
$PYTHON << 'PYEOF'
import os, glob, pandas as pd

outdir = "eval_results/gcd_original"
methods = ["alpha_delta", "speed"]

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
