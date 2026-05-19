# Concept Erasure for Stable Diffusion

Closed-form model editing methods for erasing specific concepts (objects, styles, celebrities) from pretrained Stable Diffusion models while preserving general generation capability.

## Methods

This repository implements four closed-form concept erasure methods:

| Method | File | Core Mechanism |
|--------|------|----------------|
| **AlphaEdit** | `Alphaedit.py` | Closed-form SVD-based editing of UNet cross-attention `to_k`/`to_v` weights |
| **Alpha_delta** | `Alpha_delta.py` | AlphaEdit + DeltaEdit history projection to prevent catastrophic forgetting |
| **Alpha_delta_v2** | `Alpha_delta_v2.py` | Enhanced Alpha_delta with edit direction analysis and top-1 historical projection |
| **SPEED** | `speed.py` | Closed-form editing + DFA perturbation generation + IPF retain sample filtering |

All methods support **sequential editing** (erasing concepts one by one) by loading previously edited checkpoints.

## Environment Requirements

> ⚠️ **This project requires TWO separate Python environments.**

### Environment 1: Main Project Environment

For running editing methods, sampling, and general evaluation (CLIP Score / FID).

```bash
# Using conda (recommended)
conda create -n speed python=3.10
conda activate speed
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.3.0 + CUDA 12.1
- diffusers 0.32.2
- transformers 4.48.0
- torch-fidelity, lpips, kmeans-pytorch

### Environment 2: GCD Evaluation Environment

For celebrity face recognition evaluation using the original GCD codebase (TensorFlow 1.x MTCNN + PyTorch ResNet50).

```bash
# Using conda
conda create -n gcd_tf1 python=3.7
conda activate gcd_tf1
pip install -r celeb-detection-oss/requirements_gpu.txt
```

> **Why two environments?** The original GCD code relies on TensorFlow 1.15.2, which is incompatible with PyTorch 2.x and requires CUDA 10.0. These cannot coexist in a single environment.

## Installation

```bash
git clone <repository-url>
cd Instance_log

# 1. Install main environment
conda create -n speed python=3.10
conda activate speed
pip install -r requirements.txt

# 2. Download InceptionV3 weights for FID computation
# Place at: cache/weights-inception-2015-12-05-6726825d.pth

# 3. (Optional) Install GCD environment for celebrity evaluation
conda create -n gcd_tf1 python=3.7
conda activate gcd_tf1
pip install -r celeb-detection-oss/requirements_gpu.txt
```

## Datasets

| Dataset | File | Size | Description |
|---------|------|------|-------------|
| **Instance** | `data/instance.csv` | 600 concepts | id 1-200 = target (erase), 201-600 = retain |
| **Style** | `data/style.csv` | ~1700 concepts | Artistic styles for erasure |
| **Celebrity** | `data/celebrity.csv` | 220 celebrities | For face recognition evaluation |
| **MS COCO** | `data/mscoco.csv` | 30K pairs | General generation capability evaluation |

## Quick Start

### 1. Concept Erasure (Editing)

Edit the UNet to erase target concepts while preserving retain concepts:

```bash
# Usage: bash scripts/train.sh <method> <dataset> <group_size> <gpu>

# AlphaEdit on Instance, 1 concept per step
bash scripts/train.sh alphaedit instance 1 0

# SPEED on Instance, 1 concept per step
bash scripts/train.sh speed instance 1 0

# Alpha_delta on Instance, 1 concept per step
bash scripts/train.sh alpha_delta instance 1 0

# Alpha_delta_v2 on Style, 2 concepts per step
bash scripts/train.sh alpha_delta_v2 style 2 0

# SPEED on Celebrity, 2 concepts per step
bash scripts/train.sh speed celebrity 2 0
```

**Advanced options:**

```bash
# Resume from step 50
START_STEP=50 bash scripts/train.sh speed instance 1 0

# Keep only checkpoints at step 10/20/.../100 (saves disk space)
KEEP_STEPS="10 20 30 40 50 60 70 80 90 100" bash scripts/train.sh alpha_delta instance 4 0

# Override hyperparameters
SAVE_ROOT=logs/MyExp DELTA_COEF=0.95 bash scripts/train.sh alpha_delta style 2 0
```

Edited checkpoints are saved under `logs/{method}/{dataset}/step_*/weight.pt`.

### 2. Image Sampling

Generate images using the edited model to evaluate erasure quality:

```bash
# Usage: bash scripts/sample.sh <method> <dataset> <split> <mode> <gpus>

# Instance target/retain (concept sharding across GPUs)
bash scripts/sample.sh alpha_delta instance target edit 0,1,2,3
bash scripts/sample.sh speed instance retain edit 0,1,2,3

# Style target/retain
bash scripts/sample.sh alpha_delta_v2 style target edit 0,1,2,3
bash scripts/sample.sh speed style retain edit 0,1,2,3

# Celebrity target/retain/all
bash scripts/sample.sh alpha_delta celebrity target edit 0,1,2,3
bash scripts/sample.sh speed celebrity retain edit 0,1,2,3
bash scripts/sample.sh alpha_delta_v2 celebrity all edit 0,1,2,3

# COCO generalization sampling (single GPU)
bash scripts/sample.sh speed coco coco edit 0

# Original baseline (no edit_ckpt)
bash scripts/sample.sh alphaedit instance target original 0,1
bash scripts/sample.sh speed style retain original 0,1
```

**Advanced options:**

```bash
# Use a specific step checkpoint
STEP=50 bash scripts/sample.sh alpha_delta instance target edit 0

# Resume: skip already-sampled concepts
RESUME=true bash scripts/sample.sh speed instance retain edit 0,1
```

Generated images are organized as:
```
logs/{method}/{dataset}/{step}/{concept}/edit/     # Edited model generation
data/pretrain/{dataset}/{step}/{concept}/original/ # Original model generation (baseline)
```

### 3. Evaluation

#### 3.1 CLIP Score + FID (Main Environment)

```bash
# Usage: bash scripts/eval.sh <method> <dataset> <split> <gpu>

# Instance retain (id 201-600)
bash scripts/eval.sh alpha_delta instance retain 0

# Instance target (id 1-200)
bash scripts/eval.sh speed instance target 0

# Style retain
bash scripts/eval.sh alpha_delta_v2 style retain 1

# COCO
bash scripts/eval.sh speed coco coco 0

# Evaluate all methods sequentially
bash scripts/eval.sh all instance retain 0
```

**Advanced options:**

```bash
# Use a specific step
STEP=50 bash scripts/eval.sh alpha_delta style retain 0

# Evaluate specific concept(s)
CONCEPTS="SpongeBob" bash scripts/eval.sh alpha_delta instance target 0
CONCEPTS="SpongeBob;Mickey Mouse" bash scripts/eval.sh speed instance target 0
```

Metrics:
- **CLIP Score (CS)**: Text-image alignment. Higher is better for retain, lower for target.
- **FID**: Distribution distance between edited and original generations. Lower is better.

#### 3.2 GCD Celebrity Evaluation (GCD Environment)

```bash
# Activate GCD environment
conda activate gcd_tf1

# Run original GCD evaluation (Acc_e, Acc_r, H_o)
CUDA_VISIBLE_DEVICES=0 python eval_gcd_original.py \
    --method alpha_delta \
    --erase_type celebrity \
    --step_name step_100 \
    --gpu 0
```

Metrics:
- **Acc_e**: % of target images still recognized (lower is better)
- **Acc_r**: % of retain images correctly recognized (higher is better)
- **H_o**: Harmonic mean = 2 × Acc_r × (100 - Acc_e) / (Acc_r + (100 - Acc_e))

#### 3.3 Parallel Comprehensive Evaluation

```bash
# 4-GPU parallel GCD evaluation
bash scripts/eval_celebrity.sh
```

### 4. Full Pipeline

Run the complete train → sample → evaluate pipeline:

```bash
# Speed: train + COCO sample/eval + retain20 sample/eval
METHOD=speed GPU=0 bash scripts/run_method_instance_step100_pipeline.sh

# Alpha_delta_v2: full 3-dataset pipeline
bash scripts/run_all_alpha_delta_v2.sh
```

## Evaluation Metrics

| Metric | Script | Description | Target | Retain |
|--------|--------|-------------|--------|--------|
| **CLIP Score** | `src/clip_score_cal.py` | Text-image alignment | Lower ↓ | Higher ↑ |
| **FID** | `src/clip_score_cal.py` | Fréchet Inception Distance | Lower ↓ | Lower ↓ |
| **LPIPS** | Perceptual similarity metric | - | Lower ↓ |
| **Acc_e** | `eval_gcd_original.py` | Target recognition rate | Lower ↓ | - |
| **Acc_r** | `eval_gcd_original.py` | Retain recognition rate | - | Higher ↑ |
| **H_o** | `eval_gcd_original.py` | Harmonic mean | Higher ↑ | Higher ↑ |

## Project Structure

```
Instance_log/
├── Alphaedit.py              # AlphaEdit method
├── Alpha_delta.py            # Alpha_delta method
├── Alpha_delta_v2.py         # Alpha_delta_v2 method (with direction analysis)
├── speed.py                  # SPEED method
├── sample.py                 # Basic sampling script
├── sample2.py                # DataLoader-based batch sampling
├── sample_fast.py            # Fast multi-prompt batched sampling
├── eval_gcd_original.py      # Original GCD evaluation (requires gcd_tf1 env)
├── requirements.txt          # Main environment dependencies
│
├── scripts/                  # Training / Sampling / Evaluation / Pipeline scripts
│   ├── tra_*.sh              # Training / editing scripts
│   ├── sample_*.sh           # Sampling scripts
│   ├── eval_*.sh             # Evaluation scripts
│   ├── run_*.sh              # Orchestration / pipeline scripts
│   └── auto_*.sh             # Auto-monitoring scripts
│
├── src/                      # Source utilities
│   ├── clip_score_cal.py     # CLIP Score + FID computation
│   ├── template.py           # Prompt templates (instance/style/celebrity)
│   ├── utils.py              # Common utilities (seed, token encoding, image processing)
│   └── gcd_concept_mapping_v2.json  # Celebrity → GCD label mapping
│
├── data/                     # Dataset CSV files
│   ├── instance.csv
│   ├── style.csv
│   ├── celebrity.csv
│   └── mscoco.csv
│
├── celeb-detection-oss/      # Original GCD codebase (TF1.x + PyTorch)
│   ├── examples/inference.py # Single-image GCD inference
│   ├── requirements_cpu.txt  # GCD environment (CPU)
│   └── requirements_gpu.txt  # GCD environment (GPU)
│
├── cache/                    # Model weights cache
├── logs/                     # Training checkpoints and sampled images
└── eval_results/             # Evaluation CSV outputs
```

## Notes

### Sequential Editing

All methods support sequential editing by loading a previously edited checkpoint:

```bash
# After step_001, edit step_002 starting from step_001's checkpoint
python speed.py \
    --target_concept step_002 \
    --edit_ckpt logs/Speed/instance/step_001/weight.pt \
    ...
```

### Checkpoint Formats

- AlphaEdit / Alpha_delta / Alpha_delta_v2 / SPEED: PyTorch `.pt` files
- The `sample*.py` scripts can load `.pt` or `.safetensors` checkpoints via `--edit_ckpt`


