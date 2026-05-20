# Concept Erasure for Stable Diffusion

Closed-form model editing methods for erasing specific concepts (objects, styles, celebrities) from pretrained Stable Diffusion models while preserving general generation capability.

## Methods

| Method | File | Core Mechanism |
|--------|------|----------------|
| **AlphaEdit** | `Alphaedit.py` | Closed-form SVD-based editing of UNet cross-attention `to_k`/`to_v` weights |
| **SPEED** | `speed.py` | Closed-form editing + DFA perturbation generation + IPF retain sample filtering |
| **Alpha_delta** | `Alpha_delta.py` | AlphaEdit + DeltaEdit history projection to prevent catastrophic forgetting |
| **Alpha_delta_v2** | `Alpha_delta_v2.py` | Enhanced Alpha_delta with edit direction analysis and top-1 historical projection |

All methods support **sequential editing** (erasing concepts one by one) by loading previously edited checkpoints.

## Installation

```bash
git clone https://github.com/XXXia-art/Sequential_unlearning.git
cd Sequential_unlearning

# Main environment (training, sampling, CLIP/FID evaluation)
conda create -n speed python=3.10
conda activate speed
pip install -r requirements.txt
```

> **GCD environment (optional):** Celebrity face recognition evaluation requires TensorFlow 1.x in a separate conda env. See [GCD Evaluation](#gcd-celebrity-evaluation) below.

## Quick Start

```bash
# 1. Train: erase 2 concepts per step on GPU 0
bash scripts/train.sh alpha_delta instance 2 0

# 2. Sample: generate edited images
bash scripts/sample.sh alpha_delta instance target edit 0,1

# 3. Sample: generate original baseline images
bash scripts/sample.sh alpha_delta instance target original 0,1

# 4. Evaluate: CLIP Score + FID
bash scripts/eval.sh alpha_delta instance target 0
```

> Use `CONCEPTS="SpongeBob;Mickey Mouse"` before any command to evaluate on specific concepts only.

## Usage

### Training (`scripts/train.sh`)

```bash
bash scripts/train.sh <method> <dataset> <group_size> <gpu>
```

- `method`: `alphaedit` | `speed` | `alpha_delta` | `alpha_delta_v2`
- `dataset`: `instance` | `style` | `celebrity`
- `group_size`: concepts per step (`2` or `1`)
- `gpu`: GPU id

Checkpoints are saved to `logs/{method}/{dataset}/step_*/weight.pt`.

**Common environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `STEP` | `100` | Target step for sampling/evaluation |
| `START_STEP` | `1` | Resume training from this step |
| `KEEP_STEPS` | — | Prune intermediate checkpoints (e.g. `"10 20 30"`) |
| `SD_CKPT` | `CompVis/stable-diffusion-v1-4` | SD checkpoint (HF Hub id or local path) |
| `SAVE_ROOT` | `logs/{method}/{dataset}` | Custom output directory |
| `DELTA_COEF` | `0.90` | DeltaEdit coefficient (alpha_delta only) |
| `ETA` | `1.0` | History noise threshold multiplier (alpha_delta only) |

### Sampling (`scripts/sample.sh`)

```bash
bash scripts/sample.sh <method> <dataset> <split> <mode> <gpus>
```

- `split`: `target` | `retain` | `all` | `coco`
- `mode`: `edit` (edited model) | `original` (base model baseline)
- `gpus`: comma-separated GPU ids, e.g. `0,1,2,3`

```bash
# Edited model sampling (multi-GPU sharding)
bash scripts/sample.sh alpha_delta instance target edit 0,1,2,3

# Original baseline (for FID reference)
bash scripts/sample.sh alpha_delta instance all original 0,1,2,3

# COCO generalization sampling
bash scripts/sample.sh speed coco coco edit 0

# Resume: skip already-sampled concepts
RESUME=true bash scripts/sample.sh speed instance retain edit 0,1
```

**Output structure:**
```
logs/{method}/{dataset}/step_100/{concept}/edit/     # Edited model
data/pretrain/{dataset}/{concept}/original/          # Original baseline
```

### Evaluation

#### CLIP Score + FID (`scripts/eval.sh`)

```bash
bash scripts/eval.sh <method> <dataset> <split> <gpu>
```

```bash
# Target concepts (id 1-200)
bash scripts/eval.sh alpha_delta instance target 0

# Retain concepts (id 201-600)
bash scripts/eval.sh speed instance retain 0

# COCO generalization
bash scripts/eval.sh speed coco coco 0

# Evaluate all methods sequentially
bash scripts/eval.sh all instance retain 0
```

Metrics:
- **CLIP Score (CS)**: Text-image alignment. Higher is better for retain, lower for target.
- **FID**: Distribution distance between edited and original generations. Lower is better.

#### GCD Celebrity Evaluation (`scripts/eval_celebrity.sh`)

Celebrity face recognition evaluation using the original GIPHY `celeb-detection-oss` codebase. Requires a separate TensorFlow 1.x environment.

```bash
# 1. Install GCD environment (once)
git clone https://github.com/Giphy/celeb-detection-oss.git celeb-detection-oss
conda create -n gcd_tf1 python=3.7
conda activate gcd_tf1
pip install -r celeb-detection-oss/requirements_gpu.txt

# 2. Run evaluation
bash scripts/eval_celebrity.sh alpha_delta 0

# All methods in parallel
bash scripts/eval_celebrity.sh all 0,1,2,3
```

Metrics:
- **Acc_e**: % of target images still recognized (lower is better)
- **Acc_r**: % of retain images correctly recognized (higher is better)
- **H_o**: Harmonic mean = `2 × Acc_r × (100 - Acc_e) / (Acc_r + (100 - Acc_e))`

## Project Structure

```
.
├── data/
│   ├── instance.csv              # 600 instance concepts
│   ├── style.csv                 # 600 style concepts
│   ├── celebrity.csv             # 600 celebrities
│   ├── mscoco.csv                # 30K COCO captions
│   └── pretrain/
│       ├── instance/{concept}/original/
│       ├── style/{concept}/original/
│       └── celebrity/{concept}/original/
├── logs/
│   ├── alphaedit/
│   ├── speed/
│   ├── alpha_delta/
│   └── alpha_delta_v2/
│       └── {dataset}/
│           ├── config.txt
│           ├── noise_e_log.txt
│           ├── noise_E_log.txt
│           └── step_*/
│               └── weight.pt
├── scripts/
│   ├── train.sh
│   ├── sample.sh
│   ├── eval.sh
│   └── eval_celebrity.sh
└── src/
    └── clip_score_cal.py
```

## Datasets

| Dataset | File | Size | Description |
|---------|------|------|-------------|
| **Instance** | `data/instance.csv` | 600 concepts | id 1-200 = target (erase), 201-600 = retain |
| **Style** | `data/style.csv` | 600 concepts | id 1-200 = target, 201-600 = retain |
| **Celebrity** | `data/celebrity.csv` | 600 celebrities | id 1-200 = target, 201-600 = retain |
| **MS COCO** | `data/mscoco.csv` | 30K pairs | General generation capability evaluation |

## Default Hyperparameters

The following hyperparameters are shared across all methods (overridable via environment variables):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `params` | `V` | Edited projection: `V` = to_v, `K` = to_k, `KV` = to_k + to_v |
| `threshold` | `1e-1` | SVD energy threshold for constructing retain projection matrix |
| `retain_scale` | `1.0` | Scaling factor for retain concept loss |
| `seed` | `0` | Random seed for reproducibility |
| `dtype` | `float32` | Model weight dtype |

**Method-specific extra parameters:**

| Parameter | Default | Methods | Description |
|-----------|---------|---------|-------------|
| `delta_coef` | `0.90` | Alpha_delta, Alpha_delta_v2 | DeltaEdit update coefficient |
| `eta` | `1.0` | Alpha_delta, Alpha_delta_v2 | History noise threshold multiplier |

**Dataset-specific anchors:**

| Dataset | Anchor | Usage |
|---------|--------|-------|
| Instance | `" "` (space) | Null anchor for instance concepts |
| Style | `"art"` | Art-style anchor for style concepts |
| Celebrity | `"person"` | Person anchor for celebrity concepts |
