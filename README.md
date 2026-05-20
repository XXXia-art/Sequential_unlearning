# Concept Erasure for Stable Diffusion

Closed-form model editing methods for erasing specific concepts (objects, styles, celebrities) from pretrained Stable Diffusion models while preserving general generation capability.

## Methods

This repository implements four closed-form concept erasure methods:

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

# 1. Install main environment (for training, sampling, CLIP/FID evaluation)
conda create -n speed python=3.10
conda activate speed
pip install -r requirements.txt
```

## Datasets

| Dataset | File | Size | Description |
|---------|------|------|-------------|
| **Instance** | `data/instance.csv` | 600 concepts | id 1-200 = target (erase), 201-600 = retain |
| **Style** | `data/style.csv` | 600 concepts | id 1-200 = target, 201-600 = retain |
| **Celebrity** | `data/celebrity.csv` | 600 celebrities | id 1-200 = target, 201-600 = retain |
| **MS COCO** | `data/mscoco.csv` | 30K pairs | General generation capability evaluation |

## Quick Start

### 1. Concept Erasure (Editing)

Edit the UNet to erase target concepts while preserving retain concepts:
|  | optional parameters |
|---------|------|
| **method** | `alphaedit, alpha_delta, alpha_delta_v2, speed` | 
| **dataset** | `instance, style, celebrity` | 
| **group_size** | `2,1` |
| **gpu** | `/` | 


```bash
# Usage: bash scripts/train.sh <method> <dataset> <group_size> <gpu>
# Alpha_delta on Instance, 2 concepts per step
bash scripts/train.sh alpha_delta instance 2 0
```

**Advanced options:**

```bash
# Resume from step 50
START_STEP=50 bash scripts/train.sh speed instance 2 0

# Keep only checkpoints at step 10/20/.../100 (saves disk space)
KEEP_STEPS="10 20 30 40 50 60 70 80 90 100" bash scripts/train.sh alpha_delta instance 2 0

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

# COCO generalization sampling (single GPU)
bash scripts/sample.sh speed coco coco edit 0

# Original baseline (no edit_ckpt)
bash scripts/sample.sh alphaedit instance all original 0,1
```

**Advanced options:**

```bash
# Use a specific step checkpoint (defult step = 100)
STEP=50 bash scripts.sample.sh alpha_delta instance target edit 0

# Resume: skip already-sampled concepts
RESUME=true bash scripts.sample.sh speed instance retain edit 0,1

# Specify custom concepts for sampling, still need split
CONCEPTS="SpongeBob" bash scripts/sample.sh alpha_delta instance target edit 0
CONCEPTS="SpongeBob;Mickey Mouse" bash scripts/sample.sh speed instance target edit 0,1
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

#### 3.2 GCD Celebrity Evaluation (Requires Second Environment)

Celebrity evaluation uses the **original GIPHY celeb-detection-oss** codebase to perform face recognition on generated images. It answers the question: *"After erasing a celebrity, does the model still generate images that a face recognizer can identify as that celebrity?"*

**Prerequisite: you must generate the images first.** GCD eval reads the sampled images from disk; it does not run the diffusion model itself.

```bash
# 1. Generate images with the edited model (if not already done)
bash scripts/sample.sh alpha_delta celebrity all edit 0,1,2,3

# 2. (Optional) Generate baseline images with the original model
bash scripts/sample.sh alpha_delta celebrity all original 0,1,2,3
```

Images are expected at:
```
logs/{method}/celebrity/step_100/{concept}/edit/
```

---

**Step 1: Install GCD environment**

The GCD codebase requires **TensorFlow 1.x**, which is incompatible with the main PyTorch 2.x environment. You must create a separate conda env.

```bash
# Clone GCD code (not included in this repo due to large model files)
git clone https://github.com/Giphy/celeb-detection-oss.git celeb-detection-oss

# Create and activate GCD environment
conda create -n gcd_tf1 python=3.7
conda activate gcd_tf1
pip install -r celeb-detection-oss/requirements_gpu.txt
```

> **Why a separate environment?** The original GCD code relies on TensorFlow 1.15.2, which is incompatible with PyTorch 2.x and requires CUDA 10.0. The face detector (MTCNN) runs on CPU even on modern GPUs to avoid CUDA version conflicts; only the PyTorch ResNet recognizer uses GPU.

---

**Step 2: Run GCD evaluation (single method)**

Run `eval_gcd_original.py` inside the `gcd_tf1` environment. It will:
1. Load the GCD face detector + recognizer.
2. Scan every generated image under `logs/{method}/celebrity/step_100/`.
3. Check whether the erased celebrity is still recognized (top-1 and top-5).
4. Output per-concept and summary CSVs to `eval_results/gcd_original/`.

```bash
# Single method (equivalent to the direct python call above)
bash scripts/eval_celebrity.sh alpha_delta step_100 0

# Evaluate another method on GPU 1
bash scripts/eval_celebrity.sh speed step_100 1
```

Arguments:
- `method`: `alphaedit` | `speed` | `alpha_delta` | `alpha_delta_v2` | `all`
- `step`: Step directory to evaluate, e.g., `step_100`
- `gpus`: Comma-separated GPU ids, e.g., `0` or `0,1,2,3`

Metrics:
- **Acc_e** (erase accuracy): % of target images where the erased celebrity is **still recognized**. Lower is better.
- **Acc_r** (retain accuracy): % of retain images where the celebrity is **correctly recognized**. Higher is better.
- **H_o** (harmonic mean): Overall trade-off = `2 × Acc_r × (100 - Acc_e) / (Acc_r + (100 - Acc_e))`. Higher is better.

Results:
- `eval_results/gcd_original/{method}_per_concept.csv` — per-celebrity accuracy
- `eval_results/gcd_original/{method}_summary.csv` — aggregated top-1 / top-5 summary

---

**Step 3: Parallel GCD evaluation (multi-GPU, multi-method)**

Evaluate all methods in parallel, one per GPU:

```bash
bash scripts/eval_celebrity.sh all step_100 0,1,2,3
```

Advanced:

```bash
# Evaluate only a subset of methods
GCD_METHODS="alpha_delta speed" bash scripts/eval_celebrity.sh all step_100 0,1

# Use a specific Python binary (if conda auto-detection fails)
GCD_PYTHON=/opt/conda/envs/gcd_tf1/bin/python bash scripts/eval_celebrity.sh all step_100 0,1,2,3
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


## Default Hyperparameters

The following hyperparameters are shared across all methods (overridable via environment variables when calling `scripts/train.sh`):

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
