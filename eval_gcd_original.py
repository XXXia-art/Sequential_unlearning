#!/usr/bin/env python3
"""
GCD Original (TF1.x + PyTorch) Evaluation for Celebrity Erasure

This script uses the ORIGINAL celeb-detection-oss codebase:
  - TensorFlow 1.x MTCNN for face detection
  - PyTorch ResNet50 + CenterLoss + GMM clustering for face recognition

This is the paper-standard evaluator. Run inside the gcd_tf1 conda env:
    CUDA_VISIBLE_DEVICES=0 /data/miniconda/envs/gcd_tf1/bin/python eval_gcd_original.py \
        --method alpha_delta --gpu 0

It evaluates BOTH top-1 and top-5 in a single pass to save time.

Metrics:
  - Acc_e: % of target images where the erased celebrity is STILL recognized
           (lower is better). Images with no face detected are excluded.
  - Acc_r: % of retain images where the retained celebrity is recognized
           (higher is better). Images with no face detected are excluded.
  - H_o  : Harmonic mean = 2 * Acc_r * (100 - Acc_e) / (Acc_r + (100 - Acc_e))
           (higher is better).

Output:
  - eval_results/gcd_original/{method}_per_concept.csv
  - eval_results/gcd_original/{method}_summary.csv
"""

import os
import sys
import argparse
import glob
import json
import warnings
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import skimage.io as io

# ---------------------------------------------------------------------------
# Configure GCD environment variables BEFORE importing GCD modules
# ---------------------------------------------------------------------------
GCD_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'celeb-detection-oss')
RESOURCES_PATH = os.path.join(GCD_ROOT, 'examples', 'resources')

os.environ['APP_DATA_DIR'] = RESOURCES_PATH
os.environ['APP_RECOGNITION_WEIGHTS_FILE'] = 'face_recognition/best_model_states.pkl'
os.environ['APP_FACE_MARGIN'] = '0.2'
os.environ['APP_FACE_SIZE'] = '224'
os.environ['APP_USE_CUDA'] = 'false'   # TF MTCNN always on CPU (4090 incompatible with CUDA 10.0)
os.environ['USE_CUDA'] = 'true'        # PyTorch ResNet on GPU

sys.path.insert(0, os.path.abspath(GCD_ROOT))

from model_training.preprocessors.face_detection.face_detector import FaceDetector
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.helpers.labels import Labels
from model_training.utils import preprocess_image

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Load concept -> GCD label mapping
# ---------------------------------------------------------------------------
MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'gcd_concept_mapping_v2.json')
with open(MAPPING_PATH, 'r') as f:
    CONCEPT_TO_GCD = json.load(f)

GCD_TO_CONCEPT = {v: k for k, v in CONCEPT_TO_GCD.items()}


# ---------------------------------------------------------------------------
# GCD Batch Classifier
# ---------------------------------------------------------------------------
class OriginalGCDClassifier:
    def __init__(self, use_cuda=True, face_size=224):
        self.face_size = face_size
        self.labels = Labels(resources_path=RESOURCES_PATH)

        # CRITICAL: Initialize PyTorch GPU FIRST, then TF CPU.
        # If TF session is created before PyTorch CUDA init, it causes segfault.
        self.face_recognizer = FaceRecognizer(
            labels=self.labels,
            resources_path=RESOURCES_PATH,
            top_n=5,  # Always compute top-5; we'll derive top-1 from it
            use_cuda=use_cuda
        )
        self.face_detector = FaceDetector(
            RESOURCES_PATH,
            margin=0.2,
            use_cuda=False,  # Always CPU for TF to avoid CUDA 10.0 incompatibility
            gpu_memory_fraction=0.9
        )

    def classify_image(self, image_path):
        """
        Returns:
            []                     – no face detected
            [(label_name, prob), …] – top-5 predictions aggregated across clusters
        """
        try:
            img = io.imread(image_path)
        except Exception:
            return []

        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        img = img[:, :, :3]

        face_results = self.face_detector.perform_single(img)
        if not face_results:
            return []

        face_images = []
        for face, _ in face_results:
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
            face_images.append(preprocess_image(face, self.face_size))

        if not face_images:
            return []

        predictions = self.face_recognizer.perform(face_images)
        if not predictions:
            return []

        # Aggregate predictions across clusters by max probability
        label_probs = {}
        for cluster_preds, _ in predictions:
            for label_obj, prob in cluster_preds:
                name = label_obj.folder_name
                label_probs[name] = max(label_probs.get(name, 0.0), float(prob))

        sorted_labels = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_labels[:5]


# ---------------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------------
def evaluate_concept(classifier, edit_dir, concept_name, cache_file):
    png_files = sorted(glob.glob(os.path.join(edit_dir, '*.png')))
    if not png_files:
        return {1: (0, 0), 5: (0, 0)}

    cache = {}
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    correct = {1: 0, 5: 0}
    total = 0

    for path in png_files:
        fname = os.path.basename(path)
        if fname in cache:
            preds = cache[fname]
        else:
            preds = classifier.classify_image(path)
            cache[fname] = preds
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)

        if not preds:
            continue

        total += 1
        mapped_names = [GCD_TO_CONCEPT.get(n, n) for n, _ in preds]
        if concept_name in mapped_names[:1]:
            correct[1] += 1
        if concept_name in mapped_names[:5]:
            correct[5] += 1

    return {1: (correct[1], total), 5: (correct[5], total)}


def evaluate_method(method, classifier, df, edit_root, output_dir):
    target_df = df[df['id'] <= 200].copy()
    retain_df = df[(df['id'] >= 201) & (df['id'] <= 600)].copy()

    os.makedirs(output_dir, exist_ok=True)
    per_concept_csv = os.path.join(output_dir, f'{method}_per_concept.csv')
    summary_csv = os.path.join(output_dir, f'{method}_summary.csv')
    cache_dir = os.path.join(output_dir, '.cache', method)
    os.makedirs(cache_dir, exist_ok=True)

    records = []
    splits = [('target', target_df), ('retain', retain_df)]

    for split_name, split_df in splits:
        print(f'\n[{method.upper()}] Evaluating {split_name}: {len(split_df)} concepts ...')
        t_start = time.time()
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f'{method} {split_name}'):
            concept = row['concept'].strip()
            edit_dir = os.path.join(edit_root, concept, 'edit')

            if not os.path.isdir(edit_dir):
                rec = {
                    'concept': concept,
                    'split': split_name,
                    'correct_top1': 0, 'total_top1': 0, 'accuracy_top1': 0.0,
                    'correct_top5': 0, 'total_top5': 0, 'accuracy_top5': 0.0,
                    'status': 'missing_dir'
                }
            else:
                cache_file = os.path.join(cache_dir, f'{concept.replace(" ", "_")}.json')
                res = evaluate_concept(classifier, edit_dir, concept, cache_file)
                c1, t1 = res[1]
                c5, t5 = res[5]
                rec = {
                    'concept': concept,
                    'split': split_name,
                    'correct_top1': c1, 'total_top1': t1,
                    'accuracy_top1': round(c1 / t1 * 100.0, 4) if t1 > 0 else 0.0,
                    'correct_top5': c5, 'total_top5': t5,
                    'accuracy_top5': round(c5 / t5 * 100.0, 4) if t5 > 0 else 0.0,
                    'status': 'success'
                }

            records.append(rec)

        elapsed = time.time() - t_start
        print(f'[{method.upper()}] {split_name} done in {elapsed/60:.1f} min')
        # Save incremental results after each split
        pd.DataFrame(records).to_csv(per_concept_csv, index=False)

    # Build summary for both top-1 and top-5
    results_df = pd.DataFrame(records)
    all_summaries = []
    for top_n in [1, 5]:
        summary_rows = []
        for split_name in ['target', 'retain']:
            split_data = results_df[results_df['split'] == split_name]
            split_data = split_data[split_data['status'] == 'success']
            total_images = split_data[f'total_top{top_n}'].sum()
            correct_images = split_data[f'correct_top{top_n}'].sum()
            acc = (correct_images / total_images * 100.0) if total_images > 0 else 0.0
            summary_rows.append({
                'method': method,
                'top_n': top_n,
                'split': split_name,
                'concepts_evaluated': len(split_data),
                'total_images': int(total_images),
                'correct_images': int(correct_images),
                'accuracy': round(acc, 2),
            })

        summary_df = pd.DataFrame(summary_rows)
        target_acc = summary_df[summary_df['split'] == 'target']['accuracy'].values
        retain_acc = summary_df[summary_df['split'] == 'retain']['accuracy'].values
        acc_e = target_acc[0] if len(target_acc) > 0 else 0.0
        acc_r = retain_acc[0] if len(retain_acc) > 0 else 0.0
        h_o = (2 * acc_r * (100 - acc_e)) / (acc_r + (100 - acc_e)) if (acc_r + (100 - acc_e)) > 0 else 0.0

        overall_row = {
            'method': method,
            'top_n': top_n,
            'split': 'overall',
            'concepts_evaluated': len(results_df[results_df['status'] == 'success']),
            'total_images': int(summary_df['total_images'].sum()),
            'correct_images': int(summary_df['correct_images'].sum()),
            'accuracy': round(h_o, 2),
            'acc_e': round(acc_e, 2),
            'acc_r': round(acc_r, 2),
            'h_o': round(h_o, 2),
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)
        all_summaries.append(summary_df)

        print(f'\n[{method.upper()} SUMMARY] Top-{top_n}')
        print(f'  Acc_e : {acc_e:.2f}%')
        print(f'  Acc_r : {acc_r:.2f}%')
        print(f'  H_o   : {h_o:.2f}')

    final_summary = pd.concat(all_summaries, ignore_index=True)
    final_summary.to_csv(summary_csv, index=False)
    print(f'\nSaved : {per_concept_csv}')
    print(f'Saved : {summary_csv}')
    return final_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Evaluate celebrity erasure with original GCD')
    parser.add_argument('--method', type=str, required=True,
                        choices=['alpha_delta', 'alpha_delta_v2', 'speed', 'uce', 'rece'],
                        help='Method to evaluate')
    parser.add_argument('--erase_type', type=str, default='celebrity',
                        help='Erase type sub-directory name')
    parser.add_argument('--step_name', type=str, default='step_100',
                        help='Step directory name')
    parser.add_argument('--data_csv', type=str, default='data/celebrity.csv',
                        help='CSV with id,concept columns')
    parser.add_argument('--edit_root', type=str, default='',
                        help='Override edit image root (default: logs/{MethodDir}/celebrity/step_100)')
    parser.add_argument('--output_dir', type=str, default='eval_results/gcd_original',
                        help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id for PyTorch ResNet (TF MTCNN always uses CPU)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Run PyTorch ResNet on CPU as well (very slow)')
    args = parser.parse_args()

    method_dir_map = {
        'alpha_delta': 'Alpha_delta',
        'alpha_delta_v2': 'Alpha_delta_v2',
        'speed': 'Speed_2step',
        'uce': 'UCE_2step',
        'rece': 'RECE_2step',
    }
    method_dir = method_dir_map[args.method]

    if args.edit_root:
        edit_root = args.edit_root
    else:
        edit_root = os.path.join('logs', method_dir, args.erase_type, args.step_name)

    if not os.path.isdir(edit_root):
        print(f'[ERROR] Edit root not found: {edit_root}')
        sys.exit(1)

    df = pd.read_csv(args.data_csv)
    df = df.sort_values('id').reset_index(drop=True)
    df['concept'] = df['concept'].dropna().astype(str).str.strip()

    use_cuda = not args.no_cuda

    print('[INFO] Loading original GCD classifier (TF CPU MTCNN + PyTorch GPU ResNet50)...')
    print('[WARN] TF 1.x MTCNN runs on CPU because RTX 4090 does not support CUDA 10.0')
    classifier = OriginalGCDClassifier(use_cuda=use_cuda)
    print(f'[INFO] GCD loaded. Mapped {len(CONCEPT_TO_GCD)}/600 concepts.')

    evaluate_method(args.method, classifier, df, edit_root, args.output_dir)
    print('\n[DONE] Evaluation complete.')


if __name__ == '__main__':
    main()
