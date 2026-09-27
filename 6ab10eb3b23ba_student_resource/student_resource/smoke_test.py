"""
Quick smoke test using first 50k rows of training data to validate the pipeline works end-to-end.
"""
import sys, gc, time, pickle
sys.path.insert(0, r'd:\AMAZON ML CHALLENGE\6ab10eb3b23ba_student_resource\student_resource\code\business_entity_resolution\src')
sys.stdout.reconfigure(encoding='utf-8')

# Monkey-patch paths for this test
import final_solution as fs
from pathlib import Path

TRAIN_DIR = Path(r'd:\AMAZON ML CHALLENGE\6ab10eb3b23ba_student_resource\student_resource\dataset\train')
OUT_DIR   = Path(r'd:\AMAZON ML CHALLENGE\6ab10eb3b23ba_student_resource\student_resource\output')
MODEL_DIR = Path(r'd:\AMAZON ML CHALLENGE\6ab10eb3b23ba_student_resource\student_resource\models')
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

SAMPLE_S1  = 5_000
SAMPLE_S23 = 20_000

print("Loading sample data...")
s1 = fs.preprocess(pd.read_csv(TRAIN_DIR / 'train_source1.tsv', sep='\t', encoding='utf-8', nrows=SAMPLE_S1, dtype=str).fillna(''))
s2 = fs.preprocess(pd.read_csv(TRAIN_DIR / 'train_source2.tsv', sep='\t', encoding='utf-8', nrows=SAMPLE_S23, dtype=str).fillna(''))
s3 = fs.preprocess(pd.read_csv(TRAIN_DIR / 'train_source3.tsv', sep='\t', encoding='utf-8', nrows=SAMPLE_S23, dtype=str).fillna(''))
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3; gc.collect()

gt = fs.load_gt(TRAIN_DIR / 'train_ground_truth.tsv')
# Filter GT to entities in s1 sample
s1_ids_set = set(s1['entity_id'])
gt_sample = {k: v for k, v in gt.items() if k in s1_ids_set}
print(f"GT entries in sample: {len(gt_sample)}")

# Split
ids_tr, ids_vl = train_test_split(s1['entity_id'].tolist(), test_size=0.2, random_state=42)
s1_tr = s1[s1['entity_id'].isin(set(ids_tr))].reset_index(drop=True)
s1_vl = s1[s1['entity_id'].isin(set(ids_vl))].reset_index(drop=True)
gt_tr = {k: gt_sample[k] for k in ids_tr if k in gt_sample}
gt_vl = {k: gt_sample[k] for k in ids_vl if k in gt_sample}

print(f"S1 train: {len(s1_tr):,}  S1 val: {len(s1_vl):,}")
print(f"GT train: {len(gt_tr):,}  GT val: {len(gt_vl):,}")

# Build pipeline
pipe = fs.Pipeline()
pipe.blocker.build(s23)

print("\nGenerating train candidates...")
t0 = time.time()
tr_cands = pipe.blocker.query_all(s1_tr, chunk=1000)
print(f"  Done in {time.time()-t0:.1f}s")

print("Generating val candidates...")
t0 = time.time()
vl_cands = pipe.blocker.query_all(s1_vl, chunk=1000)
print(f"  Done in {time.time()-t0:.1f}s")

# Blocking recall
recall, missed = fs.blocking_recall_audit(vl_cands, gt_vl)

# Features
print("\nBuilding training features...")
X_tr, y_tr, meta_tr = pipe._pairs_to_features(s1_tr, s23, tr_cands, gt_tr)
print("Building val features...")
X_vl, _, meta_vl    = pipe._pairs_to_features(s1_vl, s23, vl_cands)

# Train
pipe.train(X_tr, y_tr)

# Evaluate
probs_vl = pipe.predict_proba(X_vl)
pipe.tune_threshold(probs_vl, meta_vl, gt_vl, ids_vl)
preds_vl = pipe.threshold_to_preds(probs_vl, meta_vl, ids_vl, pipe.threshold)
m = fs.macro_f05(preds_vl, gt_vl)
print(f"\n=== SMOKE TEST RESULT ===")
print(f"F_0.5={m['f05']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
print(f"Blocking recall = {recall:.4f}")
print(f"Threshold = {pipe.threshold:.3f}")
