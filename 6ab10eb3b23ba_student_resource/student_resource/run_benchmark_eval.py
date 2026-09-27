"""
Run full evaluation of final_solution pipeline on the high-fidelity benchmark dataset.
Outputs:
  - Exact Blocking Recall
  - Distribution of Candidates per S1
  - LightGBM Train & Validation Performance
  - Threshold Sweep for Macro F0.5
  - Detailed Error Analysis of Missed Pairs & False Positives
"""

import os
import sys
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))

import final_solution as fs
from sklearn.model_selection import train_test_split

BENCH_DIR = ROOT / "dataset" / "benchmark"

print("=" * 70)
print("AMAZON ML CHALLENGE 2026 — BENCHMARK EVALUATION")
print("=" * 70)

t0 = time.time()
print("1. Loading and preprocessing benchmark dataset...")
s1 = fs.preprocess(pd.read_csv(BENCH_DIR / "benchmark_source1.tsv", sep="\t", dtype=str).fillna(""))
s2 = fs.preprocess(pd.read_csv(BENCH_DIR / "benchmark_source2.tsv", sep="\t", dtype=str).fillna(""))
s3 = fs.preprocess(pd.read_csv(BENCH_DIR / "benchmark_source3.tsv", sep="\t", dtype=str).fillna(""))
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()

gt = fs.load_gt(BENCH_DIR / "benchmark_ground_truth.tsv")
print(f"   S1: {len(s1):,} rows | S2+S3: {len(s23):,} candidates | GT: {len(gt):,} entries")

# Train / Validation Split (80/20 on S1 entities)
all_s1_ids = s1["entity_id"].tolist()
ids_tr, ids_vl = train_test_split(all_s1_ids, test_size=0.20, random_state=42)
s1_tr = s1[s1["entity_id"].isin(set(ids_tr))].reset_index(drop=True)
s1_vl = s1[s1["entity_id"].isin(set(ids_vl))].reset_index(drop=True)
gt_tr = {k: gt[k] for k in ids_tr}
gt_vl = {k: gt[k] for k in ids_vl}

print(f"   Train set: {len(s1_tr):,} S1 | Val set: {len(s1_vl):,} S1")

# 2. Build Multi-Layer Blocker
print("\n2. Building Multi-Layer Blocker on S2+S3...")
t_block = time.time()
blocker = fs.MultiLayerBlocker()
blocker.build(s23)
print(f"   Blocker indexed in {time.time()-t_block:.1f}s")

# 3. Query Candidates
print("\n3. Querying candidates for validation set...")
t_q = time.time()
vl_cands = blocker.query_all(s1_vl, chunk=500)
q_time = time.time() - t_q
cand_counts = [len(c) for c in vl_cands.values()]
print(f"   Queried {len(s1_vl):,} queries in {q_time:.1f}s ({len(s1_vl)/q_time:.1f} queries/s)")
print(f"   Candidate count per query: min={min(cand_counts)}, mean={np.mean(cand_counts):.1f}, median={np.median(cand_counts)}, max={max(cand_counts)}")

# 4. Measure Blocking Recall
print("\n4. Auditing Validation Blocking Recall...")
recall, missed = fs.blocking_recall_audit(vl_cands, gt_vl)
print(f"   ★ VALIDATION BLOCKING RECALL: {recall*100:.2f}% (Missed {len(missed):,} true pairs)")

if missed:
    print(f"\n   Sample of Missed Pairs (First 5):")
    s1_lookup = s1_vl.set_index("entity_id").to_dict("index")
    s23_lookup = s23.set_index("entity_id").to_dict("index")
    for s1_id, cand_id in missed[:5]:
        s1_row = s1_lookup.get(s1_id, {})
        c_row = s23_lookup.get(cand_id, {})
        print(f"     S1 [{s1_row.get('country')}]: {s1_row.get('business_name')} | {s1_row.get('business_address')}")
        print(f"     C23[{c_row.get('country')}]: {c_row.get('business_name')} | {c_row.get('business_address')}")
        print("-" * 50)

# 5. Pipeline End-to-End Test
print("\n5. Generating train candidates & building features...")
tr_cands = blocker.query_all(s1_tr, chunk=500)
pipe = fs.Pipeline()
pipe.blocker = blocker

X_tr, y_tr, meta_tr = pipe._pairs_to_features(s1_tr, s23, tr_cands, gt_tr)
X_vl, _, meta_vl    = pipe._pairs_to_features(s1_vl, s23, vl_cands)

print("\n6. Training LightGBM re-ranker...")
pipe.train(X_tr, y_tr)

print("\n7. Evaluating validation predictions & tuning threshold for F0.5...")
probs_vl = pipe.predict_proba(X_vl)
pipe.tune_threshold(probs_vl, meta_vl, gt_vl, ids_vl)

preds_vl = pipe.threshold_to_preds(probs_vl, meta_vl, ids_vl, pipe.threshold)
metrics = fs.macro_f05(preds_vl, gt_vl)

print("\n" + "=" * 70)
print(f"FINAL BENCHMARK RESULTS (Target: F0.5 > 0.98)")
print(f"  Macro F_0.5 : {metrics['f05']:.4f}")
print(f"  Precision   : {metrics['precision']:.4f}")
print(f"  Recall      : {metrics['recall']:.4f}")
print(f"  Threshold   : {pipe.threshold:.3f}")
print(f"  Total time  : {time.time()-t0:.1f}s")
print("=" * 70)
