"""
Create a high-fidelity Benchmark Dataset for fast, reliable local evaluation.

Strategy:
  1. Sample 5,000 S1 entities (mix of US and India, singletons and multi-match entities).
  2. For those 5,000 S1 entities, get ALL their true matched entity IDs from ground truth.
  3. Extract ALL true matches from Source 2 and Source 3 so blocking recall can legitimately reach 100%.
  4. Add ~40,000 negative distractors from Source 2 and Source 3.
  5. Save to student_resource/dataset/benchmark/
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "dataset" / "train"
BENCH_DIR = ROOT / "dataset" / "benchmark"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

N_S1 = 5_000
N_DISTRACTORS = 40_000

print(f"Creating Benchmark Dataset in {BENCH_DIR} ...")
t0 = time.time()

# 1. Load Ground Truth
print("1. Loading ground truth...")
gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
print(f"   Loaded {len(gt_df):,} GT rows in {time.time()-t0:.1f}s")

# 2. Sample 5,000 S1 IDs (matching + singletons)
matching = gt_df[gt_df["matched_entity_ids"].str.strip() != ""]
singletons = gt_df[gt_df["matched_entity_ids"].str.strip() == ""]
n_singletons = int(N_S1 * 0.056)
n_matching = N_S1 - n_singletons
s1_sampled_df = pd.concat([
    matching.sample(n_matching, random_state=42),
    singletons.sample(n_singletons, random_state=42)
], ignore_index=True)

target_s1_ids = set(s1_sampled_df["source1_entity_id"])
print(f"   Selected {len(target_s1_ids):,} S1 entities ({n_matching:,} matching, {n_singletons:,} singletons)")

# Collect all true positive target IDs for S2 and S3
target_s2_ids = set()
target_s3_ids = set()
for mstr in s1_sampled_df["matched_entity_ids"]:
    mstr = mstr.strip()
    if not mstr:
        continue
    for mid in mstr.split(","):
        mid = mid.strip()
        if mid.startswith("S2-"):
            target_s2_ids.add(mid)
        elif mid.startswith("S3-"):
            target_s3_ids.add(mid)

print(f"   True matching candidates needed: {len(target_s2_ids):,} in S2, {len(target_s3_ids):,} in S3")

# 3. Filter Source 1
print("2. Filtering Source 1...")
t1 = time.time()
s1_full = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str).fillna("")
s1_bench = s1_full[s1_full["entity_id"].isin(target_s1_ids)].copy().reset_index(drop=True)
s1_bench.to_csv(BENCH_DIR / "benchmark_source1.tsv", sep="\t", index=False)
del s1_full
print(f"   Saved benchmark_source1.tsv ({len(s1_bench):,} rows) in {time.time()-t1:.1f}s")

# 4. Filter Source 2 (True positives + random distractors)
print("3. Filtering Source 2...")
t2 = time.time()
s2_chunks = []
n_dist_s2 = N_DISTRACTORS // 2
dist_s2_collected = 0

for chunk in pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str, chunksize=500_000):
    chunk.fillna("", inplace=True)
    # Extract true matches
    matches = chunk[chunk["entity_id"].isin(target_s2_ids)]
    if len(matches) > 0:
        s2_chunks.append(matches)
    # Collect distractors
    if dist_s2_collected < n_dist_s2:
        non_matches = chunk[~chunk["entity_id"].isin(target_s2_ids)]
        take = min(n_dist_s2 - dist_s2_collected, len(non_matches) // 50)
        if take > 0:
            s2_chunks.append(non_matches.sample(take, random_state=42))
            dist_s2_collected += take

s2_bench = pd.concat(s2_chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])
s2_bench.to_csv(BENCH_DIR / "benchmark_source2.tsv", sep="\t", index=False)
del s2_chunks
found_s2 = s2_bench["entity_id"].isin(target_s2_ids).sum()
print(f"   Saved benchmark_source2.tsv ({len(s2_bench):,} rows, {found_s2}/{len(target_s2_ids)} true matches present) in {time.time()-t2:.1f}s")

# 5. Filter Source 3 (True positives + random distractors)
print("4. Filtering Source 3...")
t3 = time.time()
s3_chunks = []
n_dist_s3 = N_DISTRACTORS // 2
dist_s3_collected = 0

for chunk in pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str, chunksize=500_000):
    chunk.fillna("", inplace=True)
    matches = chunk[chunk["entity_id"].isin(target_s3_ids)]
    if len(matches) > 0:
        s3_chunks.append(matches)
    if dist_s3_collected < n_dist_s3:
        non_matches = chunk[~chunk["entity_id"].isin(target_s3_ids)]
        take = min(n_dist_s3 - dist_s3_collected, len(non_matches) // 50)
        if take > 0:
            s3_chunks.append(non_matches.sample(take, random_state=42))
            dist_s3_collected += take

s3_bench = pd.concat(s3_chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])
s3_bench.to_csv(BENCH_DIR / "benchmark_source3.tsv", sep="\t", index=False)
del s3_chunks
found_s3 = s3_bench["entity_id"].isin(target_s3_ids).sum()
print(f"   Saved benchmark_source3.tsv ({len(s3_bench):,} rows, {found_s3}/{len(target_s3_ids)} true matches present) in {time.time()-t3:.1f}s")

# 6. Save benchmark ground truth
gt_bench = s1_sampled_df[["source1_entity_id", "matched_entity_ids"]].copy()
gt_bench.to_csv(BENCH_DIR / "benchmark_ground_truth.tsv", sep="\t", index=False)
print(f"5. Saved benchmark_ground_truth.tsv ({len(gt_bench):,} rows)")
print(f"All done in {time.time()-t0:.1f}s! Benchmark is ready for ultra-fast validation.")
