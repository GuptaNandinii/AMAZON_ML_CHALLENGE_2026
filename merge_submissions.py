"""
Amazon ML Challenge 2026 — Master Submission Stitcher
Merges France (Account 1), US (Account 1), and India (Account 2) checkpoints
into the official 1,732,545-row submission files with verified original order.
"""

import json
import os
import sys
import time
from pathlib import Path
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    root = Path(__file__).resolve().parent
    sub_dir = root / "final_submission"
    sub_dir.mkdir(exist_ok=True)

    # 1. Locate test_source1.tsv to get exact ground-truth row order
    test_s1_path = root / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test" / "test_source1.tsv"
    if not test_s1_path.exists():
        for p in root.glob("**/test_source1.tsv"):
            test_s1_path = p
            break

    print(f"Loading reference test_source1 from {test_s1_path.name}...")
    df_s1 = pd.read_csv(test_s1_path, sep="\t", dtype=str)
    orig_s1_ids = df_s1["entity_id"].tolist()
    total_queries = len(orig_s1_ids)
    print(f"Reference entity count: {total_queries:,}")

    # 2. Check and load country JSON files
    countries = ["France", "US", "India"]
    results_map = {}
    cands_map = {}

    missing_files = []
    for c in countries:
        res_file = sub_dir / f"results_{c}.json"
        cand_file = sub_dir / f"cands_{c}.json"

        if not res_file.exists():
            missing_files.append(res_file.name)
        else:
            with open(res_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results_map.update(data)
                print(f"  Loaded {res_file.name}: {len(data):,} queries")

        if not cand_file.exists():
            missing_files.append(cand_file.name)
        else:
            with open(cand_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                cands_map.update(data)
                print(f"  Loaded {cand_file.name}: {len(data):,} queries")

    if missing_files:
        print(f"\n[!] Missing {len(missing_files)} file(s) in {sub_dir}:")
        for mf in missing_files:
            print(f"   - {mf}")
        print("\nPlease place the downloaded JSON files into this folder and run again!")
        if len(results_map) == 0:
            return

    # 3. Write final submission files
    out_matching = sub_dir / "matching_results.tsv"
    out_cands = sub_dir / "candidate_pairs.tsv"

    print(f"\nWriting official submission files ({total_queries:,} rows)...")
    t0 = time.time()
    matched_count = 0
    with open(out_matching, "w", encoding="utf-8") as f_match, \
         open(out_cands, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for idx, sid in enumerate(orig_s1_ids, 1):
            m = results_map.get(sid, "")
            c = cands_map.get(sid, "")
            if m: matched_count += 1
            f_match.write(f"{sid}\t{m}\n")
            f_cand.write(f"{sid}\t{c}\n")

            if idx % 350_000 == 0 or idx == total_queries:
                pct = idx / total_queries * 100
                print(f"  Progress: {idx:,}/{total_queries:,} ({pct:.1f}%) in {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print("SUBMISSION FILES GENERATED SUCCESSFULLY!")
    print(f"  Matching Results: {out_matching}")
    print(f"  Candidate Pairs:  {out_cands}")
    print(f"  Total queries written: {total_queries:,}")
    print(f"  Entities with matches: {matched_count:,} ({matched_count/total_queries*100:.1f}%)")
    print("=" * 60)

    # 4. Copy directly to FINALLLL folder
    finallll_dir = root / "FINALLLL"
    finallll_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(out_matching, finallll_dir / "matching_results.tsv")
    print(f"\n>>> DROP-ZONE READY: {finallll_dir / 'matching_results.tsv'} <<<")

    # 4. Automatically run official validator
    validator_path = root / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py"
    test_dir = test_s1_path.parent
    if validator_path.exists() and test_dir.exists():
        print("\nRunning Official Amazon ML Challenge Submission Validator...")
        import subprocess
        cmd = [
            sys.executable, str(validator_path),
            "--matching", str(out_matching),
            "--candidate", str(out_cands),
            "--test-dir", str(test_dir)
        ]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            print("\n[OK] OFFICIAL VALIDATION PASSED! FILE IS SAFE TO SUBMIT.")
        else:
            print("\n[!] Validation returned non-zero code. Check warnings above.")

if __name__ == "__main__":
    main()
