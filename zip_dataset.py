"""
Compress dataset into a single .zip file for fast, error-free Kaggle upload.
"""
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(r"d:\AMAZON ML CHALLENGE")
DATA_DIR = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
ZIP_PATH = ROOT / "amazon_ml_dataset.zip"

print(f"Creating zip file at {ZIP_PATH} ...")
t0 = time.time()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for folder in ["train", "test"]:
        folder_path = DATA_DIR / folder
        if not folder_path.exists():
            continue
        for file_path in folder_path.glob("*.tsv"):
            arcname = f"{folder}/{file_path.name}"
            print(f"  Adding {arcname} ({file_path.stat().st_size / 1e6:.1f} MB) ...")
            zf.write(file_path, arcname=arcname)

zip_size_mb = ZIP_PATH.stat().st_size / 1e6
print(f"\nDone! Created {ZIP_PATH.name} ({zip_size_mb:.1f} MB) in {time.time()-t0:.1f}s.")
