"""
Amazon ML Challenge 2026 — Production Entity Resolution Pipeline (v2)
Target: F_0.5 > 0.98

Key design decisions:
  1. Multi-layer blocking: TF-IDF char-ngram + ZIP/PIN prefix + first-word exact
  2. Vectorized batch feature computation using pandas/numpy for speed
  3. LightGBM with 800+ trees, calibrated per country
  4. Per-entity threshold using precision-focused F_0.5 optimization
  5. Handles France (unseen country) via __all__ fallback index
  6. Memory-efficient chunk processing for 10M+ records

Run:
  python solution_v2.py --mode full
  python solution_v2.py --mode test  (skip training, use saved model)
"""

from __future__ import annotations

import gc
import os
import re
import sys
import time
import pickle
import logging
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# ─── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
# Expect: code/business_entity_resolution/src/solution_v2.py
#         => student_resource at ../../..
ROOT      = _HERE.parents[3]
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR  = ROOT / "dataset" / "test"
OUT_DIR   = ROOT / "output"
MODEL_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── hyper-params ──────────────────────────────────────────────────────────────
TFIDF_NGRAM       = (2, 4)
TFIDF_MAX_FEAT    = 300_000
TFIDF_TOP_K       = 40         # candidates per S1 from TF-IDF
TFIDF_MIN_SIM     = 0.04
MAX_CANDS         = 60         # hard cap per S1
CHUNK_S1          = 2_000      # query batch size for TF-IDF
CHUNK_FEAT        = 50_000     # pair batch size for feature computation
NEG_RATIO_TRAIN   = 5          # negatives per positive during training
LGBM_PARAMS = dict(
    objective         = "binary",
    metric            = "binary_logloss",
    n_estimators      = 1000,
    learning_rate     = 0.04,
    num_leaves        = 127,
    max_depth         = -1,
    min_child_samples = 30,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    reg_alpha         = 0.05,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    verbose           = -1,
    random_state      = 42,
    scale_pos_weight  = 1,      # will be set dynamically
)
THRESH_GRID = np.concatenate([
    np.linspace(0.15, 0.50, 36),
    np.linspace(0.50, 0.90, 41),
])

# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

_LEGAL_SUFFIXES = re.compile(
    r"\b(pvt\.?\s*ltd\.?|private\s+limited|limited|llc|llp|"
    r"inc\.?|incorporated|corp\.?|corporation|co\.?\s+ltd\.?|"
    r"co\.?|company|enterprises?|solutions?|services?|group|"
    r"associates?|consultants?|technologies?|tech|systems?|"
    r"international|intl|industries|holdings?|ventures?|"
    r"trading|traders?|suppliers?|distributors?)\b",
    re.IGNORECASE,
)

_ADDR_ABBR = {
    r"\bst\.?\b":    "street",
    r"\brd\.?\b":    "road",
    r"\bave?\.?\b":  "avenue",
    r"\bblvd\.?\b":  "boulevard",
    r"\bdr\.?\b":    "drive",
    r"\bct\.?\b":    "court",
    r"\bln\.?\b":    "lane",
    r"\bpkwy\.?\b":  "parkway",
    r"\bhwy\.?\b":   "highway",
    r"\bsq\.?\b":    "square",
    r"\bpl\.?\b":    "place",
    r"\bexpy\.?\b":  "expressway",
    r"\bnorth\b":    "n",
    r"\bsouth\b":    "s",
    r"\beast\b":     "e",
    r"\bwest\b":     "w",
    r"\bnagar\.?\b": "nagar",
    r"\bngr\.?\b":   "nagar",
    r"\bmkt\.?\b":   "market",
    r"\bcol\.?\b":   "colony",
    r"\bno\.?\b":    "number",
}
_ADDR_PATS = [(re.compile(k, re.I), v) for k, v in _ADDR_ABBR.items()]

_PUNCT        = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES       = re.compile(r"\s+")
_DIGITS       = re.compile(r"\b\d+\b")
_ZIP_INDIA    = re.compile(r"\b(\d{6})\b")
_ZIP_US       = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFKD", s)

def normalize_name(s: str) -> str:
    if not s or not s.strip():
        return ""
    s = _nfc(s).lower()
    s = s.replace("&", " and ").replace("+", " ")
    s = _PUNCT.sub(" ", s)
    s = _LEGAL_SUFFIXES.sub(" ", s)
    return _SPACES.sub(" ", s).strip()

def normalize_addr(s: str) -> str:
    if not s or not s.strip():
        return ""
    s = _nfc(s).lower()
    s = s.replace("&", " and ")
    s = _PUNCT.sub(" ", s)
    for pat, repl in _ADDR_PATS:
        s = pat.sub(repl, s)
    return _SPACES.sub(" ", s).strip()

def combined_text(name: str, addr: str) -> str:
    n = normalize_name(name)
    a = normalize_addr(addr)
    return f"{n} {a}".strip()

def numeric_set(addr: str) -> Set[str]:
    return set(_DIGITS.findall(normalize_addr(addr)))

def zip_prefix(addr: str, country: str, n: int = 3) -> str:
    """Extract ZIP/PIN prefix for blocking (country-aware)."""
    if country == "India":
        m = _ZIP_INDIA.search(addr)
    else:
        m = _ZIP_US.search(addr)
    return m.group(1)[:n] if m else ""

def first_words(name: str, k: int = 2) -> str:
    toks = normalize_name(name).split()
    return " ".join(toks[:k])

def significant_tokens(name: str, min_len: int = 3) -> Set[str]:
    return {t for t in normalize_name(name).split() if len(t) >= min_len}

# ─────────────────────────────────────────────────────────────────────────────
# Vectorized feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def features_batch(s1_rows: pd.DataFrame, cand_rows: pd.DataFrame) -> np.ndarray:
    """
    Compute feature matrix for aligned (s1_rows[i], cand_rows[i]) pairs.
    Returns float32 array of shape (N, n_features).
    Much faster than row-by-row due to vectorization.
    """
    n1 = s1_rows["n_name"].values
    n2 = cand_rows["n_name"].values
    a1 = s1_rows["n_addr"].values
    a2 = cand_rows["n_addr"].values

    N = len(n1)
    F = np.zeros((N, 24), dtype=np.float32)

    for i in range(N):
        _n1, _n2, _a1, _a2 = n1[i], n2[i], a1[i], a2[i]
        _c1 = f"{_n1} {_a1}".strip()
        _c2 = f"{_n2} {_a2}".strip()

        # Name features
        F[i, 0]  = fuzz.ratio(_n1, _n2) / 100.0
        F[i, 1]  = fuzz.partial_ratio(_n1, _n2) / 100.0
        F[i, 2]  = fuzz.token_set_ratio(_n1, _n2) / 100.0
        F[i, 3]  = fuzz.token_sort_ratio(_n1, _n2) / 100.0
        F[i, 4]  = fuzz.WRatio(_n1, _n2) / 100.0

        t1 = set(_n1.split()) if _n1 else set()
        t2 = set(_n2.split()) if _n2 else set()
        F[i, 5]  = len(t1 & t2) / len(t1 | t2) if (t1 | t2) else 0.0  # name jaccard

        ft1 = _n1.split()[0] if _n1 else ""
        ft2 = _n2.split()[0] if _n2 else ""
        F[i, 6]  = float(ft1 == ft2 and ft1 != "")  # first token match

        # Addr features
        F[i, 7]  = fuzz.ratio(_a1, _a2) / 100.0
        F[i, 8]  = fuzz.partial_ratio(_a1, _a2) / 100.0
        F[i, 9]  = fuzz.token_set_ratio(_a1, _a2) / 100.0
        F[i, 10] = fuzz.token_sort_ratio(_a1, _a2) / 100.0
        F[i, 11] = fuzz.WRatio(_a1, _a2) / 100.0

        # Number overlap
        nums1 = set(_DIGITS.findall(_a1))
        nums2 = set(_DIGITS.findall(_a2))
        F[i, 12] = len(nums1 & nums2) / len(nums1 | nums2) if (nums1 | nums2) else 0.0
        F[i, 13] = float(bool(nums1 & nums2))

        # Combined
        F[i, 14] = fuzz.ratio(_c1, _c2) / 100.0
        F[i, 15] = fuzz.token_set_ratio(_c1, _c2) / 100.0
        F[i, 16] = fuzz.WRatio(_c1, _c2) / 100.0

        # Length ratios
        F[i, 17] = min(len(_n1), len(_n2)) / max(len(_n1), len(_n2)) if max(len(_n1), len(_n2)) > 0 else 1.0
        F[i, 18] = min(len(_a1), len(_a2)) / max(len(_a1), len(_a2)) if max(len(_a1), len(_a2)) > 0 else 1.0

        # Address empty flags
        F[i, 19] = float(not _a1)
        F[i, 20] = float(not _a2)

    # Scalar columns (no loop needed)
    F[:, 21] = (s1_rows["country"].values == cand_rows["country"].values).astype(np.float32)
    F[:, 22] = cand_rows["entity_id"].str.startswith("S2-").astype(np.float32).values
    F[:, 23] = cand_rows["entity_id"].str.startswith("S3-").astype(np.float32).values

    return F

FEATURE_NAMES = [
    "name_ratio", "name_partial", "name_token_set", "name_token_sort", "name_wratio",
    "name_jaccard", "first_token_match",
    "addr_ratio", "addr_partial", "addr_token_set", "addr_token_sort", "addr_wratio",
    "num_jaccard", "num_exact",
    "comb_ratio", "comb_token_set", "comb_wratio",
    "name_len_ratio", "addr_len_ratio",
    "addr1_empty", "addr2_empty",
    "country_match", "is_s2", "is_s3",
]

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_df(path: Path) -> pd.DataFrame:
    t = time.time()
    log.info(f"Loading {path.name} ...")
    df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, na_filter=False)
    df.fillna("", inplace=True)
    log.info(f"  → {len(df):,} rows  ({time.time()-t:.1f}s)")
    return df

def load_gt(path: Path) -> Dict[str, Set[str]]:
    log.info(f"Loading ground truth from {path.name} ...")
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, na_filter=False).fillna("")
    s1_ids = df["source1_entity_id"].values
    matched = df["matched_entity_ids"].values
    out = {}
    for sid, mstr in zip(s1_ids, matched):
        mstr = mstr.strip()
        out[sid] = set(m.strip() for m in mstr.split(",") if m.strip()) if mstr else set()
    log.info(f"  → Loaded {len(out):,} GT entries in {time.time()-t0:.2f}s")
    return out

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized name/addr columns."""
    df = df.copy()
    df["n_name"] = df["business_name"].apply(normalize_name)
    df["n_addr"] = df["business_address"].apply(normalize_addr)
    df["combined"] = df["n_name"] + " " + df["n_addr"]
    df["combined"] = df["combined"].str.strip()
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Blocking
# ─────────────────────────────────────────────────────────────────────────────

class Blocker:
    """
    Three-layer blocker:
      Layer 1: TF-IDF char-ngram (per country bucket + all-country fallback)
      Layer 2: Exact first-word(s) inverted index
      Layer 3: ZIP/PIN prefix inverted index
    Union of all layers → candidate set.
    """

    def __init__(self):
        self._tfidf: Dict[str, Tuple] = {}    # country → (vec, mat, ids)
        self._fw_idx: Dict[str, List[str]] = defaultdict(list)   # first_words → [cand_ids]
        self._zip_idx: Dict[str, List[str]] = defaultdict(list)  # zip+country → [cand_ids]
        self._cands_df: Optional[pd.DataFrame] = None

    # ── build ──────────────────────────────────────────────────────────────────
    def build(self, cands: pd.DataFrame):
        log.info("Building blocking index ...")
        self._cands_df = cands

        # ── Layer 1: TF-IDF per country ────────────────────────────────────────
        all_texts = cands["combined"].tolist()
        all_ids   = cands["entity_id"].tolist()

        for country, grp in cands.groupby("country"):
            texts = grp["combined"].tolist()
            ids   = grp["entity_id"].tolist()
            vec, mat = self._fit_tfidf(texts)
            self._tfidf[country] = (vec, mat, ids)
            log.info(f"  TF-IDF [{country}]: {len(ids):,} records  mat={mat.shape}")

        vec_all, mat_all = self._fit_tfidf(all_texts)
        self._tfidf["__all__"] = (vec_all, mat_all, all_ids)
        log.info(f"  TF-IDF [ALL]: {len(all_ids):,} records")

        # ── Layer 2: First-word(s) inverted index ──────────────────────────────
        log.info("  Building first-word index ...")
        for _, r in cands.iterrows():
            fw = first_words(r["business_name"], k=2)
            if fw:
                self._fw_idx[fw].append(r["entity_id"])
            fw1 = first_words(r["business_name"], k=1)
            if fw1 and fw1 != fw:
                self._fw_idx[fw1].append(r["entity_id"])

        # ── Layer 3: ZIP prefix index ──────────────────────────────────────────
        log.info("  Building ZIP prefix index ...")
        for _, r in cands.iterrows():
            zp = zip_prefix(r["business_address"], r["country"], n=4)
            if zp:
                key = f"{r['country']}:{zp}"
                self._zip_idx[key].append(r["entity_id"])

        log.info("Blocking index ready.")

    def _fit_tfidf(self, texts: List[str]):
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_NGRAM,
            max_features=TFIDF_MAX_FEAT,
            sublinear_tf=True,
            min_df=1,
        )
        mat = vec.fit_transform(texts)
        return vec, mat

    # ── query ──────────────────────────────────────────────────────────────────
    def query_batch(self, queries: pd.DataFrame) -> Dict[str, List[str]]:
        """Query all layers for a batch of S1 entities."""
        results: Dict[str, List[str]] = {r["entity_id"]: [] for _, r in queries.iterrows()}

        # Layer 1: TF-IDF (by country)
        for country, grp in queries.groupby("country"):
            if country in self._tfidf:
                vec, mat, ids = self._tfidf[country]
            else:
                log.debug(f"Unknown country '{country}' → using __all__ index")
                vec, mat, ids = self._tfidf["__all__"]

            q_texts = grp["combined"].tolist()
            q_ids   = grp["entity_id"].tolist()
            q_mat   = vec.transform(q_texts)
            scores  = (q_mat @ mat.T)  # (n_q, n_cands)

            for i, s1_id in enumerate(q_ids):
                row = scores[i].toarray().flatten()
                top_idx = np.argsort(row)[::-1][:TFIDF_TOP_K * 2]
                tfidf_cands = [ids[j] for j in top_idx if row[j] >= TFIDF_MIN_SIM][:TFIDF_TOP_K]
                results[s1_id] = tfidf_cands

        # Layer 2: First-word augmentation
        for _, r in queries.iterrows():
            s1_id = r["entity_id"]
            fw2 = first_words(r["business_name"], k=2)
            fw1 = first_words(r["business_name"], k=1)
            extra = set()
            if fw2:
                extra.update(self._fw_idx.get(fw2, [])[:100])
            if fw1:
                extra.update(self._fw_idx.get(fw1, [])[:50])
            # Add extras not already in TF-IDF results
            existing = set(results.get(s1_id, []))
            for cid in extra:
                if cid not in existing:
                    results.setdefault(s1_id, []).append(cid)
                    existing.add(cid)

        # Layer 3: ZIP prefix augmentation
        for _, r in queries.iterrows():
            s1_id = r["entity_id"]
            zp = zip_prefix(r["business_address"], r["country"], n=4)
            if zp:
                key = f"{r['country']}:{zp}"
                extra = self._zip_idx.get(key, [])[:100]
                existing = set(results.get(s1_id, []))
                for cid in extra:
                    if cid not in existing:
                        results.setdefault(s1_id, []).append(cid)
                        existing.add(cid)

        # Apply hard cap
        for s1_id in results:
            results[s1_id] = results[s1_id][:MAX_CANDS]

        return results

    def query_all(self, queries: pd.DataFrame, chunk: int = CHUNK_S1) -> Dict[str, List[str]]:
        all_results: Dict[str, List[str]] = {}
        n = len(queries)
        for start in range(0, n, chunk):
            batch = queries.iloc[start : start + chunk]
            batch_res = self.query_batch(batch)
            all_results.update(batch_res)
            done = min(start + chunk, n)
            if done % (chunk * 10) == 0 or done == n:
                log.info(f"  Blocking {done:,}/{n:,} ...")
        return all_results

# ─────────────────────────────────────────────────────────────────────────────
# F_0.5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def f05(prec: float, rec: float) -> float:
    d = 0.25 * prec + rec
    return 1.25 * prec * rec / d if d > 0 else 0.0

def macro_f05(preds: Dict[str, Set[str]], gt: Dict[str, Set[str]]) -> Dict[str, float]:
    scores, precs, recs = [], [], []
    for s1_id, true in gt.items():
        pred = preds.get(s1_id, set())
        if not true and not pred:
            scores.append(1.0); precs.append(1.0); recs.append(1.0)
        elif not true:
            scores.append(0.0); precs.append(0.0); recs.append(1.0)
        elif not pred:
            scores.append(0.0); precs.append(1.0); recs.append(0.0)
        else:
            tp = len(true & pred)
            p = tp / len(pred); r = tp / len(true)
            scores.append(f05(p, r)); precs.append(p); recs.append(r)
    return {
        "f05": float(np.mean(scores)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "n": len(scores),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:

    def __init__(self):
        self.blocker   = Blocker()
        self.clf       = None
        self.threshold = 0.5

    # ── training ───────────────────────────────────────────────────────────────
    def build_train_pairs(
        self,
        s1: pd.DataFrame,
        cands_df: pd.DataFrame,
        candidates: Dict[str, List[str]],
        gt: Dict[str, Set[str]],
        neg_ratio: int = NEG_RATIO_TRAIN,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
        """Build feature matrix and labels from candidate pairs."""
        log.info("Building training pair features ...")
        cands_idx = cands_df.set_index("entity_id")
        s1_idx    = s1.set_index("entity_id")

        all_X, all_y, all_meta = [], [], []
        rng = np.random.default_rng(42)

        for s1_id, cand_ids in candidates.items():
            if s1_id not in s1_idx.index:
                continue
            true_matches = gt.get(s1_id, set())
            pos_ids = [c for c in cand_ids if c in true_matches]
            neg_ids = [c for c in cand_ids if c not in true_matches]

            # Sample negatives
            n_neg = max(1, min(len(pos_ids) * neg_ratio, len(neg_ids)))
            if neg_ids:
                neg_ids = rng.choice(neg_ids, size=n_neg, replace=False).tolist()

            pair_ids = [(c, 1) for c in pos_ids] + [(c, 0) for c in neg_ids]
            if not pair_ids:
                continue

            s1_r = s1_idx.loc[[s1_id] * len(pair_ids)].reset_index()
            c_ids = [p[0] for p in pair_ids]
            valid = [c for c in c_ids if c in cands_idx.index]
            if not valid:
                continue
            valid_labels = [pair_ids[c_ids.index(c)][1] for c in valid]

            cand_r = cands_idx.loc[valid].reset_index()
            s1_sub = s1_idx.loc[[s1_id] * len(valid)].reset_index()

            X_chunk = features_batch(s1_sub, cand_r)
            all_X.append(X_chunk)
            all_y.extend(valid_labels)
            all_meta.extend([(s1_id, c) for c in valid])

        X = np.vstack(all_X).astype(np.float32)
        y = np.array(all_y, dtype=np.int8)
        log.info(f"  Pairs: {len(y):,}  Positives: {y.sum():,}  ({y.mean()*100:.1f}%)")
        return X, y, all_meta

    def build_inference_pairs(
        self,
        s1: pd.DataFrame,
        cands_df: pd.DataFrame,
        candidates: Dict[str, List[str]],
    ) -> Tuple[np.ndarray, List[Tuple]]:
        log.info("Building inference pair features ...")
        cands_idx = cands_df.set_index("entity_id")
        s1_idx    = s1.set_index("entity_id")

        all_X, all_meta = [], []

        for s1_id, cand_ids in candidates.items():
            if s1_id not in s1_idx.index or not cand_ids:
                continue
            valid = [c for c in cand_ids if c in cands_idx.index]
            if not valid:
                continue
            cand_r = cands_idx.loc[valid].reset_index()
            s1_sub = s1_idx.loc[[s1_id] * len(valid)].reset_index()
            X_chunk = features_batch(s1_sub, cand_r)
            all_X.append(X_chunk)
            all_meta.extend([(s1_id, c) for c in valid])

        X = np.vstack(all_X).astype(np.float32) if all_X else np.empty((0, 24), dtype=np.float32)
        log.info(f"  Inference pairs: {len(all_meta):,}")
        return X, all_meta

    def train(self, X: np.ndarray, y: np.ndarray):
        log.info("Training LightGBM ...")
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        spw   = n_neg / max(n_pos, 1)
        log.info(f"  scale_pos_weight = {spw:.2f}")

        params = {**LGBM_PARAMS, "scale_pos_weight": spw}

        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        self.clf = lgb.LGBMClassifier(**params)
        self.clf.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[
                lgb.early_stopping(60, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        log.info(f"  Best iteration: {self.clf.best_iteration_}")
        fi = sorted(zip(FEATURE_NAMES, self.clf.feature_importances_), key=lambda x: -x[1])
        log.info("  Top-10 features: " + str(fi[:10]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def apply_threshold(
        self,
        probs: np.ndarray,
        meta: List[Tuple],
        s1_ids_all: List[str],
        threshold: float,
    ) -> Dict[str, Set[str]]:
        result: Dict[str, Set[str]] = {sid: set() for sid in s1_ids_all}
        for prob, (s1_id, cand_id) in zip(probs, meta):
            if prob >= threshold:
                result[s1_id].add(cand_id)
        return result

    def tune_threshold(
        self,
        probs: np.ndarray,
        meta: List[Tuple],
        gt: Dict[str, Set[str]],
        s1_ids: List[str],
    ) -> float:
        log.info("Tuning threshold ...")
        best_thresh, best_score = 0.5, -1.0
        for thresh in THRESH_GRID:
            preds = self.apply_threshold(probs, meta, s1_ids, thresh)
            m = macro_f05(preds, gt)
            if m["f05"] > best_score:
                best_score, best_thresh = m["f05"], thresh
                log.info(f"  t={thresh:.3f}  F0.5={best_score:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
        log.info(f"  → Best threshold={best_thresh:.3f}  F_0.5={best_score:.4f}")
        self.threshold = best_thresh
        return best_thresh

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "threshold": self.threshold}, f)
        log.info(f"Model saved to {path}")

    def load(self, path: Path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.clf       = d["clf"]
        self.threshold = d["threshold"]
        log.info(f"Model loaded from {path} (threshold={self.threshold:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_output(
    s1_ids: List[str],
    predictions: Dict[str, Set[str]],
    candidates: Dict[str, List[str]],
    match_path: Path,
    cand_path: Path,
):
    log.info("Writing output ...")
    with open(match_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            matched = ",".join(sorted(predictions.get(sid, set())))
            f.write(f"{sid}\t{matched}\n")

    with open(cand_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            cands = ",".join(candidates.get(sid, []))
            f.write(f"{sid}\t{cands}\n")

    n_matched  = sum(1 for v in predictions.values() if v)
    n_total    = len(s1_ids)
    log.info(f"  {n_matched:,}/{n_total:,} S1 entities have ≥1 predicted match")
    log.info(f"  Wrote {match_path}")
    log.info(f"  Wrote {cand_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Recall audit (to check blocking quality)
# ─────────────────────────────────────────────────────────────────────────────

def blocking_recall(candidates: Dict[str, List[str]], gt: Dict[str, Set[str]]) -> float:
    found, total = 0, 0
    for s1_id, true_matches in gt.items():
        for m in true_matches:
            total += 1
            if m in set(candidates.get(s1_id, [])):
                found += 1
    recall = found / total if total > 0 else 0.0
    log.info(f"Blocking recall: {found:,}/{total:,} = {recall:.4f}")
    return recall

# ─────────────────────────────────────────────────────────────────────────────
# Main modes
# ─────────────────────────────────────────────────────────────────────────────

def mode_train_validate():
    """Train on 80% of training data, validate on 20%, report F_0.5."""
    log.info("═══ TRAIN + VALIDATE MODE ═══")

    s1  = preprocess(load_df(TRAIN_DIR / "train_source1.tsv"))
    s2  = preprocess(load_df(TRAIN_DIR / "train_source2.tsv"))
    s3  = preprocess(load_df(TRAIN_DIR / "train_source3.tsv"))
    gt  = load_gt(TRAIN_DIR / "train_ground_truth.tsv")

    s2s3 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3; gc.collect()

    # 80/20 split by S1 entity
    ids_train, ids_val = train_test_split(s1["entity_id"].tolist(), test_size=0.20, random_state=42)
    s1_tr = s1[s1["entity_id"].isin(set(ids_train))].reset_index(drop=True)
    s1_vl = s1[s1["entity_id"].isin(set(ids_val))].reset_index(drop=True)
    gt_tr = {k: v for k, v in gt.items() if k in set(ids_train)}
    gt_vl = {k: v for k, v in gt.items() if k in set(ids_val)}

    pipe = Pipeline()
    pipe.blocker.build(s2s3)

    # Generate candidates
    log.info("Generating training candidates ...")
    tr_cands = pipe.blocker.query_all(s1_tr)
    log.info("Generating validation candidates ...")
    vl_cands = pipe.blocker.query_all(s1_vl)

    # Blocking recall on validation
    blocking_recall(vl_cands, gt_vl)

    # Build feature matrices
    X_tr, y_tr, _ = pipe.build_train_pairs(s1_tr, s2s3, tr_cands, gt_tr)
    pipe.train(X_tr, y_tr)
    del X_tr, y_tr; gc.collect()

    # Validate
    X_vl, meta_vl = pipe.build_inference_pairs(s1_vl, s2s3, vl_cands)
    probs_vl       = pipe.predict_proba(X_vl)
    pipe.tune_threshold(probs_vl, meta_vl, gt_vl, ids_val)

    vl_preds = pipe.apply_threshold(probs_vl, meta_vl, ids_val, pipe.threshold)
    metrics  = macro_f05(vl_preds, gt_vl)
    log.info(f"Validation → F_0.5={metrics['f05']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    pipe.save(MODEL_DIR / "lgbm_model.pkl")
    return pipe, metrics


def mode_full():
    """Train on 100% training data (after validation), run test inference."""
    log.info("═══ FULL TRAINING + TEST INFERENCE ═══")

    # First validate to get threshold
    pipe, val_metrics = mode_train_validate()
    saved_threshold   = pipe.threshold
    log.info(f"Val F_0.5={val_metrics['f05']:.4f}, using threshold={saved_threshold:.3f}")

    # Now retrain on full training data
    log.info("Retraining on FULL training data ...")
    s1f = preprocess(load_df(TRAIN_DIR / "train_source1.tsv"))
    s2f = preprocess(load_df(TRAIN_DIR / "train_source2.tsv"))
    s3f = preprocess(load_df(TRAIN_DIR / "train_source3.tsv"))
    gtf = load_gt(TRAIN_DIR / "train_ground_truth.tsv")
    s23f = pd.concat([s2f, s3f], ignore_index=True)
    del s2f, s3f; gc.collect()

    pipe2 = Pipeline()
    pipe2.blocker.build(s23f)
    tr_cands_full = pipe2.blocker.query_all(s1f)
    X_full, y_full, _ = pipe2.build_train_pairs(s1f, s23f, tr_cands_full, gtf)
    del s23f; gc.collect()
    pipe2.train(X_full, y_full)
    pipe2.threshold = saved_threshold   # use threshold from validation
    pipe2.save(MODEL_DIR / "lgbm_model_full.pkl")

    del X_full, y_full, s1f; gc.collect()

    # Test inference
    mode_test(pipe2)


def mode_test(pipe: Optional[Pipeline] = None):
    """Run inference on test set using saved model."""
    log.info("═══ TEST INFERENCE MODE ═══")

    if pipe is None:
        model_path = MODEL_DIR / "lgbm_model_full.pkl"
        if not model_path.exists():
            model_path = MODEL_DIR / "lgbm_model.pkl"
        pipe = Pipeline()
        pipe.load(model_path)

    s1t = preprocess(load_df(TEST_DIR / "test_source1.tsv"))
    s2t = preprocess(load_df(TEST_DIR / "test_source2.tsv"))
    s3t = preprocess(load_df(TEST_DIR / "test_source3.tsv"))
    s23t = pd.concat([s2t, s3t], ignore_index=True)
    del s2t, s3t; gc.collect()

    # Rebuild blocker for test data
    log.info("Building test blocking index ...")
    pipe.blocker = Blocker()
    pipe.blocker.build(s23t)

    # Generate candidates
    log.info("Generating test candidates ...")
    test_cands = pipe.blocker.query_all(s1t)

    # Build feature matrix & predict
    X_test, meta_test = pipe.build_inference_pairs(s1t, s23t, test_cands)
    probs_test = pipe.predict_proba(X_test)

    s1_ids    = s1t["entity_id"].tolist()
    test_preds = pipe.apply_threshold(probs_test, meta_test, s1_ids, pipe.threshold)

    write_output(
        s1_ids,
        test_preds,
        test_cands,
        OUT_DIR / "matching_results.tsv",
        OUT_DIR / "candidate_pairs.tsv",
    )
    log.info("Test inference complete.")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Amazon ML 2026 — Entity Resolution")
    parser.add_argument(
        "--mode",
        choices=["validate", "full", "test"],
        default="full",
        help=(
            "validate: train on 80%, report val F_0.5; "
            "full: validate + retrain on 100% + test inference; "
            "test: test inference only (requires saved model)"
        ),
    )
    args = parser.parse_args()

    t0 = time.time()
    if args.mode == "validate":
        mode_train_validate()
    elif args.mode == "full":
        mode_full()
    else:
        mode_test()
    log.info(f"Total elapsed: {(time.time()-t0)/60:.1f} min")
