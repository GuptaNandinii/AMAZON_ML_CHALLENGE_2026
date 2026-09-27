"""
Amazon ML Challenge 2026 — Business Entity Resolution Pipeline
Author: Team Solution

High-level strategy:
  1. Text normalization + abbreviation expansion
  2. Multi-key blocking using TF-IDF trigram inverted index (per country bucket)
  3. Rich feature computation (RapidFuzz, token overlap, Jaccard, length ratios)
  4. LightGBM binary classifier trained on labeled pairs
  5. Threshold search optimized for F_0.5 on held-out validation fold
  6. Output generation (matching_results.tsv + candidate_pairs.tsv)
"""

import os
import re
import sys
import time
import logging
import argparse
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, utils as rfutils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / knobs
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[3]   # student_resource/
DATA_TRAIN = ROOT / "dataset" / "train"
DATA_TEST  = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_S1  = DATA_TRAIN / "train_source1.tsv"
TRAIN_S2  = DATA_TRAIN / "train_source2.tsv"
TRAIN_S3  = DATA_TRAIN / "train_source3.tsv"
TRAIN_GT  = DATA_TRAIN / "train_ground_truth.tsv"

TEST_S1   = DATA_TEST / "test_source1.tsv"
TEST_S2   = DATA_TEST / "test_source2.tsv"
TEST_S3   = DATA_TEST / "test_source3.tsv"

# Blocking
MAX_CANDIDATES_PER_S1 = 50       # recall ceiling per entity
TFIDF_TOP_K           = 30       # top-k TF-IDF candidates per query
MIN_TFIDF_SCORE       = 0.05     # minimum cosine similarity for blocking

# LightGBM
LGBM_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)

# F_0.5 threshold sweep
THRESHOLD_GRID = np.linspace(0.20, 0.85, 66)

# ─────────────────────────────────────────────────────────────────────────────
# Text normalization
# ─────────────────────────────────────────────────────────────────────────────

# Common abbreviation expansions (add more as needed)
ABBREV = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave?\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bct\b": "court",
    r"\bln\b": "lane",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bsq\b": "square",
    r"\bpl\b": "place",
    r"\bfwy\b": "freeway",
    r"\bexpy\b": "expressway",
    r"\bnte\b": "northeast",
    r"\bnw\b": "northwest",
    r"\bse\b": "southeast",
    r"\bsw\b": "southwest",
    r"\bne\b": "northeast",
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    # Business suffixes
    r"\bllc\b": "llc",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bprivate ltd\b": "private limited",
    r"\bpvt ltd\b": "private limited",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\bcompy\b": "company",
    r"\bassoc\b": "associates",
    r"\benterprises\b": "enterprises",
    r"\benterp\b": "enterprises",
    r"\bservices\b": "services",
    r"\bsvcs\b": "services",
    r"\bdept\b": "department",
    r"\bintl\b": "international",
    r"\bgrp\b": "group",
    r"\bnational\b": "national",
    r"\bnat\b": "national",
    r"\btech\b": "technology",
    r"\btechnologies\b": "technology",
    r"\bsol\b": "solutions",
    r"\bsoln\b": "solutions",
    r"\bsolutions\b": "solutions",
    r"\bsys\b": "systems",
    r"\bmgmt\b": "management",
    r"\btrng\b": "training",
    # Address-specific India
    r"\bnagar\b": "nagar",
    r"\bnagara\b": "nagar",
    r"\bngr\b": "nagar",
    r"\bcolony\b": "colony",
    r"\bcol\b": "colony",
    r"\bmarket\b": "market",
    r"\bmkt\b": "market",
    r"\broad\b": "road",
}

_ABBREV_PATTERNS = [(re.compile(k, re.IGNORECASE), v) for k, v in ABBREV.items()]

# Characters to strip
_STRIP_CHARS = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, unicode-normalize, remove punctuation, expand abbreviations."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Unicode normalise: decompose accented characters
    text = unicodedata.normalize("NFKD", text)
    # Lowercase
    text = text.lower()
    # Replace & with and
    text = text.replace("&", " and ")
    # Remove punctuation (keep spaces and word chars)
    text = _STRIP_CHARS.sub(" ", text)
    # Expand abbreviations
    for pat, repl in _ABBREV_PATTERNS:
        text = pat.sub(repl, text)
    # Collapse spaces
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def make_combined_text(row) -> str:
    """Combine name + address into a single searchable string."""
    name = normalize_text(str(row.get("business_name", "") or ""))
    addr = normalize_text(str(row.get("business_address", "") or ""))
    return f"{name} {addr}".strip()


def extract_numeric_tokens(text: str) -> set:
    """Extract digit sequences (street numbers, PIN codes, ZIP codes)."""
    return set(re.findall(r"\b\d+\b", text))


def first_digits(text: str, n: int = 5) -> str:
    """Return first n-digit sequence found (useful for ZIP/PIN blocking)."""
    m = re.search(r"\b(\d{4,6})\b", text)
    return m.group(1)[:n] if m else ""


def name_tokens(name: str) -> set:
    """Tokenize normalized name into a set of word tokens (len >= 2)."""
    return {t for t in normalize_text(name).split() if len(t) >= 2}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_source(path: Path) -> pd.DataFrame:
    log.info(f"Loading {path} ...")
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, na_filter=False)
    df = df.fillna("")
    log.info(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def load_ground_truth(path: Path) -> dict:
    """Return {s1_id: set(matched_ids)} dict."""
    log.info(f"Loading ground truth from {path} ...")
    gt = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, na_filter=False).fillna("")
    result = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        mids_str = row.get("matched_entity_ids", "").strip()
        mids = set(m.strip() for m in mids_str.split(",") if m.strip()) if mids_str else set()
        result[s1_id] = mids
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(s1_row: dict, cand_row: dict) -> dict:
    """Compute rich similarity features for a (S1, candidate) pair."""
    name1 = str(s1_row.get("business_name", "") or "")
    name2 = str(cand_row.get("business_name", "") or "")
    addr1 = str(s1_row.get("business_address", "") or "")
    addr2 = str(cand_row.get("business_address", "") or "")

    n1 = normalize_text(name1)
    n2 = normalize_text(name2)
    a1 = normalize_text(addr1)
    a2 = normalize_text(addr2)

    # Combined text
    c1 = f"{n1} {a1}".strip()
    c2 = f"{n2} {a2}".strip()

    # --- Name features ---
    name_ratio        = fuzz.ratio(n1, n2) / 100.0
    name_partial      = fuzz.partial_ratio(n1, n2) / 100.0
    name_token_set    = fuzz.token_set_ratio(n1, n2) / 100.0
    name_token_sort   = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_wratio       = fuzz.WRatio(n1, n2) / 100.0

    # Token overlap (Jaccard)
    t1 = name_tokens(name1)
    t2 = name_tokens(name2)
    name_jaccard = len(t1 & t2) / len(t1 | t2) if (t1 | t2) else 0.0

    # First token match (usually the most distinctive word)
    ft1 = n1.split()[0] if n1 else ""
    ft2 = n2.split()[0] if n2 else ""
    first_token_match = float(ft1 == ft2 and ft1 != "")

    # --- Address features ---
    addr_ratio        = fuzz.ratio(a1, a2) / 100.0
    addr_partial      = fuzz.partial_ratio(a1, a2) / 100.0
    addr_token_set    = fuzz.token_set_ratio(a1, a2) / 100.0
    addr_token_sort   = fuzz.token_sort_ratio(a1, a2) / 100.0
    addr_wratio       = fuzz.WRatio(a1, a2) / 100.0

    # Numeric token overlap (house numbers, PINs, ZIPs)
    nums1 = extract_numeric_tokens(a1)
    nums2 = extract_numeric_tokens(a2)
    num_jaccard = len(nums1 & nums2) / len(nums1 | nums2) if (nums1 | nums2) else 0.0
    num_exact   = float(bool(nums1 & nums2))

    # --- Combined features ---
    comb_ratio     = fuzz.ratio(c1, c2) / 100.0
    comb_token_set = fuzz.token_set_ratio(c1, c2) / 100.0
    comb_wratio    = fuzz.WRatio(c1, c2) / 100.0

    # --- Length features ---
    name_len_ratio  = min(len(n1), len(n2)) / max(len(n1), len(n2)) if max(len(n1), len(n2)) > 0 else 0.0
    addr_len_ratio  = min(len(a1), len(a2)) / max(len(a1), len(a2)) if max(len(a1), len(a2)) > 0 else 0.0
    addr1_empty     = float(not a1)
    addr2_empty     = float(not a2)

    # --- Country match ---
    country_match   = float(s1_row.get("country", "") == cand_row.get("country", ""))

    # --- Source indicator ---
    cand_src = cand_row.get("entity_id", "")
    is_s2   = float(cand_src.startswith("S2-"))
    is_s3   = float(cand_src.startswith("S3-"))

    return {
        "name_ratio":        name_ratio,
        "name_partial":      name_partial,
        "name_token_set":    name_token_set,
        "name_token_sort":   name_token_sort,
        "name_wratio":       name_wratio,
        "name_jaccard":      name_jaccard,
        "first_token_match": first_token_match,
        "addr_ratio":        addr_ratio,
        "addr_partial":      addr_partial,
        "addr_token_set":    addr_token_set,
        "addr_token_sort":   addr_token_sort,
        "addr_wratio":       addr_wratio,
        "num_jaccard":       num_jaccard,
        "num_exact":         num_exact,
        "comb_ratio":        comb_ratio,
        "comb_token_set":    comb_token_set,
        "comb_wratio":       comb_wratio,
        "name_len_ratio":    name_len_ratio,
        "addr_len_ratio":    addr_len_ratio,
        "addr1_empty":       addr1_empty,
        "addr2_empty":       addr2_empty,
        "country_match":     country_match,
        "is_s2":             is_s2,
        "is_s3":             is_s3,
    }


FEATURE_COLS = [
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
# Blocking / Candidate Generation
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFBlocker:
    """
    Build a TF-IDF (character n-gram) index over S2+S3 records per country bucket.
    For each S1 query, retrieve top-K candidates by cosine similarity.
    """

    def __init__(self, ngram_range=(2, 4), max_features=200_000, top_k=TFIDF_TOP_K, min_score=MIN_TFIDF_SCORE):
        self.ngram_range  = ngram_range
        self.max_features = max_features
        self.top_k        = top_k
        self.min_score    = min_score
        # Per-country index data
        self._indices  = {}   # country -> (vectorizer, tfidf_matrix, entity_ids)

    def build(self, cands: pd.DataFrame):
        """Build TF-IDF index grouped by country."""
        log.info("Building TF-IDF blocking index ...")
        cands = cands.copy()
        cands["_text"] = cands.apply(make_combined_text, axis=1)

        for country, grp in cands.groupby("country"):
            texts     = grp["_text"].tolist()
            ids       = grp["entity_id"].tolist()
            if not texts:
                continue
            vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                sublinear_tf=True,
                min_df=2,
            )
            try:
                mat = vec.fit_transform(texts)
            except ValueError:
                # too few documents
                mat = vec.fit_transform(texts)
            self._indices[country] = (vec, mat, ids)
            log.info(f"  [{country}] {len(ids):,} candidates indexed, matrix {mat.shape}")

        # Also build a combined index for mismatched / unknown countries
        all_texts = cands["_text"].tolist()
        all_ids   = cands["entity_id"].tolist()
        vec_all = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=2,
        )
        mat_all = vec_all.fit_transform(all_texts)
        self._indices["__all__"] = (vec_all, mat_all, all_ids)
        log.info(f"  [ALL] {len(all_ids):,} candidates indexed")

    def query_batch(self, queries: pd.DataFrame) -> dict:
        """
        For each S1 row in queries, return {s1_id: [candidate_ids]}.
        Uses the country-specific index; falls back to __all__ if country not in index.
        """
        log.info(f"Querying blocking index for {len(queries):,} S1 entities ...")
        queries = queries.copy()
        queries["_text"] = queries.apply(make_combined_text, axis=1)

        results = {}
        for country, grp in queries.groupby("country"):
            if country in self._indices:
                vec, mat, ids = self._indices[country]
            else:
                log.warning(f"Unknown country '{country}' — using combined index")
                vec, mat, ids = self._indices["__all__"]

            q_texts  = grp["_text"].tolist()
            q_ids    = grp["entity_id"].tolist()
            q_mat    = vec.transform(q_texts)

            # Batch dot-product (sparse × sparse^T)
            scores = (q_mat @ mat.T)  # shape (n_queries, n_cands)

            for i, s1_id in enumerate(q_ids):
                row_scores = scores[i].toarray().flatten()
                top_idx    = np.argsort(row_scores)[::-1][:self.top_k * 2]
                cand_ids   = [
                    ids[j] for j in top_idx
                    if row_scores[j] >= self.min_score
                ][:self.top_k]
                results[s1_id] = cand_ids

        # Handle any countries not covered
        for s1_id in queries["entity_id"]:
            if s1_id not in results:
                results[s1_id] = []

        return results


class NameTokenBlocker:
    """
    Secondary blocker: inverted index on normalized name tokens.
    Retrieves candidates sharing >= 1 significant name token.
    Used to augment TF-IDF results.
    """

    def __init__(self, max_per_token: int = 200):
        self.max_per_token = max_per_token
        self._index: dict[str, list] = defaultdict(list)

    def build(self, cands: pd.DataFrame):
        log.info("Building name-token inverted index ...")
        for _, row in cands.iterrows():
            cid   = row["entity_id"]
            toks  = name_tokens(str(row.get("business_name", "") or ""))
            # Filter stop-like tokens
            toks  = {t for t in toks if len(t) >= 3}
            for tok in toks:
                lst = self._index[tok]
                if len(lst) < self.max_per_token:
                    lst.append(cid)

    def query(self, s1_row: dict) -> set:
        toks = name_tokens(str(s1_row.get("business_name", "") or ""))
        toks = {t for t in toks if len(t) >= 3}
        result = set()
        for tok in toks:
            result.update(self._index.get(tok, []))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# F_0.5 evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def f05_score(precision: float, recall: float) -> float:
    beta2 = 0.25
    denom = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denom if denom > 0 else 0.0


def evaluate_predictions(predictions: dict, ground_truth: dict) -> dict:
    """
    Macro-average F_0.5 across all S1 entities in ground_truth.
    predictions: {s1_id: set(matched_ids)}
    ground_truth: {s1_id: set(matched_ids)}  (empty set = singleton)
    """
    f05_list, prec_list, rec_list = [], [], []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        if not true_matches and not pred_matches:
            f05_list.append(1.0)
            prec_list.append(1.0)
            rec_list.append(1.0)
        elif not true_matches:
            f05_list.append(0.0)
            prec_list.append(0.0)
            rec_list.append(1.0)
        elif not pred_matches:
            f05_list.append(0.0)
            prec_list.append(1.0)
            rec_list.append(0.0)
        else:
            tp = len(true_matches & pred_matches)
            prec = tp / len(pred_matches)
            rec  = tp / len(true_matches)
            f05_list.append(f05_score(prec, rec))
            prec_list.append(prec)
            rec_list.append(rec)
    return {
        "f05":       float(np.mean(f05_list)),
        "precision": float(np.mean(prec_list)),
        "recall":    float(np.mean(rec_list)),
        "n":         len(f05_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class EntityResolutionPipeline:

    def __init__(self):
        self.tfidf_blocker = TFIDFBlocker()
        self.name_blocker  = NameTokenBlocker()
        self.clf           = None
        self.threshold     = 0.5

    # ── Step 1: Build blocking index ──────────────────────────────────────────
    def build_blocking_index(self, s2: pd.DataFrame, s3: pd.DataFrame):
        cands = pd.concat([s2, s3], ignore_index=True)
        self.tfidf_blocker.build(cands)
        # Name token index (only for the combined frame)
        # We skip this for 10M+ rows due to memory; rely on TF-IDF
        log.info("Blocking index ready.")
        return cands

    # ── Step 2: Generate candidates ───────────────────────────────────────────
    def generate_candidates(
        self,
        s1: pd.DataFrame,
        cands_df: pd.DataFrame,
        batch_size: int = 5000
    ) -> dict:
        """Return {s1_id: [cand_ids]} using TF-IDF blocking (batched by country)."""
        all_candidates = {}
        n = len(s1)
        for start in range(0, n, batch_size):
            batch = s1.iloc[start : start + batch_size]
            batch_results = self.tfidf_blocker.query_batch(batch)
            all_candidates.update(batch_results)
            if (start // batch_size) % 20 == 0:
                log.info(f"  Blocking progress: {min(start+batch_size, n):,}/{n:,}")
        return all_candidates

    # ── Step 3: Build feature matrix ─────────────────────────────────────────
    def build_feature_matrix(
        self,
        s1: pd.DataFrame,
        cands_df: pd.DataFrame,
        candidates: dict,
        ground_truth: dict = None,
        max_neg_per_s1: int = 5,
    ) -> tuple:
        """
        For each (S1, candidate) pair, compute features.
        If ground_truth is provided, label the pairs (for training).
        Returns (X DataFrame, y array-or-None, pair_meta list).
        """
        log.info("Building feature matrix ...")
        s1_dict    = s1.set_index("entity_id").to_dict("index")
        cands_dict = cands_df.set_index("entity_id").to_dict("index")

        rows = []
        meta = []   # (s1_id, cand_id)
        labels = [] if ground_truth is not None else None

        for s1_id, cand_ids in candidates.items():
            if s1_id not in s1_dict:
                continue
            s1_row = s1_dict[s1_id]
            true_matches = set(ground_truth.get(s1_id, set())) if ground_truth is not None else None

            pos_ids = [c for c in cand_ids if true_matches and c in true_matches] if true_matches else []
            neg_ids = [c for c in cand_ids if not true_matches or c not in true_matches]

            # Balance negatives during training
            if ground_truth is not None and neg_ids:
                rng = np.random.default_rng(hash(s1_id) % (2**31))
                n_neg = max(max_neg_per_s1, len(pos_ids) * 3)
                neg_ids = rng.choice(neg_ids, size=min(n_neg, len(neg_ids)), replace=False).tolist()

            for cand_id in pos_ids + neg_ids:
                if cand_id not in cands_dict:
                    continue
                cand_row = cands_dict[cand_id]
                feats = compute_features(s1_row, cand_row)
                rows.append(feats)
                meta.append((s1_id, cand_id))
                if ground_truth is not None:
                    labels.append(1 if cand_id in true_matches else 0)

        X = pd.DataFrame(rows, columns=FEATURE_COLS)
        y = np.array(labels) if labels is not None else None
        log.info(f"  Feature matrix shape: {X.shape}, positives: {sum(labels) if labels else 'N/A'}")
        return X, y, meta

    # ── Step 4: Train classifier ──────────────────────────────────────────────
    def train(self, X: pd.DataFrame, y: np.ndarray):
        log.info("Training LightGBM classifier ...")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        self.clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        self.clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )
        log.info(f"  Best iteration: {self.clf.best_iteration_}")
        # Feature importance
        fi = sorted(zip(FEATURE_COLS, self.clf.feature_importances_), key=lambda x: -x[1])
        log.info("  Top features: " + str(fi[:10]))

    # ── Step 5: Predict & threshold ───────────────────────────────────────────
    def predict(
        self,
        s1_ids_all: list,
        X: pd.DataFrame,
        meta: list,
        threshold: float = None,
    ) -> dict:
        """Return {s1_id: set(matched_ids)} using the trained classifier."""
        thresh = threshold or self.threshold
        probs = self.clf.predict_proba(X)[:, 1]
        result = {sid: set() for sid in s1_ids_all}
        for prob, (s1_id, cand_id) in zip(probs, meta):
            if prob >= thresh:
                result[s1_id].add(cand_id)
        return result

    # ── Step 6: Tune threshold ────────────────────────────────────────────────
    def tune_threshold(
        self,
        X_val: pd.DataFrame,
        meta_val: list,
        ground_truth_val: dict,
        s1_ids_val: list,
    ) -> float:
        log.info("Tuning decision threshold for F_0.5 ...")
        probs = self.clf.predict_proba(X_val)[:, 1]
        best_thresh, best_f05 = 0.5, -1.0

        for thresh in THRESHOLD_GRID:
            preds = {sid: set() for sid in s1_ids_val}
            for prob, (s1_id, cand_id) in zip(probs, meta_val):
                if prob >= thresh:
                    preds[s1_id].add(cand_id)
            metrics = evaluate_predictions(preds, ground_truth_val)
            if metrics["f05"] > best_f05:
                best_f05, best_thresh = metrics["f05"], thresh

        log.info(f"  Best threshold = {best_thresh:.3f}  →  F_0.5 = {best_f05:.4f}")
        self.threshold = best_thresh
        return best_thresh

    # ── Step 7: Write output ──────────────────────────────────────────────────
    def write_output(
        self,
        s1_ids: list,
        predictions: dict,
        candidates: dict,
        matching_path: Path,
        candidate_path: Path,
    ):
        log.info("Writing output files ...")

        # matching_results.tsv
        with open(matching_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in s1_ids:
                matched = predictions.get(s1_id, set())
                f.write(f"{s1_id}\t{','.join(sorted(matched))}\n")

        # candidate_pairs.tsv
        with open(candidate_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in s1_ids:
                cands = candidates.get(s1_id, [])
                f.write(f"{s1_id}\t{','.join(cands)}\n")

        log.info(f"  Wrote {matching_path}")
        log.info(f"  Wrote {candidate_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_training_validation():
    """Train + validate on training data (80/20 split by S1 entity)."""
    log.info("=== Training + Validation Mode ===")

    # Load training data
    s1  = load_source(TRAIN_S1)
    s2  = load_source(TRAIN_S2)
    s3  = load_source(TRAIN_S3)
    gt  = load_ground_truth(TRAIN_GT)

    # Split S1 into train/val
    s1_train, s1_val = train_test_split(s1, test_size=0.2, random_state=42)
    log.info(f"Train S1: {len(s1_train):,}, Val S1: {len(s1_val):,}")

    gt_train = {k: v for k, v in gt.items() if k in set(s1_train["entity_id"])}
    gt_val   = {k: v for k, v in gt.items() if k in set(s1_val["entity_id"])}

    pipe = EntityResolutionPipeline()

    # Build blocking index on S2+S3
    cands_df = pipe.build_blocking_index(s2, s3)

    # Generate candidates for training split
    log.info("Generating candidates for training S1 ...")
    train_candidates = pipe.generate_candidates(s1_train, cands_df)

    # Also generate for validation (so we can tune threshold)
    log.info("Generating candidates for validation S1 ...")
    val_candidates = pipe.generate_candidates(s1_val, cands_df)

    # Build feature matrices
    X_train, y_train, meta_train = pipe.build_feature_matrix(
        s1_train, cands_df, train_candidates, gt_train, max_neg_per_s1=8
    )
    X_val, y_val, meta_val = pipe.build_feature_matrix(
        s1_val, cands_df, val_candidates, gt_val, max_neg_per_s1=8
    )

    # Train
    pipe.train(X_train, y_train)

    # Tune threshold on validation set
    s1_val_ids = s1_val["entity_id"].tolist()
    pipe.tune_threshold(X_val, meta_val, gt_val, s1_val_ids)

    # Evaluate on validation
    val_preds = pipe.predict(s1_val_ids, X_val, meta_val)
    metrics = evaluate_predictions(val_preds, gt_val)
    log.info(f"Validation F_0.5={metrics['f05']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    return pipe, metrics


def run_test_inference(pipe: "EntityResolutionPipeline" = None):
    """Run inference on test set, write output files."""
    log.info("=== Test Inference Mode ===")

    s1  = load_source(TEST_S1)
    s2  = load_source(TEST_S2)
    s3  = load_source(TEST_S3)

    if pipe is None:
        # Standalone: train first on full training data
        log.info("Training on full training data before test inference ...")
        s1_tr = load_source(TRAIN_S1)
        s2_tr = load_source(TRAIN_S2)
        s3_tr = load_source(TRAIN_S3)
        gt_tr = load_ground_truth(TRAIN_GT)

        pipe = EntityResolutionPipeline()
        cands_df_tr = pipe.build_blocking_index(s2_tr, s3_tr)
        train_cands = pipe.generate_candidates(s1_tr, cands_df_tr)
        X_tr, y_tr, meta_tr = pipe.build_feature_matrix(s1_tr, cands_df_tr, train_cands, gt_tr)
        pipe.train(X_tr, y_tr)
        # Threshold from training CV (default 0.5, or tune on full)
    
    # Build index for test S2+S3
    cands_df = pipe.build_blocking_index(s2, s3)

    # Generate candidates
    log.info("Generating candidates for test S1 ...")
    test_candidates = pipe.generate_candidates(s1, cands_df)

    # Build feature matrix (no labels)
    X_test, _, meta_test = pipe.build_feature_matrix(s1, cands_df, test_candidates)

    # Predict
    s1_ids = s1["entity_id"].tolist()
    test_preds = pipe.predict(s1_ids, X_test, meta_test)

    # Write output
    pipe.write_output(
        s1_ids,
        test_preds,
        test_candidates,
        OUTPUT_DIR / "matching_results.tsv",
        OUTPUT_DIR / "candidate_pairs.tsv",
    )
    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 — Entity Resolution")
    parser.add_argument("--mode", choices=["train", "test", "full"], default="full",
                        help="train=validate only, test=inference only, full=train then infer")
    args = parser.parse_args()

    if args.mode == "train":
        run_training_validation()
    elif args.mode == "test":
        run_test_inference(pipe=None)
    else:
        # Full: train on 80%, validate, then retrain on 100% and infer on test
        pipe, val_metrics = run_training_validation()
        log.info(f"Validation complete. F_0.5={val_metrics['f05']:.4f}")
        # Now run test inference with the same pipeline (trained on 80%)
        # For best results, retrain on full data — but start with this
        run_test_inference(pipe=pipe)


if __name__ == "__main__":
    main()
