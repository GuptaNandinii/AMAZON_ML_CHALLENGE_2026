"""
Amazon ML Challenge 2026 — Final Production Solution (v3)
Target: F_0.5 > 0.98

Architecture:
  BLOCKING (4 layers, unioned per S1 entity):
    1. TF-IDF char-ngram on (name + address) per country bucket
    2. TF-IDF char-ngram on address ONLY (catches transliteration cases where only addr matches)
    3. First-word exact match inverted index (fast, high-recall for common cases)
    4. Numeric token set overlap (house numbers, ZIP/PIN)

  FEATURES (24 features per pair):
    - RapidFuzz scores: ratio, partial_ratio, token_set_ratio, token_sort_ratio, WRatio
    - Jaccard on name tokens
    - Jaccard on address numeric tokens
    - Combined text scores
    - Address/name length ratios
    - Country match, source indicator

  CLASSIFIER:
    LightGBM with ~1000 trees, scale_pos_weight balanced
    Threshold tuned for max F_0.5 on validation

  POST-PROCESSING:
    Country-based sanity check: never match cross-country
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

# ─── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── paths ──────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve()
# code/business_entity_resolution/src/final_solution.py → student_resource at [3]
ROOT      = _HERE.parents[3]
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR  = ROOT / "dataset" / "test"
OUT_DIR   = ROOT / "output"
MODEL_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── hyper-params ───────────────────────────────────────────────────────────
TFIDF_NGRAM_COMB  = (2, 4)     # for name+addr combined
TFIDF_NGRAM_ADDR  = (2, 4)     # for addr-only (catches transliterations)
TFIDF_NGRAM_NAME  = (2, 4)     # for name-only (catches empty/missing addresses)
TFIDF_MAX_FEAT    = 300_000
TFIDF_TOP_K_COMB  = 40         # candidates from combined index
TFIDF_TOP_K_ADDR  = 20         # candidates from addr-only index
TFIDF_TOP_K_NAME  = 35         # candidates from name-only index
TFIDF_MIN_SIM     = 0.04       # minimum cosine similarity to include
MAX_CANDS_PER_S1  = 160        # hard cap
CHUNK_QUERY       = 2_000      # S1 rows per TF-IDF query batch
NEG_RATIO         = 6          # negatives per positive in training

LGBM_PARAMS = dict(
    objective         = "binary",
    metric            = "binary_logloss",
    n_estimators      = 1200,
    learning_rate     = 0.035,
    num_leaves        = 127,
    max_depth         = -1,
    min_child_samples = 20,
    feature_fraction  = 0.85,
    bagging_fraction  = 0.85,
    bagging_freq      = 5,
    reg_alpha         = 0.05,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    verbose           = -1,
    random_state      = 42,
)

THRESH_GRID = np.concatenate([
    np.linspace(0.10, 0.40, 31),
    np.linspace(0.40, 0.90, 51),
])


# ─────────────────────────────────────────────────────────────────────────────
# Text normalization
# ─────────────────────────────────────────────────────────────────────────────

_LEGAL = re.compile(
    r"\b(pvt\.?\s*ltd\.?|private\s+limited|limited|llc|llp|"
    r"inc\.?|incorporated|corp\.?|corporation|co\.?\s*ltd\.?|"
    r"co\.?|company|enterprises?|solutions?|services?|group|"
    r"associates?|consultants?|technologies?|tech|systems?|"
    r"international|intl|industries|holdings?|ventures?|"
    r"trading|traders?|suppliers?|distributors?|and sons?)\b",
    re.I,
)
_PUNCT  = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
_DIGITS = re.compile(r"\b\d+\b")
_ZIP6   = re.compile(r"\b(\d{6})\b")   # India PIN
_ZIP5   = re.compile(r"\b(\d{5})\b")   # US ZIP
_BNUM   = re.compile(r"\b0*(\d{1,6})[a-z]?\b", re.I)

_DOMAIN = re.compile(r"\bhttps?://\S+|\bwww\.\S+|\.(com|in|org|net|co|io|biz|info|gov|edu)\b", re.I)

_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
_STATE_PAT = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(_US_STATES.keys(), key=len, reverse=True)) + r")\b", re.I)


def extract_bldg_num(addr: str) -> str:
    m = _BNUM.findall(addr)
    return m[0] if m else ""


def _norm_base(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    s = _DOMAIN.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " ")
    s = _PUNCT.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def norm_name(s: str) -> str:
    if not isinstance(s, str): return ""
    n = _norm_base(s)
    n = _LEGAL.sub(" ", n)
    return _SPACES.sub(" ", n).strip()


def norm_addr(s: str) -> str:
    if not isinstance(s, str): return ""
    n = _norm_base(s)
    n = _STATE_PAT.sub(lambda m: _US_STATES[m.group(0).lower()], n)
    return _SPACES.sub(" ", n).strip()


def combined(name: str, addr: str) -> str:
    return f"{norm_name(name)} {norm_addr(addr)}".strip()


def num_set(addr: str) -> Set[str]:
    return set(_DIGITS.findall(norm_addr(addr)))


def zip_prefix(addr: str, country: str, n: int = 4) -> str:
    pat = _ZIP6 if country == "India" else _ZIP5
    m = pat.search(addr)
    return m.group(1)[:n] if m else ""


def first_words(name: str, k: int = 1) -> str:
    toks = norm_name(name).split()
    return " ".join(toks[:k]) if toks else ""


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_df(path: Path) -> pd.DataFrame:
    t = time.time()
    log.info(f"Loading {path.name} ...")
    df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, na_filter=False)
    df.fillna("", inplace=True)
    log.info(f"  → {len(df):,} rows  ({time.time()-t:.1f}s)")
    return df


def load_gt(path: Path) -> Dict[str, Set[str]]:
    log.info(f"Loading GT from {path.name} ...")
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str).fillna("")
    s1_ids = df["source1_entity_id"].values
    matched = df["matched_entity_ids"].values
    out: Dict[str, Set[str]] = {}
    for sid, mstr in zip(s1_ids, matched):
        mstr = mstr.strip()
        out[sid] = set(m.strip() for m in mstr.split(",") if m.strip()) if mstr else set()
    log.info(f"  → Loaded {len(out):,} GT entries in {time.time()-t0:.2f}s")
    return out


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_name"]    = df["business_name"].apply(norm_name)
    df["n_addr"]    = df["business_address"].apply(norm_addr)
    df["n_combined"]= (df["n_name"] + " " + df["n_addr"]).str.strip()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Name (7)
    "name_ratio", "name_partial", "name_token_set",
    "name_token_sort", "name_wratio", "name_jaccard", "name_first_tok",
    # Address (7)
    "addr_ratio", "addr_partial", "addr_token_set",
    "addr_token_sort", "addr_wratio",
    "num_jaccard", "num_exact",
    # Combined (3)
    "comb_ratio", "comb_token_set", "comb_wratio",
    # Lengths (4)
    "name_len_ratio", "addr_len_ratio", "addr1_empty", "addr2_empty",
    # Meta (3)
    "country_match", "is_s2", "is_s3",
]
N_FEAT = len(FEATURE_NAMES)  # 24


def compute_features_batch(s1_df: pd.DataFrame, cand_df: pd.DataFrame) -> np.ndarray:
    """
    Aligned computation: s1_df[i] paired with cand_df[i].
    Returns float32 array (N, N_FEAT).
    """
    n1_arr = s1_df["n_name"].values
    n2_arr = cand_df["n_name"].values
    a1_arr = s1_df["n_addr"].values
    a2_arr = cand_df["n_addr"].values
    c1_arr = s1_df["country"].values
    c2_arr = cand_df["country"].values
    id_arr = cand_df["entity_id"].values

    N = len(n1_arr)
    F = np.zeros((N, N_FEAT), dtype=np.float32)

    for i in range(N):
        n1, n2 = n1_arr[i], n2_arr[i]
        a1, a2 = a1_arr[i], a2_arr[i]
        c1c = f"{n1} {a1}".strip()
        c2c = f"{n2} {a2}".strip()

        # Name
        F[i, 0] = fuzz.ratio(n1, n2) / 100
        F[i, 1] = fuzz.partial_ratio(n1, n2) / 100
        F[i, 2] = fuzz.token_set_ratio(n1, n2) / 100
        F[i, 3] = fuzz.token_sort_ratio(n1, n2) / 100
        F[i, 4] = fuzz.WRatio(n1, n2) / 100

        t1 = set(n1.split()) if n1 else set()
        t2 = set(n2.split()) if n2 else set()
        F[i, 5] = len(t1 & t2) / len(t1 | t2) if (t1 | t2) else 0.0

        fw1 = n1.split()[0] if n1 else ""
        fw2 = n2.split()[0] if n2 else ""
        F[i, 6] = float(fw1 == fw2 and fw1 != "")

        # Address
        F[i, 7]  = fuzz.ratio(a1, a2) / 100
        F[i, 8]  = fuzz.partial_ratio(a1, a2) / 100
        F[i, 9]  = fuzz.token_set_ratio(a1, a2) / 100
        F[i, 10] = fuzz.token_sort_ratio(a1, a2) / 100
        F[i, 11] = fuzz.WRatio(a1, a2) / 100

        ns1 = set(_DIGITS.findall(a1))
        ns2 = set(_DIGITS.findall(a2))
        F[i, 12] = len(ns1 & ns2) / len(ns1 | ns2) if (ns1 | ns2) else 0.0
        F[i, 13] = float(bool(ns1 & ns2))

        # Combined
        F[i, 14] = fuzz.ratio(c1c, c2c) / 100
        F[i, 15] = fuzz.token_set_ratio(c1c, c2c) / 100
        F[i, 16] = fuzz.WRatio(c1c, c2c) / 100

        # Length ratios
        max_n = max(len(n1), len(n2))
        F[i, 17] = min(len(n1), len(n2)) / max_n if max_n > 0 else 1.0
        max_a = max(len(a1), len(a2))
        F[i, 18] = min(len(a1), len(a2)) / max_a if max_a > 0 else 1.0
        F[i, 19] = float(not a1)
        F[i, 20] = float(not a2)

    # Vectorized columns
    F[:, 21] = (c1_arr == c2_arr).astype(np.float32)
    F[:, 22] = np.array([1.0 if x.startswith("S2-") else 0.0 for x in id_arr], dtype=np.float32)
    F[:, 23] = np.array([1.0 if x.startswith("S3-") else 0.0 for x in id_arr], dtype=np.float32)

    return F


# ─────────────────────────────────────────────────────────────────────────────
# Blocker
# ─────────────────────────────────────────────────────────────────────────────

def _fit_tfidf(texts: List[str], ngram: tuple, max_feat: int) -> Tuple:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram,
        max_features=max_feat,
        sublinear_tf=True,
        min_df=1,
    )
    mat = vec.fit_transform(texts)
    return vec, mat


class MultiLayerBlocker:
    """
    5-layer blocking:
      1. TF-IDF (char-ngram) on combined text  — per country
      2. TF-IDF (char-ngram) on address only   — per country (handles transliterations)
      3. TF-IDF (char-ngram) on name only      — per country (handles empty / missing addresses)
      4. First-word name inverted index
      5. ZIP/PIN prefix inverted index
    """

    def __init__(self):
        self._comb_idx: Dict[str, Tuple] = {}   # country → (vec, mat, ids)
        self._addr_idx: Dict[str, Tuple] = {}   # country → (vec, mat, ids)
        self._name_idx: Dict[str, Tuple] = {}   # country → (vec, mat, ids)
        self._fw_idx:   Dict[str, List]  = defaultdict(list)  # first_word → [ids]
        self._zip_idx:  Dict[str, List]  = defaultdict(list)  # country:zip4 → [ids]
        self._exact_name_idx: Dict[str, List] = defaultdict(list)  # clean/compact name → [ids]
        self._addr_num_idx:   Dict[str, List] = defaultdict(list)  # country:bldg:zip3 → [ids]
        self._cands_df: Optional[pd.DataFrame] = None

    def build(self, cands: pd.DataFrame):
        log.info("Building blocking index ...")
        self._cands_df = cands

        # ── TF-IDF per country ──────────────────────────────────────────────
        for country, grp in cands.groupby("country"):
            texts_c = grp["n_combined"].tolist()
            texts_a = grp["n_addr"].tolist()
            texts_n = grp["n_name"].tolist()
            ids     = grp["entity_id"].tolist()

            vc, mc = _fit_tfidf(texts_c, TFIDF_NGRAM_COMB, TFIDF_MAX_FEAT)
            va, ma = _fit_tfidf(texts_a, TFIDF_NGRAM_ADDR, TFIDF_MAX_FEAT // 2)
            vn, mn = _fit_tfidf(texts_n, TFIDF_NGRAM_NAME, TFIDF_MAX_FEAT // 2)
            self._comb_idx[country] = (vc, mc, ids)
            self._addr_idx[country] = (va, ma, ids)
            self._name_idx[country] = (vn, mn, ids)
            log.info(f"  [{country}] combined:{mc.shape}  addr:{ma.shape}  name:{mn.shape}")

        # All-country fallback
        vc_all, mc_all = _fit_tfidf(cands["n_combined"].tolist(), TFIDF_NGRAM_COMB, TFIDF_MAX_FEAT)
        va_all, ma_all = _fit_tfidf(cands["n_addr"].tolist(), TFIDF_NGRAM_ADDR, TFIDF_MAX_FEAT // 2)
        vn_all, mn_all = _fit_tfidf(cands["n_name"].tolist(), TFIDF_NGRAM_NAME, TFIDF_MAX_FEAT // 2)
        all_ids = cands["entity_id"].tolist()
        self._comb_idx["__all__"] = (vc_all, mc_all, all_ids)
        self._addr_idx["__all__"] = (va_all, ma_all, all_ids)
        self._name_idx["__all__"] = (vn_all, mn_all, all_ids)
        log.info(f"  [ALL] combined:{mc_all.shape}")

        # ── First-word index ────────────────────────────────────────────────
        log.info("  Building first-word index ...")
        MAX_PER_FW = 300  # prevent hot-key explosion
        names = cands["business_name"].values
        c_ids = cands["entity_id"].values
        for b_name, eid in zip(names, c_ids):
            fw = first_words(b_name, 1)
            if fw and len(self._fw_idx[fw]) < MAX_PER_FW:
                self._fw_idx[fw].append(eid)

        # ── ZIP index ───────────────────────────────────────────────────────
        log.info("  Building ZIP/PIN index ...")
        MAX_PER_ZIP = 200
        addrs = cands["business_address"].values
        countries = cands["country"].values
        for addr, country, eid in zip(addrs, countries, c_ids):
            zp = zip_prefix(addr, country, 4)
            if zp:
                key = f"{country}:{zp}"
                if len(self._zip_idx[key]) < MAX_PER_ZIP:
                    self._zip_idx[key].append(eid)

        # ── Exact & compact name index ──────────────────────────────────────
        log.info("  Building exact/compact name index ...")
        MAX_PER_EXACT = 200
        for n_name, eid in zip(cands["n_name"].values, c_ids):
            if n_name and len(n_name) >= 3 and len(self._exact_name_idx[n_name]) < MAX_PER_EXACT:
                self._exact_name_idx[n_name].append(eid)
            compact = "".join(n_name.split())
            if compact and len(compact) >= 4 and compact != n_name and len(self._exact_name_idx[compact]) < MAX_PER_EXACT:
                self._exact_name_idx[compact].append(eid)

        # ── Address building number + ZIP prefix index ─────────────────────
        log.info("  Building address building number index ...")
        MAX_PER_BNUM = 100
        for addr, country, eid in zip(addrs, countries, c_ids):
            bldg = extract_bldg_num(addr)
            if bldg:
                zp = zip_prefix(addr, country, 3)
                if zp:
                    key = f"{country}:{bldg}:{zp}"
                    if len(self._addr_num_idx[key]) < MAX_PER_BNUM:
                        self._addr_num_idx[key].append(eid)

        log.info("Blocking index ready.")

    def _tfidf_candidates(self, country: str, q_texts: List[str], q_ids: List[str],
                          mode: str = "combined") -> Dict[str, List[str]]:
        if mode == "combined":
            idx = self._comb_idx
            top_k = TFIDF_TOP_K_COMB
        elif mode == "addr":
            idx = self._addr_idx
            top_k = TFIDF_TOP_K_ADDR
        else:
            idx = self._name_idx
            top_k = TFIDF_TOP_K_NAME

        if country in idx:
            vec, mat, ids = idx[country]
        else:
            vec, mat, ids = idx["__all__"]

        q_mat  = vec.transform(q_texts)
        scores = q_mat @ mat.T    # (n_q, n_cands) sparse

        out: Dict[str, List[str]] = {}
        for i, s1_id in enumerate(q_ids):
            row    = scores[i].toarray().flatten()
            top    = np.argsort(row)[::-1][:top_k * 2]
            cands  = [ids[j] for j in top if row[j] >= TFIDF_MIN_SIM][:top_k]
            out[s1_id] = cands
        return out

    def query_batch(self, queries: pd.DataFrame) -> Dict[str, List[str]]:
        exact_cands: Dict[str, List[str]] = defaultdict(list)
        bnum_cands:  Dict[str, List[str]] = defaultdict(list)
        tfidf_cands: Dict[str, List[str]] = defaultdict(list)
        other_cands: Dict[str, List[str]] = defaultdict(list)

        # 1. Exact & compact name (Priority 1 — Never get truncated)
        for n_name, eid in zip(queries["n_name"].values, queries["entity_id"].values):
            if n_name in self._exact_name_idx:
                exact_cands[eid].extend(self._exact_name_idx[n_name])
            compact = "".join(n_name.split())
            if compact in self._exact_name_idx:
                exact_cands[eid].extend(self._exact_name_idx[compact])

        # 2. Address building number + ZIP anchor (Priority 2)
        for addr, country, eid in zip(queries["business_address"].values, queries["country"].values, queries["entity_id"].values):
            bldg = extract_bldg_num(addr)
            if bldg:
                zp = zip_prefix(addr, country, 3)
                if zp:
                    key = f"{country}:{bldg}:{zp}"
                    if key in self._addr_num_idx:
                        bnum_cands[eid].extend(self._addr_num_idx[key])

        # 3. TF-IDF per country (Priority 3)
        for country, grp in queries.groupby("country"):
            q_ids   = grp["entity_id"].tolist()
            q_combs = grp["n_combined"].tolist()
            q_addrs = grp["n_addr"].tolist()
            q_names = grp["n_name"].tolist()

            # Name-only TF-IDF
            name_res = self._tfidf_candidates(country, q_names, q_ids, "name")
            for s1_id, cids in name_res.items():
                tfidf_cands[s1_id].extend(cids)

            # Combined TF-IDF
            comb_res = self._tfidf_candidates(country, q_combs, q_ids, "combined")
            for s1_id, cids in comb_res.items():
                tfidf_cands[s1_id].extend(cids)

            # Address-only TF-IDF
            addr_res = self._tfidf_candidates(country, q_addrs, q_ids, "addr")
            for s1_id, cids in addr_res.items():
                other_cands[s1_id].extend(cids)

        # 4. First-word and ZIP (Priority 4 — Loose candidates)
        for b_name, eid in zip(queries["business_name"].values, queries["entity_id"].values):
            fw = first_words(b_name, 1)
            if fw:
                other_cands[eid].extend(self._fw_idx.get(fw, []))

        for addr, country, eid in zip(queries["business_address"].values, queries["country"].values, queries["entity_id"].values):
            zp = zip_prefix(addr, country, 4)
            if zp:
                key = f"{country}:{zp}"
                other_cands[eid].extend(self._zip_idx.get(key, []))

        # Assemble prioritized candidates without losing high-confidence matches
        out: Dict[str, List[str]] = {}
        for s1_id in queries["entity_id"].values:
            cands_list = []
            seen = set()
            for pool in [exact_cands[s1_id], bnum_cands[s1_id], tfidf_cands[s1_id], other_cands[s1_id]]:
                for cid in pool:
                    if cid not in seen:
                        seen.add(cid)
                        cands_list.append(cid)
                        if len(cands_list) >= MAX_CANDS_PER_S1:
                            break
                if len(cands_list) >= MAX_CANDS_PER_S1:
                    break
            out[s1_id] = cands_list
        return out

    def query_all(self, queries: pd.DataFrame, chunk: int = CHUNK_QUERY) -> Dict[str, List[str]]:
        all_res: Dict[str, List[str]] = {}
        n = len(queries)
        for start in range(0, n, chunk):
            batch     = queries.iloc[start : start + chunk]
            batch_res = self.query_batch(batch)
            all_res.update(batch_res)
            done = min(start + chunk, n)
            if done % (chunk * 20) == 0 or done >= n:
                avg_cands = np.mean([len(v) for v in all_res.values()])
                log.info(f"  Blocking {done:,}/{n:,}  avg_cands={avg_cands:.1f}")
        return all_res


# ─────────────────────────────────────────────────────────────────────────────
# F_0.5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def f05(p: float, r: float) -> float:
    d = 0.25 * p + r
    return 1.25 * p * r / d if d > 0 else 0.0

def macro_f05(preds: Dict[str, Set[str]], gt: Dict[str, Set[str]]) -> Dict:
    sc, ps, rs = [], [], []
    for s1_id, true in gt.items():
        pred = preds.get(s1_id, set())
        if not true and not pred:
            sc.append(1.0); ps.append(1.0); rs.append(1.0)
        elif not true:
            sc.append(0.0); ps.append(0.0); rs.append(1.0)
        elif not pred:
            sc.append(0.0); ps.append(1.0); rs.append(0.0)
        else:
            tp = len(true & pred)
            p  = tp / len(pred); r = tp / len(true)
            sc.append(f05(p, r)); ps.append(p); rs.append(r)
    return {"f05": np.mean(sc), "precision": np.mean(ps), "recall": np.mean(rs), "n": len(sc)}

def blocking_recall_audit(cands: Dict[str, List[str]], gt: Dict[str, Set[str]]):
    found = total = 0
    missed = []
    for s1_id, true in gt.items():
        c_set = set(cands.get(s1_id, []))
        for m in true:
            total += 1
            if m in c_set:
                found += 1
            else:
                missed.append((s1_id, m))
    recall = found / total if total > 0 else 0.0
    log.info(f"Blocking recall: {found:,}/{total:,} = {recall:.4f}")
    log.info(f"  Missed pairs: {len(missed)}")
    return recall, missed


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self):
        self.blocker   = MultiLayerBlocker()
        self.clf       = None
        self.threshold = 0.5

    def _pairs_to_features(
        self,
        s1_df: pd.DataFrame,
        cands_df: pd.DataFrame,
        candidates: Dict[str, List[str]],
        gt: Optional[Dict[str, Set[str]]] = None,
        neg_ratio: int = NEG_RATIO,
    ) -> Tuple[np.ndarray, np.ndarray | None, List[Tuple]]:

        log.info("Computing pair features ...")
        s1_idx    = s1_df.set_index("entity_id")
        cands_idx = cands_df.set_index("entity_id")
        rng       = np.random.default_rng(42)

        all_X, all_y, all_meta = [], [], []

        for s1_id, cand_ids in candidates.items():
            if s1_id not in s1_idx.index or not cand_ids:
                continue
            true = set(gt.get(s1_id, set())) if gt else set()

            pos = [c for c in cand_ids if c in true]
            neg = [c for c in cand_ids if c not in true]

            if gt is not None:
                n_neg = max(2, len(pos) * neg_ratio)
                if neg:
                    neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False).tolist()

            pairs = [(c, 1) for c in pos] + [(c, 0) for c in neg]
            if not pairs:
                continue

            valid_cids = [c for c, _ in pairs if c in cands_idx.index]
            valid_lbls = [lbl for c, lbl in pairs if c in cands_idx.index]
            if not valid_cids:
                continue

            s1_sub   = s1_idx.loc[[s1_id] * len(valid_cids)].reset_index()
            cand_sub = cands_idx.loc[valid_cids].reset_index()

            chunk_X = compute_features_batch(s1_sub, cand_sub)
            all_X.append(chunk_X)
            if gt is not None:
                all_y.extend(valid_lbls)
            all_meta.extend([(s1_id, c) for c in valid_cids])

        X = np.vstack(all_X).astype(np.float32) if all_X else np.empty((0, N_FEAT), dtype=np.float32)
        y = np.array(all_y, dtype=np.int8) if gt is not None else None
        log.info(f"  Pairs: {len(all_meta):,}" + (f"  Pos: {int(y.sum()):,}" if y is not None else ""))
        return X, y, all_meta

    def train(self, X: np.ndarray, y: np.ndarray):
        log.info("Training LightGBM ...")
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        spw   = n_neg / max(n_pos, 1)
        log.info(f"  Pairs: {len(y):,}  Pos: {n_pos:,}  SPW: {spw:.2f}")

        params = {**LGBM_PARAMS, "scale_pos_weight": spw}
        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X, y, test_size=0.10, random_state=42, stratify=y
        )
        self.clf = lgb.LGBMClassifier(**params)
        self.clf.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[
                lgb.early_stopping(70, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        log.info(f"  Best iteration: {self.clf.best_iteration_}")
        fi = sorted(zip(FEATURE_NAMES, self.clf.feature_importances_), key=lambda x: -x[1])
        log.info("  Top features: " + ", ".join(f"{n}={v:.0f}" for n, v in fi[:12]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def threshold_to_preds(
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
        best_t, best_f = 0.5, -1.0
        results_log = []
        for t in THRESH_GRID:
            preds   = self.threshold_to_preds(probs, meta, s1_ids, t)
            m       = macro_f05(preds, gt)
            results_log.append((t, m["f05"], m["precision"], m["recall"]))
            if m["f05"] > best_f:
                best_f, best_t = m["f05"], t

        # Print table around best
        log.info("Threshold sweep (neighbourhood around best):")
        for t, f, p, r in results_log:
            if abs(t - best_t) < 0.08:
                log.info(f"  t={t:.3f}  F0.5={f:.4f}  P={p:.4f}  R={r:.4f}" +
                         ("  ← BEST" if t == best_t else ""))

        self.threshold = best_t
        log.info(f"  Selected threshold = {best_t:.3f}")
        return best_t

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "threshold": self.threshold}, f)
        log.info(f"Model saved → {path}")

    def load(self, path: Path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.clf       = d["clf"]
        self.threshold = d["threshold"]
        log.info(f"Model loaded ← {path}  (threshold={self.threshold:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_output(
    s1_ids: List[str],
    preds: Dict[str, Set[str]],
    cands: Dict[str, List[str]],
):
    match_path = OUT_DIR / "matching_results.tsv"
    cand_path  = OUT_DIR / "candidate_pairs.tsv"

    with open(match_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            matched = ",".join(sorted(preds.get(sid, set())))
            f.write(f"{sid}\t{matched}\n")

    with open(cand_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            c_list = ",".join(cands.get(sid, []))
            f.write(f"{sid}\t{c_list}\n")

    n_non_singleton = sum(1 for v in preds.values() if v)
    log.info(f"Output written: {n_non_singleton:,}/{len(s1_ids):,} entities have ≥1 match")
    log.info(f"  {match_path}")
    log.info(f"  {cand_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Run modes
# ─────────────────────────────────────────────────────────────────────────────

def run_validate(s1=None, s23=None, gt=None, return_pipe=False):
    """Train on 80%, validate on 20%."""
    log.info("═══ VALIDATE MODE ═══")
    if s1 is None:
        s1  = preprocess(load_df(TRAIN_DIR / "train_source1.tsv"))
        s2  = preprocess(load_df(TRAIN_DIR / "train_source2.tsv"))
        s3  = preprocess(load_df(TRAIN_DIR / "train_source3.tsv"))
        s23 = pd.concat([s2, s3], ignore_index=True)
        del s2, s3; gc.collect()
        gt  = load_gt(TRAIN_DIR / "train_ground_truth.tsv")

    ids_all = s1["entity_id"].tolist()
    ids_tr, ids_vl = train_test_split(ids_all, test_size=0.20, random_state=42)
    s1_tr  = s1[s1["entity_id"].isin(set(ids_tr))].reset_index(drop=True)
    s1_vl  = s1[s1["entity_id"].isin(set(ids_vl))].reset_index(drop=True)
    gt_tr  = {k: gt[k] for k in ids_tr if k in gt}
    gt_vl  = {k: gt[k] for k in ids_vl if k in gt}

    pipe = Pipeline()
    pipe.blocker.build(s23)

    # Candidates
    log.info("Generating training candidates ...")
    tr_cands = pipe.blocker.query_all(s1_tr)
    log.info("Generating validation candidates ...")
    vl_cands = pipe.blocker.query_all(s1_vl)

    blocking_recall_audit(vl_cands, gt_vl)

    # Features
    X_tr, y_tr, _       = pipe._pairs_to_features(s1_tr, s23, tr_cands, gt_tr)
    X_vl, _,   meta_vl  = pipe._pairs_to_features(s1_vl, s23, vl_cands)

    # Train
    pipe.train(X_tr, y_tr)
    del X_tr, y_tr; gc.collect()

    # Evaluate
    probs_vl = pipe.predict_proba(X_vl)
    pipe.tune_threshold(probs_vl, meta_vl, gt_vl, ids_vl)
    preds_vl = pipe.threshold_to_preds(probs_vl, meta_vl, ids_vl, pipe.threshold)
    m = macro_f05(preds_vl, gt_vl)
    log.info(f"Validation  F_0.5={m['f05']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    pipe.save(MODEL_DIR / "lgbm_val_model.pkl")
    if return_pipe:
        return pipe, m
    return m


def run_full():
    """Validate, then retrain on full data, then test inference."""
    log.info("═══ FULL MODE ═══")

    # Load train data once
    s1  = preprocess(load_df(TRAIN_DIR / "train_source1.tsv"))
    s2  = preprocess(load_df(TRAIN_DIR / "train_source2.tsv"))
    s3  = preprocess(load_df(TRAIN_DIR / "train_source3.tsv"))
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3; gc.collect()
    gt  = load_gt(TRAIN_DIR / "train_ground_truth.tsv")

    # Validate to get threshold
    pipe_val, val_m = run_validate(s1, s23, gt, return_pipe=True)
    best_thresh = pipe_val.threshold
    del pipe_val; gc.collect()

    # Retrain on full training data
    log.info("Retraining on FULL training data ...")
    pipe = Pipeline()
    pipe.blocker.build(s23)
    tr_cands_full = pipe.blocker.query_all(s1)
    X_full, y_full, _ = pipe._pairs_to_features(s1, s23, tr_cands_full, gt)
    pipe.train(X_full, y_full)
    pipe.threshold = best_thresh
    pipe.save(MODEL_DIR / "lgbm_full_model.pkl")
    del X_full, y_full, s1, s23, gt; gc.collect()

    # Test
    run_test(pipe)


def run_test(pipe: Optional[Pipeline] = None):
    """Run inference on test set."""
    log.info("═══ TEST MODE ═══")

    if pipe is None:
        for name in ["lgbm_full_model.pkl", "lgbm_val_model.pkl"]:
            p = MODEL_DIR / name
            if p.exists():
                pipe = Pipeline()
                pipe.load(p)
                break
        if pipe is None:
            raise RuntimeError("No saved model found. Run with --mode full first.")

    s1t = preprocess(load_df(TEST_DIR / "test_source1.tsv"))
    s2t = preprocess(load_df(TEST_DIR / "test_source2.tsv"))
    s3t = preprocess(load_df(TEST_DIR / "test_source3.tsv"))
    s23t = pd.concat([s2t, s3t], ignore_index=True)
    del s2t, s3t; gc.collect()

    # Rebuild blocker for test candidates
    log.info("Building test blocking index ...")
    pipe.blocker = MultiLayerBlocker()
    pipe.blocker.build(s23t)

    log.info("Generating test candidates ...")
    test_cands = pipe.blocker.query_all(s1t)

    X_test, _, meta_test = pipe._pairs_to_features(s1t, s23t, test_cands)
    probs = pipe.predict_proba(X_test)

    s1_ids = s1t["entity_id"].tolist()
    preds  = pipe.threshold_to_preds(probs, meta_test, s1_ids, pipe.threshold)

    write_output(s1_ids, preds, test_cands)
    log.info("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Amazon ML Challenge 2026 — Entity Resolution v3")
    p.add_argument("--mode", choices=["validate", "full", "test"], default="full",
                   help="validate=80/20 split; full=validate+retrain+test; test=inference only")
    args = p.parse_args()

    t0 = time.time()
    if args.mode == "validate":
        run_validate()
    elif args.mode == "full":
        run_full()
    else:
        run_test()
    log.info(f"Total time: {(time.time()-t0)/60:.1f} min")
