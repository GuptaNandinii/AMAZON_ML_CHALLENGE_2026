"""
Analyze blocking strategies on a training sample.
Helps understand what % of true matches we can find via different blocking keys.
"""
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

TRAIN = Path(r'd:\AMAZON ML CHALLENGE\6ab10eb3b23ba_student_resource\student_resource\dataset\train')

print("Loading data (samples)...")
SAMPLE = 100_000  # use first 100k rows of each file to speed analysis

s1 = pd.read_csv(TRAIN / 'train_source1.tsv', sep='\t', encoding='utf-8', nrows=SAMPLE)
s2 = pd.read_csv(TRAIN / 'train_source2.tsv', sep='\t', encoding='utf-8', nrows=SAMPLE*3)
s3 = pd.read_csv(TRAIN / 'train_source3.tsv', sep='\t', encoding='utf-8', nrows=SAMPLE*3)
gt = pd.read_csv(TRAIN / 'train_ground_truth.tsv', sep='\t', encoding='utf-8', nrows=SAMPLE)

s2 = s2.fillna('')
s3 = s3.fillna('')
s1 = s1.fillna('')
gt = gt.fillna('')

# Build lookup
s2_dict = s2.set_index('entity_id').to_dict('index')
s3_dict = s3.set_index('entity_id').to_dict('index')
s1_dict = s1.set_index('entity_id').to_dict('index')

# Build index of which S1 IDs appear in our sample
s1_set = set(s1_dict.keys())

# Filter GT to S1 entities we loaded
gt_sub = gt[gt['source1_entity_id'].isin(s1_set)].copy()
print(f"GT entries for sample S1: {len(gt_sub)}")

# Parse matched IDs
def parse_matches(s):
    if not s or str(s).strip() == '':
        return []
    return [x.strip() for x in str(s).split(',') if x.strip()]

# Collect all true pairs (S1, Sx) where Sx is in our S2/S3 sample
all_pairs = []
for _, row in gt_sub.iterrows():
    s1_id = row['source1_entity_id']
    for mid in parse_matches(row['matched_entity_ids']):
        d = s2_dict if mid.startswith('S2-') else s3_dict
        if mid in d:
            all_pairs.append((s1_id, mid))

print(f"True pairs findable in sample: {len(all_pairs)}")

# ───────────────────────────────────────────────────────────────────────────
# Utility functions for blocking keys
# ───────────────────────────────────────────────────────────────────────────

def norm(s):
    if not s: return ''
    s = unicodedata.normalize('NFKD', str(s)).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def first_word(name, k=1):
    toks = norm(name).split()
    return ' '.join(toks[:k]) if toks else ''

def zip_key(addr, country):
    if country == 'India':
        m = re.search(r'\b(\d{6})\b', addr)
    else:
        m = re.search(r'\b(\d{5})\b', addr)
    return m.group(1)[:4] if m else ''

def num_tokens(addr):
    return set(re.findall(r'\b\d+\b', norm(addr)))

def trigrams(s, n=3):
    s = f' {s} '
    return set(s[i:i+n] for i in range(len(s)-n+1))

def trigram_overlap(a, b):
    ta = trigrams(norm(a))
    tb = trigrams(norm(b))
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0

# ───────────────────────────────────────────────────────────────────────────
# Check different blocking keys
# ───────────────────────────────────────────────────────────────────────────

stats = {
    'country_match': 0,
    'first_word_name_match': 0,
    'first_2words_name_match': 0,
    'zip_match': 0,
    'any_num_overlap': 0,
    'name_trigram_sim_gt0.2': 0,
    'combined_trigram_sim_gt0.1': 0,
}

for (s1_id, mid) in all_pairs:
    r1 = s1_dict[s1_id]
    d  = s2_dict if mid.startswith('S2-') else s3_dict
    r2 = d[mid]

    n1 = str(r1.get('business_name','') or '')
    n2 = str(r2.get('business_name','') or '')
    a1 = str(r1.get('business_address','') or '')
    a2 = str(r2.get('business_address','') or '')
    c1 = r1.get('country','')
    c2 = r2.get('country','')

    if c1 == c2:
        stats['country_match'] += 1
    if first_word(n1, 1) and first_word(n1, 1) == first_word(n2, 1):
        stats['first_word_name_match'] += 1
    if first_word(n1, 2) and first_word(n1, 2) == first_word(n2, 2):
        stats['first_2words_name_match'] += 1
    if zip_key(a1, c1) and zip_key(a1, c1) == zip_key(a2, c2):
        stats['zip_match'] += 1
    ns1 = num_tokens(a1); ns2 = num_tokens(a2)
    if ns1 and ns2 and ns1 & ns2:
        stats['any_num_overlap'] += 1
    if trigram_overlap(n1, n2) > 0.2:
        stats['name_trigram_sim_gt0.2'] += 1
    combined1 = norm(n1) + ' ' + norm(a1)
    combined2 = norm(n2) + ' ' + norm(a2)
    if trigram_overlap(combined1, combined2) > 0.1:
        stats['combined_trigram_sim_gt0.1'] += 1

n = len(all_pairs)
print(f"\nBlocking key recall analysis (n={n:,} true pairs):")
for k, v in stats.items():
    print(f"  {k}: {v:,} / {n:,} = {v/n*100:.1f}%")

# ───────────────────────────────────────────────────────────────────────────
# What makes true pairs hard to match? (low string similarity)
# ───────────────────────────────────────────────────────────────────────────
print("\nAnalyzing difficult pairs (name trigram sim < 0.2):")
hard_pairs = []
for (s1_id, mid) in all_pairs[:5000]:
    r1 = s1_dict[s1_id]
    d  = s2_dict if mid.startswith('S2-') else s3_dict
    r2 = d[mid]
    n1 = str(r1.get('business_name','') or '')
    n2 = str(r2.get('business_name','') or '')
    if trigram_overlap(n1, n2) < 0.2:
        hard_pairs.append((n1, n2, str(r1.get('business_address','')), str(r2.get('business_address','')), r1.get('country','')))

print(f"  Hard pairs: {len(hard_pairs)}")
for p in hard_pairs[:10]:
    print(f"    '{p[0]}' vs '{p[1]}'  | addr: '{p[2][:40]}' vs '{p[3][:40]}' | {p[4]}")
