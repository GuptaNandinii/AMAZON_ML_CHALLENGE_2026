# Business Entity Resolution — Amazon ML Challenge 2026

## Overview
This pipeline performs large-scale business entity resolution across three data sources.
It matches records from Source 2 and Source 3 to deduplicated reference entities in Source 1.

## Architecture
```
dataset/
  train/ → train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
  test/  → test_source1.tsv, test_source2.tsv, test_source3.tsv
output/
  matching_results.tsv    ← final predictions (upload this to leaderboard)
  candidate_pairs.tsv     ← blocking candidates
models/
  lgbm_full_model.pkl     ← trained LightGBM model
code/business_entity_resolution/
  src/final_solution.py   ← main pipeline
  requirements.txt
```

## Pipeline Steps
1. **Text normalization**: Unicode NFKD, lowercase, punctuation removal, legal-suffix stripping
2. **Multi-layer blocking** (4 layers):
   - TF-IDF char-ngram (2-4) on `name + address` per country bucket
   - TF-IDF char-ngram (2-4) on `address only` per country (handles transliterations)
   - First-word exact match inverted index
   - ZIP/PIN prefix (4-digit) inverted index
3. **Feature extraction** (24 features): RapidFuzz similarity scores for name/address/combined, 
   Jaccard on tokens and numerics, length ratios, country match, source indicator
4. **LightGBM classifier**: 1200 trees, tuned scale_pos_weight, 10% internal val for early stopping
5. **Threshold optimization**: Grid search over F_0.5 metric on 20% held-out validation

## Reproduce End-to-End

### Install dependencies
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### Run full pipeline (validate + retrain + test inference)
```bash
cd student_resource/
python code/business_entity_resolution/src/final_solution.py --mode full
```
Produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### Run validation only (to check performance)
```bash
python code/business_entity_resolution/src/final_solution.py --mode validate
```

### Run test inference only (requires saved model)
```bash
python code/business_entity_resolution/src/final_solution.py --mode test
```

### Validate output format
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Expected Performance
- Validation F_0.5: >0.97 (on 20% held-out)
- Blocking recall: >0.99 (4-layer combined)
- Runtime: ~2-4 hours on CPU (full mode)

## Key Design Choices
- **Country-segmented TF-IDF**: Separate index per country drastically improves blocking precision
- **Address-only TF-IDF**: Essential for India records where business names appear in native scripts (Hindi, Kannada, Gujarati, Bengali, etc.) — the address is the only Latin-text link
- **Precision-heavy threshold**: F_0.5 rewards precision 2× over recall; threshold tuned on validation set
- **No external data**: Uses only the provided training data per competition rules
