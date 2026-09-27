# Amazon ML Challenge 2026 — Multi-Source Business Entity Resolution

Comprehensive, production-grade entity resolution pipeline developed for the **Amazon ML Challenge 2026**.

## Problem Formulation
Given records across three heterogeneous business directories:
- **Source 1**: Primary queries
- **Source 2**: Secondary catalog
- **Source 3**: Tertiary catalog

The objective is to map each business in Source 1 to its corresponding duplicate entities in Source 2 and Source 3 across three major geographic markets: **India**, **United States**, and **France**, evaluated under **Macro $F_{0.5}$**.

---

## Architecture & Engineering Innovations

### 1. Country-Isolated Streaming Architecture
- Partitions the multi-million record test set by country bucket (`France`, `US`, `India`) to prevent cross-country false collisions and contain peak RAM below **4.5 GiB**.
- Employs streaming query chunks to completely eliminate out-of-memory (OOM) failures under 30 GB container constraints.

### 2. Streamed Sparse Blocker
- **Precomputed Transpose Matrices ($C^T$)**: Eliminates redundant on-the-fly CSR transpositions.
- **Word-Level N-Grams (1, 2) with Dynamic Stop-Word Pruning (`max_df`)**: Filters ultra-high-frequency stop-words (e.g., street suffixes, common address tokens) occurring across millions of records.
- **Inverted Indices**:
  - Exact Normalized Name Index
  - Compact (Whitespace-Stripped) Name Index
  - Building Number + PIN/ZIP prefix Index

### 3. Feature Engineering & Gradient Boosting
- **25 Vectorized Features**: RapidFuzz fuzzy ratios (Ratio, Partial Ratio, Token Set, Token Sort, WRatio), token Jaccard similarities, first-word matching, numeric address overlap, length ratios, and source provenance indicators.
- **LightGBM Classifier**: Trained with balanced scale-pos-weights, early stopping, and fine grid search optimization for the high-precision $F_{0.5}$ metric.

### 4. Distributed Multi-Instance Parallel Execution
- Implements country-isolated checkpointing (`results_{country}.json` and `cands_{country}.json`).
- Enables parallel multi-GPU execution across Kaggle instances with deterministic order-preserving assembly.

---

## File Structure
- `kaggle_solution.py`: Full end-to-end training and multi-country test inference pipeline.
- `kaggle_india_only.py`: High-throughput single-matrix pipeline specialized for large-scale Indian entity resolution.
- `merge_submissions.py`: Master order-preserving stitcher that validates and outputs official `matching_results.tsv` and `candidate_pairs.tsv`.
- `kaggle_gpu_test.ipynb`: Interactive Jupyter notebook ready for Kaggle GPU environments.
- `6ab10eb3b23ba_student_resource/`: Official starter code, evaluation utilities, and benchmark validator.

---

## License
MIT License
