# FTEC5660 Reproducibility Assignment: A-MEM Retrieval Quality

This repository contains the reproducibility work for the A-MEM (Agentic Memory) system, reproduced on the LoCoMo long-conversation QA dataset.

## Project Overview

- **Paper:** A-MEM: Agentic Memory for LLM Agents (Xu et al., NeurIPS 2025)
- **Original repo:** https://github.com/WujiangXu/AgenticMemory
- **What we reproduce:** The retrieval layer's Recall@k on the LoCoMo 10-sample subset
- **Our modification:** Hybrid BM25 + embedding retrieval vs. the original pure-embedding baseline

## Repository Structure

```
.
├── A-mem-main/              # Original A-MEM evaluation code (unmodified)
│   ├── data/
│   │   └── locomo10.json    # LoCoMo 10-sample subset
│   ├── memory_layer.py      # Core memory system
│   ├── test_advanced_robust.py  # Robust evaluation script
│   └── ...
├── reproduce_experiment.py  # Our reproduction experiment script
├── report.md                # Reproducibility report (source)
└── README.md                # This file
```

## Setup Instructions

### 1. Create a virtual environment

```bash
cd E:\FTEC5660_Reproducibility
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install numpy scikit-learn rank-bm25 nltk rouge-score tqdm sentence-transformers
```

### 3. Run the reproduction experiment

```bash
python reproduce_experiment.py
```

This will:
- Load the LoCoMo 10-sample dataset
- Build memory corpora from each conversation
- Evaluate Recall@k for three retrieval methods:
  1. Pure embedding (baseline, original A-MEM)
  2. Hybrid BM25 + embedding (our modification)
  3. Pure BM25 (reference)
- Print results and save to `reproduction_results.json`

## What We Modified

**One isolated change:** The retrieval scoring function.

- **Baseline:** `SimpleEmbeddingRetriever` — pure cosine similarity on all-MiniLM-L6-v2 embeddings (as in the original A-MEM codebase)
- **Modified:** `HybridRetriever` — blend of BM25 keyword scores and embedding cosine similarity (alpha=0.5)

**Why:** Pure embedding retrieval can miss exact keyword matches (names, dates, specific terms) that are critical for factoid QA. BM25 excels at exact term matching but fails on semantic paraphrase. A hybrid combines both strengths.

**How it's measured:** Recall@k (5, 10, 20) on the LoCoMo QA pairs, broken down by question category.

## Results

See `report.md` for full results.

## Notes

- No LLM API key is required for this reproduction — the retrieval layer is deterministic.
- The first run will download the all-MiniLM-L6-v2 embedding model (~90 MB).
- Total runtime: ~5-10 minutes on a modern laptop.

## References

- Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., & Zhang, Y. (2025). A-Mem: Agentic memory for llm agents. Advances in Neural Information Processing Systems.
- LoCoMo dataset: https://github.com/snap-research/locomo
