"""
A-MEM Reproducibility Experiment: Retrieval Quality Evaluation

This script reproduces the retrieval component of A-MEM (Agentic Memory) and tests
a modification: hybrid retrieval (BM25 + embedding) vs. the original pure-embedding
retrieval.

Reproduction target: Recall@k on LoCoMo dataset - does the retriever surface the
evidence turns needed to answer each question?

Modification: Replace SimpleEmbeddingRetriever with a hybrid BM25 + embedding retriever.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add the A-MEM repo to path
AMEM_DIR = Path(__file__).parent / "A-mem-main"
sys.path.insert(0, str(AMEM_DIR))

from load_dataset import load_locomo_dataset, LoCoMoSample

# ---------------------------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------------------------

class EmbeddingRetriever:
    """Original A-MEM retriever: pure embedding (cosine similarity)."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None

    def build(self, documents: List[str]):
        self.corpus = documents
        self.embeddings = self.model.encode(documents, show_progress_bar=False)

    def search(self, query: str, k: int = 10) -> List[int]:
        if not self.corpus:
            return []
        query_emb = self.model.encode([query], show_progress_bar=False)[0]
        # cosine similarity
        sims = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        top_k = np.argsort(sims)[-k:][::-1]
        return top_k.tolist()


class HybridRetriever:
    """Modified retriever: BM25 + embedding hybrid (alpha blend)."""

    def __init__(self, model_name='all-MiniLM-L6-v2', alpha: float = 0.5):
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi
        self.model = SentenceTransformer(model_name)
        self.BM25Okapi = BM25Okapi
        self.alpha = alpha  # weight for BM25 (1-alpha for embedding)
        self.corpus = []
        self.embeddings = None
        self.bm25 = None

    def build(self, documents: List[str]):
        self.corpus = documents
        # Embedding index
        self.embeddings = self.model.encode(documents, show_progress_bar=False)
        # BM25 index
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = self.BM25Okapi(tokenized)

    def search(self, query: str, k: int = 10) -> List[int]:
        if not self.corpus:
            return []
        # BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        if len(bm25_scores) > 0 and bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        # Embedding scores
        query_emb = self.model.encode([query], show_progress_bar=False)[0]
        emb_sims = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        # Normalize embedding scores to [0,1]
        if emb_sims.max() > emb_sims.min():
            emb_sims = (emb_sims - emb_sims.min()) / (emb_sims.max() - emb_sims.min())
        # Hybrid score
        hybrid = self.alpha * bm25_scores + (1 - self.alpha) * emb_sims
        top_k = np.argsort(hybrid)[-k:][::-1]
        return top_k.tolist()


class BM25Retriever:
    """Baseline: pure BM25 (keyword matching)."""

    def __init__(self):
        from rank_bm25 import BM25Okapi
        self.BM25Okapi = BM25Okapi
        self.corpus = []
        self.bm25 = None

    def build(self, documents: List[str]):
        self.corpus = documents
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = self.BM25Okapi(tokenized)

    def search(self, query: str, k: int = 10) -> List[int]:
        if not self.corpus:
            return []
        scores = np.array(self.bm25.get_scores(query.lower().split()))
        top_k = np.argsort(scores)[-k:][::-1]
        return top_k.tolist()


# ---------------------------------------------------------------------------
# Experiment logic
# ---------------------------------------------------------------------------

def build_memory_corpus(sample: LoCoMoSample) -> Tuple[List[str], Dict[str, int]]:
    """Build a memory corpus from a conversation: one note per turn.

    Returns:
        corpus: list of note texts
        dia_id_to_idx: mapping from dia_id (turn id) to corpus index
    """
    corpus = []
    dia_id_to_idx = {}
    for session_id, session in sample.conversation.sessions.items():
        for turn in session.turns:
            text = f"Speaker {turn.speaker} says: {turn.text}"
            dia_id_to_idx[turn.dia_id] = len(corpus)
            corpus.append(text)
    return corpus, dia_id_to_idx


def get_evidence_indices(sample: LoCoMoSample, qa, dia_id_to_idx: Dict[str, int]) -> List[int]:
    """Convert evidence list (dia_ids like 'D1:3') to corpus indices."""
    evidence_indices = []
    for ev in qa.evidence:
        # evidence format matches dia_id exactly (e.g. 'D1:3')
        if ev in dia_id_to_idx:
            idx = dia_id_to_idx[ev]
            if idx not in evidence_indices:
                evidence_indices.append(idx)
    return evidence_indices


def evaluate_recall(retriever, samples: List[LoCoMoSample], k_values: List[int]) -> Dict:
    """Evaluate recall@k for a retriever across all samples and questions."""
    results = defaultdict(lambda: defaultdict(list))  # category -> k -> recall list

    for sample in tqdm(samples, desc="Samples"):
        # Build corpus for this conversation
        corpus, dia_id_to_idx = build_memory_corpus(sample)
        if len(corpus) == 0:
            continue

        # Build retriever index
        retriever.build(corpus)

        # Evaluate each question
        for qa in sample.qa:
            if qa.category is None or qa.category not in [1, 2, 3, 4, 5]:
                continue
            if not qa.evidence:
                continue

            evidence_indices = get_evidence_indices(sample, qa, dia_id_to_idx)
            if not evidence_indices:
                continue

            query = qa.question
            for k in k_values:
                retrieved = retriever.search(query, k=min(k, len(corpus)))
                # Recall: fraction of evidence that appears in top-k
                hits = sum(1 for ev_idx in evidence_indices if ev_idx in retrieved)
                recall = hits / len(evidence_indices)
                results[qa.category][k].append(recall)
                results["overall"][k].append(recall)

    # Aggregate: mean recall per category and per k
    aggregated = {}
    for category, k_dict in results.items():
        aggregated[category] = {}
        for k, recalls in k_dict.items():
            aggregated[category][f"recall@{k}"] = {
                "mean": float(np.mean(recalls)),
                "std": float(np.std(recalls)) if len(recalls) > 1 else 0.0,
                "count": len(recalls),
            }
    return aggregated


def main():
    print("=" * 60)
    print("A-MEM Reproducibility Experiment: Retrieval Quality")
    print("=" * 60)

    # Load dataset
    dataset_path = AMEM_DIR / "data" / "locomo10.json"
    print(f"\nLoading dataset from {dataset_path}...")
    samples = load_locomo_dataset(dataset_path)
    print(f"Loaded {len(samples)} conversations")

    # Count questions per category
    cat_counts = defaultdict(int)
    total_q = 0
    for s in samples:
        for qa in s.qa:
            if qa.category in [1,2,3,4,5]:
                cat_counts[qa.category] += 1
                total_q += 1
    print(f"Total questions: {total_q}")
    for c in sorted(cat_counts):
        print(f"  Category {c}: {cat_counts[c]}")

    k_values = [5, 10, 20]
    all_results = {}

    # --- Baseline: Pure Embedding (original A-MEM) ---
    print("\n" + "=" * 60)
    print("BASELINE: Pure Embedding Retrieval (original A-MEM)")
    print("=" * 60)
    emb_retriever = EmbeddingRetriever()
    emb_results = evaluate_recall(emb_retriever, samples, k_values)
    all_results["pure_embedding"] = emb_results

    # --- Modification: Hybrid BM25 + Embedding ---
    print("\n" + "=" * 60)
    print("MODIFICATION: Hybrid BM25 + Embedding (alpha=0.5)")
    print("=" * 60)
    hybrid_retriever = HybridRetriever(alpha=0.5)
    hybrid_results = evaluate_recall(hybrid_retriever, samples, k_values)
    all_results["hybrid_bm25_emb"] = hybrid_results

    # --- Additional: Pure BM25 ---
    print("\n" + "=" * 60)
    print("ADDITIONAL: Pure BM25")
    print("=" * 60)
    bm25_retriever = BM25Retriever()
    bm25_results = evaluate_recall(bm25_retriever, samples, k_values)
    all_results["pure_bm25"] = bm25_results

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: Overall Recall@k")
    print("=" * 60)
    print(f"{'Method':<25} {'Recall@5':>10} {'Recall@10':>10} {'Recall@20':>10}")
    print("-" * 60)
    for method, res in all_results.items():
        row = f"{method:<25}"
        for k in k_values:
            key = f"recall@{k}"
            if key in res.get("overall", {}):
                row += f" {res['overall'][key]['mean']:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # Per-category breakdown for key methods
    print("\n" + "=" * 60)
    print("PER-CATEGORY: Pure Embedding vs Hybrid")
    print("=" * 60)
    for cat in [1, 2, 3, 4, 5]:
        cat_str = str(cat)
        print(f"\nCategory {cat}:")
        print(f"  {'Method':<20} {'R@5':>8} {'R@10':>8} {'R@20':>8}")
        for method in ["pure_embedding", "hybrid_bm25_emb", "pure_bm25"]:
            res = all_results[method]
            if cat_str in res:
                row = f"  {method:<20}"
                for k in k_values:
                    key = f"recall@{k}"
                    if key in res[cat_str]:
                        row += f" {res[cat_str][key]['mean']:>8.4f}"
                    else:
                        row += f" {'N/A':>8}"
                print(row)

    # Save results
    output_path = Path(__file__).parent / "reproduction_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
