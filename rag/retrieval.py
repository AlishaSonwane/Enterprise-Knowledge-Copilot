"""Retrieval: query understanding, search, and reranking."""

import re
from dataclasses import dataclass

import numpy as np

import config
from rag.embeddings import embed_query
from rag.indexing import hybrid_search, vector_search


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    chunks: list[dict]
    query_original: str
    query_rewritten: str
    retrieval_scores: list[float]
    top_score: float
    is_confident: bool


# Pre-compiled regex patterns for query expansion
_EXPANSIONS = [
    (re.compile(r"\bAPI\b"), "API Application Programming Interface"),
    (re.compile(r"\bML\b"), "ML Machine Learning"),
    (re.compile(r"\bNLP\b"), "NLP Natural Language Processing"),
    (re.compile(r"\bRAG\b"), "RAG Retrieval Augmented Generation"),
    (re.compile(r"\bDB\b"), "DB Database"),
    (re.compile(r"\bUI\b"), "UI User Interface"),
    (re.compile(r"\bLLM\b"), "LLM Large Language Model"),
]


def rewrite_query(query: str) -> str:
    """Simple query rewriting: clean up and expand the query.

    In production, you'd use an LLM for this. Here we do rule-based cleanup.
    """
    # Remove filler words for search
    query = query.strip().rstrip("?").strip()

    expanded = query
    for pattern, replacement in _EXPANSIONS:
        expanded = pattern.sub(replacement, expanded)

    return expanded


def evidence_cosine(query_embedding, document_embedding):
    """Return cosine evidence, or None for invalid vectors. No rank fallback."""
    try:
        query = np.asarray(query_embedding, dtype=np.float64)
        document = np.asarray(document_embedding, dtype=np.float64)
        if (query.ndim != 1 or document.shape != query.shape or query.size == 0
                or not np.isfinite(query).all() or not np.isfinite(document).all()):
            return None
        denominator = np.linalg.norm(query) * np.linalg.norm(document)
        if not np.isfinite(denominator) or denominator <= 0:
            return None
        score = float(np.dot(query, document) / denominator)
        return float(np.clip(score, -1.0, 1.0)) if np.isfinite(score) else None
    except (TypeError, ValueError, OverflowError):
        return None


def rerank_results(query_embedding: np.ndarray, results: list[dict],
                   top_k: int = None) -> list[dict]:
    """Rerank results using embedding similarity.

    Uses pre-fetched embeddings from search results to avoid redundant
    embedding computations.
    """
    top_k = top_k or config.TOP_K_RERANK

    if not results:
        return []

    # Use validated cosine evidence; ranking remains RRF + cosine.
    validated = []
    for result in results:
        score = evidence_cosine(query_embedding, result.get("embedding"))
        if score is not None:
            result["evidence_score"] = score
            validated.append(result)
    results = validated
    similarities = [r["evidence_score"] for r in results]

    # Combine original score with re-ranking score
    for i, result in enumerate(results):
        original_score = result.get("score", 0)
        rerank_score = float(similarities[i])
        # Weighted combination: favor rerank score
        result["rerank_score"] = rerank_score
        result["combined_score"] = 0.4 * original_score + 0.6 * rerank_score

    # Sort by combined score
    results.sort(key=lambda x: x["combined_score"], reverse=True)

    return results[:top_k]


def retrieve(query: str, top_k: int = None,
             use_reranking: bool = True) -> RetrievalResult:
    """Full retrieval pipeline: rewrite -> search -> rerank."""
    top_k_final = top_k or config.TOP_K_RERANK

    # Step 1: Query understanding / rewriting
    rewritten = rewrite_query(query)

    # Step 2: Embed and search
    query_embedding = embed_query(rewritten)
    candidates = hybrid_search(query_embedding, rewritten, top_k=config.TOP_K_RETRIEVAL)
    valid_candidates = []
    for candidate in candidates:
        evidence = evidence_cosine(query_embedding, candidate.get("embedding"))
        if evidence is not None:
            candidate = dict(candidate)
            candidate["evidence_score"] = evidence
            valid_candidates.append(candidate)
    candidates = valid_candidates

    if not candidates:
        return RetrievalResult(
            chunks=[],
            query_original=query,
            query_rewritten=rewritten,
            retrieval_scores=[],
            top_score=0.0,
            is_confident=False,
        )

    # Step 3: Reranking (pass query_embedding to avoid re-embedding)
    if use_reranking and len(candidates) > top_k_final:
        ranked = rerank_results(query_embedding, candidates, top_k=top_k_final)
    else:
        ranked = candidates[:top_k_final]

    # Evidence scale is identical with and without reranking; never use RRF.
    scores = [r["evidence_score"] for r in ranked]
    top_score = max(scores) if scores else 0.0
    is_confident = top_score >= config.RETRIEVAL_SCORE_THRESHOLD

    return RetrievalResult(
        chunks=ranked,
        query_original=query,
        query_rewritten=rewritten,
        retrieval_scores=scores,
        top_score=top_score,
        is_confident=is_confident,
    )
