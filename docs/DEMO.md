# Interview demonstration and verified evidence

## What to show

1. Explain hybrid BM25 + vector retrieval and RRF ranking.
2. Index the three included public upstream sample documents.
3. Ask the RAG definition/hallucination question and inspect cited source passages.
4. Ask about supervised versus unsupervised learning, then HNSW.
5. Ask for the company's private subscription price: show fail-closed abstention.
6. Explain why cosine evidence is not a calibrated probability and why citation
   membership alone does not prove a claim is supported.

## Observed results on the owner's Windows machine

- Three documents, seven indexed chunks.
- RAG full FastAPI query: HTTP 200, about 1.56 seconds, cosine evidence 0.363,
  three cited answer sentences referencing rag_system_guide.md.
- ML and HNSW questions returned source-attributed answers; unsupported private
  pricing was blocked with insufficient_evidence_support and zero cloud calls.
- ONNX embeddings on a copied seven-chunk index matched the prior embedding
  space: minimum cosine 0.99999994, maximum absolute difference 1.49e-7.
- 71 Python offline regressions and 12 frontend checks passed at the completion
  checkpoint. Launcher smoke checks confirmed HTTP readiness and cleanup.

These are small-sample observations, not p95 benchmarks or production guarantees.
Live answers can vary. Wording can overstate claims despite citations; manually
check evidence. The provisional 0.30 threshold is uncalibrated. Large-corpus,
multi-user, OCR, complex document layouts and clean-machine setup remain unverified.
Raw logs, local paths, keys and databases are deliberately excluded from Git.

## Attribution and scope of contributions

Based on [sugeerth/production-rag](https://github.com/sugeerth/production-rag),
inspected upstream commit `5025fe334e4b35ccd4a42f68a1f17f46378e66ee`, MIT licensed.
Preserve LICENSE and acknowledge the upstream author, Sugeerth Murugesan.

This adaptation adds consistent cosine evidence/abstention independent of reranking,
narrow pricing support checks, configurable Ollama limits, opt-in Groq generation,
structured claim/citation validation, pinned offline ONNX embeddings, a bounded
Windows launcher, clearer evidence/privacy UI, and focused regression tests.
Do not present the upstream retrieval architecture as entirely original work.
