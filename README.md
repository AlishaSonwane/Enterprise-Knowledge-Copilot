# Production RAG

> Retrieval-Augmented Generation over your private documents — hybrid search, reranking, and traceable, cited answers.

## Overview

A Retrieval-Augmented Generation system for answering natural-language questions over private documents. It combines hybrid BM25 + vector retrieval with embedding-similarity reranking, and returns answers with inline citations and full traceability back to source chunks. Runs locally (FastAPI + Ollama) with hallucination guardrails, query logging, and a monitoring dashboard.

## Features

- Hybrid retrieval: BM25 lexical search + dense vector search fused via Reciprocal Rank Fusion
- Reranking stage to lift the most relevant chunks before generation
- Inline citations (`[Source: filename]`) with full answer-to-source traceability
- Multi-format ingestion: PDF, Markdown, DOCX, HTML, plain text
- Incremental re-indexing on document upload
- Safety guardrails: prompt-injection detection and confidence-based abstention
- Built-in evaluation suite and a live monitoring dashboard

## Tech Stack

- **Language:** Python
- **API / Server:** FastAPI, Uvicorn
- **Vector store:** ChromaDB (HNSW)
- **Lexical search:** BM25 (`rank-bm25`)
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM:** Ollama (llama3.2)

---

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure Ollama is running with llama3.2
ollama pull llama3.2
ollama serve  # if not already running

# 3. Launch the application
python app.py

# 4. Open http://localhost:8000
```

The app ships with 3 sample documents in `documents/`. Upload more via the UI or drop files into the `documents/` folder and click "Re-index All Documents".

---

## ML System Design

### Scenario

Design a RAG system that answers user questions using a private corpus (internal docs, PDFs, knowledge base articles).

**Requirements:**
- Natural-language Q&A over private documents
- Handle frequent document updates (new/changed docs)
- Provide citations and traceability to sources
- Low latency for interactive use (target p95 < 2s)
- Reduce hallucinations; answers grounded in retrieved context

### Scope & Assumptions

| Parameter | Value |
|---|---|
| Corpus scale | 10^5 - 10^7 chunks |
| Formats | PDF, Markdown, DOCX, HTML, plain text |
| Update frequency | Near-real-time (upload triggers re-indexing) |
| Query types | Fact lookup, multi-section synthesis, summarization |
| Latency budget | p95 < 2s retrieval, < 10s end-to-end with generation |
| Output | Natural language answer + source citations |

---

## End-to-End Architecture

```
                        OFFLINE PIPELINE
  ┌──────────┐    ┌────────┐    ┌─────────┐    ┌──────────┐    ┌───────────┐
  │ Document  │───>│ Parser │───>│ Chunker │───>│ Embedder │───>│ Vector DB │
  │ Sources   │    │        │    │         │    │          │    │ + BM25    │
  └──────────┘    └────────┘    └─────────┘    └──────────┘    └───────────┘
  PDF/MD/DOCX/     Format-       Semantic +     MiniLM-L6       ChromaDB
  HTML/TXT         specific      overlap        384-dim         HNSW index

                        ONLINE PIPELINE
  ┌───────┐    ┌─────────┐    ┌────────┐    ┌────────┐    ┌─────────┐    ┌────────┐
  │ Query │───>│ Rewrite │───>│ Hybrid │───>│Reranker│───>│ Context │───>│Ollama  │───> Answer
  │       │    │ +Expand │    │ Search │    │        │    │Assembly │    │LLM     │    +Citations
  └───────┘    └─────────┘    └────────┘    └────────┘    └─────────┘    └────────┘
  User input    Abbreviation   Vector+BM25   Embedding     Dedupe +      llama3.2
                expansion      RRF fusion    re-score      compress      T=0.1
```

### Data Flow

1. **User uploads document** -> Parser extracts text -> Chunker splits into segments -> Embedder creates vectors -> Stored in ChromaDB + BM25 index
2. **User asks question** -> Query rewritten/expanded -> Hybrid search (vector + BM25 via RRF) -> Top-20 candidates reranked to top-5 -> Context assembled with citations -> Ollama generates grounded answer -> Safety checks applied -> Answer with sources returned

---

## Component Details

### 1. Document Ingestion & Preprocessing (`rag/ingestion.py`)

**Parsing:**
- Format-specific parsers for PDF (PyPDF2), DOCX (python-docx), HTML (BeautifulSoup), Markdown, plain text
- Structure preservation: headings, page numbers, section boundaries
- Text cleaning: whitespace normalization, control character removal, artifact cleanup

**Chunking strategy:**
- **Semantic-first**: Split by markdown headings/sections to preserve topical coherence
- **Sentence-boundary splitting**: Large sections split at sentence boundaries (not mid-word)
- **Size**: ~300 tokens per chunk with ~50 token overlap (configurable)
- **Minimum filter**: Chunks below 50 tokens are discarded
- **Metadata**: Each chunk tagged with `source`, `filepath`, `file_type`, `chunk_index`, `total_chunks`, `modified`, `word_count`

**Pitfalls addressed:**
- Too-small chunks lose context; too-large chunks reduce retrieval precision
- PDF extraction noise handled by cleaning pipeline
- Duplicate/boilerplate text handled by deduplication in context assembly

### 2. Embedding Strategy (`rag/embeddings.py`)

**Model:** `all-MiniLM-L6-v2` (384 dimensions)
- Good balance of speed and quality
- Normalized embeddings for cosine similarity
- Batch encoding for ingestion efficiency
- Lazy model loading to reduce startup time

**Design decisions:**
- L2-normalized embeddings so cosine similarity = dot product
- Single model for both documents and queries (symmetric embedding)
- Batch size of 64 for throughput during ingestion

### 3. Indexing (`rag/indexing.py`)

**Vector index:** ChromaDB with HNSW (cosine space)
- Persistent storage on disk
- Upsert support for incremental updates
- Metadata filtering capability

**Lexical index:** BM25Okapi
- Built from all indexed chunks
- Rebuilt on each indexing operation
- Handles exact keyword matching that vector search may miss

**Hybrid search:** Reciprocal Rank Fusion (RRF)
- `RRF_score(d) = alpha * 1/(k + rank_vector(d)) + (1-alpha) * 1/(k + rank_bm25(d))`
- Default alpha = 0.7 (favors vector search)
- k = 60 (standard RRF constant)

### 4. Retrieval & Reranking (`rag/retrieval.py`)

**Query understanding:**
- Abbreviation expansion (API, ML, NLP, RAG, DB, UI, LLM)
- Question mark removal for search optimization
- (Production: LLM-based query rewriting for multi-query retrieval)

**Retrieval:**
- Top-20 hybrid search candidates
- Configurable alpha between vector and BM25 weights

**Reranking:**
- Reuse the query and candidate embeddings from retrieval (no re-embedding)
- Compute direct embedding similarity as rerank score
- Combined score: `0.4 * original_score + 0.6 * rerank_score`
- Select top-5 after reranking

### 5. Generation & Grounding (`rag/generation.py`)

**Context assembly:**
- Deduplicate near-identical chunks (token-set overlap > 0.85)
- Format with document number, source filename, section, relevance score
- Separated by `---` dividers for clarity

**System prompt enforces:**
- Answer ONLY from provided context
- Cite sources using `[Source: filename]` format
- Say "I don't know" when information is missing
- Note conflicting sources
- Be concise but thorough

**LLM configuration:**
- Ollama with llama3.2 (configurable via `OLLAMA_MODEL` env var)
- Temperature: 0.1 (low for factual consistency)
- Max prediction tokens: 1024

### 6. Safety & Guardrails (`rag/guardrails.py`)

**Prompt injection detection:**
- Pattern matching for common injection attempts ("ignore previous instructions", "pretend you are", etc.)
- Query length limit (5000 chars)

**Retrieval confidence guardrails:**
- Abstention threshold: if best retrieval score < 0.10, refuse to answer
- No-results handling: explicit "no relevant documents found" message

**Fallback behavior:**
- Graceful error messages for Ollama connection failures
- Descriptive abstention messages guiding users to rephrase or upload relevant docs

---

## Evaluation Plan

### A. Ingestion/Chunking Quality

| Metric | Method | Target |
|---|---|---|
| Chunk coherence | Manual audit (sample 50 chunks) | >90% rated "coherent" |
| Duplication rate | Automated near-duplicate detection | <5% duplicate chunks |
| Avg chunk size | Automated measurement | 200-400 tokens |
| Parser error rate | Count failures per format | <1% failure rate |
| Metadata accuracy | Spot-check source/timestamp | 100% accurate |

### B. Retrieval Quality

| Metric | Method | Target |
|---|---|---|
| Recall@5 | Labeled (query, relevant_chunk) pairs | >80% |
| Recall@20 | Same labeled dataset | >95% |
| MRR | Rank of first relevant result | >0.7 |
| nDCG@5 | Graded relevance labels | >0.75 |
| Latency (retrieval only) | Timer on hybrid search | p95 < 200ms |

**Creating labels:** Generate synthetic Q/A pairs from documents using an LLM, with human spot-checks.

### C. Reranking Quality

| Metric | Method | Target |
|---|---|---|
| nDCG@5 improvement | Compare before/after reranking | >10% relative improvement |
| Precision@1 improvement | Same | >15% relative improvement |
| Reranking latency | Timer | <100ms |

### D. Generation Quality

| Metric | Method | Target |
|---|---|---|
| Answer correctness | Human rating (1-5 scale) or LLM-as-judge | >4.0 avg |
| Faithfulness/grounding | Check each claim against context | >95% claims grounded |
| Citation accuracy | Verify cited sources contain the claimed info | >90% citations valid |
| Readability | Human rating | >4.0 avg |
| Hallucination rate | Claims not in any retrieved context | <5% |

### E. End-to-End Metrics

| Metric | Method | Target |
|---|---|---|
| Task success rate | User can get correct answer | >80% |
| User satisfaction | Thumbs up/down | >75% positive |
| End-to-end latency | Timer | p95 < 10s |
| Abstention rate | Count refusals / total queries | 10-20% (balanced) |
| Cost per query | Track compute usage | Within budget |

---

## Online Monitoring & Continuous Improvement

### Logging (built-in)

Every query logs:
- Original query, rewritten query
- Retrieved chunk IDs and scores
- Reranking scores
- Final answer text and length
- Citation count
- Latency (ms)
- Status (success/error/blocked)
- User feedback (thumbs up/down)

### Live Metrics Dashboard

The **Monitoring** tab shows:
- Total indexed chunks
- Total queries served
- Average latency, retrieval score, chunks used, citations
- Abstention rate
- Positive feedback rate
- Recent query logs table

### Continuous Improvement Loop

1. **Collect failure cases** from user feedback (thumbs down) and low-confidence queries
2. **Add to evaluation set** for regression testing
3. **Tune parameters:**
   - Chunk size and overlap
   - Top-k retrieval and reranking counts
   - Hybrid alpha (vector vs BM25 weight)
   - Retrieval confidence thresholds
4. **Expand domain coverage:**
   - Add synonym/abbreviation expansions
   - Upload additional documents for coverage gaps
   - Consider domain-adapted embedding models
5. **Monitor for drift:**
   - Embedding distribution shifts over time
   - Retrieval score distributions
   - Document freshness and staleness

---

## Project Structure

```
RAG/
├── app.py                  # FastAPI application (API + static files)
├── config.py               # All configuration parameters
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── rag/
│   ├── __init__.py
│   ├── ingestion.py        # Document parsing, cleaning, chunking
│   ├── embeddings.py       # Embedding model management
│   ├── indexing.py         # ChromaDB vector store + BM25 index
│   ├── retrieval.py        # Query rewriting, hybrid search, reranking
│   ├── generation.py       # Context assembly + Ollama LLM generation
│   ├── guardrails.py       # Safety checks + fallback behavior
│   └── evaluation.py       # Query logging + metrics computation
├── static/
│   ├── index.html          # Web UI
│   ├── style.css           # Styling
│   └── app.js              # Frontend logic
├── documents/              # Document corpus (upload files here)
│   ├── rag_system_guide.md
│   ├── machine_learning_basics.md
│   └── vector_databases.md
└── data/                   # Persistent storage (ChromaDB, logs)
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the web UI |
| `POST` | `/api/query` | Ask a question `{query, top_k, use_reranking}` |
| `POST` | `/api/upload` | Upload a document (multipart form) |
| `POST` | `/api/ingest` | Re-index all documents in `documents/` |
| `GET` | `/api/stats` | System stats, metrics, document list, config |
| `GET` | `/api/logs?n=20` | Recent query logs |
| `POST` | `/api/feedback` | Submit feedback `{query_index, feedback}` |
| `POST` | `/api/clear` | Clear all indexed data |
| `DELETE` | `/api/documents/{filename}` | Delete a document and re-index |

## Configuration

All parameters in `config.py` or via environment variables:

| Parameter | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model for generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | 300 | Target tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Token overlap between chunks |
| `TOP_K_RETRIEVAL` | 20 | Initial retrieval candidates |
| `TOP_K_RERANK` | 5 | Final results after reranking |
| `HYBRID_ALPHA` | 0.7 | Vector weight (1-alpha for BM25) |
| `TEMPERATURE` | 0.1 | LLM generation temperature |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.30 | Provisional cosine evidence threshold |
| `ABSTENTION_THRESHOLD` | 0.30 | Provisional cosine threshold below which system abstains |

## Common Edge Cases Handled

- **Conflicting sources**: System prompt instructs LLM to present multiple perspectives
- **No relevant documents**: Graceful abstention with helpful message
- **Prompt injection**: Pattern-based detection and blocking
- **Ollama unavailable**: Clear error message with instructions
- **Unsupported file types**: Explicit error listing supported formats
- **Empty chunks**: Filtered out during ingestion
- **Duplicate chunks**: Deduplicated during context assembly


## Evidence scores and bounded local generation

Hybrid vector/BM25 search, RRF ordering and optional reranking remain in place.
RRF `score` and `combined_score` are ranking signals. Confidence fields
(`retrieval_scores`, `top_score`, API `confidence`/`retrieval_score`, source
`score` and context relevance) now use query-document cosine similarity from
validated embeddings, independent of whether reranking runs. These values are
not probabilities or proof that a passage answers a question. Historical scores
and logs are not directly comparable. Invalid/missing embeddings are excluded;
empty results and below-threshold evidence abstain without calling Ollama.
Unvalidated evidence also fails closed. The shared 0.30 threshold is PROVISIONAL,
not calibrated: relevant and no-evidence examples need broader validation before
any production use. Semantically similar but unsupported questions can still
pass; the context-only generation prompt retains its abstention instruction.

Environment variables are read from the launching process; `.env.example` is a
template, not an automatically loaded file. `OLLAMA_NUM_PREDICT` defaults to 1024;
`OLLAMA_NUM_CTX` defaults to unset (omitted from the API payload, preserving the
Ollama default). Explicit values must be positive decimal integers. Blank,
negative, zero, fractional and unlimited values fail startup with a clear error.

Suggested low-memory trial settings (not a memory-safety guarantee):
`OLLAMA_MODEL=llama3.2:1b`, `OLLAMA_NUM_PREDICT=128`, `OLLAMA_NUM_CTX=2048`.
Set them only for the application process. Recheck RAM and use bounded requests.
Smaller context reduces KV-cache memory; shorter output can reduce latency but
can truncate answers/citations. Model weights and embeddings still require RAM.
`MAX_CONTEXT_TOKENS` is currently unused: context assembly does not enforce a
token budget. Do not assume all evidence/instructions fit a small context window.
Citation extraction preserves the existing format but does not validate invented
citations or factual correctness. Context budgeting and citation validation are
separate follow-ups, not implemented here. No index migration is required.

Offline regression tests (standard library unittest; existing dependencies):
`python -B -m unittest discover -s tests -v`. Tests stub embedding/search modules
and mock Ollama; they do not load models, connect to services or touch ChromaDB.


## Company-specific subscription pricing restriction

The offline evidence-support gate is narrowly scoped to company-specific
subscription/service pricing requests. It runs after query safety and before
cosine confidence. Such requests return `blocked` with
`insufficient_evidence_support`, so Ollama is never called for them.
The current corpus and configuration contain no independently verified company
identity or authoritative pricing records. Therefore ALL detected scoped pricing
requests fail closed, even if retrieved text contains an amount, currency,
billing period, company name or a self-declared `trusted` metadata flag.
No identities, prices or trusted metadata were added to the real corpus.

Scope detection uses conservative language patterns only to decide when to
restrict a request; keyword overlap is never treated as proof of factual support.
This is not an extractor, verifier of arbitrary facts, or general entailment
model. Unusual phrasing may evade detection; conceptual pricing discussion may
be conservatively blocked. Before enabling any positive pricing response, a
separate reviewed mechanism must verify company/service identity, provenance,
amount, currency, billing period and the actual supporting source passage.
There is deliberately no public API or metadata switch to mark a source trusted.
The 0.30 provisional cosine threshold, retrieval ranking, source attribution,
citations and existing index remain unchanged. General RAG/ML/HNSW behavior is
covered by offline tests; wider real-index validation remains separate.


## Optional cloud generation on low-memory Windows

Set `GENERATION_PROVIDER=groq`, `GROQ_FREE_PLAN_CONFIRMED=1` and
`GROQ_API_KEY` in the application subprocess environment. `.env` is not
loaded automatically. Do not paste credentials into chat or commit them.
This uses the existing httpx dependency; no new SDK is required.
The fixed model is `openai/gpt-oss-20b`; `GROQ_MAX_COMPLETION_TOKENS`
defaults to 1024 and is limited to positive integers up to 2048.

Embeddings, retrieval and guardrails remain local. The question, grounding
instructions, selected retrieved passages and source filenames are transmitted
to Groq over HTTPS. Start with public sample documents; review provider data
policies before using confidential documents. Pricing questions still fail
closed without independently trusted pricing evidence.

Free-only operation depends on keeping the organization on Groq's Free plan.
The confirmation variable is an operator acknowledgement, not an API billing
tier verification. Upgrading that account can permit billing; this application
cannot prevent account changes. Rate limits produce an explicit error, with no
retry, local model fallback or alternate paid provider. Missing keys, bad
credentials, timeouts, incomplete answers and invalid citations also fail closed.
Correct source names alone do not prove an answer is factually supported.

Original local Ollama behavior remains the default. Cloud integration has offline
tests; real /api/query answer quality must be validated separately with a key.

Private GitHub publishing must exclude secrets, uploaded documents, runtime
logs, databases, environment directories and model files. Keep LICENSE and
original upstream attribution. Private visibility does not replace secret hygiene.


### Verified status of this Windows/cloud checkpoint

65 offline tests passed. On 2026-10-10 a real loopback FastAPI `/api/query`
test passed using the optional ONNX backend, a copy of the existing seven-chunk
index and one Groq request. Startup, OpenAPI and query HTTP statuses passed;
fresh retrieval top cosine was 0.36298578 and endpoint latency was 1558.7 ms.
Structured answer citation/output checks passed, and all three claims cited
`rag_system_guide.md`. The definition and retrieval mechanism match the source;
the hallucination-reduction explanation is broadly supported, though "ensuring"
is stronger wording than ideal and must not be read as a correctness guarantee.
This is one question's integration evidence, not general factuality validation.
Initial available RAM was 1.5766 GiB; minimum sampled RAM was 1.3429 GiB
(one-second samples, not an exact instantaneous peak), maximum sampled load 77%.
Server shutdown and port release passed, with no original project file changes.
The independent 1.5 GiB startup gate and 0.5 GiB severe-pressure stop remain.
No original indexed data, models or secrets are included in this repository.

Additional real endpoint checks (2026-10-10): supervised versus unsupervised
learning returned source-supported definitions/examples with citations to
`machine_learning_basics.md` in 1298.3 ms (top cosine 0.56456). HNSW returned
graph structure, speed/recall and memory tradeoffs with `vector_databases.md`
citations in 1106.6 ms (top cosine 0.31344); its recall statement omitted the
source's "typically" qualifier, so treat this as a remaining wording limitation.
The unsupported company-private pricing question scored 0.37944 but correctly
returned `blocked / insufficient_evidence_support`, with zero cloud calls.
All three endpoint expectations passed; only two cloud calls occurred. Server
shutdown, port release and original-file integrity checks passed. These sample
checks do not calibrate the provisional 0.30 threshold or validate all factual
questions.

Run offline tests: `python -I -B tests/run_offline.py` using the project's compatible
Python 3.11 environment and installed dependencies. Native model imports are
blocked and no live services are required.

Windows prerequisites include compatible x64 native runtime dependencies for
ONNX Runtime. The earlier process-local DLL cache workaround was specific to the
original workspace and is not a portable installation requirement. Do not copy
arbitrary DLLs or Windows virtual environments to other machines.
# Citation quality limitation

UI scores now show cosine evidence as decimals (for example 0.363), not
"Confidence 36.3%". The API's legacy `confidence` field remains compatible.
Blocked responses can have high topical similarity without answer support.
Monitoring and architecture show the configured generation provider/backend;
the cloud notice explains that questions and retrieved passages go to Groq.
Metadata is escaped before HTML rendering and HTTP errors are not shown as
generated answers. Private documents require permission to share their content.

Frontend offline checks: `node tests/frontend_checks.cjs` (no extra libraries or
live requests). The Python suite also checks public provider metadata without
exposing keys or loading native infrastructure.

### Demo completion and remaining limitations

Final UI/configuration checks: 71 Python offline tests and 12 JavaScript checks
passed. Core live RAG, ML/HNSW answers, pricing abstention and launcher startup
were verified as documented above. Restart the launcher after backend changes
and refresh the browser to load updated UI assets. No folders were removed or
migrated, and no indexed documents were changed by these finishing edits.

This is a functioning local demonstration, not a production certification.
The cosine threshold is provisional; pricing support checks are narrowly scoped,
and citation formatting/source membership are not factual entailment checks.
Free cloud quotas can produce rate-limit errors. PDF/DOCX upload edge cases,
large corpora, multi-user access and private-data compliance have not undergone
comprehensive validation. Keep the app bound to localhost, use approved
documents and review factual claims before relying on them.

Cloud transport requests a JSON `claims` array with one sentence and an exact source filename per claim. It rejects malformed structures and renders `[Source: filename]` itself. The application still checks source membership and output validation before returning success. No JSON API mode is assumed: a model that ignores the requested structure fails closed, without repair or retry. Rendered citations do not establish factual support.

Cloud output now fails closed on absent/unknown citations, uncited lines, multiple sentences sharing one trailing citation, and a narrow set of overstated guarantee phrases. Generation requests one sentence per line. These conservative formatting/phrase heuristics may reject abbreviations or valid negated statements; they do not determine entailment or catch all unsupported claims. Rejections return an error with zero confidence, suppress the rejected answer, and do not retry or fall back to Ollama.

Generation instructions request a source citation immediately after every factual sentence and avoid claims that grounding guarantees correctness. Cloud validation rejects absent citations and filenames outside retrieved sources. This filename check does not establish sentence-level evidence support; citation placement and factual support still require review. Prompt instructions are not an enforced factuality guarantee.
# Low-memory embedding backend

### Windows daily launcher

The prepared Windows shortcut is `Start-Copilot.cmd` in the project folder.
Double-click it, then open the `Ready:` localhost URL printed in its terminal.
Keep that terminal open while using the app; Ctrl+C stops the server. Launcher
offline tests bring the total to 70 passing tests. A bounded real launcher
smoke test passed homepage, OpenAPI and logs HTTP checks, clean shutdown, port
release and original database integrity, without cloud generation.

`run_windows.py` loads only the Groq settings from the private local `.env` and
uses a private `.launcher.json` profile containing local Python, ONNX model,
tokenizer and optional verified native-runtime directory paths. These local
files are ignored by Git. It never installs or downloads anything, prints API
keys, changes PATH, or starts Ollama generation. A per-machine
`Start-Copilot.cmd` starts the prepared interpreter without PowerShell policies.

The launcher checks at least 1.5 GiB available RAM before startup, binds only
127.0.0.1 on a free port, prints the browser URL when ready and monitors memory.
Use Ctrl+C in its terminal to stop cleanly. It stops at severe pressure below
0.5 GiB, after a 90-second startup timeout, or after one hour by default
(`--max-seconds` allows 1–14400 seconds). Logs remain in ignored `data/launcher/`.
`--smoke-test` checks homepage, OpenAPI and logs using a copied index, then exits;
it never calls cloud generation. Normal sessions use the original project
index, so uploads and indexing persist. The Free-plan acknowledgment is not
proof of billing tier; keep the Groq account on Free. Cloud queries transmit
retrieved text, so use public sample documents until data-sharing permission
has been established for private material.

Set `EMBEDDING_BACKEND=onnx` with explicit local `ONNX_MODEL_PATH` and `ONNX_TOKENIZER_PATH`. The default remains `torch`. ONNX uses already installed ONNX Runtime and tokenizers directly, avoiding Torch imports, CPU only, batches capped at eight. The pinned official model SHA-256 is checked before loading; no automatic download or fallback occurs. Matching MiniLM tokenization, 256-token truncation, masked mean pooling and L2 normalization preserve the embedding space. A copied seven-chunk Windows index comparison gave minimum cosine 0.99999994 and maximum absolute difference 1.49e-7. This is sample compatibility evidence, not a universal equivalence or memory guarantee. Model files and paths containing secrets must not be committed.
