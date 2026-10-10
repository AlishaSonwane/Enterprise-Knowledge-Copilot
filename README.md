# Enterprise Knowledge Copilot

A working Windows RAG demonstration: retrieve evidence from documents, generate
source-attributed answers through opt-in Groq cloud generation, and abstain when
configured evidence checks fail. Local CPU ONNX embeddings avoid loading Torch
on the tested low-memory machine. FastAPI serves a loopback-only browser UI.

## Start here

- [Windows setup and fresh-machine reproduction](docs/WINDOWS_SETUP.md)
- [Interview demo, measured results and contributions](docs/DEMO.md)
- [Detailed implementation history and upstream design notes](docs/IMPLEMENTATION_NOTES.md)

The repository includes application code, static UI, original dependency pins,
three public sample documents, tests and configuration templates. It intentionally
excludes API keys, personal uploads, model binaries, virtual environments, local
databases and machine-specific launch paths. Downloading the repo alone is not a
ready-to-run binary installation: follow the setup guide and rebuild the sample index.

## Architecture

Documents -> parsing/chunking -> MiniLM embeddings -> persistent ChromaDB + BM25
-> hybrid RRF retrieval -> optional reranking -> cosine evidence/guardrails
-> source context -> Groq or Ollama -> citation/output checks -> browser answer.

Ranking and evidence scores have different meanings. The cosine threshold 0.30
is provisional, not calibrated. Evidence is shown as a decimal, not a probability.
Company-specific subscription pricing requires trusted scoped support; missing
provenance fails closed. This narrow gate does not validate all factual questions.

## Verified demonstration

The existing Windows setup indexed three sample documents into seven chunks and
passed full FastAPI question-answer checks for RAG, ML and HNSW. Unsupported private
pricing abstained before generation. At the completion checkpoint, 71 Python offline
regressions and 12 frontend checks passed. See the demo guide for timing and caveats.
A fresh-machine installation has not been verified. This is a demonstration,
not production certification or a factuality guarantee.

## Privacy and cost

Groq is opt-in and receives the question and retrieved document passages. Use only
approved/public material. Keep the account on Free; rate limits can interrupt a
session, and the local Free-plan acknowledgment is not proof of account billing.
No automatic provider fallback or generation retry is used. Never upload `.env`
even to a private repository. Models and indexes are rebuilt locally.

## Attribution

Adapted from [sugeerth/production-rag](https://github.com/sugeerth/production-rag)
at commit `5025fe334e4b35ccd4a42f68a1f17f46378e66ee`. Original copyright belongs to
Sugeerth Murugesan; [MIT LICENSE](LICENSE) is retained. The demo guide distinguishes
upstream architecture from this adaptation's changes. Historical implementation
notes contain original design targets; they are not measured results of this project.