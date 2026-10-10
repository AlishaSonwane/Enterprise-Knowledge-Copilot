"""RAG System Configuration."""

import os
import re


def _positive_int_env(name, default=None):
    """Accept only explicit positive decimal integers; never unlimited values."""
    value = os.environ.get(name)
    if value is None:
        return default
    if not re.fullmatch(r"[0-9]+", value) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking
CHUNK_SIZE = 300  # tokens (approx words)
CHUNK_OVERLAP = 50  # tokens overlap between chunks
MIN_CHUNK_SIZE = 50  # minimum chunk size to keep

# Retrieval
TOP_K_RETRIEVAL = 20  # initial retrieval count
TOP_K_RERANK = 5  # after reranking
HYBRID_ALPHA = 0.7  # weight for vector search (1-alpha for BM25)

# Generation
MAX_CONTEXT_TOKENS = 3000
TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = _positive_int_env("OLLAMA_NUM_PREDICT", 1024)
OLLAMA_NUM_CTX = _positive_int_env("OLLAMA_NUM_CTX")

# Guardrails
# PROVISIONAL cosine evidence thresholds; not calibrated probabilities.
RETRIEVAL_SCORE_THRESHOLD = 0.30
ABSTENTION_THRESHOLD = 0.30

# Collection name
COLLECTION_NAME = "rag_documents"

# Cloud generation is opt-in; existing local behavior remains the default.
GENERATION_PROVIDER = os.environ.get("GENERATION_PROVIDER", "ollama").strip().lower()
if GENERATION_PROVIDER not in ("ollama", "groq"):
    raise ValueError("GENERATION_PROVIDER must be ollama or groq")
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MAX_COMPLETION_TOKENS = _positive_int_env("GROQ_MAX_COMPLETION_TOKENS", 1024)
if GROQ_MAX_COMPLETION_TOKENS > 2048:
    raise ValueError("GROQ_MAX_COMPLETION_TOKENS must be <= 2048")
