"""
Application configuration (drivers layer).

Loads all configuration from environment variables (.env file).
This is the ONLY place where dotenv and os.environ are accessed.

Lower layers (ports, adapters) should NOT import configuration directly.
Configuration should be injected by dependency.py at construction time.

.env file location:
    resume-pipeline/
    ├── .env          ← project root, same level as main.py
    ├── main.py
    └── src/
        └── drivers/
            └── config.py  ← this file
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (three levels up: src/drivers/ → src/ → drivers/ → root)
_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_ROOT / ".env")


# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_LLM_TEMPERATURE: int = int(os.getenv("OPENAI_LLM_TEMPERATURE", "0"))


# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
GEMINI_LLM_TEMPERATURE: float = float(os.getenv("GEMINI_LLM_TEMPERATURE", "0"))


# ── Embeddings ────────────────────────────────────────────────────────────────
OPEN_AI_EMBEDDING_MODEL: str = os.getenv("OPEN_AI_EMBEDDING_MODEL", "text-embedding-3-small")
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")


# ── Reranking ─────────────────────────────────────────────────────────────────
# Cohere reranker (cloud, best quality)
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")

# Cross-encoder reranker (local, free)
CROSS_ENCODER_MODEL: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_DEVICE: str = os.getenv("CROSS_ENCODER_DEVICE", "cpu")

# Overfetch multiplier: candidates retrieved before reranking
RERANK_FETCH_K: int = int(os.getenv("RERANK_FETCH_K", "20"))


# ── Storage paths ─────────────────────────────────────────────────────────────
UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./uploads"))
VECTORSTORE_PATH: Path = Path(os.getenv("VECTORSTORE_PATH", "./resume_chroma_db"))

# Ensure storage directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Chroma HTTP server (docker-compose / remote) ──────────────────────────────
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8001"))
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "resumes")
CHROMA_SERVER_TOKEN: str = os.getenv("CHROMA_SERVER_TOKEN", "")


# ── File handling ─────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".txt"})


# ── Text splitting ────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))


# ── Batch processing ──────────────────────────────────────────────────────────
BATCH_MAX_WORKERS: int = int(os.getenv("BATCH_MAX_WORKERS", "4"))
BATCH_MAX_FILES: int = int(os.getenv("BATCH_MAX_FILES", "50"))


# ── API server (FastAPI) ──────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"


# ── CORS ──────────────────────────────────────────────────────────────────────
CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
