# Resume Pipeline

A production-grade resume ingestion, extraction, and semantic search system built with **LangGraph**, **LangChain**, **ChromaDB**, and **FastAPI** — structured using Clean Architecture (Ports & Adapters).

---

## Architecture

```
src/
├── domain/       # Pure Python — entities, state, config. Zero external deps.
├── ports/        # Abstract interfaces (ABCs) the domain depends on.
├── adapters/     # Concrete implementations of every port.
└── drivers/      # Entry points: FastAPI (HTTP) and CLI (batch).
```

The dependency rule is strictly inward: `drivers → adapters → ports → domain`. Nothing in `domain` or `ports` imports from `adapters` or `drivers`.

---

## Features

- **Single resume upload** — async (background) or sync
- **Batch processing** — parallel ingestion via `ThreadPoolExecutor`
- **LLM extraction** — structured JSON extraction of name, summary, skills, job history, education
- **Multiple LLM providers** — OpenAI (`gpt-4o-mini`) or Google Gemini (`gemini-2.5-flash-lite`)
- **Multiple embedding providers** — OpenAI, Gemini, or local cross-encoder (free)
- **Vector store** — section-aware chunking with rich metadata, persisted to ChromaDB
- **Semantic reranking** — Cohere cloud reranker or local cross-encoder for ranking search results
- **ChromaDB HTTP client** — connects to a remote ChromaDB server (Docker / production)
- **Semantic search** — full-text, by section, by candidate, by skill, by experience
- **Job polling** — async job status via in-memory repository (swap to Redis/SQL with one-line change)
- **REST API** — FastAPI with auto-generated Swagger docs at `/docs`
- **CLI** — `python -m src.drivers.cli ./resumes/`
- **Docker Compose** — ChromaDB + API server as separate containerised services

---

## Project Structure

```
resume-pipeline/
│
├── .env                        # Secrets and config (never commit)
├── .env.example                # Safe template — commit this
├── .gitignore
├── .dockerignore
├── docker-compose.yml          # ChromaDB + API services
├── Dockerfile                  # API image
├── main.py                     # Uvicorn entry point
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
│
└── src/
    │
    ├── domain/
    │   ├── entities.py         # JobEntry, ResumeData, ProcessingResult, BatchSummary (@dataclass, no Pydantic)
    │   └── pipeline_state.py   # LangGraph ResumeState (TypedDict)
    │
    ├── ports/
    │   ├── document_loader.py  # DocumentLoader ABC + RawDocument
    │   ├── llm_extractor.py    # LLMExtractor ABC + ExtractionError
    │   ├── vector_store.py     # VectorStore ABC + Chunk + SearchResult
    │   ├── job_repository.py   # JobRepository ABC + Job + JobStatus
    │   └── chunker.py          # Chunker ABC
    │
    ├── adapters/
    │   ├── langchain_loader.py         # DocumentLoader → PyPDF / Docx2txt / TextLoader
    │   ├── openai_extractor.py         # LLMExtractor → ChatOpenAI (gpt-4o-mini)
    │   ├── gemini_extractor.py         # LLMExtractor → ChatGoogleGenerativeAI
    │   ├── section_chunker.py          # Chunker → section-aware + sliding-window fallback
    │   ├── chroma_http_store.py        # VectorStore → ChromaDB HTTP client (production)
    │   ├── openai_vector_store.py      # VectorStore → Chroma + OpenAI embeddings
    │   ├── gemini_vector_store.py      # VectorStore → Chroma + Google embeddings
    │   ├── cohere_reranker.py          # Reranker → Cohere cloud reranker (best quality)
    │   ├── in_memory_job_repo.py       # JobRepository → plain dict (swap for Redis/SQL)
    │   ├── langgraph_pipeline.py       # Wires all ports into a LangGraph StateGraph
    │   ├── retrieval_pipeline.py       # Search + retrieval with semantic ranking
    │   └── batch_processor.py          # Fans out pipeline.process() via ThreadPoolExecutor
    │
    └── drivers/
        ├── config.py           # All settings loaded from .env (single source of truth)
        ├── cli.py              # CLI entry point for batch processing
        └── api/
            ├── app.py          # FastAPI factory (create_app)
            ├── dependencies.py # @lru_cache singletons + request helpers + dependency injection
            ├── job_runner.py   # Background task runners (single + batch)
            ├── schemas.py      # Pydantic DTO models (request/response validation), separate from domain
            └── routes/
                ├── resumes.py    # POST /resumes/upload, /upload/sync
                ├── batch.py      # POST /resumes/batch, GET /batch/{id}
                ├── jobs.py       # GET /jobs, /jobs/{id}
                ├── search.py     # GET /search, /search/skills, /search/experience
                └── candidates.py # GET /candidates, /candidates/{name}
```

---

## Setup (local development)

### 1. Clone the repository

```bash
git clone https://github.com/your-org/resume-pipeline.git
cd resume-pipeline
```

### 2. Create a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Your prompt should show `(.venv)` when the environment is active.
> Run `deactivate` to exit it.

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install dependencies

```bash
# Production only
pip install -r requirements.txt

# Production + dev/test tools (recommended)
pip install -r requirements.txt -r requirements-dev.txt

# Or install as an editable package
pip install -e ".[dev]"
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values — at minimum:

```bash
OPENAI_API_KEY=sk-...
```

---

## Running with Docker Compose (recommended)

Docker Compose starts ChromaDB and the API together — no manual ChromaDB setup needed.

### 1. Configure `.env`

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY (and GEMINI_API_KEY if using Gemini)
```

### 2. Build and start

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Resume API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| ChromaDB | http://localhost:8001 |

### 3. Stop and clean up

```bash
# Stop services (data is preserved in volumes)
docker compose down

# Stop and remove volumes (wipes all ChromaDB data)
docker compose down -v
```

### 4. Rebuild after code changes

```bash
docker compose up --build
```

### How the services connect

```
┌─────────────────────────────────────┐
│        docker-compose network       │
│                                     │
│  ┌───────────┐   HTTP    ┌────────┐ │
│  │    api    │ ────────► │ chroma │ │
│  │  :8000    │           │  :8000 │ │
│  └───────────┘           └────────┘ │
│        │                     │      │
│   uploads_data          chroma_data │
│    (volume)               (volume)  │
└─────────────────────────────────────┘
```

The `api` container connects to ChromaDB using the service name `chromadb` as the host. This is injected automatically by `docker-compose.yml` and overrides whatever `CHROMA_HOST` is set to in `.env`.

---

## Quickstart (local, without Docker)

Once setup is complete (virtual environment active, `.env` configured):

### Start ChromaDB locally

```bash
pip install chromadb
chroma run --host localhost --port 8001 --path ./resume_chroma_db
```

### Start the API server

```bash
uvicorn main:app --reload --port 8000
```

Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)

### Process resumes via CLI

```bash
# Single folder
python -m src.drivers.cli ./resumes/

# Specific files
python -m src.drivers.cli cv1.pdf cv2.docx cv3.txt
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/resumes/upload` | Upload single resume (async) |
| `POST` | `/resumes/upload/sync` | Upload single resume (sync, blocks) |
| `POST` | `/resumes/batch` | Upload up to 50 resumes in parallel |
| `GET` | `/jobs/{job_id}` | Poll a single job |
| `GET` | `/jobs` | List all jobs (filter by status) |
| `GET` | `/batch/{batch_id}` | Poll a batch job |
| `GET` | `/search` | Semantic search (filter by section / candidate) |
| `GET` | `/search/skills` | Find candidates by skill |
| `GET` | `/search/experience` | Search job history |
| `GET` | `/candidates` | List all indexed candidates |
| `GET` | `/candidates/{name}` | Get all sections for a candidate |

---

## Embedding providers

| Adapter | Model | Cost | Privacy | Quality |
|---------|-------|------|---------|---------|
| `OpenAIVectorStore` | `text-embedding-3-small` | Paid | Cloud | Excellent |
| `GeminiVectorStore` | `gemini-embedding-2-preview` | Paid | Cloud | Excellent |

Swap embedding provider in `src/drivers/api/dependencies.py`:

```python
# Example: switch to Gemini embeddings
from src.adapters import GeminiVectorStore

@lru_cache(maxsize=1)
def get_vector_store() -> GeminiVectorStore:
    return GeminiVectorStore()
```

---

## Swapping adapters

Every adapter is injected via `src/drivers/api/dependencies.py` with constructor parameters. To swap any implementation, update the dependency:

```python
# Example: swap LLM extractor from OpenAI to Gemini
from src.adapters import GeminiLLMExtractor
from src.drivers.config import GEMINI_API_KEY, GEMINI_LLM_MODEL, GEMINI_LLM_TEMPERATURE

@lru_cache(maxsize=1)
def get_pipeline() -> LangGraphPipeline:
    return LangGraphPipeline(
        loader=       LangChainDocumentLoader(),
        extractor=    GeminiLLMExtractor(api_key=GEMINI_API_KEY, model=GEMINI_LLM_MODEL, temperature=GEMINI_LLM_TEMPERATURE),  # ← changed
        chunker=      SectionChunker(),
        vector_store= get_vector_store(),
    )
```

**Key principle:** Configuration is injected as constructor parameters, not imported as module-level constants. This makes testing easy (mock any adapter) and swapping providers trivial.

---

## Architecture patterns

### Clean Architecture (Ports & Adapters)

The project follows strict unidirectional dependencies:

```
drivers  →  adapters  →  ports  →  domain
```

- **domain/** — Pure Python, no external dependencies, no I/O, no frameworks. Only business logic.
- **ports/** — Interface contracts (ABCs) that domain depends on.
- **adapters/** — Concrete implementations (OpenAI, Gemini, ChromaDB, etc.). Zero config imports; config injected via constructor.
- **drivers/** — Entry points (HTTP, CLI, config). Dependency injection wiring here.

### Constructor Injection (No Config Imports in Adapters)

Configuration flows **into** adapters, not **from** them:

```python
# ❌ Bad: adapter imports config
class OpenAILLMExtractor:
    def __init__(self):
        from src.drivers.config import OPENAI_API_KEY  # Tight coupling

# ✅ Good: adapter receives config
class OpenAILLMExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
```

Benefits:
- **Testability**: Mock any config value in tests
- **Portability**: No module-level side effects
- **Flexibility**: Use same adapter with different configs

### Domain Entities as Pure `@dataclass`

Domain entities are framework-free Python dataclasses:

```python
@dataclass
class ResumeData:
    name: str
    email: Optional[str] = None
    summary: str = ""
    skills: List[str] = field(default_factory=list)
    job_history: List[JobEntry] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for boundary crossing."""
        return asdict(self)
```

### DTO Layer (Pydantic at API Boundary Only)

HTTP request/response validation is **separate** from domain entities:

```python
# Domain (pure, no Pydantic)
@dataclass
class ResumeData:
    name: str
    summary: str

# API (Pydantic DTOs)
class ResumeDataSchema(BaseModel):  # ← Pydantic here
    name: str
    summary: str

# Conversion at route boundary
@router.post("/resumes/upload/sync")
async def upload_resume_sync(pipeline) -> ResumeResponse:
    result = pipeline.process(...)
    return ResumeResponse(
        candidate=ResumeDataSchema(**result.resume_data.to_dict()),  # ← Explicit conversion
        ...
    )
```

Benefits:
- **Decoupling**: Swap Pydantic for another validator without touching domain
- **API flexibility**: DTOs can differ from domain models
- **Defensive**: `OPT Optional[ResumeDataSchema] = None` prevents API crashes

---

## Configuration reference

All settings are loaded from `.env` via `src/drivers/config.py` (the only place where environment variables are accessed).

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required for OpenAI adapters) |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | OpenAI model for extraction |
| `OPENAI_LLM_TEMPERATURE` | `0` | OpenAI LLM sampling temperature |
| `OPEN_AI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `GEMINI_API_KEY` | — | Google Gemini API key (required for Gemini adapters) |
| `GEMINI_LLM_MODEL` | `gemini-2.5-flash-lite` | Gemini model for extraction |
| `GEMINI_LLM_TEMPERATURE` | `0` | Gemini LLM sampling temperature |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2-preview` | Gemini embedding model |
| `COHERE_API_KEY` | — | Cohere API key (required for Cohere reranker) |
| `COHERE_RERANK_MODEL` | `rerank-english-v3.0` | Cohere reranking model |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder reranker (free) |
| `CROSS_ENCODER_DEVICE` | `cpu` | Device for cross-encoder (`cpu`, `cuda`, `mps`) |
| `RERANK_FETCH_K` | `20` | Overfetch multiplier: candidates retrieved before reranking |
| `CHROMA_HOST` | `localhost` | ChromaDB server host |
| `CHROMA_PORT` | `8001` | ChromaDB server port |
| `CHROMA_COLLECTION` | `resumes` | Chroma collection name |
| `CHROMA_SERVER_TOKEN` | — | ChromaDB auth token (leave blank to disable) |
| `UPLOAD_DIR` | `./uploads` | Uploaded file storage |
| `VECTORSTORE_PATH` | `./resume_chroma_db` | Local Chroma path (non-Docker only) |
| `CHUNK_SIZE` | `500` | Text splitter chunk size |
| `CHUNK_OVERLAP` | `50` | Text splitter overlap |
| `BATCH_MAX_WORKERS` | `4` | Thread pool size for batch processing |
| `BATCH_MAX_FILES` | `50` | Max files per batch request |
| `API_HOST` | `0.0.0.0` | API server bind host |
| `API_PORT` | `8000` | API server port |
| `API_RELOAD` | `true` | Uvicorn auto-reload (disable in production) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Running tests

```bash
pytest tests/ -v
```

Tests inject mock adapters directly into `LangGraphPipeline` — no OpenAI calls, no ChromaDB, no disk I/O required:

```python
pipeline = LangGraphPipeline(
    loader=       MockLoader(),
    extractor=    MockExtractor(),
    chunker=      SectionChunker(),
    vector_store= MockVectorStore(),
)
```

---

## Requirements

- Python 3.11+
- Docker + Docker Compose (for containerised deployment)
- At least one LLM provider key: `OPENAI_API_KEY` or `GEMINI_API_KEY`
- At least one embedding provider: OpenAI (paid), Gemini (paid), or local cross-encoder (free)