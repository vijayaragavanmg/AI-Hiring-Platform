"""
FastAPI application factory.

create_app() returns a fully configured FastAPI instance.
Using a factory (rather than a bare module-level `app`) makes the
app testable — tests can call create_app() with overridden
dependencies without side effects.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.drivers.api.routes import batch, candidates, jobs, resumes, search


def create_app() -> FastAPI:
    app = FastAPI(
        title="Resume Processing API",
        description=(
            "Ingest resumes (PDF / DOCX / TXT), extract structured data with an LLM, "
            "embed into a vector store, and run semantic search — "
            "powered by LangGraph + LangChain + Chroma."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ─────────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    # ── Routers ────────────────────────────────────────────────────────────
    app.include_router(resumes.router)
    app.include_router(batch.router)
    app.include_router(jobs.router)
    app.include_router(search.router)
    app.include_router(candidates.router)

    return app
