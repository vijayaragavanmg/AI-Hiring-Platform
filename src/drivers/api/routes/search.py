"""Routes: semantic search endpoints with optional reranking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.adapters import ChromaHttpVectorStore, RetrievalPipeline
from src.drivers.api.dependencies import get_retrieval_pipeline, get_vector_store
from src.drivers.api.schemas import SearchResponse, SearchResultSchema
from src.ports.reranker import RankedResult
from src.ports.vector_store import SearchResult

router = APIRouter(prefix="/search", tags=["Search"])


# ── Helpers ────────────────────────────────────────────────────────────────

def _to_schema(results):
    return [
        SearchResultSchema(
            content=   r.content,
            section=   r.metadata.get("section",   "unknown"),
            candidate= r.metadata.get("candidate", "unknown"),
            metadata=  r.metadata,
            score=     getattr(r, "relevance_score", None) or getattr(r, "score", None),
        )
        for r in results
    ]


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=SearchResponse,
    summary="Semantic search with optional reranking",
)
async def search(
    q:         str            = Query(..., min_length=2, description="Natural language query"),
    k:         int            = Query(5, ge=1, le=20,   description="Number of final results"),
    section:   Optional[str]  = Query(None,              description="summary | skills | job_history | education"),
    candidate: Optional[str]  = Query(None,              description="Filter by candidate name"),
    rerank:    bool           = Query(True,              description="Apply cross-encoder reranking (recommended)"),
    retrieval: RetrievalPipeline    = Depends(get_retrieval_pipeline),
    store:     ChromaHttpVectorStore = Depends(get_vector_store),
):
    """
    Two-stage retrieval when rerank=true (default):
      1. Vector search fetches 20 candidates from ChromaDB.
      2. Cross-encoder reranker scores all 20 and returns top k.

    Set rerank=false to return raw vector results without reranking.
    """
    filter_dict = None
    if candidate:
        filter_dict = {"candidate": candidate}
    elif section:
        filter_dict = {"section": section}

    try:
        results = (
            retrieval.search(q, top_n=k, filter=filter_dict)
            if rerank
            else store.search(q, k=k, filter=filter_dict)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    return SearchResponse(query=q, total=len(results), results=_to_schema(results))


@router.get(
    "/skills",
    summary="Find candidates who list a specific skill",
)
async def search_skills(
    skill:     str  = Query(..., description="e.g. 'Kubernetes'"),
    k:         int  = Query(5, ge=1, le=20),
    rerank:    bool = Query(True, description="Apply reranking"),
    retrieval: RetrievalPipeline    = Depends(get_retrieval_pipeline),
    store:     ChromaHttpVectorStore = Depends(get_vector_store),
):
    skill_filter = {"section": "skills"}
    try:
        results = (
            retrieval.search(skill, top_n=k, filter=skill_filter)
            if rerank
            else store.search(skill, k=k, filter=skill_filter)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    return {
        "skill":    skill,
        "total":    len(results),
        "reranked": rerank,
        "matches": [
            {
                "candidate":   r.metadata.get("candidate"),
                "skills_text": r.content,
                "score":       getattr(r, "relevance_score", None) or getattr(r, "score", None),
            }
            for r in results
        ],
    }


@router.get(
    "/experience",
    summary="Search job history across all candidates",
)
async def search_experience(
    q:         str  = Query(..., description="e.g. 'led a team', 'ML pipeline'"),
    k:         int  = Query(5, ge=1, le=20),
    rerank:    bool = Query(True, description="Apply reranking"),
    retrieval: RetrievalPipeline    = Depends(get_retrieval_pipeline),
    store:     ChromaHttpVectorStore = Depends(get_vector_store),
):
    exp_filter = {"section": "job_history"}
    try:
        results = (
            retrieval.search(q, top_n=k, filter=exp_filter)
            if rerank
            else store.search(q, k=k, filter=exp_filter)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    return {
        "query":    q,
        "total":    len(results),
        "reranked": rerank,
        "matches": [
            {
                "candidate": r.metadata.get("candidate"),
                "company":   r.metadata.get("company"),
                "title":     r.metadata.get("title"),
                "excerpt":   r.content[:200],
                "score":     getattr(r, "relevance_score", None) or getattr(r, "score", None),
            }
            for r in results
        ],
    }
