"""Routes: candidate listing and detail."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.adapters import OpenAIVectorStore
from src.drivers.api.dependencies import get_vector_store

router = APIRouter(prefix="/candidates", tags=["Candidates"])


@router.get(
    "",
    summary="List all unique candidates in the vector store",
)
async def list_candidates(
    store: OpenAIVectorStore = Depends(get_vector_store),
):
    try:
        names = store.list_candidates()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"total": len(names), "candidates": names}


@router.get(
    "/{name}",
    summary="Get all stored sections for a specific candidate",
)
async def get_candidate(
    name:  str,
    store: OpenAIVectorStore = Depends(get_vector_store),
):
    results = store.search(name, k=20, filter={"candidate": name})
    if not results:
        raise HTTPException(status_code=404, detail=f"Candidate '{name}' not found.")

    sections: dict = {}
    for r in results:
        sec = r.metadata.get("section", "other")
        sections.setdefault(sec, []).append(r.content)

    return {"candidate": name, "sections": sections}
