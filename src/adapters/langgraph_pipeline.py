"""
Adapter: LangGraphPipeline

Wires the four port implementations together into a LangGraph
StateGraph and exposes process_single_resume() as the single
callable the rest of the application uses.

Dependency injection: all port implementations are passed in at
construction time, making this fully testable with mocks.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.domain.entities import ProcessingResult, ResumeData
from src.domain.pipeline_state import ResumeState
from src.ports.chuncker import Chunker
from src.ports.document_loader import DocumentLoader
from src.ports.llm_extractor import ExtractionError, LLMExtractor
from src.ports.vector_store import VectorStore

log = logging.getLogger(__name__)


class LangGraphPipeline:
    """
    Builds and runs the LangGraph resume processing pipeline.

    All I/O is delegated to the injected port implementations —
    the graph wiring logic here stays completely framework-agnostic
    with respect to loaders, LLMs, and vector stores.
    """

    def __init__(
        self,
        loader: DocumentLoader,
        extractor: LLMExtractor,
        chunker: Chunker,
        vector_store: VectorStore,
        supported_extensions: set = None,
    ) -> None:
        """Initialize pipeline with port implementations.
        
        Args:
            loader: DocumentLoader implementation
            extractor: LLMExtractor implementation
            chunker: Chunker implementation
            vector_store: VectorStore implementation
            supported_extensions: Set of allowed file extensions (default: {.pdf, .docx, .txt})
        """
        self._loader = loader
        self._extractor = extractor
        self._chunker = chunker
        self._vector_store = vector_store
        self._supported_extensions = supported_extensions or {".pdf", ".docx", ".txt"}
        self._graph = self._build()

    # ── Public entry point ─────────────────────────────────────────────────

    def process(
        self,
        file_path: str,
        vectorstore_path: str = "./resume_chroma_db",
    ) -> ProcessingResult:
        """Run the full pipeline for one resume file.
        
        Args:
            file_path: Path to resume file
            vectorstore_path: Path to vectorstore (default: ./resume_chroma_db)
        
        Returns:
            ProcessingResult with success/failure details
        """
        start = time.time()
        path = Path(file_path)

        if not path.exists():
            return ProcessingResult(
                file_path=file_path,
                success=False,
                error=f"File not found: {file_path}",
            )

        if path.suffix.lower() not in self._supported_extensions:
            return ProcessingResult(
                file_path=file_path,
                success=False,
                error=(
                    f"Unsupported extension '{path.suffix}'. "
                    f"Accepted: {', '.join(sorted(self._supported_extensions))}"
                ),
            )

        try:
            final_state = self._graph.invoke({
                "file_path":        str(file_path),
                "raw_text":         "",
                "raw_documents":    [],
                "resume_data":      None,
                "extraction_error": None,
                "chunks":           [],
                "status_log":       [],
                "vectorstore_path": vectorstore_path,
            })

            rd: ResumeData | None = final_state.get("resume_data")

            return ProcessingResult(
                file_path=        file_path,
                success=          True,
                candidate_name=   rd.name if rd else path.stem,
                skills_count=     len(rd.skills) if rd else 0,
                jobs_count=       len(rd.job_history) if rd else 0,
                chunks_count=     len(final_state.get("chunks", [])),
                duration_seconds= round(time.time() - start, 2),
                status_log=       final_state.get("status_log", []),
                resume_data=      rd,
            )

        except Exception as exc:
            log.exception("Pipeline failed for %s", file_path)
            return ProcessingResult(
                file_path=        file_path,
                success=          False,
                error=            str(exc),
                duration_seconds= round(time.time() - start, 2),
            )

    # ── Graph construction ─────────────────────────────────────────────────

    def _build(self) -> CompiledStateGraph:
        workflow = StateGraph(ResumeState)

        workflow.add_node("load_document",         self._node_load)
        workflow.add_node("extract_resume_fields", self._node_extract)
        workflow.add_node("build_chunks",          self._node_chunk)
        workflow.add_node("embed_and_store",       self._node_store)

        workflow.set_entry_point("load_document")
        workflow.add_edge("load_document", "extract_resume_fields")
        workflow.add_conditional_edges(
            "extract_resume_fields",
            lambda state: "build_chunks",   # always advance; fallback handled in node
            {"build_chunks": "build_chunks"},
        )
        workflow.add_edge("build_chunks",    "embed_and_store")
        workflow.add_edge("embed_and_store", END)

        return workflow.compile()

    # ── Node implementations ───────────────────────────────────────────────

    def _node_load(self, state: ResumeState) -> dict:
        docs     = self._loader.load(state["file_path"])
        raw_text = "\n".join(d.content for d in docs)
        return {
            "raw_documents": docs,
            "raw_text":      raw_text,
            "status_log":    [f"Loaded {Path(state['file_path']).name} ({len(raw_text)} chars)"],
        }

    def _node_extract(self, state: ResumeState) -> dict:
        try:
            resume_data = self._extractor.extract(state["raw_text"])
            return {
                "resume_data":      resume_data,
                "extraction_error": None,
                "status_log": [
                    f"Extracted: {resume_data.name} | "
                    f"{len(resume_data.skills)} skills | "
                    f"{len(resume_data.job_history)} roles"
                ],
            }
        except ExtractionError as exc:
            return {
                "resume_data":      None,
                "extraction_error": str(exc),
                "status_log":       [f"Extraction failed — falling back to raw chunking: {exc}"],
            }

    def _node_chunk(self, state: ResumeState) -> dict:
        chunks = self._chunker.chunk(
            resume_data=   state["resume_data"],
            raw_documents= state["raw_documents"],
        )
        return {
            "chunks":     chunks,
            "status_log": [f"Built {len(chunks)} chunks"],
        }

    def _node_store(self, state: ResumeState) -> dict:
        self._vector_store.store(state["chunks"])
        return {
            "status_log": [
                f"Stored {len(state['chunks'])} vectors → {state['vectorstore_path']}"
            ],
        }
