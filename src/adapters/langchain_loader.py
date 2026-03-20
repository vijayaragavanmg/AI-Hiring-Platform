"""
Adapter: LangChainDocumentLoader

Implements DocumentLoader using LangChain community loaders.
Supports .pdf (PyPDF), .docx (Docx2txt), .txt (plain text).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

from src.ports.document_loader import DocumentLoader, RawDocument


class LangChainDocumentLoader(DocumentLoader):
    """Load documents using LangChain loaders (.pdf, .docx, .txt)."""

    def __init__(
        self,
        supported_extensions: set = None,
    ) -> None:
        """Initialize with supported file extensions.
        
        Args:
            supported_extensions: Set of allowed extensions (default: {.pdf, .docx, .txt})
        """
        self._supported_extensions = supported_extensions or {".pdf", ".docx", ".txt"}

    def load(self, file_path: str) -> List[RawDocument]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self._supported_extensions:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Accepted: {', '.join(sorted(self._supported_extensions))}"
            )

        _loaders = {
            ".pdf":  lambda: PyPDFLoader(str(path)),
            ".docx": lambda: Docx2txtLoader(str(path)),
            ".txt":  lambda: TextLoader(str(path)),
        }

        lc_docs = _loaders[ext]().load()

        return [
            RawDocument(
                content=doc.page_content,
                page=doc.metadata.get("page", i),
                metadata=dict(doc.metadata),
            )
            for i, doc in enumerate(lc_docs)
        ]
