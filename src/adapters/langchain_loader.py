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

from src.domain.config import SUPPORTED_EXTENSIONS
from src.ports.document_loader import DocumentLoader, RawDocument


class LangChainDocumentLoader(DocumentLoader):

    def load(self, file_path: str) -> List[RawDocument]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
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
