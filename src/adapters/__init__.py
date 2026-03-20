"""
Adapters — concrete implementations of the port interfaces.

Each class in this package implements exactly one port ABC.
Nothing outside this package should import third-party framework
types (LangChain Documents, Chroma, OpenAI, etc.) directly.
"""

from src.adapters.batch_processor import BatchProcessor
from src.adapters.openai_vector_store import OpenAIVectorStore
from src.adapters.in_memory_job_repo import InMemoryJobRepository
from src.adapters.langchain_loader import LangChainDocumentLoader
from src.adapters.langgraph_pipeline import LangGraphPipeline
from src.adapters.openai_extractor import OpenAILLMExtractor
from src.adapters.section_chunker import SectionChunker
from src.adapters.gemini_extractor import GeminiLLMExtractor
from src.adapters.gemini_vector_store import GeminiVectorStore
from src.adapters.chroma_http_store import ChromaHttpVectorStore
from src.adapters.cohere_reranker import CohereReranker
from src.adapters.retrieval_pipeline import RetrievalPipeline

__all__ = [
    "BatchProcessor",
    "ChromaHttpVectorStore",
    "OpenAIVectorStore",
    "InMemoryJobRepository",
    "LangChainDocumentLoader",
    "LangGraphPipeline",
    "OpenAILLMExtractor",
    "SectionChunker",
    "GeminiLLMExtractor",
    "GeminiVectorStore",
    "CohereReranker",
    "RetrievalPipeline",
]
