"""Configuration defaults for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_BOOK_PATH = "Harry Potter - Book 1 - The Sorcerers Stone.pdf"


@dataclass(frozen=True)
class ModelConfig:
    openai_chat_model: str = "gpt-4o"
    openai_summarization_model: str = "gpt-3.5-turbo-0125"
    groq_chat_model: str = "llama3-70b-8192"


@dataclass(frozen=True)
class VectorStorePaths:
    chunks: Path = Path("chunks_vector_store")
    chapter_summaries: Path = Path("chapter_summaries_vector_store")
    book_quotes: Path = Path("book_quotes_vectorstore")


@dataclass(frozen=True)
class RetrievalConfig:
    chunk_k: int = 1
    summary_k: int = 1
    quote_k: int = 10


@dataclass(frozen=True)
class PipelineConfig:
    book_path: Path = Path(DEFAULT_BOOK_PATH)
    models: ModelConfig = ModelConfig()
    vectorstores: VectorStorePaths = VectorStorePaths()
    retrieval: RetrievalConfig = RetrievalConfig()
