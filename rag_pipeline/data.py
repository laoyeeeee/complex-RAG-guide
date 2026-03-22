"""Data loading and preprocessing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

from helper_functions import (
    extract_book_quotes_as_documents,
    replace_t_with_space,
    split_into_chapters,
)


@dataclass(frozen=True)
class BookArtifacts:
    chapters: List[Document]
    quotes: List[Document]
    pages: List[Document]


def load_pdf_pages(book_path: str | Path) -> List[Document]:
    loader = PyPDFLoader(str(book_path))
    pages = loader.load()
    return replace_t_with_space(pages)


def load_chapters(book_path: str | Path) -> List[Document]:
    chapters = split_into_chapters(str(book_path))
    return replace_t_with_space(chapters)


def extract_quotes(pages: Iterable[Document]) -> List[Document]:
    return extract_book_quotes_as_documents(list(pages))


def build_book_artifacts(book_path: str | Path) -> BookArtifacts:
    pages = load_pdf_pages(book_path)
    chapters = load_chapters(book_path)
    quotes = extract_quotes(pages)
    return BookArtifacts(chapters=chapters, quotes=quotes, pages=pages)
