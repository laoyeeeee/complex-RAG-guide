"""Vector store helpers for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from helper_functions import replace_t_with_space
from .config import RetrievalConfig, VectorStorePaths


@dataclass(frozen=True)
class VectorStores:
    chunks: FAISS
    chapter_summaries: FAISS
    book_quotes: FAISS


@dataclass(frozen=True)
class Retrievers:
    chunks: object
    chapter_summaries: object
    book_quotes: object


def encode_book(path: str | Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    loader = PyPDFLoader(str(path))
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(cleaned_texts, embeddings)


def encode_chapter_summaries(chapter_summaries: Iterable) -> FAISS:
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(list(chapter_summaries), embeddings)


def encode_quotes(book_quotes_list: Iterable) -> FAISS:
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(list(book_quotes_list), embeddings)


def load_or_build_vectorstores(
    book_path: str | Path,
    chapter_summaries: Iterable,
    book_quotes: Iterable,
    paths: VectorStorePaths,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> VectorStores:
    if paths.chunks.exists() and paths.chapter_summaries.exists() and paths.book_quotes.exists():
        embeddings = OpenAIEmbeddings()
        chunks = FAISS.load_local(
            str(paths.chunks),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        chapter_summaries_store = FAISS.load_local(
            str(paths.chapter_summaries),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        book_quotes_store = FAISS.load_local(
            str(paths.book_quotes),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return VectorStores(
            chunks=chunks,
            chapter_summaries=chapter_summaries_store,
            book_quotes=book_quotes_store,
        )

    chunks = encode_book(book_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chapter_summaries_store = encode_chapter_summaries(chapter_summaries)
    book_quotes_store = encode_quotes(book_quotes)

    chunks.save_local(str(paths.chunks))
    chapter_summaries_store.save_local(str(paths.chapter_summaries))
    book_quotes_store.save_local(str(paths.book_quotes))

    return VectorStores(
        chunks=chunks,
        chapter_summaries=chapter_summaries_store,
        book_quotes=book_quotes_store,
    )


def build_retrievers(vectorstores: VectorStores, retrieval: RetrievalConfig) -> Retrievers:
    return Retrievers(
        chunks=vectorstores.chunks.as_retriever(search_kwargs={"k": retrieval.chunk_k}),
        chapter_summaries=vectorstores.chapter_summaries.as_retriever(
            search_kwargs={"k": retrieval.summary_k}
        ),
        book_quotes=vectorstores.book_quotes.as_retriever(search_kwargs={"k": retrieval.quote_k}),
    )
