"""Core pipeline operations extracted from the notebook."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from helper_functions import escape_quotes, text_wrap
from .chains import CoreChains
from .vectorstores import Retrievers


def _get_value(output: Any, field: str) -> Any:
    if isinstance(output, dict):
        return output[field]
    return getattr(output, field)


@dataclass
class PipelineOps:
    retrievers: Retrievers
    chains: CoreChains

    def retrieve_context_per_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]

        docs = self.retrievers.chunks.get_relevant_documents(question)
        context = " ".join(doc.page_content for doc in docs)

        docs_summaries = self.retrievers.chapter_summaries.get_relevant_documents(question)
        context_summaries = " ".join(
            f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries
        )

        docs_book_quotes = self.retrievers.book_quotes.get_relevant_documents(question)
        book_quotes = " ".join(doc.page_content for doc in docs_book_quotes)

        all_contexts = escape_quotes(context + context_summaries + book_quotes)
        return {"context": all_contexts, "question": question}

    def keep_only_relevant_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        context = state["context"]

        output = self.chains.keep_only_relevant_content_chain.invoke(
            {"query": question, "retrieved_documents": context}
        )
        relevant_content = _get_value(output, "relevant_content")
        relevant_content = escape_quotes("".join(relevant_content))

        return {
            "relevant_context": relevant_content,
            "context": context,
            "question": question,
        }

    def rewrite_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        result = self.chains.question_rewriter.invoke({"question": question})
        new_question = _get_value(result, "rewritten_question")
        return {"question": new_question}

    def answer_question_from_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        context = state.get("aggregated_context", state["context"])

        output = self.chains.question_answer_chain.invoke(
            {"question": question, "context": context}
        )
        answer = _get_value(output, "answer_based_on_content")
        return {"answer": answer, "context": context, "question": question}

    def is_relevant_content(self, state: Dict[str, Any]) -> str:
        question = state["question"]
        context = state["context"]

        output = self.chains.is_relevant_content_chain.invoke(
            {"query": question, "context": context}
        )
        is_relevant = _get_value(output, "is_relevant")
        return "relevant" if is_relevant else "not relevant"

    def grade_generation_v_documents_and_question(self, state: Dict[str, Any]) -> str:
        context = state["context"]
        answer = state["answer"]
        question = state["question"]

        result = self.chains.is_grounded_on_facts_chain.invoke(
            {"context": context, "answer": answer}
        )
        grounded_on_facts = _get_value(result, "grounded_on_facts")
        if not grounded_on_facts:
            return "hallucination"

        output = self.chains.can_be_answered_chain.invoke(
            {"question": question, "context": context}
        )
        can_be_answered = _get_value(output, "can_be_answered")
        return "useful" if can_be_answered else "not_useful"

    def is_distilled_content_grounded_on_content(self, state: Dict[str, Any]) -> str:
        distilled_content = state["relevant_context"]
        original_context = state["context"]

        output = self.chains.is_distilled_content_grounded_chain.invoke(
            {"distilled_content": distilled_content, "original_context": original_context}
        )
        grounded = _get_value(output, "grounded")
        return (
            "grounded on the original context" if grounded else "not grounded on the original context"
        )

    def retrieve_chunks_context_per_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        docs = self.retrievers.chunks.get_relevant_documents(question)
        context = escape_quotes(" ".join(doc.page_content for doc in docs))
        return {"context": context, "question": question}

    def retrieve_summaries_context_per_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        docs_summaries = self.retrievers.chapter_summaries.get_relevant_documents(question)
        context_summaries = " ".join(
            f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries
        )
        context_summaries = escape_quotes(context_summaries)
        return {"context": context_summaries, "question": question}

    def retrieve_book_quotes_context_per_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        docs_book_quotes = self.retrievers.book_quotes.get_relevant_documents(question)
        book_quotes = " ".join(doc.page_content for doc in docs_book_quotes)
        book_quotes_context = escape_quotes(book_quotes)
        return {"context": book_quotes_context, "question": question}

    def is_answer_grounded_on_context(self, state: Dict[str, Any]) -> str:
        context = state["context"]
        answer = state["answer"]

        result = self.chains.is_grounded_on_facts_chain.invoke(
            {"context": context, "answer": answer}
        )
        grounded_on_facts = _get_value(result, "grounded_on_facts")
        return "grounded on context" if grounded_on_facts else "hallucination"

    def log_final_answer(self, response: str) -> str:
        return text_wrap(f" the final answer is: {response}")
