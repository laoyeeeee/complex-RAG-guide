"""LangGraph workflows for qualitative retrieval and answering."""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from ..operations import PipelineOps


class QualitativeRetrievalAnswerGraphState(TypedDict):
    question: str
    context: str
    answer: str


class QualitativeRetrievalGraphState(TypedDict):
    question: str
    context: str
    relevant_context: str


class QualitativeAnswerGraphState(TypedDict):
    question: str
    context: str
    answer: str


class QualitativeWorkflows:
    def __init__(self, ops: PipelineOps) -> None:
        self.ops = ops
        self.retrieval_answer_app = self._build_retrieval_answer_app()
        self.chunks_retrieval_app = self._build_chunks_retrieval_app()
        self.summaries_retrieval_app = self._build_summaries_retrieval_app()
        self.quotes_retrieval_app = self._build_quotes_retrieval_app()
        self.answer_app = self._build_answer_app()

    def _build_retrieval_answer_app(self):
        workflow = StateGraph(QualitativeRetrievalAnswerGraphState)
        workflow.add_node("retrieve_context_per_question", self.ops.retrieve_context_per_question)
        workflow.add_node("keep_only_relevant_content", self.ops.keep_only_relevant_content)
        workflow.add_node("rewrite_question", self.ops.rewrite_question)
        workflow.add_node("answer_question_from_context", self.ops.answer_question_from_context)

        workflow.set_entry_point("retrieve_context_per_question")
        workflow.add_edge("retrieve_context_per_question", "keep_only_relevant_content")
        workflow.add_conditional_edges(
            "keep_only_relevant_content",
            self.ops.is_relevant_content,
            {
                "relevant": "answer_question_from_context",
                "not relevant": "rewrite_question",
            },
        )
        workflow.add_edge("rewrite_question", "retrieve_context_per_question")
        workflow.add_conditional_edges(
            "answer_question_from_context",
            self.ops.grade_generation_v_documents_and_question,
            {
                "hallucination": "answer_question_from_context",
                "not_useful": "rewrite_question",
                "useful": END,
            },
        )
        return workflow.compile()

    def _build_chunks_retrieval_app(self):
        workflow = StateGraph(QualitativeRetrievalGraphState)
        workflow.add_node(
            "retrieve_chunks_context_per_question",
            self.ops.retrieve_chunks_context_per_question,
        )
        workflow.add_node("keep_only_relevant_content", self.ops.keep_only_relevant_content)

        workflow.set_entry_point("retrieve_chunks_context_per_question")
        workflow.add_edge("retrieve_chunks_context_per_question", "keep_only_relevant_content")
        workflow.add_conditional_edges(
            "keep_only_relevant_content",
            self.ops.is_distilled_content_grounded_on_content,
            {
                "grounded on the original context": END,
                "not grounded on the original context": "keep_only_relevant_content",
            },
        )
        return workflow.compile()

    def _build_summaries_retrieval_app(self):
        workflow = StateGraph(QualitativeRetrievalGraphState)
        workflow.add_node(
            "retrieve_summaries_context_per_question",
            self.ops.retrieve_summaries_context_per_question,
        )
        workflow.add_node("keep_only_relevant_content", self.ops.keep_only_relevant_content)

        workflow.set_entry_point("retrieve_summaries_context_per_question")
        workflow.add_edge("retrieve_summaries_context_per_question", "keep_only_relevant_content")
        workflow.add_conditional_edges(
            "keep_only_relevant_content",
            self.ops.is_distilled_content_grounded_on_content,
            {
                "grounded on the original context": END,
                "not grounded on the original context": "keep_only_relevant_content",
            },
        )
        return workflow.compile()

    def _build_quotes_retrieval_app(self):
        workflow = StateGraph(QualitativeRetrievalGraphState)
        workflow.add_node(
            "retrieve_book_quotes_context_per_question",
            self.ops.retrieve_book_quotes_context_per_question,
        )
        workflow.add_node("keep_only_relevant_content", self.ops.keep_only_relevant_content)

        workflow.set_entry_point("retrieve_book_quotes_context_per_question")
        workflow.add_edge("retrieve_book_quotes_context_per_question", "keep_only_relevant_content")
        workflow.add_conditional_edges(
            "keep_only_relevant_content",
            self.ops.is_distilled_content_grounded_on_content,
            {
                "grounded on the original context": END,
                "not grounded on the original context": "keep_only_relevant_content",
            },
        )
        return workflow.compile()

    def _build_answer_app(self):
        workflow = StateGraph(QualitativeAnswerGraphState)
        workflow.add_node("answer_question_from_context", self.ops.answer_question_from_context)

        workflow.set_entry_point("answer_question_from_context")
        workflow.add_conditional_edges(
            "answer_question_from_context",
            self.ops.is_answer_grounded_on_context,
            {
                "hallucination": "answer_question_from_context",
                "grounded on context": END,
            },
        )
        return workflow.compile()
