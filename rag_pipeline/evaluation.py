"""Evaluation helpers for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_recall,
    faithfulness,
)

from helper_functions import analyse_metric_results
from .config import ModelConfig
from .workflows.plan_execute import PlanExecuteWorkflow


@dataclass(frozen=True)
class EvaluationResult:
    generated_answers: List[str]
    retrieved_documents: List[str]
    results_df: object


def run_plan_execute_answers(
    plan_execute: PlanExecuteWorkflow,
    questions: Iterable[str],
) -> Tuple[List[str], List[str]]:
    generated_answers = []
    retrieved_documents = []

    for question in questions:
        inputs = {"question": question}
        final_answer, final_state = plan_execute.execute_plan_and_print_steps(inputs)
        generated_answers.append(final_answer)
        retrieved_documents.append(final_state["aggregated_context"])

    return generated_answers, retrieved_documents


def evaluate_with_ragas(
    questions: List[str],
    ground_truth_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str],
    model_config: ModelConfig,
) -> EvaluationResult:
    data_samples = {
        "question": questions,
        "answer": generated_answers,
        "contexts": retrieved_documents,
        "ground_truth": ground_truth_answers,
    }

    data_samples["contexts"] = [
        [context] if isinstance(context, str) else context for context in data_samples["contexts"]
    ]

    dataset = Dataset.from_dict(data_samples)
    metrics = [
        answer_correctness,
        faithfulness,
        answer_relevancy,
        context_recall,
        answer_similarity,
    ]

    llm = ChatOpenAI(temperature=0, model_name=model_config.openai_chat_model, max_tokens=4000)
    score = evaluate(dataset, metrics=metrics, llm=llm)
    results_df = score.to_pandas()
    analyse_metric_results(results_df)

    return EvaluationResult(
        generated_answers=generated_answers,
        retrieved_documents=retrieved_documents,
        results_df=results_df,
    )
