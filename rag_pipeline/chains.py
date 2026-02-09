"""LLM chain factories for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from .config import ModelConfig
from .prompts import (
    ANONYMIZE_QUESTION_TEMPLATE,
    BREAK_DOWN_PLAN_TEMPLATE,
    CAN_BE_ANSWERED_ALREADY_TEMPLATE,
    CAN_BE_ANSWERED_TEMPLATE,
    DEANONYMIZE_PLAN_TEMPLATE,
    IS_DISTILLED_GROUNDED_TEMPLATE,
    IS_GROUNDED_ON_FACTS_TEMPLATE,
    IS_RELEVANT_CONTENT_TEMPLATE,
    KEEP_ONLY_RELEVANT_CONTENT_TEMPLATE,
    PLANNER_TEMPLATE,
    QUESTION_ANSWER_COT_TEMPLATE,
    REPLANNER_TEMPLATE,
    REWRITE_QUESTION_TEMPLATE,
    TASK_HANDLER_TEMPLATE,
    build_prompt,
)


class KeepRelevantContent(BaseModel):
    relevant_content: str = Field(
        description="The relevant content from the retrieved documents that is relevant to the query."
    )


class RewriteQuestion(BaseModel):
    rewritten_question: str = Field(
        description="The improved question optimized for vectorstore retrieval."
    )
    explanation: str = Field(
        description="The explanation of the rewritten question."
    )


class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(
        description="Generates an answer to a query based on a given context."
    )


class Relevance(BaseModel):
    is_relevant: bool = Field(description="Whether the document is relevant to the query.")
    explanation: str = Field(description="An explanation of why the document is relevant or not.")


class IsGroundedOnFacts(BaseModel):
    grounded_on_facts: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


class QuestionAnswer(BaseModel):
    can_be_answered: bool = Field(
        description="binary result of whether the question can be fully answered or not"
    )
    explanation: str = Field(
        description="An explanation of why the question can be fully answered or not."
    )


class IsDistilledContentGroundedOnContent(BaseModel):
    grounded: bool = Field(
        description="Whether the distilled content is grounded on the original context."
    )
    explanation: str = Field(
        description="An explanation of why the distilled content is or is not grounded on the original context."
    )


class Plan(BaseModel):
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class ActPossibleResults(BaseModel):
    plan: Plan = Field(description="Plan to follow in future.")
    explanation: str = Field(description="Explanation of the action.")


class TaskHandlerOutput(BaseModel):
    query: str = Field(
        description="The query to be either retrieved from the vector store, or the question that should be answered from context."
    )
    curr_context: str = Field(
        description="The context to be based on in order to answer the query."
    )
    tool: str = Field(
        description=(
            "The tool to be used should be either retrieve_chunks, retrieve_summaries, "
            "retrieve_quotes, or answer_from_context."
        )
    )


class AnonymizeQuestion(BaseModel):
    anonymized_question: str = Field(description="Anonymized question.")
    mapping: dict = Field(description="Mapping of original name entities to variables.")
    explanation: str = Field(description="Explanation of the action.")


class DeAnonymizePlan(BaseModel):
    plan: List = Field(
        description="Plan to follow in future. with all the variables replaced with the mapped words."
    )


class CanBeAnsweredAlready(BaseModel):
    can_be_answered: bool = Field(
        description="Whether the question can be fully answered or not based on the given context."
    )


@dataclass(frozen=True)
class CoreChains:
    keep_only_relevant_content_chain: object
    question_rewriter: object
    question_answer_chain: object
    is_relevant_content_chain: object
    is_grounded_on_facts_chain: object
    can_be_answered_chain: object
    is_distilled_content_grounded_chain: object
    planner: object
    break_down_plan_chain: object
    replanner: object
    task_handler_chain: object
    anonymize_question_chain: object
    de_anonymize_plan_chain: object
    can_be_answered_already_chain: object


class ChainFactory:
    def __init__(self, model_config: ModelConfig, groq_api_key: str | None = None) -> None:
        self.model_config = model_config
        self.groq_api_key = groq_api_key

    def build(self) -> CoreChains:
        keep_prompt = build_prompt(
            KEEP_ONLY_RELEVANT_CONTENT_TEMPLATE,
            ["query", "retrieved_documents"],
        )
        keep_llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_config.openai_chat_model,
            max_tokens=2000,
        )
        keep_chain = keep_prompt | keep_llm.with_structured_output(KeepRelevantContent)

        rewrite_parser = JsonOutputParser(pydantic_object=RewriteQuestion)
        rewrite_prompt = build_prompt(
            REWRITE_QUESTION_TEMPLATE,
            ["question"],
            partial_variables={"format_instructions": rewrite_parser.get_format_instructions()},
        )
        rewrite_llm = ChatGroq(
            temperature=0,
            model_name=self.model_config.groq_chat_model,
            groq_api_key=self.groq_api_key,
            max_tokens=4000,
        )
        question_rewriter = rewrite_prompt | rewrite_llm | rewrite_parser

        answer_prompt = build_prompt(QUESTION_ANSWER_COT_TEMPLATE, ["context", "question"])
        answer_llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_config.openai_chat_model,
            max_tokens=2000,
        )
        question_answer_chain = answer_prompt | answer_llm.with_structured_output(QuestionAnswerFromContext)

        is_relevant_parser = JsonOutputParser(pydantic_object=Relevance)
        is_relevant_prompt = build_prompt(
            IS_RELEVANT_CONTENT_TEMPLATE,
            ["query", "context"],
            partial_variables={"format_instructions": is_relevant_parser.get_format_instructions()},
        )
        is_relevant_llm = ChatGroq(
            temperature=0,
            model_name=self.model_config.groq_chat_model,
            groq_api_key=self.groq_api_key,
            max_tokens=4000,
        )
        is_relevant_content_chain = is_relevant_prompt | is_relevant_llm | is_relevant_parser

        is_grounded_prompt = build_prompt(IS_GROUNDED_ON_FACTS_TEMPLATE, ["context", "answer"])
        is_grounded_llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_config.openai_chat_model,
            max_tokens=2000,
        )
        is_grounded_chain = is_grounded_prompt | is_grounded_llm.with_structured_output(IsGroundedOnFacts)

        can_be_answered_parser = JsonOutputParser(pydantic_object=QuestionAnswer)
        can_be_answered_prompt = build_prompt(
            CAN_BE_ANSWERED_TEMPLATE,
            ["question", "context"],
            partial_variables={"format_instructions": can_be_answered_parser.get_format_instructions()},
        )
        can_be_answered_llm = ChatGroq(
            temperature=0,
            model_name=self.model_config.groq_chat_model,
            groq_api_key=self.groq_api_key,
            max_tokens=4000,
        )
        can_be_answered_chain = can_be_answered_prompt | can_be_answered_llm | can_be_answered_parser

        distilled_parser = JsonOutputParser(pydantic_object=IsDistilledContentGroundedOnContent)
        distilled_prompt = build_prompt(
            IS_DISTILLED_GROUNDED_TEMPLATE,
            ["distilled_content", "original_context"],
            partial_variables={"format_instructions": distilled_parser.get_format_instructions()},
        )
        distilled_llm = ChatGroq(
            temperature=0,
            model_name=self.model_config.groq_chat_model,
            groq_api_key=self.groq_api_key,
            max_tokens=4000,
        )
        distilled_chain = distilled_prompt | distilled_llm | distilled_parser

        planner_prompt = build_prompt(PLANNER_TEMPLATE, ["question"])
        planner_llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_config.openai_chat_model,
            max_tokens=2000,
        )
        planner = planner_prompt | planner_llm.with_structured_output(Plan)

        break_down_prompt = build_prompt(BREAK_DOWN_PLAN_TEMPLATE, ["plan"])
        break_down_llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_config.openai_chat_model,
            max_tokens=2000,
        )
        break_down_plan_chain = break_down_prompt | break_down_llm.with_structured_output(Plan)

        act_parser = JsonOutputParser(pydantic_object=ActPossibleResults)
        replanner_prompt = build_prompt(
            REPLANNER_TEMPLATE,
            ["question", "plan", "past_steps", "aggregated_context"],
            partial_variables={"format_instructions": act_parser.get_format_instructions()},
        )
        replanner_llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_config.openai_chat_model,
            max_tokens=2000,
        )
        replanner = replanner_prompt | replanner_llm | act_parser

        task_handler_prompt = build_prompt(
            TASK_HANDLER_TEMPLATE,
            ["curr_task", "aggregated_context", "last_tool", "past_steps", "question"],
        )
        task_handler_llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_config.openai_chat_model,
            max_tokens=2000,
        )
        task_handler_chain = task_handler_prompt | task_handler_llm.with_structured_output(TaskHandlerOutput)

        anonymize_parser = JsonOutputParser(pydantic_object=AnonymizeQuestion)
        anonymize_prompt = build_prompt(
            ANONYMIZE_QUESTION_TEMPLATE,
            ["question"],
            partial_variables={"format_instructions": anonymize_parser.get_format_instructions()},
        )
        anonymize_llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_config.openai_chat_model,
            max_tokens=2000,
        )
        anonymize_question_chain = anonymize_prompt | anonymize_llm | anonymize_parser

        de_anonymize_prompt = build_prompt(DEANONYMIZE_PLAN_TEMPLATE, ["plan", "mapping"])
        de_anonymize_llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_config.openai_chat_model,
            max_tokens=2000,
        )
        de_anonymize_plan_chain = de_anonymize_prompt | de_anonymize_llm.with_structured_output(DeAnonymizePlan)

        can_be_answered_already_prompt = build_prompt(
            CAN_BE_ANSWERED_ALREADY_TEMPLATE,
            ["question", "context"],
        )
        can_be_answered_already_llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_config.openai_chat_model,
            max_tokens=2000,
        )
        can_be_answered_already_chain = (
            can_be_answered_already_prompt
            | can_be_answered_already_llm.with_structured_output(CanBeAnsweredAlready)
        )

        return CoreChains(
            keep_only_relevant_content_chain=keep_chain,
            question_rewriter=question_rewriter,
            question_answer_chain=question_answer_chain,
            is_relevant_content_chain=is_relevant_content_chain,
            is_grounded_on_facts_chain=is_grounded_chain,
            can_be_answered_chain=can_be_answered_chain,
            is_distilled_content_grounded_chain=distilled_chain,
            planner=planner,
            break_down_plan_chain=break_down_plan_chain,
            replanner=replanner,
            task_handler_chain=task_handler_chain,
            anonymize_question_chain=anonymize_question_chain,
            de_anonymize_plan_chain=de_anonymize_plan_chain,
            can_be_answered_already_chain=can_be_answered_already_chain,
        )
