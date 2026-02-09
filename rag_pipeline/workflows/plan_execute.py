"""Plan-and-execute workflow for complex question answering."""
from __future__ import annotations

from typing import Dict, List, TypedDict

import langgraph
from langgraph.graph import END, StateGraph

from ..chains import CoreChains
from ..operations import PipelineOps
from .qualitative import QualitativeWorkflows


class PlanExecute(TypedDict):
    curr_state: str
    question: str
    anonymized_question: str
    query_to_retrieve_or_answer: str
    plan: List[str]
    past_steps: List[str]
    mapping: Dict[str, str]
    curr_context: str
    aggregated_context: str
    tool: str
    response: str


class PlanExecuteWorkflow:
    def __init__(
        self,
        ops: PipelineOps,
        chains: CoreChains,
        qualitative_workflows: QualitativeWorkflows,
    ) -> None:
        self.ops = ops
        self.chains = chains
        self.qualitative_workflows = qualitative_workflows
        self.app = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(PlanExecute)

        workflow.add_node("anonymize_question", self.anonymize_queries)
        workflow.add_node("planner", self.plan_step)
        workflow.add_node("de_anonymize_plan", self.deanonymize_queries)
        workflow.add_node("break_down_plan", self.break_down_plan_step)
        workflow.add_node("task_handler", self.run_task_handler_chain)
        workflow.add_node("retrieve_chunks", self.run_qualitative_chunks_retrieval_workflow)
        workflow.add_node("retrieve_summaries", self.run_qualitative_summaries_retrieval_workflow)
        workflow.add_node("retrieve_book_quotes", self.run_qualitative_book_quotes_retrieval_workflow)
        workflow.add_node("answer", self.run_qualtative_answer_workflow)
        workflow.add_node("replan", self.replan_step)
        workflow.add_node("get_final_answer", self.run_qualtative_answer_workflow_for_final_answer)

        workflow.set_entry_point("anonymize_question")
        workflow.add_edge("anonymize_question", "planner")
        workflow.add_edge("planner", "de_anonymize_plan")
        workflow.add_edge("de_anonymize_plan", "break_down_plan")
        workflow.add_edge("break_down_plan", "task_handler")

        workflow.add_conditional_edges(
            "task_handler",
            self.retrieve_or_answer,
            {
                "chosen_tool_is_retrieve_chunks": "retrieve_chunks",
                "chosen_tool_is_retrieve_summaries": "retrieve_summaries",
                "chosen_tool_is_retrieve_quotes": "retrieve_book_quotes",
                "chosen_tool_is_answer": "answer",
            },
        )

        workflow.add_edge("retrieve_chunks", "replan")
        workflow.add_edge("retrieve_summaries", "replan")
        workflow.add_edge("retrieve_book_quotes", "replan")
        workflow.add_edge("answer", "replan")

        workflow.add_conditional_edges(
            "replan",
            self.can_be_answered,
            {
                "can_be_answered_already": "get_final_answer",
                "cannot_be_answered_yet": "break_down_plan",
            },
        )

        workflow.add_edge("get_final_answer", END)

        return workflow.compile()

    def run_task_handler_chain(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "task_handler"

        if not state.get("past_steps"):
            state["past_steps"] = []

        curr_task = state["plan"][0]
        inputs = {
            "curr_task": curr_task,
            "aggregated_context": state["aggregated_context"],
            "last_tool": state["tool"],
            "past_steps": state["past_steps"],
            "question": state["question"],
        }

        output = self.chains.task_handler_chain.invoke(inputs)
        state["past_steps"].append(curr_task)
        state["plan"].pop(0)

        if output.tool == "retrieve_chunks":
            state["query_to_retrieve_or_answer"] = output.query
            state["tool"] = "retrieve_chunks"
        elif output.tool == "retrieve_summaries":
            state["query_to_retrieve_or_answer"] = output.query
            state["tool"] = "retrieve_summaries"
        elif output.tool == "retrieve_quotes":
            state["query_to_retrieve_or_answer"] = output.query
            state["tool"] = "retrieve_quotes"
        elif output.tool == "answer_from_context":
            state["query_to_retrieve_or_answer"] = output.query
            state["curr_context"] = output.curr_context
            state["tool"] = "answer"
        else:
            raise ValueError(
                "Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'"
            )
        return state

    def retrieve_or_answer(self, state: PlanExecute) -> str:
        state["curr_state"] = "decide_tool"
        if state["tool"] == "retrieve_chunks":
            return "chosen_tool_is_retrieve_chunks"
        if state["tool"] == "retrieve_summaries":
            return "chosen_tool_is_retrieve_summaries"
        if state["tool"] == "retrieve_quotes":
            return "chosen_tool_is_retrieve_quotes"
        if state["tool"] == "answer":
            return "chosen_tool_is_answer"
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")

    def run_qualitative_chunks_retrieval_workflow(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "retrieve_chunks"
        question = state["query_to_retrieve_or_answer"]
        inputs = {"question": question}
        for output in self.qualitative_workflows.chunks_retrieval_app.stream(inputs):
            for _, _ in output.items():
                pass
        if not state["aggregated_context"]:
            state["aggregated_context"] = ""
        state["aggregated_context"] += output["relevant_context"]
        return state

    def run_qualitative_summaries_retrieval_workflow(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "retrieve_summaries"
        question = state["query_to_retrieve_or_answer"]
        inputs = {"question": question}
        for output in self.qualitative_workflows.summaries_retrieval_app.stream(inputs):
            for _, _ in output.items():
                pass
        if not state["aggregated_context"]:
            state["aggregated_context"] = ""
        state["aggregated_context"] += output["relevant_context"]
        return state

    def run_qualitative_book_quotes_retrieval_workflow(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "retrieve_book_quotes"
        question = state["query_to_retrieve_or_answer"]
        inputs = {"question": question}
        for output in self.qualitative_workflows.quotes_retrieval_app.stream(inputs):
            for _, _ in output.items():
                pass
        if not state["aggregated_context"]:
            state["aggregated_context"] = ""
        state["aggregated_context"] += output["relevant_context"]
        return state

    def run_qualtative_answer_workflow(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "answer"
        question = state["query_to_retrieve_or_answer"]
        context = state["curr_context"]
        inputs = {"question": question, "context": context}
        for output in self.qualitative_workflows.answer_app.stream(inputs):
            for _, _ in output.items():
                pass
        if not state["aggregated_context"]:
            state["aggregated_context"] = ""
        state["aggregated_context"] += output["answer"]
        return state

    def run_qualtative_answer_workflow_for_final_answer(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "get_final_answer"
        question = state["question"]
        context = state["aggregated_context"]
        inputs = {"question": question, "context": context}
        for output in self.qualitative_workflows.answer_app.stream(inputs):
            for _, value in output.items():
                pass
        state["response"] = value
        return state

    def anonymize_queries(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "anonymize_question"
        anonymized_question_output = self.chains.anonymize_question_chain.invoke(
            {"question": state["question"]}
        )
        state["anonymized_question"] = anonymized_question_output["anonymized_question"]
        state["mapping"] = anonymized_question_output["mapping"]
        return state

    def deanonymize_queries(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "de_anonymize_plan"
        deanonimzed_plan = self.chains.de_anonymize_plan_chain.invoke(
            {"plan": state["plan"], "mapping": state["mapping"]}
        )
        state["plan"] = deanonimzed_plan.plan
        return state

    def plan_step(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "planner"
        plan = self.chains.planner.invoke({"question": state["anonymized_question"]})
        state["plan"] = plan.steps
        return state

    def break_down_plan_step(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "break_down_plan"
        refined_plan = self.chains.break_down_plan_chain.invoke({"plan": state["plan"]})
        state["plan"] = refined_plan.steps
        return state

    def replan_step(self, state: PlanExecute) -> PlanExecute:
        state["curr_state"] = "replan"
        inputs = {
            "question": state["question"],
            "plan": state["plan"],
            "past_steps": state["past_steps"],
            "aggregated_context": state["aggregated_context"],
        }
        output = self.chains.replanner.invoke(inputs)
        state["plan"] = output["plan"]["steps"]
        return state

    def can_be_answered(self, state: PlanExecute) -> str:
        state["curr_state"] = "can_be_answered_already"
        question = state["question"]
        context = state["aggregated_context"]
        output = self.chains.can_be_answered_already_chain.invoke(
            {"question": question, "context": context}
        )
        if output.can_be_answered:
            return "can_be_answered_already"
        return "cannot_be_answered_yet"

    def execute_plan_and_print_steps(self, inputs: Dict[str, str], recursion_limit: int = 45):
        config = {"recursion_limit": recursion_limit}
        try:
            for plan_output in self.app.stream(inputs, config=config):
                for _, agent_state_value in plan_output.items():
                    pass
            response = agent_state_value["response"]
        except langgraph.pregel.GraphRecursionError:
            response = "The answer wasn't found in the data."
        final_state = agent_state_value
        return response, final_state
