"""Pipeline assembly for the refactored RAG workflow."""
from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv

from .chains import ChainFactory, CoreChains
from .config import PipelineConfig
from .data import build_book_artifacts
from .operations import PipelineOps
from .summarization import ChapterSummarizer
from .vectorstores import VectorStores, build_retrievers, load_or_build_vectorstores
from .workflows.plan_execute import PlanExecuteWorkflow
from .workflows.qualitative import QualitativeWorkflows


@dataclass
class RAGPipeline:
    config: PipelineConfig
    chains: CoreChains
    vectorstores: VectorStores
    ops: PipelineOps
    qualitative_workflows: QualitativeWorkflows
    plan_execute_workflow: PlanExecuteWorkflow

    @classmethod
    def build(cls, config: PipelineConfig, groq_api_key: str | None = None) -> "RAGPipeline":
        load_dotenv(override=True)
        artifacts = build_book_artifacts(config.book_path)

        summarizer = ChapterSummarizer(config.models)
        chapter_summaries = summarizer.summarize_chapters(artifacts.chapters)

        vectorstores = load_or_build_vectorstores(
            book_path=config.book_path,
            chapter_summaries=chapter_summaries,
            book_quotes=artifacts.quotes,
            paths=config.vectorstores,
        )
        retrievers = build_retrievers(vectorstores, config.retrieval)

        chains = ChainFactory(config.models, groq_api_key=groq_api_key).build()
        ops = PipelineOps(retrievers=retrievers, chains=chains)
        qualitative_workflows = QualitativeWorkflows(ops)
        plan_execute_workflow = PlanExecuteWorkflow(ops, chains, qualitative_workflows)

        return cls(
            config=config,
            chains=chains,
            vectorstores=vectorstores,
            ops=ops,
            qualitative_workflows=qualitative_workflows,
            plan_execute_workflow=plan_execute_workflow,
        )
