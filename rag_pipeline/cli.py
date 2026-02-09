"""Command-line entry point for the RAG pipeline."""
from __future__ import annotations

import argparse
import os

from .config import PipelineConfig
from .pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the plan-and-execute RAG pipeline.")
    parser.add_argument(
        "question",
        help="The question to ask the RAG pipeline.",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    pipeline = RAGPipeline.build(config, groq_api_key=os.getenv("GROQ_API_KEY"))

    response, _ = pipeline.plan_execute_workflow.execute_plan_and_print_steps(
        {"question": args.question}
    )
    print(response)


if __name__ == "__main__":
    main()
