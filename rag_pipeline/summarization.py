"""Summarization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI

from helper_functions import num_tokens_from_string, replace_double_lines_with_one_line
from .config import ModelConfig
from .prompts import SUMMARY_TEMPLATE, build_prompt


@dataclass(frozen=True)
class SummarizationConfig:
    max_tokens: int = 16000
    verbose: bool = False


class ChapterSummarizer:
    def __init__(
        self,
        model_config: ModelConfig,
        summarization_config: SummarizationConfig | None = None,
    ) -> None:
        self.model_config = model_config
        self.config = summarization_config or SummarizationConfig()
        self.prompt = build_prompt(SUMMARY_TEMPLATE, ["text"])

    def create_chapter_summary(self, chapter: Document) -> Document:
        chapter_txt = chapter.page_content
        llm = ChatOpenAI(temperature=0, model_name=self.model_config.openai_summarization_model)
        num_tokens = num_tokens_from_string(chapter_txt, self.model_config.openai_summarization_model)

        if num_tokens < self.config.max_tokens:
            chain = load_summarize_chain(
                llm,
                chain_type="stuff",
                prompt=self.prompt,
                verbose=self.config.verbose,
            )
        else:
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=self.prompt,
                combine_prompt=self.prompt,
                verbose=self.config.verbose,
            )

        summary_result = chain.invoke([Document(page_content=chapter_txt)])
        summary_text = replace_double_lines_with_one_line(summary_result["output_text"])
        return Document(page_content=summary_text, metadata=chapter.metadata)

    def summarize_chapters(self, chapters: Iterable[Document]) -> List[Document]:
        summaries = []
        for chapter in chapters:
            summaries.append(self.create_chapter_summary(chapter))
        return summaries
