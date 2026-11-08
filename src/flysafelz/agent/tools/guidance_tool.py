"""search_guidance (G13): the passages of published guidance (CAP 1535, CAP 2562, ...) that best answer a question.

    build_search_guidance(index)  -> the tool over this guidance index (guidance/index.py)

The model's only argument is `query`. The place, the time zone and everything else are the thread's and are not arguments (Q8, F2). The tool returns the
top `top_k` chunks (settings.yaml), best first, each as one heading line `<document> <edition> §<section> <title>` followed by its text, and an artifact with
the same fields as data, so the guard can treat the section numbers printed here as sources (FR6).

Guidance explains terms and rules; it is not a reading of today's conditions. The tools give facts and `assess` gives the verdict.

Nothing is loaded when this module is imported or the tool is built: the index builds the embedder at the first search, once (about a second). The text
goes through the same redaction boundary as every other tool's (P1). A problem the model should relay is text, never an exception: the index is missing or
unreadable ("Guidance index is not available"), the embedding model is not stored (`make setup`), or the query is empty.
"""

import logging

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from flysafelz.agent.settings import SETTINGS
from flysafelz.agent.tools.artifacts import GuidanceArtifact, GuidanceChunk
from flysafelz.agent.tools.common import Redactor, reply
from flysafelz.guidance.corpus import IndexedChunk
from flysafelz.guidance.embedder import EmbeddingModelMissing
from flysafelz.guidance.index import GuidanceIndex

log = logging.getLogger(__name__)

NOT_AVAILABLE = "Guidance index is not available"
EMPTY_QUERY = "The guidance search needs a question or some key words. Say what to look up and search again."
NOTHING_FOUND = "No guidance passage was found for that query."
INTRO = "Passages from published guidance, best match first:"

DESCRIPTION = (
    "Search the published guidance documents (the CAA's Skyway Code and Paramotor Code, and other guidance on airspace, NOTAMs and danger areas) for the passages "
    "that best answer a question, and return them with their document, edition and section number. Use it to explain a term or a rule, and to cite where it is "
    "written. It says nothing about today's weather, airspace or NOTAMs: those come from the other tools. Give a short question or key words."
)


class GuidanceArgs(BaseModel):
    """The only argument of search_guidance is the query. `runtime` is filled in by LangGraph and is not part of the schema the model is shown."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    query: str = Field(description="What to look up in the guidance: a short question or some key words, such as 'what is a danger area'.")
    runtime: ToolRuntime = Field(default=None)  # type: ignore[assignment]  # injected; never part of the model-facing schema


def _heading(chunk: IndexedChunk) -> str:
    return f"{chunk.doc} {chunk.edition} §{chunk.section} {chunk.title}"


def guidance_answer(query: str, found: list[IndexedChunk]) -> tuple[str, GuidanceArtifact]:
    """The text and the artifact for the chunks that came back (best first). Pure."""
    artifact = GuidanceArtifact(
        query=query,
        chunks=[GuidanceChunk(document=c.doc, edition=c.edition, section=c.section, title=c.title, text=c.text) for c in found],
    )
    if not found:
        return NOTHING_FOUND, artifact
    blocks = [f"{_heading(chunk)}\n{chunk.text}" for chunk in found]
    return "\n\n".join([INTRO, *blocks]), artifact


def build_search_guidance(index: GuidanceIndex) -> BaseTool:
    """The search_guidance tool over this index."""

    @tool("search_guidance", args_schema=GuidanceArgs, description=DESCRIPTION)
    def search_guidance(runtime: ToolRuntime, query: str) -> Command:
        redactor = Redactor(runtime.state.get("phone_map"))
        if not index.available:
            return reply(runtime, NOT_AVAILABLE, GuidanceArtifact(query=query, chunks=[]), redactor)
        if not query.strip():
            return reply(runtime, EMPTY_QUERY, GuidanceArtifact(query=query, chunks=[]), redactor)
        try:
            found = index.search(query, SETTINGS.top_k)
        except EmbeddingModelMissing as error:
            log.error("Guidance search cannot run: %s", error)
            return reply(runtime, f"{NOT_AVAILABLE}: {error}.", GuidanceArtifact(query=query, chunks=[]), redactor)
        text, artifact = guidance_answer(query, found)
        return reply(runtime, text, artifact, redactor)

    return search_guidance
