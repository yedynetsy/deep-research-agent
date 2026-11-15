"""Gradio interface for the research agent."""

import os

import gradio as gr
from groq import APIStatusError, RateLimitError

from agent.core import run_agent


def research(query: str) -> str:
    """Run a research question and show the report in the browser."""

    if not query.strip():
        return "Please enter a research question."

    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        return "The Groq and Tavily API keys must be configured before starting research."

    try:
        return run_agent(query)

    except RateLimitError:
        return "Groq rate limit exceeded. Please try again later."

    except APIStatusError as error:
        if error.status_code == 413:
            return "Groq rejected the request due to token limits. Try a narrower question."
        return f"Groq API error: HTTP {error.status_code}"


app = gr.Interface(
    fn=research,
    inputs=gr.Textbox(
        label="Research question",
        placeholder="What are the main methods for reducing hallucinations in RAG systems?",
        lines=3,
    ),
    outputs=gr.Markdown(label="Research report"),
    title="Deep Research Agent",
    description=(
        "Ask a research question. The agent searches the web, reads sources, "
        "and creates a report with source URLs. Research may take a minute."
    ),
    flagging_mode="never",
    analytics_enabled=False,
)


if __name__ == "__main__":
    app.launch()
