"""Gradio interface for the research agent."""

import os
from queue import Queue
from threading import Thread

import gradio as gr
from groq import APIStatusError, RateLimitError

from agent.core import run_agent


def research(query: str):
    """Show tool calls as they happen, followed by the final report."""

    if not query.strip():
        yield "", "Please enter a research question."
        return

    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        yield "", "The Groq and Tavily API keys must be configured before starting research."
        return

    events = Queue()

    def run():
        try:
            report = run_agent(query, on_event=lambda message: events.put(("log", message)))
            events.put(("report", report))

        except RateLimitError:
            events.put(("error", "Groq rate limit exceeded. Please try again later."))

        except APIStatusError as error:
            if error.status_code == 413:
                events.put(("error", "Groq rejected the request due to token limits. Try a narrower question."))
            else:
                events.put(("error", f"Groq API error: HTTP {error.status_code}"))

        except Exception as error:
            events.put(("error", f"Research failed: {type(error).__name__}"))

    Thread(target=run, daemon=True).start()
    activity = []

    while True:
        event_type, message = events.get()

        if event_type == "log":
            activity.append(message)
            yield "\n".join(activity), ""

        else:
            yield "\n".join(activity), message
            return


app = gr.Interface(
    fn=research,
    inputs=gr.Textbox(
        label="Research question",
        placeholder="What are the main methods for reducing hallucinations in RAG systems?",
        lines=3,
    ),
    outputs=[
        gr.Textbox(label="Agent activity", lines=12, interactive=False),
        gr.Markdown(label="Research report"),
    ],
    title="Deep Research Agent",
    description=(
        "Ask a research question. The agent searches the web, reads sources, "
        "and creates a report with source URLs. Tool calls appear below as they happen."
    ),
    flagging_mode="never",
    analytics_enabled=False,
)


if __name__ == "__main__":
    app.launch()
