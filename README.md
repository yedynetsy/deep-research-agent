---
title: Deep Research Agent
emoji: 🔎
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
---

# Deep Research Agent

A Python research agent that searches the web, reads useful pages, and writes a Markdown report with source URLs. The language model decides which tool to call next. The Gradio page shows tool calls while the agent works and then displays its report; the terminal interface is also available.

## Tools

| Tool | Purpose |
| --- | --- |
| `search_web(query, max_results)` | Finds sources with Tavily. |
| `read_webpage(url)` | Reads public HTML pages. |
| `get_research_sources()` | Lists unique sources collected during the current run. |
| `save_report(report)` | Saves the final report as a Markdown file in `reports/`. |

The agent can make up to eight model steps and twelve tool calls per question. If it reaches the step limit, it asks the model for a final answer using the evidence already collected.

## Run locally

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your own Groq and Tavily API keys in `.env`. The file is ignored by Git. Then start either interface:

```bash
python app.py   # Browser UI at http://127.0.0.1:7860
python main.py  # Terminal UI
```

To run the local tests:

```bash
python -m unittest discover -s tests -v
```

The tests use fake API responses. They do not spend Groq or Tavily credits.

## Deploy on Hugging Face Spaces

Create a Gradio Space and upload the project files. `README.md` contains the Space configuration, and `app.py` is the entry point. Add `GROQ_API_KEY` and `TAVILY_API_KEY` in the Space settings as **Secrets**. Do not upload `.env`, `.venv`, or generated reports.

A public Space using your keys can consume your API quotas. Review the quotas before sharing its URL.

## Project files

- `app.py` — Gradio interface.
- `main.py` — terminal interface.
- `agent/core.py` — agent loop and tool execution.
- `agent/tools.py` — tool schemas and function registry.
- `tools/` — web search, webpage reader, source list, and report saving.
- `tests/` — local tests with fake API responses.

## Limitations

The model can make mistakes when summarizing sources. Check important claims and URLs in its report before relying on them. Webpage text is limited to keep model requests within API limits. Reports are stored locally under `reports/`; this directory is intentionally not committed to Git.
