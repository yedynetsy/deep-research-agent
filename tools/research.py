from contextvars import ContextVar

_sources: ContextVar[dict[str, dict] | None] = ContextVar("research_sources", default=None)


def start_research():
    """Create an isolated source collection for this run."""
    return _sources.set({})


def finish_research(token) -> None:
    _sources.reset(token)


def track_sources(tool_name: str, result) -> None:
    """Record only sources successfully returned to the agent."""
    sources = _sources.get()
    if sources is None:
        return

    items = result if tool_name == "search_web" else [result]
    for item in items:
        url = item.get("url")
        if not url:
            continue
        previous = sources.get(url, {})
        sources[url] = {
            "title": item.get("title") or previous.get("title", ""),
            "url": url,
            "read": tool_name == "read_webpage" or previous.get("read", False),
        }


def get_research_sources() -> list[dict]:
    """Return unique source titles and URLs collected in the current run."""
    return list((_sources.get() or {}).values())
