import os

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return relevant sources."""

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY is missing")

    if not query.strip():
        raise ValueError("Search query cannot be empty")

    max_results = max(1, min(max_results, 10))

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_answer=False,
    )

    sources = []

    for result in response.get("results", []):
        sources.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
        })

    return sources
