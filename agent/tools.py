from tools.search import search_web
from tools.reader import read_webpage
from tools.research import get_research_sources
from tools.report import save_report


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet for information relevant "
                "to a research question. Returns source titles, "
                "URLs, and content snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results (1-10)."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": (
                "Read a public HTML webpage and extract its "
                "title and text content. Use it to examine "
                "sources found through web search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full HTTP or HTTPS URL."
                    }
                },
                "required": ["url"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_research_sources",
            "description": "List unique sources found in this research run and show which pages were read.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": "Save the complete final research report as a Markdown file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "report": {"type": "string", "description": "Complete report with source URLs."},
                },
                "required": ["report"],
                "additionalProperties": False,
            },
        },
    }
]


AVAILABLE_TOOLS = {
    "search_web": search_web,
    "read_webpage": read_webpage,
    "get_research_sources": get_research_sources,
    "save_report": save_report,
}
