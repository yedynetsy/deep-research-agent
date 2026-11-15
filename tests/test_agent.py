import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent import core
from agent.tools import AVAILABLE_TOOLS, TOOL_SCHEMAS
from tools import reader, search
from tools.research import finish_research, get_research_sources, start_research
from tools.report import save_report


class FakeResponse:
    def __init__(self, content=None, calls=None):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))]


def call(name, arguments, index):
    return SimpleNamespace(id=str(index), function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


class ToolTests(unittest.TestCase):
    def test_four_registered_tools(self):
        self.assertEqual(set(AVAILABLE_TOOLS), {"search_web", "read_webpage", "get_research_sources", "save_report"})
        self.assertEqual({item["function"]["name"] for item in TOOL_SCHEMAS}, set(AVAILABLE_TOOLS))

    def test_search_web(self):
        fake_client = SimpleNamespace(search=lambda **kwargs: {"results": [{"title": "Source", "url": "https://example.com", "content": "Evidence"}]})
        with patch.dict(search.os.environ, {"TAVILY_API_KEY": "test"}), patch.object(search, "TavilyClient", return_value=fake_client):
            self.assertEqual(search.search_web("topic")[0]["url"], "https://example.com")

    def test_read_webpage(self):
        class Response:
            headers = {"content-type": "text/html"}
            encoding = "utf-8"
            def raise_for_status(self): pass
            def iter_bytes(self): yield b"<html><title>Page</title><body>Evidence</body></html>"
            def __enter__(self): return self
            def __exit__(self, *args): pass
        class Client:
            def __init__(self, **kwargs): pass
            def stream(self, *args): return Response()
            def __enter__(self): return self
            def __exit__(self, *args): pass
        with patch.object(reader, "validate_url"), patch.object(reader.httpx, "Client", Client):
            self.assertEqual(reader.read_webpage("https://example.com")["content"], "Page Evidence")

    def test_get_research_sources_and_save_report(self):
        token = start_research()
        try:
            with patch.dict(core.AVAILABLE_TOOLS, {
                "search_web": lambda **kwargs: [{"title": "Source", "url": "https://example.com", "content": "Evidence"}],
                "read_webpage": lambda **kwargs: {"title": "Page", "url": "https://example.com", "content": "More evidence"},
            }):
                core.execute_tool("search_web", '{"query":"topic"}')
                core.execute_tool("read_webpage", '{"url":"https://example.com"}')
            result = json.loads(core.execute_tool("get_research_sources", '{}'))
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0]["read"])
            with tempfile.TemporaryDirectory() as folder, patch("tools.report.REPORTS_DIR", Path(folder)):
                saved = json.loads(core.execute_tool("save_report", '{"report":"# Report\\nSource: https://example.com"}'))
                self.assertEqual(Path(saved["path"]).read_text(), "# Report\nSource: https://example.com\n")
        finally:
            finish_research(token)
        self.assertEqual(get_research_sources(), [])

    def test_agent_loop_uses_all_four_tools(self):
        responses = iter([
            FakeResponse(calls=[call("search_web", {"query": "topic"}, 1)]),
            FakeResponse(calls=[call("read_webpage", {"url": "https://example.com"}, 2)]),
            FakeResponse(calls=[call("get_research_sources", {}, 3)]),
            FakeResponse(calls=[call("save_report", {"report": "# Report\nSource: https://example.com"}, 4)]),
        ])
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: next(responses))))
        with tempfile.TemporaryDirectory() as folder, patch.dict(core.os.environ, {"GROQ_API_KEY": "test"}), patch.object(core, "Groq", return_value=fake_client), patch("tools.report.REPORTS_DIR", Path(folder)), patch.dict(core.AVAILABLE_TOOLS, {
            "search_web": lambda **kwargs: [{"title": "Source", "url": "https://example.com", "content": "Evidence"}],
            "read_webpage": lambda **kwargs: {"title": "Page", "url": "https://example.com", "content": "More evidence"},
        }):
            report = core.run_agent("topic")
            files = list(Path(folder).glob("*.md"))
            self.assertEqual(report, "# Report\nSource: https://example.com")
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_text(), report + "\n")
        self.assertEqual(get_research_sources(), [])

    def test_agent_saves_report_when_model_skips_save_tool(self):
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: FakeResponse(content="Final report"))))
        with tempfile.TemporaryDirectory() as folder, patch.dict(core.os.environ, {"GROQ_API_KEY": "test"}), patch.object(core, "Groq", return_value=fake_client), patch("tools.report.REPORTS_DIR", Path(folder)):
            self.assertEqual(core.run_agent("topic"), "Final report")
            self.assertEqual(len(list(Path(folder).glob("*.md"))), 1)


if __name__ == "__main__":
    unittest.main()
