"""Tool-calling chat, tested against a local fake LLM server (no internet, no cost).

The fake server plays the model: first it asks to call analyze_etf, then it
answers using the tool result. We check both provider formats end to end.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from maple_etf.chat import ChatSession, RuleBasedChat, AgentTools
from maple_etf.data import SampleDataProvider
from maple_etf.llm import LLMClient
from maple_etf.models import InvestorProfile


@pytest.fixture
def provider(catalog):
    return SampleDataProvider({t: catalog.get(t).category for t in catalog.tickers()})


def make_server(final_text: str):
    state = {"requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(body)
            anthropic = self.path.endswith("/messages")
            first = len(state["requests"]) == 1
            if anthropic:
                if first:
                    reply = {"content": [{"type": "tool_use", "id": "tu_1", "name": "analyze_etf",
                                          "input": {"ticker": "VOO"}}], "stop_reason": "tool_use",
                             "usage": {"input_tokens": 10, "output_tokens": 5}}
                else:
                    reply = {"content": [{"type": "text", "text": final_text}], "stop_reason": "end_turn",
                             "usage": {"input_tokens": 20, "output_tokens": 8}}
            else:
                if first:
                    msg = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "analyze_etf", "arguments": json.dumps({"ticker": "VOO"})}}]}
                else:
                    msg = {"role": "assistant", "content": final_text}
                reply = {"choices": [{"message": msg}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}", state


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_tool_calling_round_trip(kind, catalog, provider):
    httpd, url, state = make_server("VOO is HOLD for your TFSA because about 15% of US dividends are withheld.")
    try:
        llm = LLMClient(kind, "fake-model", "k", url)
        reply = ChatSession(catalog, provider, InvestorProfile(), llm=llm).ask("Why HOLD for VOO?")
    finally:
        httpd.shutdown()

    assert reply.source == "llm"
    assert reply.tool_calls == [{"name": "analyze_etf", "arguments": {"ticker": "VOO"}}]
    assert "withheld" in reply.text
    second = state["requests"][1]
    if kind == "anthropic":
        tool_msg = second["messages"][-1]["content"][0]
        assert tool_msg["type"] == "tool_result" and tool_msg["tool_use_id"] == "tu_1"
        assert second["tools"][0]["input_schema"]["type"] == "object"
    else:
        tool_msg = second["messages"][-1]
        assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"
        assert second["tools"][0]["type"] == "function"
    assert '"decision"' in tool_msg["content"]


def test_invented_numbers_are_rejected(catalog, provider):
    httpd, url, _ = make_server("VOO returned 42.7% and the MER is 0.5%.")
    try:
        reply = ChatSession(catalog, provider, InvestorProfile(), llm=LLMClient("openai", "m", "k", url)).ask("VOO?")
    finally:
        httpd.shutdown()
    assert reply.source == "llm_rejected"
    assert 42.7 in reply.trace["ungrounded_numbers"]
    assert "rules-based answer" in reply.text


def test_llm_down_falls_back_to_rules(catalog, provider):
    llm = LLMClient("openai", "m", "k", "http://127.0.0.1:9")  # nothing listens here
    llm.timeout = 2
    reply = ChatSession(catalog, provider, InvestorProfile(), llm=llm).ask("Why HOLD for VOO?")
    assert reply.source == "llm_error"
    assert "VOO" in reply.text


def test_rules_chat_routes(catalog, provider):
    chat = RuleBasedChat(AgentTools(catalog, provider, InvestorProfile()))
    assert chat.answer("Why HOLD for voo?").tool_calls[0]["name"] == "analyze_etf"
    assert chat.answer("xeqt or veqt?").tool_calls[0]["name"] == "compare_etfs"
    assert chat.answer("XEQT 60% ZAG 40%").tool_calls[0]["name"] == "check_portfolio"
    assert "XEQT.TO" in chat.answer("list ETFs").text


def test_tools_return_errors_not_exceptions(catalog, provider):
    tools = AgentTools(catalog, provider, InvestorProfile())
    assert "error" in json.loads(tools.run("analyze_etf", {"ticker": "AAPL"}))
    assert "error" in json.loads(tools.run("no_such_tool", {}))
