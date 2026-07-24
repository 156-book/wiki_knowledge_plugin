from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from test_internal_llm import test_model


def text_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))]
    )


def tool_response(call_id, name, arguments):
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))]
    )


class FakeCompletions:
    def __init__(self):
        self.responses = [
            text_response("LLM_BASIC_OK"),
            tool_response("call-search", "search_knowledge_base", {"search_key": "设备连接"}),
            tool_response(
                "call-read",
                "read_knowledge_document",
                {"url": "https://wiki.example.invalid/test"},
            ),
            text_response("根据测试资料，应先检查连接线。[1]"),
        ]

    def create(self, **kwargs):
        return self.responses.pop(0)


class InternalLLMDiagnosticTests(unittest.TestCase):
    def test_diagnostic_accepts_search_read_and_final_answer_sequence(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )

        result = test_model(client, "deepseek-v3.1-terminus-chat", "test-key")

        self.assertTrue(result.passed)
        self.assertTrue(result.basic_chat.success)
        self.assertTrue(result.tool_call.success)
        self.assertTrue(result.tool_round_trip.success)


if __name__ == "__main__":
    unittest.main()
