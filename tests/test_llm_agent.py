from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from llm_client import InternalLLMClient
from settings import LLMSettings


class FakeCompletions:
    def __init__(self):
        self.calls = []
        self.round = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.round += 1
        if self.round == 1:
            tool_call = SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="search_knowledge_base",
                    arguments=json.dumps({"search_key": "设备连接"}),
                ),
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=[tool_call])
                    )
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="根据资料，先检查设备连接。[1]", tool_calls=[])
                )
            ]
        )


class FakeOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class LLMToolLoopTests(unittest.TestCase):
    def test_model_can_call_search_tool_before_final_answer(self):
        client = InternalLLMClient(
            LLMSettings(
                base_url="http://llm/v1", api_key="test", model="deepseek-v3.1-terminus-chat"
            )
        )
        fake_openai = FakeOpenAI()
        client._client = fake_openai
        calls = []

        reply = client.answer_with_tools(
            "设备连接失败怎么办？",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "description": "搜索知识库",
                        "parameters": {
                            "type": "object",
                            "properties": {"search_key": {"type": "string"}},
                            "required": ["search_key"],
                        },
                    },
                }
            ],
            lambda name, arguments: calls.append((name, arguments)) or '{"records": []}',
        )

        self.assertEqual("根据资料，先检查设备连接。[1]", reply)
        self.assertEqual([("search_knowledge_base", {"search_key": "设备连接"})], calls)
        self.assertEqual(2, len(fake_openai.chat.completions.calls))
        self.assertEqual(
            "deepseek-v3.1-terminus-chat",
            fake_openai.chat.completions.calls[0]["model"],
        )
        self.assertIn("tools", fake_openai.chat.completions.calls[0])


if __name__ == "__main__":
    unittest.main()
