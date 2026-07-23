from __future__ import annotations

import unittest

from wiki_knowledge_plugin.knowledge_service import (
    AnswerResult,
    KnowledgeService,
    KnowledgeServiceError,
    RetrievedDocument,
    format_answer,
)
from wiki_knowledge_plugin.mcp_client import decode_tool_payload
from wiki_knowledge_plugin.settings import (
    AppSettings,
    KnowledgeSettings,
    LLMSettings,
    MCPSettings,
    WikiRoot,
)


def make_settings(*roots: WikiRoot) -> AppSettings:
    return AppSettings(
        knowledge=KnowledgeSettings(
            roots=tuple(roots),
            allowed_hosts=("wiki.huawei.com",),
            max_search_results=8,
            max_fetch_documents=3,
            max_chars_per_document=1000,
            max_context_chars=3000,
            cache_ttl_seconds=300,
            max_question_chars=500,
        ),
        mcp=MCPSettings(transport="stdio", command=("uvx",)),
        llm=LLMSettings(base_url="http://llm/v1", api_key="test", model="test-model"),
    )


class FakeWikiClient:
    def __init__(self, searches=None, documents=None):
        self.searches = searches or {}
        self.documents = documents or {}
        self.search_calls = []
        self.fetch_calls = []

    def search_documents(self, url, search_range, search_key):
        self.search_calls.append((url, search_range, search_key))
        value = self.searches.get(url, [])
        if isinstance(value, Exception):
            raise value
        return value

    def fetch_document(self, url):
        self.fetch_calls.append(url)
        value = self.documents[url]
        if isinstance(value, Exception):
            raise value
        return value


class FakeLLMClient:
    def __init__(self, reply="这是严格根据Wiki资料生成的回答。[1]"):
        self.reply = reply
        self.calls = []

    def answer(self, question, context):
        self.calls.append((question, context))
        return self.reply


class KnowledgeServiceTests(unittest.TestCase):
    def test_default_root_search_fetch_answer_and_real_links(self):
        root = WikiRoot("团队知识库", "https://wiki.huawei.com/root")
        wiki = FakeWikiClient(
            searches={
                root.url: [
                    {
                        "title": "设备连接指南",
                        "url": "https://wiki.huawei.com/doc-1",
                        "content_match_snippets": ["检查USB连接"],
                    },
                    {
                        "title": "驱动安装说明",
                        "url": "https://wiki.huawei.com/doc-2",
                        "content_match_snippets": ["安装匹配版本的驱动"],
                    },
                ]
            },
            documents={
                "https://wiki.huawei.com/doc-1": {
                    "title": "设备连接指南",
                    "content": "连接失败时先检查USB连接。",
                    "last_update_time": "2026-07-01",
                },
                "https://wiki.huawei.com/doc-2": {
                    "title": "驱动安装说明",
                    "content": "确认驱动版本与设备型号匹配。",
                },
            },
        )
        llm = FakeLLMClient()
        service = KnowledgeService(make_settings(root), wiki, llm)

        result = service.answer_question("PLR设备连接失败应该怎么处理？")
        rendered = format_answer(result)

        self.assertEqual(2, len(result.sources))
        self.assertIn("连接失败时先检查USB连接", llm.calls[0][1])
        self.assertIn("https://wiki.huawei.com/doc-1", rendered)
        self.assertIn("https://wiki.huawei.com/doc-2", rendered)
        self.assertNotIn("UserData", rendered)
        self.assertEqual("当前文档及子文档", wiki.search_calls[0][1])

    def test_direct_wiki_link_is_used_without_configured_root(self):
        direct = "https://wiki.huawei.com/direct"
        wiki = FakeWikiClient(
            searches={direct: []},
            documents={direct: {"title": "指定流程", "content": "流程必须先完成环境检查。"}},
        )
        llm = FakeLLMClient("应先完成环境检查。[1]")
        service = KnowledgeService(make_settings(), wiki, llm)

        result = service.answer_question(f"{direct} 这个流程开始前需要做什么？")

        self.assertEqual(direct, result.sources[0].url)
        self.assertEqual("这个流程开始前需要做什么？", llm.calls[0][0])

    def test_no_matching_documents_returns_clear_message(self):
        root = WikiRoot("团队知识库", "https://wiki.huawei.com/root")
        service = KnowledgeService(
            make_settings(root), FakeWikiClient(searches={root.url: []}), FakeLLMClient()
        )

        with self.assertRaisesRegex(KnowledgeServiceError, "没有找到"):
            service.answer_question("一个不存在的问题")

    def test_disallowed_url_is_rejected_before_mcp_call(self):
        wiki = FakeWikiClient()
        service = KnowledgeService(make_settings(), wiki, FakeLLMClient())

        with self.assertRaisesRegex(KnowledgeServiceError, "允许访问"):
            service.answer_question("https://example.com/wiki 请回答")
        self.assertEqual([], wiki.search_calls)

    def test_fetch_failure_falls_back_to_search_snippet(self):
        root = WikiRoot("团队知识库", "https://wiki.huawei.com/root")
        url = "https://wiki.huawei.com/doc"
        wiki = FakeWikiClient(
            searches={
                root.url: [
                    {"title": "片段文档", "url": url, "content_match_snippets": ["可用的搜索片段"]}
                ]
            },
            documents={url: RuntimeError("fetch failed")},
        )
        llm = FakeLLMClient()
        result = KnowledgeService(make_settings(root), wiki, llm).answer_question("片段")

        self.assertEqual("可用的搜索片段", result.sources[0].content)
        self.assertIn("可用的搜索片段", llm.calls[0][1])

    def test_document_cache_avoids_duplicate_fetch(self):
        root = WikiRoot("团队知识库", "https://wiki.huawei.com/root")
        url = "https://wiki.huawei.com/doc"
        wiki = FakeWikiClient(
            searches={root.url: [{"title": "文档", "url": url}]},
            documents={url: {"title": "文档", "content": "正文"}},
        )
        service = KnowledgeService(make_settings(root), wiki, FakeLLMClient())

        service.answer_question("第一次查询")
        service.answer_question("第二次查询")

        self.assertEqual([url], wiki.fetch_calls)

    def test_format_answer_uses_source_objects_not_llm_supplied_links(self):
        result = AnswerResult(
            answer="模型回答中没有来源列表。",
            sources=(
                RetrievedDocument(
                    title="真实来源", url="https://wiki.huawei.com/real", content="正文"
                ),
            ),
        )
        rendered = format_answer(result)

        self.assertIn("[1] 真实来源", rendered)
        self.assertIn("https://wiki.huawei.com/real", rendered)

    def test_decode_tool_payload_accepts_json_code_fence_and_double_encoding(self):
        self.assertEqual({"records": []}, decode_tool_payload('```json\n{"records": []}\n```'))
        self.assertEqual({"title": "测试"}, decode_tool_payload('"{\\"title\\": \\"测试\\"}"'))


if __name__ == "__main__":
    unittest.main()

