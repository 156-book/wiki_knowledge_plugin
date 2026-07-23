from __future__ import annotations

import importlib
import sys
import types
import unittest

from wiki_knowledge_plugin.knowledge_service import AnswerResult, RetrievedDocument


class FakeService:
    def __init__(self):
        self.questions = []

    def answer_question(self, question):
        self.questions.append(question)
        return AnswerResult(
            answer="回答内容。[1]",
            sources=(
                RetrievedDocument(
                    title="来源文档", url="https://wiki.huawei.com/source", content="资料"
                ),
            ),
        )


class FakeMsg:
    def __init__(self, params, first=False):
        self.params = params
        self.receiver = "receiver"
        self._first = first

    def is_first_input(self):
        return self._first


class PluginEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sent = []
        cls.received = []

        util = types.ModuleType("util")
        util_msg = types.ModuleType("util.msg")
        util_msg.Msg = FakeMsg
        util_api = types.ModuleType("util.api")
        util_by_token = types.ModuleType("util.api.by_token")
        util_api_module = types.ModuleType("util.api.by_token.api")
        util_send_module = types.ModuleType("util.api.by_token.send_msg")
        util_api_module.recv_next_msg = cls.received.append
        util_send_module.send_msg = lambda message, receiver: cls.sent.append((message, receiver))
        sys.modules.update(
            {
                "util": util,
                "util.msg": util_msg,
                "util.api": util_api,
                "util.api.by_token": util_by_token,
                "util.api.by_token.api": util_api_module,
                "util.api.by_token.send_msg": util_send_module,
            }
        )
        cls.plugin = importlib.import_module("wiki_knowledge_plugin.plugin")

    def setUp(self):
        self.sent.clear()
        self.received.clear()
        self.service = FakeService()
        self.plugin._service = self.service

    def test_first_empty_input_shows_menu_and_waits(self):
        msg = FakeMsg("/", first=True)
        self.plugin.handle(msg)

        self.assertIn("团队知识库助手", self.sent[0][0])
        self.assertEqual([msg], self.received)

    def test_question_is_completed_in_one_user_message(self):
        msg = FakeMsg("知识问答：设备连接失败怎么办？", first=False)
        self.plugin.handle(msg)

        self.assertEqual(["设备连接失败怎么办？"], self.service.questions)
        self.assertIn("正在检索", self.sent[0][0])
        self.assertIn("https://wiki.huawei.com/source", self.sent[1][0])
        self.assertEqual([msg], self.received)

    def test_first_input_with_question_is_answered_directly(self):
        msg = FakeMsg("测试报告包含什么？", first=True)
        self.plugin.handle(msg)

        self.assertEqual(["测试报告包含什么？"], self.service.questions)
        self.assertEqual(2, len(self.sent))


if __name__ == "__main__":
    unittest.main()
