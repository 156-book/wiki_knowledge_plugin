"""小鲁班团队 Wiki 知识库问答插件入口。"""

from __future__ import annotations

import os
import re
import sys
import threading
from typing import Any

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from util.api.by_token.api import recv_next_msg
from util.api.by_token.send_msg import send_msg
from util.msg import Msg

try:
    from .knowledge_service import KnowledgeService, KnowledgeServiceError, format_answer
    from .llm_client import InternalLLMClient
    from .mcp_client import WikiMCPClient
    from .onebox_catalog import OneBoxWikiRootProvider
    from .settings import ConfigurationError, load_settings
except ImportError:  # 直接运行 plugin.py 时使用同目录模块
    from knowledge_service import KnowledgeService, KnowledgeServiceError, format_answer
    from llm_client import InternalLLMClient
    from mcp_client import WikiMCPClient
    from onebox_catalog import OneBoxWikiRootProvider
    from settings import ConfigurationError, load_settings


_MENU = """团队知识库助手

直接发送问题，我会查询团队Wiki后根据资料回答，并附上来源链接。

示例：
PLR设备连接失败应该怎么处理？
测试报告需要包含哪些内容？

也可以指定一个Wiki文档及其子文档作为本次查询范围：
https://wiki.huawei.com/... 这个流程有哪些注意事项？

输入“帮助”查看说明，输入“退出”结束。"""

_CONTINUE_HINT = """你可以继续发送问题，也可以发送“Wiki链接 + 问题”。
输入“帮助”查看完整说明，输入“退出”结束。"""

_service: KnowledgeService | None = None
_service_lock = threading.Lock()


def _get_service() -> KnowledgeService:
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            settings = load_settings()
            roots_provider = None
            if settings.onebox.enabled:
                roots_provider = OneBoxWikiRootProvider(
                    onebox_url=settings.onebox.url,
                    w3_account=settings.mcp.w3_account,
                    w3_password=settings.mcp.w3_password,
                    w3_cid=settings.mcp.w3_cid,
                    search_range=settings.onebox.search_range,
                    allowed_hosts=settings.knowledge.allowed_hosts,
                    cache_ttl_seconds=settings.onebox.cache_ttl_seconds,
                )
            _service = KnowledgeService(
                settings=settings,
                wiki_client=WikiMCPClient(settings.mcp),
                llm_client=InternalLLMClient(settings.llm),
                roots_provider=roots_provider,
            )
    return _service


def _message_content(msg: Msg) -> str:
    return str(getattr(msg, "params", "") or "").strip()


def _normalize_question(content: str) -> str:
    return re.sub(
        r"^\s*(?:知识问答|知识库问答|提问|问)\s*[：:]\s*",
        "",
        content,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def _send_and_continue(message: str, msg: Msg) -> None:
    send_msg(message, msg.receiver)
    recv_next_msg(msg)


def _answer(content: str, msg: Msg) -> None:
    question = _normalize_question(content)
    if not question:
        _send_and_continue(f"问题不能为空。\n\n{_CONTINUE_HINT}", msg)
        return

    send_msg("正在检索团队Wiki并整理答案，请稍候……", msg.receiver)
    try:
        result = _get_service().answer_question(question)
        reply = format_answer(result)
    except (ConfigurationError, KnowledgeServiceError) as exc:
        reply = str(exc)
    except Exception as exc:
        print(f"[wiki-knowledge-plugin] unexpected error: {type(exc).__name__}")
        reply = "知识库查询暂时失败，请稍后重试或联系插件负责人。"
    _send_and_continue(f"{reply}\n\n{_CONTINUE_HINT}", msg)


def handle(msg: Msg) -> None:
    content = _message_content(msg)

    if msg.is_first_input() and content in {"", "/"}:
        _send_and_continue(_MENU, msg)
        return

    normalized = content.casefold().strip()
    if normalized in {"退出", "结束", "exit", "quit"}:
        send_msg("已退出团队知识库助手，再见。", msg.receiver)
        return
    if normalized in {"帮助", "菜单", "help", "?", "？"}:
        _send_and_continue(_MENU, msg)
        return
    if not content:
        _send_and_continue(f"问题不能为空。\n\n{_MENU}", msg)
        return
    _answer(content, msg)


if __name__ == "__main__":
    from util.debug.debug import debug_handle

    debug_handle(handle, "/")
