"""Wiki-MCP stdio 客户端，只暴露知识库问答所需的只读工具。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Coroutine, TypeVar

try:
    from .settings import MCPSettings
except ImportError:  # 小鲁班以单文件入口加载插件目录时
    from settings import MCPSettings


class MCPClientError(RuntimeError):
    """Wiki-MCP 连接或调用失败。"""


_T = TypeVar("_T")


def decode_tool_payload(payload: Any) -> Any:
    """将 MCP 文本结果还原为 JSON；无法解析时保留原文本。"""
    if isinstance(payload, (dict, list)):
        return payload
    if payload is None:
        return {}
    text = str(payload).strip()
    if not text:
        return {}

    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    value: Any = text
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            break
    return value


class WikiMCPClient:
    """在固定后台线程和事件循环中复用一个 stdio MCP 会话。"""

    def __init__(self, settings: MCPSettings):
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wiki-mcp")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: Any = None
        self._stdio_context: Any = None
        self._session_context: Any = None
        self._lock = threading.RLock()

    def _run_in_worker(self, factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(factory())

    def _run(self, factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
        with self._lock:
            future = self._executor.submit(self._run_in_worker, factory)
            try:
                return future.result(timeout=self._settings.timeout_seconds)
            except FutureTimeoutError as exc:
                future.cancel()
                raise MCPClientError("Wiki-MCP 调用超时，请稍后重试。") from exc

    def _runtime_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(self._settings.environment)
        if self._settings.w3_account:
            environment["w3Account"] = self._settings.w3_account
        if self._settings.w3_password:
            environment["password"] = self._settings.w3_password

        has_token = bool(environment.get("w3token") or environment.get("W3TOKEN"))
        if not has_token and not (
            environment.get("w3Account") and environment.get("password")
        ):
            raise MCPClientError(
                "Wiki-MCP 鉴权未配置：请填写 W3 账号和加密密码，或由运行环境提供 W3TOKEN。"
            )
        return environment

    async def _ensure_connected(self) -> None:
        if self._session is not None:
            return
        if not self._settings.command:
            raise MCPClientError("Wiki-MCP 启动命令尚未配置。")

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise MCPClientError("缺少 mcp 依赖，请先安装 requirements.txt。") from exc

        parameters = StdioServerParameters(
            command=self._settings.command[0],
            args=list(self._settings.command[1:]),
            env=self._runtime_environment(),
        )
        try:
            self._stdio_context = stdio_client(parameters)
            read_stream, write_stream = await self._stdio_context.__aenter__()
            self._session_context = ClientSession(read_stream, write_stream)
            self._session = await self._session_context.__aenter__()
            await self._session.initialize()
        except BaseException as exc:
            await self._close_async()
            message = str(exc).strip() or type(exc).__name__
            raise MCPClientError(f"Wiki-MCP 连接失败：{message}") from exc

    async def _close_async(self) -> None:
        session_context, stdio_context = self._session_context, self._stdio_context
        self._session = None
        self._session_context = None
        self._stdio_context = None
        if session_context is not None:
            try:
                await session_context.__aexit__(None, None, None)
            except BaseException:
                pass
        if stdio_context is not None:
            try:
                await stdio_context.__aexit__(None, None, None)
            except BaseException:
                pass

    @staticmethod
    def _extract_result(result: Any) -> Any:
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured

        texts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text is not None:
                texts.append(str(text))
        if not texts:
            return {}
        if len(texts) == 1:
            return decode_tool_payload(texts[0])
        decoded = [decode_tool_payload(text) for text in texts]
        if all(isinstance(item, dict) for item in decoded):
            merged: dict[str, Any] = {}
            for item in decoded:
                merged.update(item)
            return merged
        return "\n".join(texts)

    async def _call_async(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        await self._ensure_connected()
        assert self._session is not None
        try:
            result = await self._session.call_tool(tool_name, arguments=arguments)
        except BaseException as exc:
            await self._close_async()
            message = str(exc).strip() or type(exc).__name__
            raise MCPClientError(f"Wiki-MCP 工具 {tool_name} 调用失败：{message}") from exc
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            detail = self._extract_result(result)
            raise MCPClientError(f"Wiki-MCP 工具 {tool_name} 返回错误：{detail}")
        return self._extract_result(result)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return self._run(lambda: self._call_async(tool_name, arguments))

    def search_documents(self, url: str, search_range: str, search_key: str) -> list[dict[str, Any]]:
        payload = self.call_tool(
            "search_wiki_documents",
            {"url": url, "search_range": search_range, "search_key": search_key},
        )
        decoded = decode_tool_payload(payload)
        if not isinstance(decoded, dict):
            raise MCPClientError("Wiki-MCP 搜索结果格式异常。")
        records = decoded.get("records", [])
        if not isinstance(records, list):
            raise MCPClientError("Wiki-MCP 搜索结果缺少 records 列表。")
        return [record for record in records if isinstance(record, dict)]

    def fetch_document(self, url: str) -> dict[str, Any]:
        payload = self.call_tool("fetch_wiki_content", {"url": url})
        decoded = decode_tool_payload(payload)
        if not isinstance(decoded, dict):
            raise MCPClientError("Wiki-MCP 文档内容格式异常。")
        return decoded

    def close(self) -> None:
        try:
            self._run(self._close_async)
        except Exception:
            pass
        self._executor.shutdown(wait=False, cancel_futures=True)
