"""知识库插件配置加载与校验。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class ConfigurationError(RuntimeError):
    """配置缺失或格式错误。"""


_PLACEHOLDER_PREFIXES = ("__REPLACE_", "请替换", "YOUR_")


def _usable_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.startswith(_PLACEHOLDER_PREFIXES):
        return ""
    return text


def _decrypt_optional(value: Any, field_name: str) -> str:
    encrypted = _usable_text(value)
    if not encrypted:
        return ""
    try:
        from util.debug.local import decrypt

        return str(decrypt(encrypted) or "")
    except Exception as exc:
        raise ConfigurationError(
            f"{field_name} 解密失败，请使用小鲁班 main.py 的“加密数据”功能重新生成。"
        ) from exc


@dataclass(frozen=True)
class WikiRoot:
    name: str
    url: str
    search_range: str = "当前文档及子文档"


@dataclass(frozen=True)
class KnowledgeSettings:
    roots: tuple[WikiRoot, ...]
    allowed_hosts: tuple[str, ...]
    max_search_results: int = 8
    max_fetch_documents: int = 3
    max_chars_per_document: int = 6000
    max_context_chars: int = 18000
    cache_ttl_seconds: int = 300
    max_question_chars: int = 500

    def is_allowed_wiki_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").casefold()
        except ValueError:
            return False
        if parsed.scheme != "https" or not hostname:
            return False
        return any(
            hostname == allowed or hostname.endswith(f".{allowed}")
            for allowed in self.allowed_hosts
        )


@dataclass(frozen=True)
class MCPSettings:
    transport: str
    command: tuple[str, ...]
    environment: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 120.0
    w3_account: str = ""
    w3_password: str = ""


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 120.0
    temperature: float = 0.1


@dataclass(frozen=True)
class AppSettings:
    knowledge: KnowledgeSettings
    mcp: MCPSettings
    llm: LLMSettings


def _positive_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"wiki_knowledge.{key} 必须是正整数。")
    return value


def _positive_number(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"{key} 必须是正数。")
    return float(value)


def _load_roots(data: dict[str, Any], allowed_hosts: tuple[str, ...]) -> tuple[WikiRoot, ...]:
    raw_roots = data.get("roots", [])
    if not isinstance(raw_roots, list):
        raise ConfigurationError("wiki_knowledge.roots 必须是列表。")

    roots: list[WikiRoot] = []
    for index, item in enumerate(raw_roots, 1):
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        name = _usable_text(item.get("name")) or f"知识库{index}"
        url = _usable_text(item.get("url"))
        if not url:
            continue
        try:
            parsed_url = urlparse(url)
            hostname = (parsed_url.hostname or "").casefold()
        except ValueError as exc:
            raise ConfigurationError(f"知识库“{name}”的 Wiki URL 格式不正确。") from exc
        if parsed_url.scheme != "https" or not any(
            hostname == host or hostname.endswith(f".{host}") for host in allowed_hosts
        ):
            raise ConfigurationError(f"知识库“{name}”的 Wiki URL 不合法或域名不在白名单中。")
        search_range = _usable_text(item.get("search_range")) or "当前文档及子文档"
        if search_range not in {"知识库", "类目", "当前文档及子文档"}:
            raise ConfigurationError(f"知识库“{name}”的 search_range 配置不正确。")
        roots.append(WikiRoot(name=name, url=url, search_range=search_range))
    return tuple(roots)


def load_settings(config_path: str | os.PathLike[str] | None = None) -> AppSettings:
    path = Path(config_path) if config_path else Path(__file__).with_name("config.json")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"无法读取插件配置文件：{path}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError("config.json 顶层必须是 JSON 对象。")

    knowledge_data = raw.get("wiki_knowledge", {})
    mcp_data = raw.get("wiki_mcp", {})
    llm_data = raw.get("llm", {})
    extra = raw.get("extra_config", {})
    if not all(isinstance(item, dict) for item in (knowledge_data, mcp_data, llm_data, extra)):
        raise ConfigurationError("wiki_knowledge、wiki_mcp、llm 和 extra_config 必须是 JSON 对象。")

    allowed_hosts_raw = knowledge_data.get("allowed_hosts", ["wiki.huawei.com"])
    if not isinstance(allowed_hosts_raw, list):
        raise ConfigurationError("wiki_knowledge.allowed_hosts 必须是列表。")
    allowed_hosts = tuple(
        host.casefold() for host in (_usable_text(value) for value in allowed_hosts_raw) if host
    )
    if not allowed_hosts:
        raise ConfigurationError("至少需要配置一个允许访问的 Wiki 域名。")

    knowledge = KnowledgeSettings(
        roots=_load_roots(knowledge_data, allowed_hosts),
        allowed_hosts=allowed_hosts,
        max_search_results=_positive_int(knowledge_data, "max_search_results", 8),
        max_fetch_documents=_positive_int(knowledge_data, "max_fetch_documents", 3),
        max_chars_per_document=_positive_int(knowledge_data, "max_chars_per_document", 6000),
        max_context_chars=_positive_int(knowledge_data, "max_context_chars", 18000),
        cache_ttl_seconds=_positive_int(knowledge_data, "cache_ttl_seconds", 300),
        max_question_chars=_positive_int(knowledge_data, "max_question_chars", 500),
    )

    command = mcp_data.get("command", [])
    if not isinstance(command, list) or not all(isinstance(part, str) and part for part in command):
        raise ConfigurationError("wiki_mcp.command 必须是非空字符串列表。")
    environment = mcp_data.get("environment", {})
    if not isinstance(environment, dict):
        raise ConfigurationError("wiki_mcp.environment 必须是 JSON 对象。")
    transport = _usable_text(mcp_data.get("transport")) or "stdio"
    if transport != "stdio":
        raise ConfigurationError("当前版本仅支持 Wiki-MCP 官方推荐的 stdio 本地连接方式。")

    w3_account = os.environ.get("WIKI_W3_ACCOUNT") or _usable_text(extra.get("w3_account"))
    w3_password = os.environ.get("WIKI_W3_PASSWORD") or _decrypt_optional(
        extra.get("w3_password_encrypted"), "W3 密码"
    )
    mcp = MCPSettings(
        transport=transport,
        command=tuple(command),
        environment={str(key): str(value) for key, value in environment.items()},
        timeout_seconds=_positive_number(mcp_data, "timeout_seconds", 120.0),
        w3_account=w3_account,
        w3_password=w3_password,
    )

    base_url = os.environ.get("WIKI_LLM_BASE_URL") or _usable_text(llm_data.get("base_url"))
    model = os.environ.get("WIKI_LLM_MODEL") or _usable_text(llm_data.get("model"))
    api_key = os.environ.get("WIKI_LLM_API_KEY") or _decrypt_optional(
        extra.get("llm_api_key_encrypted"), "大模型 API Key"
    ) or _usable_text(llm_data.get("api_key"))
    temperature = llm_data.get("temperature", 0.1)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ConfigurationError("llm.temperature 必须是数字。")
    llm = LLMSettings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=_positive_number(llm_data, "timeout_seconds", 120.0),
        temperature=float(temperature),
    )
    return AppSettings(knowledge=knowledge, mcp=mcp, llm=llm)
