"""知识库检索、正文读取、回答生成与来源拼装。"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

try:
    from .settings import AppSettings, WikiRoot
except ImportError:  # 小鲁班以单文件入口加载插件目录时
    from settings import AppSettings, WikiRoot


class KnowledgeServiceError(RuntimeError):
    """可直接展示给用户的知识库业务错误。"""


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippets: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedDocument:
    title: str
    url: str
    content: str
    last_update_time: str = ""


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: tuple[RetrievedDocument, ...]


@dataclass
class _AgentState:
    """一次模型工具调用过程中的检索结果，不跨请求保存。"""

    roots: tuple[WikiRoot, ...]
    hits_by_url: dict[str, SearchHit]
    documents_by_url: dict[str, RetrievedDocument]
    allowed_urls: set[str]


_URL_PATTERN = re.compile(r"https://[^\s<>\"']+", re.IGNORECASE)
_URL_END_PUNCTUATION = "，。；：！？、,.!?;:)]}）】》>"


class KnowledgeService:
    def __init__(
        self,
        settings: AppSettings,
        wiki_client: Any,
        llm_client: Any,
        roots_provider: Callable[[], tuple[WikiRoot, ...]] | None = None,
    ):
        self._settings = settings
        self._wiki = wiki_client
        self._llm = llm_client
        self._roots_provider = roots_provider
        self._document_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()

    def _current_roots(self) -> tuple[WikiRoot, ...]:
        if self._roots_provider is not None:
            try:
                roots = tuple(self._roots_provider())
            except KnowledgeServiceError:
                raise
            except Exception as exc:
                detail = " ".join(str(exc).split())[:800] or type(exc).__name__
                raise KnowledgeServiceError(detail) from exc
            return roots
        return self._settings.knowledge.roots

    def _extract_question_and_url(self, text: str) -> tuple[str, str | None]:
        urls = [match.group(0).rstrip(_URL_END_PUNCTUATION) for match in _URL_PATTERN.finditer(text)]
        direct_url: str | None = None
        if urls:
            direct_url = urls[0]
            if not self._settings.knowledge.is_allowed_wiki_url(direct_url):
                raise KnowledgeServiceError("消息中的链接不是当前插件允许访问的 Wiki 域名。")
        question = text
        if direct_url:
            question = question.replace(direct_url, " ", 1)
        question = re.sub(r"\s+", " ", question).strip(" ：:，,。")
        if not question:
            raise KnowledgeServiceError("请在 Wiki 链接后补充需要查询的问题。")
        if len(question) > self._settings.knowledge.max_question_chars:
            raise KnowledgeServiceError(
                f"问题不能超过 {self._settings.knowledge.max_question_chars} 个字符，请精简后重试。"
            )
        return question, direct_url

    @staticmethod
    def _search_keys(question: str) -> tuple[str, ...]:
        full = question.strip()
        simplified = re.sub(
            r"请问|请帮我|麻烦|我想知道|帮我查一下|查询一下|查一下|是什么|怎么|如何|为什么|有哪些|吗|呢",
            "",
            full,
        )
        simplified = re.sub(r"[？?！!，,。；;：:]", " ", simplified)
        simplified = re.sub(r"\s+", " ", simplified).strip()
        keys = [full]
        if len(simplified) >= 2 and simplified != full:
            keys.append(simplified)
        return tuple(keys)

    @staticmethod
    def _normalize_snippets(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value.strip(),) if value.strip() else ()
        if not isinstance(value, list):
            return ()
        snippets: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                snippets.append(item.strip())
            elif isinstance(item, dict):
                text = str(item.get("content") or item.get("text") or "").strip()
                if text:
                    snippets.append(text)
        return tuple(snippets)

    def _normalize_hits(self, records: Iterable[dict[str, Any]]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for record in records:
            url = str(record.get("url") or "").strip()
            if not url or not self._settings.knowledge.is_allowed_wiki_url(url):
                continue
            title = str(record.get("title") or "未命名Wiki文档").strip()
            hits.append(
                SearchHit(
                    title=title,
                    url=url,
                    snippets=self._normalize_snippets(record.get("content_match_snippets")),
                )
            )
        return hits

    def _search_one_root(self, root: WikiRoot, question: str) -> list[SearchHit]:
        for key in self._search_keys(question):
            records = self._wiki.search_documents(root.url, root.search_range, key)
            hits = self._normalize_hits(records)
            if hits:
                return hits
        return []

    @staticmethod
    def _round_robin(groups: list[list[SearchHit]], limit: int) -> list[SearchHit]:
        result: list[SearchHit] = []
        seen: set[str] = set()
        max_size = max((len(group) for group in groups), default=0)
        for index in range(max_size):
            for group in groups:
                if index >= len(group):
                    continue
                hit = group[index]
                if hit.url in seen:
                    continue
                result.append(hit)
                seen.add(hit.url)
                if len(result) >= limit:
                    return result
        return result

    def _search(self, roots: tuple[WikiRoot, ...], question: str) -> list[SearchHit]:
        groups: list[list[SearchHit]] = []
        errors = 0
        for root in roots:
            try:
                groups.append(self._search_one_root(root, question))
            except Exception:
                errors += 1
        hits = self._round_robin(groups, self._settings.knowledge.max_search_results)
        if not hits and errors == len(roots):
            raise KnowledgeServiceError("Wiki-MCP 当前无法完成检索，请稍后重试或联系插件负责人。")
        return hits

    def _fetch_raw(self, url: str) -> dict[str, Any]:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._document_cache.get(url)
            if cached and cached[0] > now:
                return dict(cached[1])
        raw = self._wiki.fetch_document(url)
        if not isinstance(raw, dict):
            raise KnowledgeServiceError("Wiki-MCP 返回的文档格式异常。")
        with self._cache_lock:
            self._document_cache[url] = (
                now + self._settings.knowledge.cache_ttl_seconds,
                dict(raw),
            )
        return raw

    def _fetch_hit(self, hit: SearchHit) -> RetrievedDocument:
        raw = self._fetch_raw(hit.url)
        content = str(raw.get("content") or "").strip()
        if not content:
            content = "\n".join(hit.snippets).strip()
        if not content:
            raise KnowledgeServiceError("Wiki文档正文为空。")
        return RetrievedDocument(
            title=str(raw.get("title") or hit.title).strip(),
            url=hit.url,
            content=content,
            last_update_time=str(raw.get("last_update_time") or "").strip(),
        )

    def _fetch_documents(
        self, hits: list[SearchHit], direct_url: str | None
    ) -> list[RetrievedDocument]:
        documents: list[RetrievedDocument] = []
        for hit in hits[: self._settings.knowledge.max_fetch_documents]:
            try:
                documents.append(self._fetch_hit(hit))
            except Exception:
                if hit.snippets:
                    documents.append(
                        RetrievedDocument(
                            title=hit.title,
                            url=hit.url,
                            content="\n".join(hit.snippets),
                        )
                    )

        if not documents and direct_url:
            try:
                raw = self._fetch_raw(direct_url)
                content = str(raw.get("content") or "").strip()
                if content:
                    documents.append(
                        RetrievedDocument(
                            title=str(raw.get("title") or "指定Wiki文档").strip(),
                            url=direct_url,
                            content=content,
                            last_update_time=str(raw.get("last_update_time") or "").strip(),
                        )
                    )
            except Exception:
                pass
        return documents

    def _build_context(self, documents: list[RetrievedDocument]) -> str:
        sections: list[str] = []
        used = 0
        for index, document in enumerate(documents, 1):
            content = document.content[: self._settings.knowledge.max_chars_per_document]
            header = f"[资料{index}]\n标题：{document.title}\n链接：{document.url}"
            if document.last_update_time:
                header += f"\n最近更新时间：{document.last_update_time}"
            section = f"{header}\n正文：\n{content}"
            remaining = self._settings.knowledge.max_context_chars - used
            if remaining <= 0:
                break
            section = section[:remaining]
            sections.append(section)
            used += len(section)
        return "\n\n".join(sections)

    @staticmethod
    def _safe_service_error(exc: BaseException) -> str:
        text = " ".join((str(exc).strip() or type(exc).__name__).split())
        return text[:800]

    def _answer_direct_wiki(self, question: str, direct_url: str) -> AnswerResult:
        """明确给出Wiki链接时直接读取正文，避免无意义的关键词搜索。"""
        hit = SearchHit("指定Wiki文档", direct_url, ())
        try:
            document = self._fetch_hit(hit)
        except Exception as exc:
            detail = self._safe_service_error(exc)
            raise KnowledgeServiceError(
                f"已识别到Wiki链接，但Wiki-MCP读取正文失败：{detail}"
            ) from exc

        context = self._build_context([document])
        try:
            answer = self._llm.answer(question, context)
        except Exception as exc:
            detail = self._safe_service_error(exc)
            raise KnowledgeServiceError(detail or "大模型暂时无法总结Wiki正文。") from exc
        return AnswerResult(answer=answer.strip(), sources=(document,))

    def _tool_definitions(self, roots: tuple[WikiRoot, ...]) -> list[dict[str, Any]]:
        root_names = "、".join(root.name for root in roots) or "默认知识库"
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": (
                        "在团队Wiki知识库中搜索与用户问题相关的文档。"
                        f"可用知识库：{root_names}。必须先调用此工具，再读取正文。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_key": {
                                "type": "string",
                                "description": "从用户问题提炼出的简短检索关键词。",
                            },
                            "root_name": {
                                "type": "string",
                                "description": "可选知识库名称，不确定时留空并搜索全部知识库。",
                            },
                        },
                        "required": ["search_key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_knowledge_document",
                    "description": "读取搜索结果中的Wiki文档正文，用于支撑最终回答。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "search_knowledge_base 返回的Wiki文档链接。",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    @staticmethod
    def _json_tool_result(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _select_roots(roots: tuple[WikiRoot, ...], root_name: str) -> tuple[WikiRoot, ...]:
        requested = root_name.strip().casefold()
        if not requested:
            return roots
        selected = tuple(root for root in roots if root.name.casefold() == requested)
        return selected or roots

    def _execute_search_tool(
        self, state: _AgentState, arguments: dict[str, Any], fallback_question: str
    ) -> str:
        search_key = str(arguments.get("search_key") or fallback_question).strip()
        if not search_key:
            return self._json_tool_result({"records": [], "error": "search_key不能为空"})
        root_name = str(arguments.get("root_name") or "")
        selected_roots = self._select_roots(state.roots, root_name)
        groups: list[list[SearchHit]] = []
        errors = 0
        for root in selected_roots:
            try:
                groups.append(self._search_one_root(root, search_key))
            except Exception:
                errors += 1
        hits = self._round_robin(groups, self._settings.knowledge.max_search_results)
        for hit in hits:
            state.hits_by_url[hit.url] = hit
            state.allowed_urls.add(hit.url)
        if not hits and errors == len(selected_roots):
            return self._json_tool_result(
                {"records": [], "error": "Wiki-MCP暂时无法完成检索，请稍后重试。"}
            )
        return self._json_tool_result(
            {
                "total_records": len(hits),
                "records": [
                    {
                        "title": hit.title,
                        "url": hit.url,
                        "content_match_snippets": list(hit.snippets),
                    }
                    for hit in hits
                ],
            }
        )

    def _execute_read_tool(self, state: _AgentState, arguments: dict[str, Any]) -> str:
        url = str(arguments.get("url") or "").strip()
        if not url or not self._settings.knowledge.is_allowed_wiki_url(url):
            return self._json_tool_result({"error": "该链接不是允许访问的Wiki链接。"})
        if url not in state.allowed_urls:
            return self._json_tool_result(
                {"error": "只能读取搜索结果中的Wiki链接，请先调用search_knowledge_base。"}
            )
        hit = state.hits_by_url.get(url) or SearchHit("指定Wiki文档", url, ())
        try:
            document = self._fetch_hit(hit)
        except Exception:
            return self._json_tool_result({"error": "Wiki文档读取失败，请尝试其他搜索结果。"})
        state.documents_by_url[url] = document
        return self._json_tool_result(
            {
                "title": document.title,
                "url": document.url,
                "last_update_time": document.last_update_time,
                "content": document.content[: self._settings.knowledge.max_chars_per_document],
            }
        )

    def _execute_tool(
        self, state: _AgentState, tool_name: str, arguments: dict[str, Any], question: str
    ) -> str:
        if tool_name == "search_knowledge_base":
            return self._execute_search_tool(state, arguments, question)
        if tool_name == "read_knowledge_document":
            return self._execute_read_tool(state, arguments)
        return self._json_tool_result({"error": f"不支持的知识库工具：{tool_name}"})

    def answer_question(self, raw_question: str) -> AnswerResult:
        question, direct_url = self._extract_question_and_url(raw_question)
        if direct_url:
            return self._answer_direct_wiki(question, direct_url)

        roots = self._current_roots()
        if not roots:
            raise KnowledgeServiceError("插件尚未配置团队Wiki根链接，请联系插件负责人。")

        state = _AgentState(
            roots=roots,
            hits_by_url={},
            documents_by_url={},
            allowed_urls={root.url for root in roots},
        )
        try:
            answer = self._llm.answer_with_tools(
                question,
                self._tool_definitions(roots),
                lambda tool_name, arguments: self._execute_tool(
                    state, tool_name, arguments, question
                ),
            )
        except AttributeError as exc:
            raise KnowledgeServiceError("当前大模型客户端不支持工具调用接口。") from exc
        except Exception as exc:
            message = str(exc).strip()
            raise KnowledgeServiceError(message or "大模型暂时无法完成知识库问答。") from exc

        documents = list(state.documents_by_url.values())
        if not documents and state.hits_by_url:
            # 模型已经完成搜索但未读取正文时，插件自动读取最高相关结果，
            # 避免出现“模型回答了，但没有任何可引用资料”的情况。
            documents = self._fetch_documents(
                list(state.hits_by_url.values()), None
            )[: self._settings.knowledge.max_fetch_documents]
        if not documents:
            raise KnowledgeServiceError(
                "当前知识库中没有找到足以回答该问题的资料，请更换关键词后重试。"
            )
        return AnswerResult(answer=answer.strip(), sources=tuple(documents))


def format_answer(result: AnswerResult, max_chars: int = 6000) -> str:
    answer = result.answer.strip()
    source_lines = ["参考资料："]
    for index, source in enumerate(result.sources, 1):
        title = re.sub(r"\s+", " ", source.title).strip() or "未命名Wiki文档"
        source_lines.append(f"[{index}] {title}\n{source.url}")
    sources = "\n\n".join(source_lines)
    reserved = len(sources) + 4
    if len(answer) + reserved > max_chars:
        answer = answer[: max(1, max_chars - reserved - 1)].rstrip() + "…"
    return f"{answer}\n\n{sources}"
