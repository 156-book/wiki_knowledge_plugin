"""从 OneBox 表格加载 Wiki 知识库目录。

OneBox 表格约定在某一行包含“知识库名称”和“知识库链接”两个表头，
后续每一行分别填写知识库名称和 Wiki 链接。表格可以在表头前保留标题、
说明或空行，解析器会先定位表头再读取数据。
"""

from __future__ import annotations

import ast
import os
import re
import time
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

try:
    from .settings import WikiRoot
except ImportError:  # 直接运行插件目录中的模块时
    from settings import WikiRoot


class OneBoxCatalogError(RuntimeError):
    """OneBox 知识库目录无法读取或格式不正确。"""


_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_URL_END_CHARS = "，。；：！？、,.!?;:)]}）】>"
_NAME_HEADER = "知识库名称"
_URL_HEADER = "知识库链接"


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u00a0", " ").split()).strip()


def _header_text(value: Any) -> str:
    return re.sub(r"[\s:：()（）\[\]【】]", "", _cell_text(value)).casefold()


def _as_rows(value: Any) -> list[Any]:
    """把 Excel 工具的常见返回形式统一成行列表。"""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, Mapping):
        # 某些封装会返回 {headers: [...], rows: [...]}。
        for key in ("rows", "data", "records", "values"):
            nested = value.get(key)
            if isinstance(nested, (list, tuple)):
                return list(nested)
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _row_cells(row: Any) -> list[str]:
    if isinstance(row, Mapping):
        # 字典行直接按表头名称取值；调用方会优先处理这种形式。
        return [_cell_text(value) for value in row.values()]
    if isinstance(row, (list, tuple)):
        return [_cell_text(value) for value in row]
    return [_cell_text(row)] if _cell_text(row) else []


def _is_allowed_url(url: str, allowed_hosts: Iterable[str]) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    hostname = (parsed.hostname or "").casefold()
    if parsed.scheme.casefold() != "https" or not hostname:
        return False
    return any(
        hostname == host.casefold() or hostname.endswith(f".{host.casefold()}")
        for host in allowed_hosts
    )


def _extract_url(value: Any) -> str:
    text = _cell_text(value)
    match = _URL_RE.search(text)
    if not match:
        return ""
    return match.group(0).rstrip(_URL_END_CHARS)


def _decode_onebox_response(value: Any) -> Any:
    """解析 OneBox 工具可能返回的 Python 字典字符串。"""

    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    return value


def _onebox_error_message(value: Any) -> str:
    """只提取安全策略错误文本，不把完整认证对象回显给用户。"""

    decoded = _decode_onebox_response(value)
    if not isinstance(decoded, Mapping):
        return ""
    nested = decoded.get("securityAccreditResult")
    if isinstance(nested, Mapping):
        message = _cell_text(nested.get("message"))
        if message:
            return message[:500]
    for key in ("message", "errorMessage", "error", "msg"):
        message = _cell_text(decoded.get(key))
        if message:
            return message[:500]
    return ""


def parse_wiki_roots(
    table: Any,
    *,
    search_range: str,
    allowed_hosts: Iterable[str],
) -> tuple[WikiRoot, ...]:
    """解析 OneBox 表格中的 Wiki 名称和链接。

    表头不要求位于第一行；解析器会忽略表头前的标题/说明行。
    无效域名、空行和重复链接会被跳过。若没有任何可用链接则抛出明确错误。
    """

    rows = _as_rows(table)
    if not rows:
        raise OneBoxCatalogError("OneBox 表格为空，无法加载知识库链接。")

    # 支持常见的“字典记录”返回形式：键名就是表头。
    if isinstance(rows[0], Mapping):
        name_key = next(
            (key for key in rows[0] if _header_text(key) == _header_text(_NAME_HEADER)),
            None,
        )
        url_key = next(
            (key for key in rows[0] if _header_text(key) == _header_text(_URL_HEADER)),
            None,
        )
        if name_key is not None and url_key is not None:
            roots: list[WikiRoot] = []
            seen: set[str] = set()
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                name = _cell_text(row.get(name_key))
                url = _extract_url(row.get(url_key))
                if not name or not url or url in seen or not _is_allowed_url(url, allowed_hosts):
                    continue
                roots.append(WikiRoot(name=name, url=url, search_range=search_range))
                seen.add(url)
            if not roots:
                raise OneBoxCatalogError("OneBox 表格中没有找到有效的 Wiki 链接。")
            return tuple(roots)

    name_index: int | None = None
    url_index: int | None = None
    header_row_index: int | None = None
    for row_index, row in enumerate(rows):
        cells = _row_cells(row)
        normalized = [_header_text(cell) for cell in cells]
        if name_index is None:
            for index, value in enumerate(normalized):
                if value == _header_text(_NAME_HEADER):
                    name_index = index
                    break
        if url_index is None:
            for index, value in enumerate(normalized):
                if value == _header_text(_URL_HEADER):
                    url_index = index
                    break
        if name_index is not None and url_index is not None:
            header_row_index = row_index
            break

    if header_row_index is None or name_index is None or url_index is None:
        raise OneBoxCatalogError(
            "OneBox 表格缺少“知识库名称”和“知识库链接”表头，请检查第二行表头。"
        )

    roots = []
    seen: set[str] = set()
    for row in rows[header_row_index + 1 :]:
        cells = _row_cells(row)
        if max(name_index, url_index) >= len(cells):
            continue
        name = cells[name_index]
        url = _extract_url(cells[url_index])
        if not name or not url or url in seen:
            continue
        if not _is_allowed_url(url, allowed_hosts):
            continue
        roots.append(WikiRoot(name=name, url=url, search_range=search_range))
        seen.add(url)

    if not roots:
        raise OneBoxCatalogError("OneBox 表格中没有找到有效的 Wiki 链接。")
    return tuple(roots)


class OneBoxWikiRootProvider:
    """按缓存周期读取 OneBox 表格并生成当前 Wiki 根节点。"""

    def __init__(
        self,
        *,
        onebox_url: str,
        w3_account: str,
        w3_password: str,
        w3_cid: str,
        search_range: str,
        allowed_hosts: Iterable[str],
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._onebox_url = onebox_url.strip()
        self._w3_account = w3_account.strip()
        self._w3_password = w3_password
        self._w3_cid = w3_cid.strip()
        self._search_range = search_range
        self._allowed_hosts = tuple(allowed_hosts)
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cached_roots: tuple[WikiRoot, ...] = ()
        self._cache_expire_at = 0.0

    def _read_table(self) -> Any:
        if not self._onebox_url:
            raise OneBoxCatalogError(
                "OneBox 知识库表格链接尚未配置，请填写 onebox_knowledge.url。"
            )
        if not self._w3_account or not self._w3_password or not self._w3_cid:
            raise OneBoxCatalogError(
                "读取 OneBox 表格需要 W3 工号、密码和 CID，请检查 extra_config 配置。"
            )
        try:
            from util.api.by_cookie.onebox import get_onebox_file_path
        except ImportError as exc:
            missing = getattr(exc, "name", None) or "util.api.by_cookie.onebox"
            if missing == "cachetools":
                raise OneBoxCatalogError(
                    "小鲁班 OneBox 接口本身已找到，但它依赖的公开 Python 包 "
                    "cachetools 尚未安装；请确认 requirements.txt 已包含 cachetools，"
                    "然后重新安装插件依赖。"
                ) from exc
            raise OneBoxCatalogError(
                "当前运行环境无法导入 OneBox 工具模块 "
                f"{missing}。请确认小鲁班运行时提供该接口。"
            ) from exc
        try:
            from util.com.excel import get_excel_values
        except ImportError as exc:
            missing = getattr(exc, "name", None) or "util.com.excel"
            raise OneBoxCatalogError(
                "当前运行环境无法导入 Excel 工具模块 "
                f"{missing}。请确认小鲁班运行时提供 util.com.excel，或补充表格读取依赖。"
            ) from exc

        try:
            excel_path = get_onebox_file_path(
                self._onebox_url,
                self._w3_account,
                self._w3_password,
                self._w3_cid,
            )
            security_message = _onebox_error_message(excel_path)
            if security_message:
                raise OneBoxCatalogError(
                    f"OneBox 文件下载被安全策略拒绝：{security_message}。"
                    "请联系表格负责人或平台管理员为当前账号配置合法的文件下载权限。"
                )
            decoded_path = _decode_onebox_response(excel_path)
            if isinstance(decoded_path, Mapping):
                raise OneBoxCatalogError(
                    "OneBox 文件接口未返回可读取的文件路径，请检查表格链接和访问权限。"
                )
            if not isinstance(decoded_path, (str, os.PathLike)) or not str(decoded_path).strip():
                raise OneBoxCatalogError(
                    "OneBox 文件接口未返回有效的 Excel 文件路径。"
                )
            return get_excel_values(decoded_path, need_header=True)
        except OneBoxCatalogError:
            raise
        except Exception as exc:
            detail = " ".join(str(exc).split())[:500] or type(exc).__name__
            raise OneBoxCatalogError(f"读取 OneBox 知识库表格失败：{detail}") from exc

    def get_roots(self, *, force_refresh: bool = False) -> tuple[WikiRoot, ...]:
        now = time.monotonic()
        if not force_refresh and self._cached_roots and now < self._cache_expire_at:
            return self._cached_roots

        roots = parse_wiki_roots(
            self._read_table(),
            search_range=self._search_range,
            allowed_hosts=self._allowed_hosts,
        )
        self._cached_roots = roots
        self._cache_expire_at = now + self._cache_ttl_seconds
        return roots

    def __call__(self) -> tuple[WikiRoot, ...]:
        return self.get_roots()
