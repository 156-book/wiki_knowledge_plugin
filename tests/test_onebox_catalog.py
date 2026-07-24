from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from wiki_knowledge_plugin.onebox_catalog import (
    OneBoxCatalogError,
    OneBoxWikiRootProvider,
    parse_wiki_roots,
)


class OneBoxCatalogTests(unittest.TestCase):
    def test_header_can_start_on_second_row_and_children_are_loaded(self):
        rows = [
            ["团队Wiki知识库目录", ""],
            ["知识库名称", "知识库链接"],
            ["效率专区", "https://wiki.huawei.com/domains/1/wiki/10/WIKI-A"],
            ["测试规范", "https://wiki.huawei.com/domains/2/wiki/20/WIKI-B"],
        ]

        roots = parse_wiki_roots(
            rows,
            search_range="当前文档及子文档",
            allowed_hosts=("wiki.huawei.com",),
        )

        self.assertEqual(["效率专区", "测试规范"], [root.name for root in roots])
        self.assertEqual("当前文档及子文档", roots[0].search_range)

    def test_invalid_duplicate_and_blank_rows_are_ignored(self):
        rows = [
            ["说明"],
            ["知识库名称", "知识库链接"],
            ["", "https://wiki.huawei.com/blank"],
            ["外部链接", "https://example.com/wiki"],
            ["有效链接", "https://wiki.huawei.com/doc"],
            ["重复链接", "https://wiki.huawei.com/doc"],
        ]

        roots = parse_wiki_roots(
            rows,
            search_range="知识库",
            allowed_hosts=("wiki.huawei.com",),
        )

        self.assertEqual(("有效链接", "https://wiki.huawei.com/doc"), (roots[0].name, roots[0].url))

    def test_missing_headers_are_reported(self):
        with self.assertRaisesRegex(OneBoxCatalogError, "缺少"):
            parse_wiki_roots(
                [["名称", "链接"], ["效率专区", "https://wiki.huawei.com/doc"]],
                search_range="知识库",
                allowed_hosts=("wiki.huawei.com",),
            )

    def test_provider_reads_table_and_caches_it(self):
        onebox_module = types.ModuleType("util.api.by_cookie.onebox")
        excel_module = types.ModuleType("util.com.excel")
        calls = []

        def get_onebox_file_path(url, account, password, cid):
            calls.append((url, account, password, cid))
            return "table.xlsx"

        excel_module.get_excel_values = lambda path, need_header=True: [
            ["知识库名称", "知识库链接"],
            ["效率专区", "https://wiki.huawei.com/doc"],
        ]
        onebox_module.get_onebox_file_path = get_onebox_file_path

        modules = {
            "util.api.by_cookie.onebox": onebox_module,
            "util.com.excel": excel_module,
        }
        provider = OneBoxWikiRootProvider(
            onebox_url="https://onebox.huawei.com/table",
            w3_account="employee",
            w3_password="password",
            w3_cid="cid",
            search_range="当前文档及子文档",
            allowed_hosts=("wiki.huawei.com",),
            cache_ttl_seconds=300,
        )
        with patch.dict(sys.modules, modules):
            first = provider.get_roots()
            second = provider.get_roots()

        self.assertEqual(first, second)
        self.assertEqual(1, len(calls))
        self.assertEqual("效率专区", first[0].name)


if __name__ == "__main__":
    unittest.main()

