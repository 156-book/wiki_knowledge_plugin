# 团队知识库助手

该插件通过公司内部 OpenAI 兼容模型先分析用户意图，再通过工具调用 Wiki-MCP 搜索和读取团队Wiki，最后生成有依据的回答。Wiki正文不会写入 `UserData`，回答末尾的来源链接由插件代码根据真实MCP结果生成。

## 部署前配置

编辑 `config.json`：

1. 将 `owner` 和白名单中的占位符替换为插件负责人工号。
2. 知识库目录由 OneBox 表格维护时，将 `onebox_knowledge.url` 替换为表格链接，并把 `onebox_knowledge.enabled` 改为 `true`。表格第二行需包含“知识库名称”“知识库链接”表头，从第三行开始逐行填写名称和Wiki链接。启用后检索范围使用 `onebox_knowledge.search_range`，当前默认是“当前文档及子文档”。
3. `llm.base_url` 当前按内部接口文档填写为 `http://api.openai.rnd.huawei.com/v1`，模型使用已通过独立测试的 `deepseek-v3.1-terminus-chat`。API Key直接配置在 `llm.api_key`；如果后续要求安全存储，也可以改用代码支持的加密字段 `extra_config.llm_api_key_encrypted`。
4. 在小鲁班开发工具中运行 `python main.py`，选择“加密数据”，加密W3密码、W3 CID和模型API Key后填写到 `extra_config`。
5. 将 `extra_config.w3_account` 替换为明文W3工号（例如工号格式的账号），不要加密账号；`w3_password_encrypted` 和 `w3_cid_encrypted` 填写通过开发工具加密后的结果。CID只用于OneBox文件接口，不会传给Wiki-MCP。
6. 内部AI市场当前页面的版本与本地 `uvx` 安装命令版本可能不同，发布前应把 `wiki_mcp.command` 替换为页面上最新的完整 `uvx` 命令。

不要把W3密码或模型Key以明文写进 `config.json`，也不要提交真实凭证。

## 本地配置覆盖

代码也支持使用以下环境变量覆盖敏感配置，便于联调：

- `WIKI_W3_ACCOUNT`
- `WIKI_W3_PASSWORD`
- `WIKI_W3_CID`
- `WIKI_LLM_BASE_URL`
- `WIKI_LLM_API_KEY`
- `WIKI_LLM_MODEL`

## 用户使用方式

用户进入插件后直接发送问题，例如：

```text
PLR设备连接失败应该怎么处理？
```

也可以临时指定Wiki范围：

```text
https://wiki.huawei.com/... 这个测试流程有哪些注意事项？
```

模型可调用两个只读知识库工具，插件再将其映射到Wiki-MCP：

- `search_knowledge_base` → `search_wiki_documents`
- `read_knowledge_document` → `fetch_wiki_content`

模型必须先搜索，再读取正文；插件只接受搜索结果中的链接作为正文读取目标，并由插件代码统一追加来源链接。

## OneBox知识库目录

OneBox表格只保存知识库名称和Wiki链接，不保存Wiki正文。插件首次查询时读取表格并生成全部检索根节点；后续在 `cache_ttl_seconds` 时间内复用结果，缓存过期后自动重新读取，因此表格中的增删改不需要修改插件配置或代码。

表格示例：

| 知识库名称 | 知识库链接 |
|---|---|
| 效率专区 | `https://wiki.huawei.com/...` |
| 测试规范 | `https://wiki.huawei.com/...` |

表头前允许存在标题或空行。无效域名、空行和重复链接会被忽略；如果没有任何有效Wiki链接，插件会返回明确的表格格式错误。

## 本地测试

测试使用假的MCP和模型客户端，不需要连接华为内网：

```powershell
conda run -n plugin python -m unittest discover -s wiki_knowledge_plugin/tests -v
```
