# 团队知识库助手

该插件通过 Wiki-MCP 实时搜索团队Wiki、读取相关文档，再调用公司内部 OpenAI 兼容模型生成回答。Wiki正文不会写入 `UserData`，回答末尾的来源链接由插件代码根据MCP结果生成。

## 部署前配置

编辑 `config.json`：

1. 将 `owner` 和白名单中的占位符替换为插件负责人工号。
2. 将 `wiki_knowledge.roots[0].url` 替换为团队Wiki根文档链接。
3. 根据内部模型使用说明填写 `llm.base_url` 和 `llm.model`。
4. 在小鲁班开发工具中运行 `python main.py`，选择“加密数据”，分别加密W3密码和模型API Key，然后填写到 `extra_config`。
5. 将 `extra_config.w3_account` 替换为W3账号。
6. 内部AI市场当前页面的版本与本地 `uvx` 安装命令版本可能不同，发布前应把 `wiki_mcp.command` 替换为页面上最新的完整 `uvx` 命令。

不要把W3密码或模型Key以明文写进 `config.json`，也不要提交真实凭证。

## 本地配置覆盖

代码也支持使用以下环境变量覆盖敏感配置，便于联调：

- `WIKI_W3_ACCOUNT`
- `WIKI_W3_PASSWORD`
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

插件只调用两个只读工具：

- `search_wiki_documents`
- `fetch_wiki_content`

## 本地测试

测试使用假的MCP和模型客户端，不需要连接华为内网：

```powershell
conda run -n plugin python -m unittest discover -s wiki_knowledge_plugin/tests -v
```
