"""公司内部 OpenAI 兼容模型独立诊断脚本。

该文件可以单独复制到内网机器执行，不依赖小鲁班 ``util`` 包，也不会调用
Wiki-MCP。它验证的内容与知识库插件使用模型的方式一致：

1. 查询 ``/v1/models`` 获取真实模型 ID；
2. 测试普通对话；
3. 测试模型是否会返回 ``tool_calls``；
4. 模拟执行工具，并把工具结果回传给模型生成最终回答；
5. 将结果保存为不包含 API Key 的 JSON 报告。

推荐命令：

    # 测试 config.json 中配置的模型
    python test_internal_llm.py

    # 测试所有名称中包含 claude 的模型
    python test_internal_llm.py --contains claude

    # 指定一个或多个精确模型 ID
    python test_internal_llm.py --model "模型ID一" --model "模型ID二"

    # 逐个测试接口返回的全部模型（可能耗时较长）
    python test_internal_llm.py --all

内部接口文档建议绕过系统代理，脚本默认使用 ``--network-mode direct``。
如果需要复现系统代理环境，可改为 ``--network-mode environment``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_BASE_URL = "https://www.juaiapi.com/v1"
DEFAULT_API_KEY = "sk-KbjVFwLSLZqoMMOvUpeKXsour4I4rXe66TnMajQfSwraIAqf"
DEFAULT_MODEL = "gpt-5.6-sol"


@dataclass
class CheckResult:
    success: bool
    detail: str
    elapsed_seconds: float


@dataclass
class ModelTestResult:
    model: str
    basic_chat: CheckResult
    tool_call: CheckResult
    tool_round_trip: CheckResult

    @property
    def passed(self) -> bool:
        return (
            self.basic_chat.success
            and self.tool_call.success
            and self.tool_round_trip.success
        )


def _usable(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.startswith(("__REPLACE_", "YOUR_", "请替换")):
        return ""
    return text


def load_defaults(config_path: Path) -> tuple[str, str, str]:
    base_url, api_key, model = DEFAULT_BASE_URL, DEFAULT_API_KEY, DEFAULT_MODEL
    if not config_path.exists():
        return base_url, api_key, model
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        llm = raw.get("llm", {})
        if isinstance(llm, dict):
            base_url = _usable(llm.get("base_url")) or base_url
            api_key = _usable(llm.get("api_key")) or api_key
            model = _usable(llm.get("model")) or model
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] config.json 读取失败，将使用脚本内默认配置：{exc}")
    return base_url.rstrip("/"), api_key, model


def safe_error(exc: BaseException, api_key: str) -> str:
    text = str(exc).strip() or type(exc).__name__
    if api_key:
        text = text.replace(api_key, "***")
    return f"{type(exc).__name__}: {text}"[:1500]


def create_client(base_url: str, api_key: str, network_mode: str, timeout: float) -> Any:
    try:
        import httpx
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "缺少依赖，请先执行：pip install openai httpx"
        ) from exc

    trust_env = network_mode == "environment"
    http_client = httpx.Client(
        timeout=timeout,
        trust_env=False,
        follow_redirects=True,
    )
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        http_client=http_client,
        # 内部接口文档的 curl 示例使用 x-goog-api-key；同时保留OpenAI SDK
        # 自动生成的 Authorization 头，以兼容两种网关配置。
        default_headers={"x-goog-api-key": api_key},
    )


def list_models(client: Any, api_key: str) -> tuple[list[str], CheckResult]:
    started = time.perf_counter()
    try:
        response = client.models.list()
        model_ids = sorted(
            {
                str(getattr(model, "id", "") or "").strip()
                for model in getattr(response, "data", []) or []
                if str(getattr(model, "id", "") or "").strip()
            },
            key=str.casefold,
        )
        elapsed = time.perf_counter() - started
        return model_ids, CheckResult(True, f"获取到 {len(model_ids)} 个模型", elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return [], CheckResult(False, safe_error(exc, api_key), elapsed)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def test_basic_chat(client: Any, model: str, api_key: str) -> CheckResult:
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "这是接口连通性测试。请只回复：LLM_BASIC_OK",
                }
            ],
            max_tokens=64,
            stream=False,
        )
        message = response.choices[0].message
        text = _message_text(message)
        elapsed = time.perf_counter() - started
        if not text:
            return CheckResult(False, "接口返回成功，但回答内容为空", elapsed)
        return CheckResult(True, f"回答：{text[:300]}", elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return CheckResult(False, safe_error(exc, api_key), elapsed)


def test_tool_call_and_round_trip(
    client: Any, model: str, api_key: str
) -> tuple[CheckResult, CheckResult]:
    started = time.perf_counter()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "在团队知识库中搜索资料。测试时必须调用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_key": {
                            "type": "string",
                            "description": "需要搜索的关键词",
                        }
                    },
                    "required": ["search_key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_knowledge_document",
                "description": "读取搜索结果中的Wiki文档正文。必须在搜索后调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "搜索结果返回的Wiki链接",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你正在进行知识库工具调用测试。必须先调用search_knowledge_base，"
                "然后调用read_knowledge_document读取正文，最后才能回答。"
            ),
        },
        {
            "role": "user",
            "content": "请搜索关键词“设备连接”，然后根据工具结果回答。",
        },
    ]

    search_url = "https://wiki.huawei.com/domains/189176/wiki/400668/WIKI2026070611733269"
    search_result = json.dumps(
        {
            "total_records": 1,
            "records": [
                {
                    "title": "RK3568烧机指导",
                    "url": search_url,
                    "content_match_snippets": ["对RK3568芯片的烧机流程。"],
                }
            ],
        },
        ensure_ascii=False,
    )
    document_result = json.dumps(
        {
            "title": "设备连接测试文档",
            "url": search_url,
            "content": "测试资料：设备连接前应检查连接线，并确认驱动正常。",
        },
        ensure_ascii=False,
    )
    saw_search = False
    saw_read = False
    search_detail = ""

    try:
        for round_index in range(6):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=256,
                stream=False,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            elapsed = time.perf_counter() - started

            if not tool_calls:
                final_text = _message_text(message)
                if not saw_search:
                    detail = "模型没有先调用search_knowledge_base"
                    if final_text:
                        detail += f"，而是直接回答：{final_text[:300]}"
                    failed = CheckResult(False, detail, elapsed)
                    return failed, CheckResult(False, "未进入工具结果回传阶段", 0.0)
                tool_check = CheckResult(True, search_detail, elapsed)
                if not saw_read:
                    return tool_check, CheckResult(
                        False, "模型搜索后没有调用read_knowledge_document读取正文", elapsed
                    )
                if not final_text:
                    return tool_check, CheckResult(False, "工具结果回传后回答为空", elapsed)
                return tool_check, CheckResult(
                    True, f"最终回答：{final_text[:500]}", elapsed
                )

            serialized_calls: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                tool_name = str(getattr(function, "name", "") or "")
                raw_arguments = str(getattr(function, "arguments", "{}") or "{}")
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": raw_arguments}
                if not isinstance(arguments, dict):
                    arguments = {}

                if round_index == 0 and tool_name != "search_knowledge_base":
                    failed = CheckResult(
                        False,
                        f"首次调用的工具不是search_knowledge_base，而是{tool_name}",
                        elapsed,
                    )
                    return failed, CheckResult(False, "工具调用顺序不正确", 0.0)

                serialized_calls.append(
                    {
                        "id": str(getattr(tool_call, "id", "")),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": raw_arguments,
                        },
                    }
                )
                if tool_name == "search_knowledge_base":
                    saw_search = True
                    search_detail = (
                        "成功调用search_knowledge_base，参数："
                        + json.dumps(arguments, ensure_ascii=False)
                    )
                elif tool_name == "read_knowledge_document" and saw_search:
                    saw_read = True

            messages.append(
                {
                    "role": "assistant",
                    "content": getattr(message, "content", None),
                    "tool_calls": serialized_calls,
                }
            )
            for tool_call in serialized_calls:
                tool_name = tool_call["function"]["name"]
                if tool_name == "search_knowledge_base":
                    result = search_result
                elif tool_name == "read_knowledge_document":
                    result = document_result
                else:
                    result = json.dumps({"error": f"未知测试工具：{tool_name}"}, ensure_ascii=False)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )

        elapsed = time.perf_counter() - started
        tool_check = CheckResult(saw_search, search_detail or "模型未调用搜索工具", elapsed)
        return tool_check, CheckResult(False, "超过6轮仍未生成最终回答", elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        failed = CheckResult(False, safe_error(exc, api_key), elapsed)
        return failed, CheckResult(False, "工具调用阶段异常，未完成回传", 0.0)


def test_model(client: Any, model: str, api_key: str) -> ModelTestResult:
    print(f"\n{'=' * 72}\n测试模型：{model}\n{'=' * 72}")
    basic = test_basic_chat(client, model, api_key)
    print(f"[{'PASS' if basic.success else 'FAIL'}] 普通对话：{basic.detail}")

    tool_call, round_trip = test_tool_call_and_round_trip(client, model, api_key)
    print(f"[{'PASS' if tool_call.success else 'FAIL'}] 工具调用：{tool_call.detail}")
    print(f"[{'PASS' if round_trip.success else 'FAIL'}] 工具回传：{round_trip.detail}")
    return ModelTestResult(model, basic, tool_call, round_trip)


def choose_models(
    available: list[str], configured: str, explicit: list[str], contains: str, test_all: bool
) -> list[str]:
    if explicit:
        return list(dict.fromkeys(explicit))
    if test_all:
        return available
    if contains:
        keyword = contains.casefold()
        return [model for model in available if keyword in model.casefold()]
    return [configured]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="逐个测试内部OpenAI兼容模型及工具调用能力")
    parser.add_argument("--config", type=Path, default=script_dir / "config.json")
    parser.add_argument("--base-url", help="覆盖config.json中的模型接口地址")
    parser.add_argument("--api-key", help="覆盖config.json中的API Key；不会写入报告")
    parser.add_argument("--model", action="append", default=[], help="精确模型ID，可重复指定")
    parser.add_argument("--contains", default="", help="只测试名称中包含该文本的模型")
    parser.add_argument("--all", action="store_true", help="测试/models返回的全部模型")
    parser.add_argument(
        "--network-mode",
        choices=("direct", "environment"),
        default="direct",
        help="direct绕过系统代理；environment使用系统代理配置",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--delay", type=float, default=0.2, help="测试不同模型之间的等待秒数")
    parser.add_argument("--limit", type=int, default=0, help="最多测试多少个模型，0表示不限制")
    parser.add_argument(
        "--report",
        type=Path,
        default=script_dir / "llm_test_report.json",
        help="JSON报告保存位置",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_base_url, config_api_key, configured_model = load_defaults(args.config)
    base_url = (args.base_url or config_base_url).rstrip("/")
    api_key = args.api_key or config_api_key

    if urlparse(base_url).scheme not in {"http", "https"}:
        print(f"[FAIL] base_url格式错误：{base_url}")
        return 2
    if not api_key:
        print("[FAIL] API Key为空，请在config.json中填写或使用--api-key传入。")
        return 2

    print("内部模型诊断开始")
    print(f"接口地址：{base_url}")
    print(f"网络模式：{args.network_mode}")
    print(f"配置模型：{configured_model}")
    print("API Key：已读取（不会输出到控制台或报告）")

    try:
        client = create_client(base_url, api_key, args.network_mode, args.timeout)
    except Exception as exc:
        print(f"[FAIL] 客户端创建失败：{safe_error(exc, api_key)}")
        return 2

    available, list_check = list_models(client, api_key)
    print(f"[{'PASS' if list_check.success else 'FAIL'}] 模型列表：{list_check.detail}")
    if available:
        print("\n接口返回的模型ID：")
        for index, model in enumerate(available, 1):
            configured_mark = "  <-- config.json当前配置" if model == configured_model else ""
            print(f"{index:>3}. {model}{configured_mark}")
    if available and configured_model not in available:
        similar = get_close_matches(configured_model, available, n=5, cutoff=0.2)
        print(f"\n[WARN] 配置模型“{configured_model}”不在/models返回列表中。")
        if similar:
            print("可能的模型ID：" + "、".join(similar))

    selected = choose_models(
        available, configured_model, args.model, args.contains, args.all
    )
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        print("[FAIL] 没有匹配到需要测试的模型。")
        return 2

    if args.all:
        print(f"\n[WARN] 即将逐个调用 {len(selected)} 个模型，请确认符合内部服务使用规范。")

    results: list[ModelTestResult] = []
    for index, model in enumerate(selected):
        if index and args.delay > 0:
            time.sleep(args.delay)
        results.append(test_model(client, model, api_key))

    report = {
        "tested_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "network_mode": args.network_mode,
        "model_list": asdict(list_check),
        "available_models": available,
        "results": [
            {
                **asdict(result),
                "passed": result.passed,
            }
            for result in results
        ],
    }
    try:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告已保存：{args.report.resolve()}")
    except OSError as exc:
        print(f"\n[WARN] 报告写入失败：{exc}")

    passed = sum(result.passed for result in results)
    print(f"测试完成：{passed}/{len(results)} 个模型通过普通对话和完整工具调用测试。")
    return 0 if passed == len(results) and list_check.success else 1


if __name__ == "__main__":
    sys.exit(main())
