"""最小 ReAct 循环示例。

用计算器工具验证 agent loop：
1. 用户问"15 乘以 4 再加 12 是多少"
2. LLM 决定调用 calculator 工具
3. agent loop 执行工具，把结果返回给 LLM
4. LLM 给出最终答案

运行前设置环境变量（任选一个 provider）：
    # OpenAI
    set OPENAI_API_KEY=sk-xxx
    # Ollama (本地)
    set LLM_BASE_URL=http://localhost:11434/v1
    # 智谱
    set ZHIPU_API_KEY=xxx

运行：
    python -m examples.basic_react
"""

import asyncio
import json
import os

from minagent.events import EventStream, AgentEvent
from minagent.llm import LLMProvider, ModelConfig, OpenAICompatibleProvider
from minagent.loop import AgentContext, AgentLoopConfig, agent_loop
from minagent.messages import ToolCall


# ---------------------------------------------------------------------------
# 工具定义：计算器（P1 简单版，P2 会用 ToolRegistry + pydantic schema）
# ---------------------------------------------------------------------------

def calculator_execute(tool_call: ToolCall) -> str:
    """计算器工具：支持四则运算。

    P1 版直接 eval，P2 会换成安全的 AST 解析。
    """
    expr = tool_call.arguments.get("expression", "")
    if not expr:
        return "ERROR: missing 'expression' argument"
    try:
        # P1 简单版：限制字符集后 eval
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expr):
            return f"ERROR: invalid expression: {expr}"
        result = eval(expr, {"__builtins__": {}}, {})
        return f"{expr} = {result}"
    except Exception as e:
        return f"ERROR: {e}"


# OpenAI function calling schema
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "四则运算计算器。输入数学表达式，返回计算结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '15 * 4 + 12'",
                },
            },
            "required": ["expression"],
        },
    },
}


# ---------------------------------------------------------------------------
# 事件打印器（模拟 UI 订阅事件流）
# ---------------------------------------------------------------------------

async def print_events(stream: EventStream) -> None:
    """订阅事件流，打印每个事件。"""
    async for event in stream:
        _print_event(event)


def _print_event(event: AgentEvent) -> None:
    """打印单个事件。"""
    t = event.type
    if t == "agent_start":
        print("\n=== agent_start ===")
    elif t == "agent_end":
        print(f"\n=== agent_end ({len(event.messages)} messages) ===")
    elif t == "turn_start":
        print(f"\n--- turn {event.turn_index} ---")
    elif t == "turn_end":
        n_tools = len(event.tool_results)
        print(f"--- turn end (tools: {n_tools}) ---")
    elif t == "message_start":
        print("[msg start]")
    elif t == "message_update":
        print(event.delta_text, end="", flush=True)
    elif t == "message_end":
        print("\n[msg end]")
    elif t == "tool_execution_start":
        print(f"\n  [tool] {event.tool_name}({event.arguments})")
    elif t == "tool_execution_end":
        status = "ERROR" if event.is_error else "ok"
        print(f"  [tool result: {status}]")
    elif t == "tool_result_message":
        # 打印工具结果内容
        for trc in event.message.content:
            for tc in trc.content:
                print(f"  -> {tc.text}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

async def main() -> None:
    # 1. 配置模型（自动检测 provider）
    model_config = ModelConfig(
        model=os.getenv("LLM_MODEL_ID") or "gpt-4o-mini",
        provider="auto",
    )
    provider = OpenAICompatibleProvider(model_config)
    print(f"✅ Provider: {provider.provider_name}, Model: {provider.model}")

    # 2. 配置 agent context
    context = AgentContext(
        system_prompt="你是一个能使用计算器工具的助手。需要计算时调用 calculator 工具。",
        tools=[CALCULATOR_TOOL],
    )

    # 3. 配置 agent loop
    config = AgentLoopConfig(
        model_config=model_config,
        max_turns=10,
        execute_tool=calculator_execute,
    )

    # 4. 启动 agent loop
    prompt = "请帮我计算：15 乘以 4，然后加上 12 是多少？"
    print(f"\n🤖 用户: {prompt}")

    stream = await agent_loop(prompt, context, config, provider)

    # 5. 订阅事件流（边产生边打印）
    await print_events(stream)


if __name__ == "__main__":
    asyncio.run(main())
