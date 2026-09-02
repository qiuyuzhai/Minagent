"""最小 ReAct 循环示例（P2 版：用 ToolRegistry + ToolExecutor）。

用计算器 + read_file 两个工具验证 agent loop：
1. 用户问"15 乘以 4 再加 12 是多少"
2. LLM 决定调用 calculator 工具
3. ToolExecutor 执行工具（parallel 模式），结果按原序返回
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
import os

from minagent.events import AgentEvent, EventStream
from minagent.llm import ModelConfig, OpenAICompatibleProvider
from minagent.loop import AgentContext, AgentLoopConfig, agent_loop
from minagent.tools import ToolExecutor, ToolRegistry
from minagent.tools.builtin import CalculatorTool, ReadFileTool


def build_registry() -> ToolRegistry:
    """构建工具注册表。"""
    reg = ToolRegistry()
    reg.register(CalculatorTool())   # parallel 模式
    reg.register(ReadFileTool())     # sequential 模式
    return reg


async def print_events(stream: EventStream) -> None:
    """订阅事件流，打印每个事件。"""
    async for event in stream:
        _print_event(event)


def _print_event(event: AgentEvent) -> None:
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
        for trc in event.message.content:
            for tc in trc.content:
                print(f"  -> {tc.text}")


async def main() -> None:
    # 1. 配置模型（自动检测 provider）
    model_config = ModelConfig(
        model=os.getenv("LLM_MODEL_ID") or "gpt-4o-mini",
        provider="auto",
    )
    provider = OpenAICompatibleProvider(model_config)
    print(f"Provider: {provider.provider_name}, Model: {provider.model}")

    # 2. 构建工具注册表 + 执行器
    registry = build_registry()
    print(f"Tools: {[t.name for t in registry.list_tools()]}")

    executor = ToolExecutor(
        registry,
        before_tool_call=lambda ctx: None,  # 可加权限拦截
        default_execution_mode="parallel",
    )

    # 3. 配置 agent context
    context = AgentContext(
        system_prompt="你是一个能使用计算器和文件读取工具的助手。需要计算时调用 calculator。",
        tools=registry.to_openai_schemas(),
    )

    # 4. 配置 agent loop（P2 路径：tool_executor）
    config = AgentLoopConfig(
        model_config=model_config,
        max_turns=10,
        tool_executor=executor,
    )

    # 5. 启动 agent loop
    prompt = "请帮我计算：15 乘以 4，然后加上 12 是多少？"
    print(f"\n用户: {prompt}")

    stream = await agent_loop(prompt, context, config, provider)

    # 6. 订阅事件流
    await print_events(stream)


if __name__ == "__main__":
    asyncio.run(main())
