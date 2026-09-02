"""中断控制示例：steering + abort + follow-up。

演示 P3 control 层：
1. agent 运行中用户通过 steering_queue 插话"换个方向"
2. agent 在 turn 间隙收到插话，下一轮看到
3. 用户通过 abort_controller 中断
4. follow_up_queue 让 agent 自然停止后续跑

运行：
    python -m examples.control_demo
"""

import asyncio
import os

from minagent.control import AbortController, FollowUpQueue, SteeringQueue
from minagent.events import AgentEvent, EventStream
from minagent.llm import ModelConfig, OpenAICompatibleProvider
from minagent.loop import AgentContext, AgentLoopConfig, agent_loop
from minagent.messages import UserMessage
from minagent.tools import ToolExecutor, ToolRegistry
from minagent.tools.builtin import CalculatorTool


def _print_event(event: AgentEvent) -> None:
    t = event.type
    if t == "turn_start":
        print(f"\n--- turn {event.turn_index} ---")
    elif t == "message_update":
        print(event.delta_text, end="", flush=True)
    elif t == "tool_execution_start":
        print(f"\n  [tool] {event.tool_name}({event.arguments})")
    elif t == "tool_result_message":
        for trc in event.message.content:
            for tc in trc.content:
                print(f"  -> {tc.text}")
    elif t == "agent_end":
        print(f"\n=== agent end ({len(event.messages)} msgs) ===")


async def main() -> None:
    model_config = ModelConfig(
        model=os.getenv("LLM_MODEL_ID") or "gpt-4o-mini",
        provider="auto",
    )
    provider = OpenAICompatibleProvider(model_config)
    print(f"Provider: {provider.provider_name}, Model: {provider.model}")

    # P3 控制层
    abort_controller = AbortController()
    steering_queue = SteeringQueue()
    follow_up_queue = FollowUpQueue()

    # 工具
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    executor = ToolExecutor(registry)

    context = AgentContext(
        system_prompt="你是助手，可以调用计算器。",
        tools=registry.to_openai_schemas(),
    )
    config = AgentLoopConfig(
        model_config=model_config,
        max_turns=10,
        tool_executor=executor,
        abort_controller=abort_controller,
        steering_queue=steering_queue,
        follow_up_queue=follow_up_queue,
    )

    # 启动 agent loop（后台运行）
    stream = await agent_loop("帮我算 15 * 4", context, config, provider)

    # 模拟用户中途插话（steering）
    async def user_steering():
        await asyncio.sleep(1.0)  # 等 agent 跑一会
        print("\n>>> [用户插话] 再算个 23 + 17")
        steering_queue.push_nowait(UserMessage.from_text("再算个 23 + 17"))

    # 模拟 follow-up
    async def user_follow_up():
        await asyncio.sleep(3.0)
        print("\n>>> [follow-up] 最后算个 100 / 4")
        follow_up_queue.push_nowait(UserMessage.from_text("最后算个 100 / 4"))

    asyncio.create_task(user_steering())
    asyncio.create_task(user_follow_up())

    # 订阅事件流
    async for event in stream:
        _print_event(event)


if __name__ == "__main__":
    asyncio.run(main())
