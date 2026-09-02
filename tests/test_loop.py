"""P1 loop 骨架测试：用 MockProvider 验证 agent loop 逻辑，不依赖真实 LLM。"""

import asyncio
import json
from typing import AsyncIterator

import pytest

from minagent.events import EventStream, AgentEvent
from minagent.llm import LLMProvider, ModelConfig, StreamChunk
from minagent.loop import AgentContext, AgentLoopConfig, agent_loop
from minagent.messages import ToolCall


class MockProvider(LLMProvider):
    """按预设脚本返回响应的 mock provider。"""

    def __init__(self, scripts: list[list[StreamChunk]]):
        """scripts: 每次调用的 chunk 列表列表。"""
        self._scripts = scripts
        self._call_index = 0

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        config: ModelConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if self._call_index >= len(self._scripts):
            # 默认返回空停止
            yield StreamChunk(delta_text="(无更多响应)", finish_reason="stop")
            return
        chunks = self._scripts[self._call_index]
        self._call_index += 1
        for chunk in chunks:
            yield chunk


def make_calculator_tool() -> dict:
    """计算器工具 schema。"""
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算器",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    }


def calculator_execute(tc: ToolCall) -> str:
    """计算器工具实现。"""
    expr = tc.arguments.get("expression", "")
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expr):
            return f"ERROR: invalid expression: {expr}"
        result = eval(expr, {"__builtins__": {}}, {})
        return f"{expr} = {result}"
    except Exception as e:
        return f"ERROR: {e}"


async def collect_events(stream: EventStream) -> tuple[list[AgentEvent], list]:
    """收集所有事件，返回 (events, final_messages)。"""
    events: list[AgentEvent] = []
    final_messages = []
    async for event in stream:
        events.append(event)
        if event.type == "agent_end":
            final_messages = event.messages
    return events, final_messages


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_tool_call_loop():
    """测试：LLM 调用一次工具后给出答案。"""
    # mock provider 脚本：
    # 第 1 次调用：返回工具调用
    # 第 2 次调用：返回最终文本
    scripts = [
        # turn 1: 调用 calculator
        [
            StreamChunk(tool_call_id="call_1", tool_call_name="calculator"),
            StreamChunk(tool_call_arguments_delta=json.dumps({"expression": "15 * 4 + 12"})),
            StreamChunk(finish_reason="tool_calls"),
        ],
        # turn 2: 给出答案
        [
            StreamChunk(delta_text="15 乘以 4 再加 12 等于 72。"),
            StreamChunk(finish_reason="stop"),
        ],
    ]
    provider = MockProvider(scripts)

    context = AgentContext(
        system_prompt="你是助手",
        tools=[make_calculator_tool()],
    )
    config = AgentLoopConfig(
        model_config=ModelConfig(model="mock"),
        max_turns=10,
        execute_tool=calculator_execute,
    )

    stream = await agent_loop("算一下 15*4+12", context, config, provider)
    events, final_messages = await collect_events(stream)

    # 验证事件序列
    event_types = [e.type for e in events]
    assert "agent_start" in event_types
    assert "agent_end" in event_types
    assert event_types.count("turn_start") == 2  # 两个 turn
    assert event_types.count("turn_end") == 2

    # 验证工具执行
    tool_starts = [e for e in events if e.type == "tool_execution_start"]
    assert len(tool_starts) == 1
    assert tool_starts[0].tool_name == "calculator"
    assert tool_starts[0].arguments["expression"] == "15 * 4 + 12"

    tool_ends = [e for e in events if e.type == "tool_execution_end"]
    assert len(tool_ends) == 1
    assert tool_ends[0].is_error is False

    # 验证工具结果
    tool_results = [e for e in events if e.type == "tool_result_message"]
    assert len(tool_results) == 1
    result_text = tool_results[0].message.content[0].content[0].text
    assert "72" in result_text

    # 验证最终消息数：user + assistant(工具调用) + tool_result + assistant(答案) = 4
    # (system_prompt 也会加入，所以 +1 = 5)
    assert len(final_messages) >= 4


@pytest.mark.asyncio
async def test_max_turns_protection():
    """测试：max_turns 死循环防护。"""
    # mock provider 每次都返回工具调用，永不停止
    script = [
        StreamChunk(tool_call_id="call_1", tool_call_name="calculator"),
        StreamChunk(tool_call_arguments_delta=json.dumps({"expression": "1+1"})),
        StreamChunk(finish_reason="tool_calls"),
    ]
    # 复制 100 次，确保超过 max_turns
    scripts = [script[:] for _ in range(100)]
    provider = MockProvider(scripts)

    context = AgentContext(
        system_prompt="你是助手",
        tools=[make_calculator_tool()],
    )
    config = AgentLoopConfig(
        model_config=ModelConfig(model="mock"),
        max_turns=3,  # 限制 3 轮
        execute_tool=calculator_execute,
    )

    stream = await agent_loop("死循环测试", context, config, provider)
    events, _ = await collect_events(stream)

    # 验证 turn 数不超过 max_turns
    turn_starts = [e for e in events if e.type == "turn_start"]
    assert len(turn_starts) <= 3


@pytest.mark.asyncio
async def test_no_tool_call_stops():
    """测试：LLM 不调用工具时，agent 自然停止。"""
    scripts = [
        # 唯一一次调用：直接返回文本，无工具调用
        [
            StreamChunk(delta_text="你好，我不需要计算。"),
            StreamChunk(finish_reason="stop"),
        ],
    ]
    provider = MockProvider(scripts)

    context = AgentContext(system_prompt="你是助手", tools=[make_calculator_tool()])
    config = AgentLoopConfig(
        model_config=ModelConfig(model="mock"),
        max_turns=10,
        execute_tool=calculator_execute,
    )

    stream = await agent_loop("你好", context, config, provider)
    events, _ = await collect_events(stream)

    # 只有一个 turn
    turn_starts = [e for e in events if e.type == "turn_start"]
    assert len(turn_starts) == 1

    # 没有工具执行
    tool_starts = [e for e in events if e.type == "tool_execution_start"]
    assert len(tool_starts) == 0

    # 有 agent_end
    assert any(e.type == "agent_end" for e in events)


@pytest.mark.asyncio
async def test_should_stop_after_turn_hook():
    """测试：should_stop_after_turn 钩子能优雅停止。"""
    scripts = [
        # turn 1: 工具调用
        [
            StreamChunk(tool_call_id="call_1", tool_call_name="calculator"),
            StreamChunk(tool_call_arguments_delta=json.dumps({"expression": "1+1"})),
            StreamChunk(finish_reason="tool_calls"),
        ],
        # turn 2: 文本（不应该被调用，因为钩子在 turn 1 后停止）
        [
            StreamChunk(delta_text="不应该到这里"),
            StreamChunk(finish_reason="stop"),
        ],
    ]
    provider = MockProvider(scripts)

    context = AgentContext(system_prompt="你是助手", tools=[make_calculator_tool()])
    config = AgentLoopConfig(
        model_config=ModelConfig(model="mock"),
        max_turns=10,
        execute_tool=calculator_execute,
        should_stop_after_turn=lambda ctx: True,  # 第一个 turn 后立即停止
    )

    stream = await agent_loop("测试钩子停止", context, config, provider)
    events, _ = await collect_events(stream)

    # 只有一个 turn（钩子在 turn 1 后停止）
    turn_starts = [e for e in events if e.type == "turn_start"]
    assert len(turn_starts) == 1
