"""P3 control 层测试：abort + steering + follow-up + timeout。"""

import asyncio
import json
from typing import AsyncIterator

import pytest

from minagent.control import (
    AbortController,
    FollowUpQueue,
    SteeringQueue,
    TimeoutManager,
)
from minagent.events import EventStream
from minagent.llm import LLMProvider, ModelConfig, StreamChunk
from minagent.loop import AgentContext, AgentLoopConfig, agent_loop
from minagent.messages import UserMessage


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """按脚本返回响应的 mock provider。"""
    def __init__(self, scripts: list[list[StreamChunk]]):
        self._scripts = scripts
        self._call = 0

    async def stream(self, messages, tools=None, config=None) -> AsyncIterator[StreamChunk]:
        if self._call >= len(self._scripts):
            yield StreamChunk(delta_text="(结束)", finish_reason="stop")
            return
        chunks = self._scripts[self._call]
        self._call += 1
        for chunk in chunks:
            yield chunk


async def collect_events(stream: EventStream):
    events = []
    final_msgs = []
    async for event in stream:
        events.append(event)
        if event.type == "agent_end":
            final_msgs = event.messages
    return events, final_msgs


# ---------------------------------------------------------------------------
# AbortController 测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_abort_controller_basic():
    """AbortController 基本功能。"""
    controller = AbortController()
    assert not controller.is_aborted()

    controller.abort("test")
    assert controller.is_aborted()
    assert controller.reason == "test"

    controller.reset()
    assert not controller.is_aborted()


@pytest.mark.asyncio
async def test_abort_controller_wait():
    """wait_for_abort 等待中断。"""
    controller = AbortController()

    async def trigger():
        await asyncio.sleep(0.05)
        controller.abort("triggered")

    asyncio.create_task(trigger())
    result = await controller.wait_for_abort(timeout=1.0)
    assert result is True
    assert controller.reason == "triggered"


@pytest.mark.asyncio
async def test_abort_controller_timeout():
    """wait_for_abort 超时返回 False。"""
    controller = AbortController()
    result = await controller.wait_for_abort(timeout=0.05)
    assert result is False


# ---------------------------------------------------------------------------
# SteeringQueue / FollowUpQueue 测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_steering_queue_push_drain():
    """SteeringQueue push + drain。"""
    queue = SteeringQueue()
    assert queue.is_empty()

    await queue.push(UserMessage.from_text("换个方向"))
    await queue.push(UserMessage.from_text("用另一种方法"))
    assert not queue.is_empty()

    msgs = queue.drain()
    assert len(msgs) == 2
    assert msgs[0].content[0].text == "换个方向"
    assert msgs[1].content[0].text == "用另一种方法"
    assert queue.is_empty()


@pytest.mark.asyncio
async def test_follow_up_queue_push_drain():
    """FollowUpQueue push + drain。"""
    queue = FollowUpQueue()
    await queue.push(UserMessage.from_text("再算一个"))
    msgs = queue.drain()
    assert len(msgs) == 1
    assert msgs[0].content[0].text == "再算一个"


# ---------------------------------------------------------------------------
# TimeoutManager 测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_manager_tool_timeout():
    """工具级超时。"""
    tm = TimeoutManager(tool_timeout=0.1)

    async def slow_coro():
        await asyncio.sleep(1.0)
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        await tm.run_with_tool_timeout(slow_coro())


@pytest.mark.asyncio
async def test_timeout_manager_no_timeout():
    """不设超时不触发。"""
    tm = TimeoutManager(tool_timeout=None)

    async def fast_coro():
        await asyncio.sleep(0.01)
        return "done"

    result = await tm.run_with_tool_timeout(fast_coro())
    assert result == "done"


# ---------------------------------------------------------------------------
# loop 集成：abort
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_abort_stops_after_turn():
    """abort 在 turn 结束后停止循环。"""
    scripts = [
        # turn 1: 工具调用
        [
            StreamChunk(tool_call_id="c1", tool_call_name="calculator"),
            StreamChunk(tool_call_arguments_delta=json.dumps({"expression": "1+1"})),
            StreamChunk(finish_reason="tool_calls"),
        ],
        # turn 2: 不应到达（abort 后停止）
        [StreamChunk(delta_text="不应到达"), StreamChunk(finish_reason="stop")],
    ]
    provider = MockProvider(scripts)
    controller = AbortController()

    # 构造一个简单 execute_tool，执行后 abort
    from minagent.messages import ToolCall
    def execute_tool(tc: ToolCall) -> str:
        # 工具执行后触发 abort
        controller.abort("user requested")
        return "1+1 = 2"

    context = AgentContext(system_prompt="你是助手")
    config = AgentLoopConfig(
        model_config=ModelConfig(model="mock"),
        max_turns=10,
        execute_tool=execute_tool,
        abort_controller=controller,
    )

    stream = await agent_loop("算 1+1", context, config, provider)
    events, _ = await collect_events(stream)

    # 只有 1 个 turn（abort 在工具执行后，turn 结束前检查 abort）
    turn_starts = [e for e in events if e.type == "turn_start"]
    assert len(turn_starts) == 1


# ---------------------------------------------------------------------------
# loop 集成：steering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_steering_injects_message():
    """steering 消息在 turn 结束后注入，loop 继续。"""
    scripts = [
        # turn 1: 工具调用
        [
            StreamChunk(tool_call_id="c1", tool_call_name="calculator"),
            StreamChunk(tool_call_arguments_delta=json.dumps({"expression": "1+1"})),
            StreamChunk(finish_reason="tool_calls"),
        ],
        # turn 2: 看到 steering 消息后回应
        [StreamChunk(delta_text="好的，我换个方向"), StreamChunk(finish_reason="stop")],
    ]
    provider = MockProvider(scripts)
    steering = SteeringQueue()

    from minagent.messages import ToolCall
    def execute_tool(tc: ToolCall) -> str:
        # 工具执行时同步插入 steering 消息
        steering.push_nowait(UserMessage.from_text("换个方向"))
        return "1+1 = 2"

    context = AgentContext(system_prompt="你是助手")
    config = AgentLoopConfig(
        model_config=ModelConfig(model="mock"),
        max_turns=10,
        execute_tool=execute_tool,
        steering_queue=steering,
    )

    stream = await agent_loop("算 1+1", context, config, provider)
    events, final_msgs = await collect_events(stream)

    # 应该有 2 个 turn（steering 让 loop 继续）
    turn_starts = [e for e in events if e.type == "turn_start"]
    assert len(turn_starts) == 2

    # steering 消息应在 final_messages 里
    texts = []
    for m in final_msgs:
        if isinstance(m, UserMessage):
            texts.append(m.content[0].text)
    assert "换个方向" in texts


# ---------------------------------------------------------------------------
# loop 集成：follow-up
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_follow_up_continues():
    """follow-up 消息让 agent 自然停止后续跑。"""
    scripts = [
        # turn 1: 直接回答（无工具，自然停止）
        [StreamChunk(delta_text="第一次回答"), StreamChunk(finish_reason="stop")],
        # follow-up 后 turn 2: 再回答
        [StreamChunk(delta_text="第二次回答"), StreamChunk(finish_reason="stop")],
    ]
    provider = MockProvider(scripts)
    follow_up = FollowUpQueue()

    # 预先放入 follow-up 消息
    await follow_up.push(UserMessage.from_text("再问一个"))

    context = AgentContext(system_prompt="你是助手")
    config = AgentLoopConfig(
        model_config=ModelConfig(model="mock"),
        max_turns=10,
        follow_up_queue=follow_up,
    )

    stream = await agent_loop("第一次问", context, config, provider)
    events, final_msgs = await collect_events(stream)

    # 应该有 2 个 turn（follow-up 续跑）
    turn_starts = [e for e in events if e.type == "turn_start"]
    assert len(turn_starts) == 2

    # follow-up 消息应在 final_messages 里
    texts = []
    for m in final_msgs:
        if isinstance(m, UserMessage):
            texts.append(m.content[0].text)
    assert "再问一个" in texts


@pytest.mark.asyncio
async def test_loop_no_follow_up_stops():
    """无 follow-up 消息时 agent 自然停止。"""
    scripts = [
        [StreamChunk(delta_text="回答"), StreamChunk(finish_reason="stop")],
    ]
    provider = MockProvider(scripts)
    follow_up = FollowUpQueue()  # 空队列

    context = AgentContext(system_prompt="你是助手")
    config = AgentLoopConfig(
        model_config=ModelConfig(model="mock"),
        max_turns=10,
        follow_up_queue=follow_up,
    )

    stream = await agent_loop("你好", context, config, provider)
    events, _ = await collect_events(stream)

    turn_starts = [e for e in events if e.type == "turn_start"]
    assert len(turn_starts) == 1  # 只有一个 turn，自然停止
