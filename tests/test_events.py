"""events EventStream 测试。"""

import asyncio

import pytest

from minagent.events import (
    AgentEndEvent,
    AgentStartEvent,
    EventStream,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from minagent.messages import AssistantMessage, TextContent


@pytest.mark.asyncio
async def test_event_stream_basic():
    """基本事件流：emit -> end -> 迭代消费。"""
    stream = EventStream()

    async def producer():
        await stream.emit(AgentStartEvent())
        await stream.emit(TurnStartEvent(turn_index=1))
        await stream.emit(MessageUpdateEvent(delta_text="hello"))
        await stream.emit(TurnEndEvent(message=AssistantMessage(content=[])))
        await stream.end([])

    asyncio.create_task(producer())

    events = []
    async for event in stream:
        events.append(event)

    assert len(events) == 5  # 4 events + agent_end
    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], TurnStartEvent)
    assert isinstance(events[2], MessageUpdateEvent)
    assert events[2].delta_text == "hello"
    assert isinstance(events[3], TurnEndEvent)
    assert isinstance(events[4], AgentEndEvent)


@pytest.mark.asyncio
async def test_event_stream_end_with_messages():
    """end 携带最终消息。"""
    stream = EventStream()
    final_msgs = [AssistantMessage(content=[TextContent(text="done")])]

    async def producer():
        await stream.emit(AgentStartEvent())
        await stream.end(final_msgs)

    asyncio.create_task(producer())

    events = []
    async for event in stream:
        events.append(event)

    assert len(events) == 2  # agent_start + agent_end
    assert isinstance(events[1], AgentEndEvent)
    assert events[1].messages == final_msgs


@pytest.mark.asyncio
async def test_event_stream_emit_after_end_ignored():
    """end 后再 emit 被忽略。"""
    stream = EventStream()

    await stream.emit(AgentStartEvent())
    await stream.end([])
    await stream.emit(MessageUpdateEvent(delta_text="should be ignored"))

    events = []
    async for event in stream:
        events.append(event)

    assert len(events) == 2  # agent_start + agent_end
    # 不应有 message_update
    assert not any(isinstance(e, MessageUpdateEvent) for e in events)
