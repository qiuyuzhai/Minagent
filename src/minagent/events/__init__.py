"""事件层：AgentEvent 类型 + EventStream。

参考 pi 的 EventStream 设计：不是'调一次 LLM 拿结果'，而是全程发事件。
UI 和后端逻辑订阅同一套事件流，工具执行中还能通过 on_update 流式推送部分结果。

事件类型层级：
    agent_start / agent_end
      └─ turn_start / turn_end
           ├─ message_start / message_update / message_end
           └─ tool_execution_start / tool_execution_end / tool_result_message

实现用 asyncio.Queue，生产者（agent loop）和消费者（UI/日志）解耦。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Union

from minagent.messages import AnyAgentMessage, ToolResultMessage


# ---------------------------------------------------------------------------
# AgentEvent：事件类型联合
# ---------------------------------------------------------------------------

@dataclass
class AgentStartEvent:
    """agent 运行开始。"""
    type: Literal["agent_start"] = "agent_start"


@dataclass
class AgentEndEvent:
    """agent 运行结束，携带本次运行新增的消息。"""
    messages: list[AnyAgentMessage]
    type: Literal["agent_end"] = "agent_end"


@dataclass
class TurnStartEvent:
    """一个 turn 开始（turn = 一次 assistant 响应 + 它的工具调用）。"""
    turn_index: int = 0
    type: Literal["turn_start"] = "turn_start"


@dataclass
class TurnEndEvent:
    """一个 turn 结束。"""
    message: AnyAgentMessage
    tool_results: list[ToolResultMessage] = field(default_factory=list)
    type: Literal["turn_end"] = "turn_end"


@dataclass
class MessageStartEvent:
    """消息开始（user/assistant/tool_result）。"""
    message: AnyAgentMessage
    type: Literal["message_start"] = "message_start"


@dataclass
class MessageUpdateEvent:
    """消息更新（assistant 流式响应的增量）。"""
    delta_text: str = ""
    type: Literal["message_update"] = "message_update"


@dataclass
class MessageEndEvent:
    """消息结束。"""
    message: AnyAgentMessage
    type: Literal["message_end"] = "message_end"


@dataclass
class ToolExecutionStartEvent:
    """工具执行开始。"""
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass
class ToolExecutionEndEvent:
    """工具执行结束。"""
    tool_call_id: str
    is_error: bool = False
    type: Literal["tool_execution_end"] = "tool_execution_end"


@dataclass
class ToolResultMessageEvent:
    """工具结果消息（作为 transcript 的一部分）。"""
    message: ToolResultMessage
    type: Literal["tool_result_message"] = "tool_result_message"


# 所有事件类型联合
AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
    ToolResultMessageEvent,
]


# ---------------------------------------------------------------------------
# EventStream：事件流
# ---------------------------------------------------------------------------

class EventStream:
    """异步事件流。

    生产者（agent loop）通过 emit() 推送事件，
    消费者（UI / 日志）通过 async for 迭代事件。

    用 asyncio.Queue 实现，支持多个消费者订阅（广播模式可选）。
    end() 后队列里的事件仍可被消费完，然后迭代结束。
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._ended = False

    async def emit(self, event: AgentEvent) -> None:
        """生产者：推送一个事件。"""
        if self._ended:
            return
        await self._queue.put(event)

    async def end(self, messages: list[AnyAgentMessage] | None = None) -> None:
        """生产者：标记流结束，可携带 agent_end 的最终消息。"""
        if self._ended:
            return
        await self._queue.put(AgentEndEvent(messages=messages or []))
        await self._queue.put(None)  # 哨兵，通知迭代结束
        self._ended = True

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        """消费者：异步迭代事件，直到流结束。"""
        return self._consume()

    async def _consume(self) -> AsyncIterator[AgentEvent]:
        while True:
            event = await self._queue.get()
            if event is None:  # 哨兵
                return
            yield event
            if isinstance(event, AgentEndEvent):
                # agent_end 后再消费一个 None 哨兵就结束
                # 但为了安全，直接 return
                return
