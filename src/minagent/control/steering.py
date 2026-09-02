"""Steering + FollowUp 队列：用户中途插话与续跑。

参考 pi 的 steering / follow-up 设计：

Steering（中途插话）：
- agent 正在跑工具时用户突然插话"换个方向"
- 消息进 SteeringQueue，当前 turn 结束后自动注入 context
- 不停止循环，下一轮 LLM 会看到插话内容

FollowUp（续跑）：
- agent 本来要停了（无工具调用，自然停止）
- FollowUpQueue 里还有消息，自动续跑
- 实现"用户连续发多条消息，agent 依次处理"

两者都用 asyncio.Queue 实现，非阻塞 drain。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from minagent.messages import UserMessage


@dataclass
class SteeringQueue:
    """Steering 队列：agent 运行中用户插话。

    用法：
        steering = SteeringQueue()
        await steering.push(UserMessage.from_text("换个方向"))
        # turn 结束后
        new_msgs = steering.drain()  # 取出所有排队消息
    """
    _queue: asyncio.Queue[UserMessage] = field(default_factory=asyncio.Queue)

    async def push(self, message: UserMessage) -> None:
        """用户插入消息（非阻塞，立刻入队）。"""
        await self._queue.put(message)

    def push_nowait(self, message: UserMessage) -> None:
        """同步入队（用于从同步代码触发，如工具执行回调）。"""
        self._queue.put_nowait(message)

    def drain(self) -> list[UserMessage]:
        """取出所有排队消息（非阻塞）。turn 结束后调用。"""
        messages: list[UserMessage] = []
        while not self._queue.empty():
            try:
                messages.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def is_empty(self) -> bool:
        """队列是否为空。"""
        return self._queue.empty()


@dataclass
class FollowUpQueue:
    """FollowUp 队列：agent 停止后续跑。

    与 SteeringQueue 区别：
    - Steering 在 turn 间隙注入，不停循环
    - FollowUp 在 agent 自然停止后检查，有消息则重启循环
    """
    _queue: asyncio.Queue[UserMessage] = field(default_factory=asyncio.Queue)

    async def push(self, message: UserMessage) -> None:
        """排队续跑消息。"""
        await self._queue.put(message)

    def push_nowait(self, message: UserMessage) -> None:
        """同步入队。"""
        self._queue.put_nowait(message)

    def drain(self) -> list[UserMessage]:
        """取出所有续跑消息。"""
        messages: list[UserMessage] = []
        while not self._queue.empty():
            try:
                messages.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def is_empty(self) -> bool:
        return self._queue.empty()
