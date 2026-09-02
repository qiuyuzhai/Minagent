"""超时管理：工具级 + turn 级超时。

参考 pi 的 timeout 设计：
- 工具级超时：单个工具执行超时，取消该工具，其他工具继续
- turn 级超时：整个 turn 超时，触发 abort

用 asyncio.wait_for 实现，超时抛 TimeoutError。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")


@dataclass
class TimeoutManager:
    """超时管理器。

    用法：
        tm = TimeoutManager(tool_timeout=30, turn_timeout=300)
        # 工具级
        result = await tm.run_with_tool_timeout(tool.execute(...))
        # turn 级
        await tm.run_with_turn_timeout(turn_coro())
    """
    tool_timeout: float | None = None  # 单工具超时（秒），None 不限制
    turn_timeout: float | None = None  # turn 超时（秒），None 不限制

    async def run_with_tool_timeout(self, coro: Awaitable[T]) -> T:
        """工具级超时。超时抛 asyncio.TimeoutError，由 executor 捕获转 error。"""
        if self.tool_timeout is None:
            return await coro
        return await asyncio.wait_for(coro, timeout=self.tool_timeout)

    async def run_with_turn_timeout(self, coro: Awaitable[T]) -> T:
        """turn 级超时。超时抛 asyncio.TimeoutError，由 loop 捕获触发 abort。"""
        if self.turn_timeout is None:
            return await coro
        return await asyncio.wait_for(coro, timeout=self.turn_timeout)
