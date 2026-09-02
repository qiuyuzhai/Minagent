"""中断控制：AbortController。

参考 pi 的 abort 设计：用 asyncio.Event 作为信号，传播到 agent loop 和工具执行。
工具在执行中要检查 signal.is_set()，及时退出。

关键设计：
- abort() 设置 Event，所有持有 signal 的协程都能感知
- 工具执行时把 signal 传下去，工具内循环检查
- agent loop 检查 signal，abort 后优雅结束当前 turn
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AbortController:
    """中断控制器。

    用法：
        controller = AbortController()
        # 传给 agent_loop / tool_executor
        await agent_loop(..., abort_controller=controller)
        # 需要中断时
        controller.abort("user requested")
        # 检查
        if controller.is_aborted():
            ...
    """
    signal: asyncio.Event = field(default_factory=asyncio.Event)
    reason: str = ""

    def abort(self, reason: str = "") -> None:
        """请求中断。设置信号，所有等待者立刻感知。"""
        self.reason = reason
        self.signal.set()

    def is_aborted(self) -> bool:
        """是否已中断。"""
        return self.signal.is_set()

    def reset(self) -> None:
        """重置（用于复用 controller）。"""
        self.signal.clear()
        self.reason = ""

    async def wait_for_abort(self, timeout: float | None = None) -> bool:
        """等待中断信号。返回 True 表示已中断，False 表示超时。"""
        try:
            if timeout is None:
                await self.signal.wait()
                return True
            await asyncio.wait_for(self.signal.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
