"""工具执行器：sequential/parallel 双模式 + before/after 钩子。

参考 pi 的 executeToolCalls 设计：

sequential 模式：
- 逐个 prepare → execute → finalize，工具间串行
- 一个工具出错不影响下一个

parallel 模式：
- 预备阶段串行（参数校验 + before_tool_call 钩子）
- 执行阶段并发（asyncio.gather）
- finalize 阶段按 assistant 原始顺序回放，保证 LLM 看到的工具结果顺序稳定

per-tool execution_mode 可强制 sequential（如写文件的工具不能并发）

before_tool_call: 可阻断执行（block=True），常用于权限拦截
after_tool_call: 可覆盖结果字段（content/details/is_error/terminate）
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable

from minagent.events import (
    EventStream,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from minagent.messages import TextContent, ToolCall, ToolResultContent, ToolResultMessage
from minagent.tools.base import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolResult,
)
from minagent.tools.registry import ToolArgumentError, ToolNotFoundError, ToolRegistry


# ---------------------------------------------------------------------------
# 单次工具执行结果
# ---------------------------------------------------------------------------

@dataclass
class ExecutedToolCall:
    """单个工具调用的执行结果（含元数据）。"""
    tool_call: ToolCall
    result: ToolResult


@dataclass
class ExecutedBatch:
    """一批工具调用的执行结果。"""
    executed: list[ExecutedToolCall] = field(default_factory=list)
    messages: list[ToolResultMessage] = field(default_factory=list)

    @property
    def terminate(self) -> bool:
        """所有工具都 terminate 时才终止。"""
        return len(self.executed) > 0 and all(e.result.terminate for e in self.executed)


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """工具执行器。

    用法：
        executor = ToolExecutor(registry, before_tool_call=..., after_tool_call=...)
        batch = await executor.execute(tool_calls, stream)
        context.messages.extend(batch.messages)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        before_tool_call: Any = None,
        after_tool_call: Any = None,
        default_execution_mode: str = "parallel",
    ) -> None:
        self.registry = registry
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.default_execution_mode = default_execution_mode

    async def execute(
        self,
        tool_calls: list[ToolCall],
        stream: EventStream | None = None,
        signal: asyncio.Event | None = None,
    ) -> ExecutedBatch:
        """执行一批工具调用。

        根据是否有 sequential 工具决定整体模式：
        - 任意工具 execution_mode=sequential → 整批 sequential
        - 否则按 default_execution_mode
        """
        if not tool_calls:
            return ExecutedBatch()

        # 检查是否有工具强制 sequential
        has_sequential = False
        for tc in tool_calls:
            if self.registry.has(tc.name):
                tool = self.registry.get(tc.name)
                if tool.execution_mode == "sequential":
                    has_sequential = True
                    break

        if has_sequential or self.default_execution_mode == "sequential":
            return await self._execute_sequential(tool_calls, stream, signal)
        return await self._execute_parallel(tool_calls, stream, signal)

    # ------------------------------------------------------------------
    # sequential 模式
    # ------------------------------------------------------------------

    async def _execute_sequential(
        self,
        tool_calls: list[ToolCall],
        stream: EventStream | None,
        signal: asyncio.Event | None,
    ) -> ExecutedBatch:
        """串行执行：prepare → execute → finalize，一个接一个。"""
        batch = ExecutedBatch()

        for tc in tool_calls:
            if stream:
                await stream.emit(ToolExecutionStartEvent(
                    tool_call_id=tc.id, tool_name=tc.name, arguments=tc.arguments,
                ))

            executed = await self._prepare_execute_finalize(tc, stream, signal)
            batch.executed.append(executed)

            # 构造 ToolResultMessage
            msg = self._make_result_message(executed)
            batch.messages.append(msg)

            if stream:
                await stream.emit(ToolExecutionEndEvent(
                    tool_call_id=tc.id, is_error=executed.result.is_error,
                ))

            if signal and signal.is_set():
                break

        return batch

    # ------------------------------------------------------------------
    # parallel 模式
    # ------------------------------------------------------------------

    async def _execute_parallel(
        self,
        tool_calls: list[ToolCall],
        stream: EventStream | None,
        signal: asyncio.Event | None,
    ) -> ExecutedBatch:
        """并发执行：预备串行 → 执行并发 → 按原序 finalize。

        关键：tool_result 按 assistant 原始顺序返回，保证 LLM 看到的顺序稳定。
        asyncio.gather 保证结果顺序与输入顺序一致。
        """
        # 阶段 1：预备（串行）—— 参数校验 + before_tool_call 钩子
        # prepared 存 (index, coroutine) 或 (index, ExecutedToolCall 立即结果)
        immediates: dict[int, ExecutedToolCall] = {}
        coros: list[tuple[int, Awaitable[ExecutedToolCall]]] = []

        for idx, tc in enumerate(tool_calls):
            if stream:
                await stream.emit(ToolExecutionStartEvent(
                    tool_call_id=tc.id, tool_name=tc.name, arguments=tc.arguments,
                ))

            prep = await self._prepare(tc, stream, signal)
            if prep is None:
                # before 阻断或工具不存在，立即构造错误结果
                immediates[idx] = ExecutedToolCall(
                    tool_call=tc,
                    result=ToolResult.error("blocked or not found"),
                )
                if stream:
                    await stream.emit(ToolExecutionEndEvent(
                        tool_call_id=tc.id, is_error=True,
                    ))
            else:
                tool, params = prep
                # 包装成 coroutine，稍后并发执行
                coros.append((idx, self._execute_and_finalize(tc, tool, params, stream, signal)))

            if signal and signal.is_set():
                break

        # 阶段 2：并发执行（asyncio.gather 保证结果顺序与输入一致）
        coro_results: list[ExecutedToolCall] = []
        if coros:
            indices = [i for i, _ in coros]
            tasks = [c for _, c in coros]
            gathered = await asyncio.gather(*tasks)
            coro_results = list(zip(indices, gathered))

        # 阶段 3：合并 immediates 和 coro_results，按原序排列
        merged: dict[int, ExecutedToolCall] = dict(immediates)
        for idx, executed in coro_results:
            merged[idx] = executed

        # 阶段 4：构造 messages（按原序）
        batch = ExecutedBatch()
        for idx in sorted(merged.keys()):
            executed = merged[idx]
            batch.executed.append(executed)
            batch.messages.append(self._make_result_message(executed))

        return batch

    # ------------------------------------------------------------------
    # 预备阶段：校验 + before 钩子
    # ------------------------------------------------------------------

    async def _prepare(
        self,
        tc: ToolCall,
        stream: EventStream | None,
        signal: asyncio.Event | None,
    ) -> tuple[AgentTool, dict[str, Any]] | None:
        """预备阶段：查找工具 + 校验参数 + before 钩子。

        返回 (tool, params) 或 None（被阻断/出错）。
        """
        try:
            tool, params = self.registry.validate(tc)
        except (ToolNotFoundError, ToolArgumentError) as e:
            return None

        # before_tool_call 钩子
        if self.before_tool_call:
            ctx = BeforeToolCallContext(tool_call=tc, tool=tool, params=params)
            result = self.before_tool_call(ctx)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
            if result and getattr(result, "block", False):
                return None

        return tool, params

    # ------------------------------------------------------------------
    # 执行 + finalize（parallel 用）
    # ------------------------------------------------------------------

    async def _execute_and_finalize(
        self,
        tc: ToolCall,
        tool: AgentTool,
        params: dict[str, Any],
        stream: EventStream | None,
        signal: asyncio.Event | None,
    ) -> ExecutedToolCall:
        """执行单个工具并 finalize（发 end 事件 + after 钩子）。"""
        try:
            result = await tool.execute(params, signal)
        except Exception as e:
            result = ToolResult.error(str(e))

        # after_tool_call 钩子
        result = await self._apply_after_hook(tc, tool, params, result)

        if stream:
            await stream.emit(ToolExecutionEndEvent(
                tool_call_id=tc.id, is_error=result.is_error,
            ))

        return ExecutedToolCall(tool_call=tc, result=result)

    # ------------------------------------------------------------------
    # sequential 的 prepare-execute-finalize
    # ------------------------------------------------------------------

    async def _prepare_execute_finalize(
        self,
        tc: ToolCall,
        stream: EventStream | None,
        signal: asyncio.Event | None,
    ) -> ExecutedToolCall:
        """串行模式：预备 → 执行 → finalize 一体。"""
        prep = await self._prepare(tc, stream, signal)
        if prep is None:
            return ExecutedToolCall(
                tool_call=tc,
                result=ToolResult.error("blocked or not found"),
            )

        tool, params = prep
        try:
            result = await tool.execute(params, signal)
        except Exception as e:
            result = ToolResult.error(str(e))

        # after_tool_call 钩子
        result = await self._apply_after_hook(tc, tool, params, result)

        return ExecutedToolCall(tool_call=tc, result=result)

    # ------------------------------------------------------------------
    # after 钩子
    # ------------------------------------------------------------------

    async def _apply_after_hook(
        self,
        tc: ToolCall,
        tool: AgentTool,
        params: dict[str, Any],
        result: ToolResult,
    ) -> ToolResult:
        """应用 after_tool_call 钩子，可覆盖结果字段。"""
        if not self.after_tool_call:
            return result

        ctx = AfterToolCallContext(tool_call=tc, tool=tool, params=params, result=result)
        override = self.after_tool_call(ctx)
        if inspect.isawaitable(override):
            override = await override  # type: ignore[assignment]
        if override is None:
            return result

        # 应用覆盖
        if override.content is not None:
            result.content = override.content
        if override.details is not None:
            result.details = override.details
        if override.is_error is not None:
            result.is_error = override.is_error
        if override.terminate is not None:
            result.terminate = override.terminate
        return result

    # ------------------------------------------------------------------
    # 构造 ToolResultMessage
    # ------------------------------------------------------------------

    @staticmethod
    def _make_result_message(executed: ExecutedToolCall) -> ToolResultMessage:
        """把 ExecutedToolCall 转成 transcript 里的 ToolResultMessage。"""
        trc = ToolResultContent(
            tool_call_id=executed.tool_call.id,
            content=executed.result.content,
            is_error=executed.result.is_error,
        )
        return ToolResultMessage(content=[trc])
