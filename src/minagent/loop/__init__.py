"""循环层：agent_loop 核心 + turn 抽象 + 生命周期钩子。

参考 pi 的 runLoop 设计：
- 外层循环：follow-up 续跑（agent 本要停，队列有消息则续跑）
- 内层循环：turn 循环（stream assistant → 执行工具 → 检查停止）

一个 turn = 一次 assistant 响应 + 它的工具调用 + 工具结果。

生命周期钩子：
- should_stop_after_turn：turn 结束后是否优雅停止（如 context 快满）
- prepare_next_turn：下一轮动态切 model / context
- transform_context：convert_to_llm 前裁剪/注入 context
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from minagent.events import (
    AgentEndEvent,
    AgentStartEvent,
    EventStream,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResultMessageEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from minagent.llm import LLMProvider, ModelConfig, StreamChunk
from minagent.messages import (
    AnyAgentMessage,
    AssistantMessage,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultContent,
    ToolResultMessage,
    UserMessage,
    convert_to_llm,
)
from minagent.tools import ToolExecutor


# ---------------------------------------------------------------------------
# AgentContext：agent 运行状态
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """agent 运行状态。"""
    system_prompt: str = ""
    messages: list[AnyAgentMessage] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)  # OpenAI tool schema


# ---------------------------------------------------------------------------
# 生命周期钩子类型
# ---------------------------------------------------------------------------

@dataclass
class TurnContext:
    """传给钩子的上下文。"""
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[AnyAgentMessage]


# 钩子签名
ShouldStopAfterTurn = Callable[[TurnContext], bool | Awaitable[bool]]
PrepareNextTurn = Callable[[TurnContext], dict[str, Any] | None | Awaitable[dict[str, Any] | None]]
TransformContext = Callable[[list[AnyAgentMessage]], list[AnyAgentMessage] | Awaitable[list[AnyAgentMessage]]]
ExecuteTool = Callable[[ToolCall], str]


# ---------------------------------------------------------------------------
# AgentLoopConfig：循环配置
# ---------------------------------------------------------------------------

@dataclass
class AgentLoopConfig:
    """agent loop 配置。"""
    model_config: ModelConfig
    max_turns: int = 20  # 死循环防护
    # 生命周期钩子（全部可选）
    should_stop_after_turn: ShouldStopAfterTurn | None = None
    prepare_next_turn: PrepareNextTurn | None = None
    transform_context: TransformContext | None = None
    # 工具执行器（P2 新增，优先使用）
    tool_executor: ToolExecutor | None = None
    # P1 简单版 execute_tool（向后兼容，tool_executor 优先）
    execute_tool: ExecuteTool | None = None


# ---------------------------------------------------------------------------
# agent_loop：核心循环
# ---------------------------------------------------------------------------

async def agent_loop(
    prompt: str | UserMessage,
    context: AgentContext,
    config: AgentLoopConfig,
    provider: LLMProvider,
    stream: EventStream | None = None,
) -> EventStream:
    """启动 agent loop。

    返回 EventStream，调用者可以 async for event in stream 迭代事件。
    流结束时（agent_end）携带本次运行新增的消息。
    """
    if stream is None:
        stream = EventStream()

    # 准备 prompt
    if isinstance(prompt, str):
        prompt_msg = UserMessage.from_text(prompt)
    else:
        prompt_msg = prompt

    # 追加到 context
    if context.system_prompt and not any(
        isinstance(m, SystemMessage) for m in context.messages
    ):
        context.messages.insert(0, SystemMessage(content=context.system_prompt))
    context.messages.append(prompt_msg)

    new_messages: list[AnyAgentMessage] = [prompt_msg]

    # 异步启动循环
    import asyncio
    asyncio.create_task(
        _run_loop(context, config, provider, stream, new_messages)
    )
    return stream


async def _run_loop(
    context: AgentContext,
    config: AgentLoopConfig,
    provider: LLMProvider,
    stream: EventStream,
    new_messages: list[AnyAgentMessage],
) -> None:
    """主循环逻辑。"""
    await stream.emit(AgentStartEvent())

    try:
        turn_index = 0
        has_more_tool_calls = True

        while has_more_tool_calls and turn_index < config.max_turns:
            turn_index += 1
            await stream.emit(TurnStartEvent(turn_index=turn_index))

            # 1. stream assistant response
            assistant_msg = await _stream_assistant(context, config, provider, stream)
            context.messages.append(assistant_msg)
            new_messages.append(assistant_msg)

            if assistant_msg.stop_reason in ("error", "aborted"):
                await stream.emit(TurnEndEvent(message=assistant_msg, tool_results=[]))
                break

            # 2. 执行工具
            tool_results: list[ToolResultMessage] = []
            tool_calls = assistant_msg.tool_calls

            if tool_calls and config.tool_executor:
                # P2 路径：用 ToolExecutor（sequential/parallel + 钩子）
                batch = await config.tool_executor.execute(tool_calls, stream)
                for tr_msg in batch.messages:
                    tool_results.append(tr_msg)
                    context.messages.append(tr_msg)
                    new_messages.append(tr_msg)
                    await stream.emit(ToolResultMessageEvent(message=tr_msg))
                has_more_tool_calls = not batch.terminate
            elif tool_calls and config.execute_tool:
                # P1 路径：简单 callable（向后兼容）
                for tc in tool_calls:
                    await stream.emit(
                        ToolExecutionStartEvent(
                            tool_call_id=tc.id, tool_name=tc.name, arguments=tc.arguments
                        )
                    )
                    result_text = config.execute_tool(tc)
                    is_error = result_text.startswith("ERROR:")
                    tr_content = ToolResultContent(
                        tool_call_id=tc.id,
                        content=[TextContent(text=result_text)],
                        is_error=is_error,
                    )
                    tr_msg = ToolResultMessage(content=[tr_content])
                    tool_results.append(tr_msg)
                    context.messages.append(tr_msg)
                    new_messages.append(tr_msg)
                    await stream.emit(
                        ToolExecutionEndEvent(tool_call_id=tc.id, is_error=is_error)
                    )
                    await stream.emit(ToolResultMessageEvent(message=tr_msg))
                has_more_tool_calls = True
            else:
                has_more_tool_calls = False

            await stream.emit(TurnEndEvent(message=assistant_msg, tool_results=tool_results))

            # 3. should_stop_after_turn 钩子
            turn_ctx = TurnContext(
                message=assistant_msg,
                tool_results=tool_results,
                context=context,
                new_messages=new_messages,
            )
            if config.should_stop_after_turn:
                stop = config.should_stop_after_turn(turn_ctx)
                if isinstance(stop, Awaitable):
                    stop = await stop  # type: ignore[assignment]
                if stop:
                    break

            # 4. prepare_next_turn 钩子（P1 简化，只打日志）
            # P3 会在这里动态切 model / context

        await stream.end(new_messages)
    except Exception as e:
        # 错误也结束流，避免消费者卡住
        await stream.end(new_messages)


async def _stream_assistant(
    context: AgentContext,
    config: AgentLoopConfig,
    provider: LLMProvider,
    stream: EventStream,
) -> AssistantMessage:
    """流式获取 assistant 响应，发事件，返回最终 AssistantMessage。"""
    # transform_context 钩子（convert 前裁剪/注入）
    messages = context.messages
    if config.transform_context:
        result = config.transform_context(messages)
        if isinstance(result, Awaitable):
            result = await result  # type: ignore[assignment]
        messages = result  # type: ignore[assignment]

    # 转换为 LLM 格式
    llm_messages = convert_to_llm(messages)
    tools = context.tools if context.tools else None

    # 流式调用
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    # 累积工具调用增量
    tc_accum: dict[int, dict[str, Any]] = {}  # index -> {id, name, arguments}
    finish_reason: str | None = None

    await stream.emit(MessageStartEvent(message=AssistantMessage(content=[])))

    async for chunk in provider.stream(llm_messages, tools, config.model_config):
        if chunk.delta_text:
            text_parts.append(chunk.delta_text)
            await stream.emit(MessageUpdateEvent(delta_text=chunk.delta_text))

        if chunk.tool_call_id is not None or chunk.tool_call_name is not None or chunk.tool_call_arguments_delta:
            # 这里简化：单工具调用流式累积
            # 实际 pi 用 index 区分多个并发 tool call，P1 先支持单工具
            if chunk.tool_call_id:
                tc_accum.setdefault(0, {})["id"] = chunk.tool_call_id
            if chunk.tool_call_name:
                tc_accum.setdefault(0, {})["name"] = chunk.tool_call_name
            if chunk.tool_call_arguments_delta:
                tc_accum.setdefault(0, {}).setdefault("arguments", "")
                tc_accum[0]["arguments"] += chunk.tool_call_arguments_delta

        if chunk.finish_reason:
            finish_reason = chunk.finish_reason

    # 构造 AssistantMessage
    content: list[TextContent | ToolCall] = []
    if text_parts:
        content.append(TextContent(text="".join(text_parts)))
    for idx in sorted(tc_accum.keys()):
        tc_data = tc_accum[idx]
        args_str = tc_data.get("arguments", "{}")
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {"_raw": args_str}
        content.append(ToolCall(
            id=tc_data.get("id", f"call_{idx}"),
            name=tc_data.get("name", ""),
            arguments=args,
        ))

    stop_reason: Literal["stop", "tool_use", "error", "aborted"] = "stop"
    if finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "error":
        stop_reason = "error"

    msg = AssistantMessage(content=content, stop_reason=stop_reason)
    await stream.emit(MessageEndEvent(message=msg))
    return msg
