"""工具基类：AgentTool / ToolResult / ExecutionMode。

参考 pi 的 AgentTool 设计：
- 用 pydantic BaseModel 替代 pi 的 TypeBox schema 做参数校验
- ToolResult 分离 content（给 LLM）和 details（给 UI/日志）
- per-tool execution_mode：sequential（如写文件）/ parallel（如读）
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Union

from pydantic import BaseModel

from minagent.messages import TextContent, ToolCall


# ---------------------------------------------------------------------------
# ExecutionMode：工具执行模式
# ---------------------------------------------------------------------------

ExecutionMode = Literal["sequential", "parallel"]


# ---------------------------------------------------------------------------
# ToolResult：工具执行结果（content/details 分离）
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """工具执行结果。

    参考 pi 的 AgentToolResult：
    - content: 给 LLM 看的（text/image）
    - details: 给 UI/日志看的（结构化数据）
    - is_error: 是否出错
    - terminate: 终止提示（所有工具都 terminate 时 agent 停止）
    """
    content: list[TextContent] = field(default_factory=list)
    details: Any = None
    is_error: bool = False
    terminate: bool = False

    @classmethod
    def text(cls, text: str, **kwargs: Any) -> ToolResult:
        """快速构造文本结果。"""
        return cls(content=[TextContent(text=text)], **kwargs)

    @classmethod
    def error(cls, error: str, **kwargs: Any) -> ToolResult:
        """快速构造错误结果。"""
        return cls(
            content=[TextContent(text=f"ERROR: {error}")],
            is_error=True,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 钩子上下文与结果类型
# ---------------------------------------------------------------------------

@dataclass
class BeforeToolCallContext:
    """before_tool_call 钩子的上下文。"""
    tool_call: ToolCall
    tool: AgentTool
    params: dict[str, Any]


@dataclass
class BeforeToolCallResult:
    """before_tool_call 钩子返回值。返回 block=True 阻止执行。"""
    block: bool = False
    reason: str = ""


@dataclass
class AfterToolCallContext:
    """after_tool_call 钩子的上下文。"""
    tool_call: ToolCall
    tool: AgentTool
    params: dict[str, Any]
    result: ToolResult


@dataclass
class AfterToolCallResult:
    """after_tool_call 钩子返回值，可覆盖结果字段。"""
    content: list[TextContent] | None = None
    details: Any = None
    is_error: bool | None = None
    terminate: bool | None = None


# 钩子签名
BeforeToolCallHook = Callable[[BeforeToolCallContext], BeforeToolCallResult | Awaitable[BeforeToolCallResult] | None | Awaitable[None]]
AfterToolCallHook = Callable[[AfterToolCallContext], AfterToolCallResult | Awaitable[AfterToolCallResult] | None | Awaitable[None]]
OnUpdateCallback = Callable[[ToolResult], None | Awaitable[None]]


# ---------------------------------------------------------------------------
# AgentTool：工具基类
# ---------------------------------------------------------------------------

class AgentTool(ABC):
    """工具基类。

    子类需要定义：
    - name: 工具名（LLM 调用时用）
    - description: 工具描述（给 LLM 看的）
    - params_schema: pydantic BaseModel，参数 schema
    - execution_mode: 默认 "parallel"，写操作用 "sequential"
    - execute(): 执行逻辑，返回 ToolResult
    """

    name: str
    description: str
    params_schema: type[BaseModel]
    execution_mode: ExecutionMode = "parallel"

    @abstractmethod
    async def execute(
        self,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: OnUpdateCallback | None = None,
    ) -> ToolResult:
        """执行工具。失败时抛异常，由 executor 捕获转成 ToolResult.error。"""
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """生成 OpenAI function calling 的 schema。

        用 pydantic 的 model_json_schema() 转换，去掉无关字段。
        """
        schema = self.params_schema.model_json_schema()
        # OpenAI 要求 schema 是 object 类型
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }
