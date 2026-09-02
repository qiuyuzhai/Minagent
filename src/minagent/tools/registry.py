"""工具注册表：ToolRegistry。

负责工具注册、查找、参数校验、生成 OpenAI schema 列表。

参考 pi 的 ToolRegistry 设计：
- register(tool): 注册工具
- get(name): 按名查找工具
- to_openai_schemas(): 生成所有工具的 OpenAI function schema 列表
- validate(tool_call): 用 pydantic 校验参数
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from minagent.messages import ToolCall
from minagent.tools.base import AgentTool


class ToolNotFoundError(Exception):
    """工具未找到。"""


class ToolArgumentError(Exception):
    """工具参数校验失败。"""


class ToolRegistry:
    """工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> AgentTool:
        """注册工具。重复名覆盖。"""
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str) -> None:
        """取消注册。"""
        self._tools.pop(name, None)

    def get(self, name: str) -> AgentTool:
        """按名查找工具。"""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found. Available: {list(self._tools)}")
        return self._tools[name]

    def list_tools(self) -> list[AgentTool]:
        """列出所有工具。"""
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        """是否存在工具。"""
        return name in self._tools

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """生成 OpenAI function calling 的 schema 列表。"""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def validate(self, tool_call: ToolCall) -> tuple[AgentTool, dict[str, Any]]:
        """校验工具调用参数。

        返回 (tool, validated_params)。
        工具不存在抛 ToolNotFoundError，参数不合规抛 ToolArgumentError。
        """
        try:
            tool = self.get(tool_call.name)
        except ToolNotFoundError:
            raise

        # pydantic 校验参数
        try:
            # 兼容 LLM 传 dict 或空参数
            raw_args = tool_call.arguments or {}
            validated = tool.params_schema.model_validate(raw_args)
            params = validated.model_dump()
            return tool, params
        except ValidationError as e:
            raise ToolArgumentError(
                f"Tool '{tool_call.name}' 参数校验失败: {e}"
            ) from e
