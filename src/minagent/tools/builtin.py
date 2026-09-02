"""内置工具：calculator + read_file。

calculator：parallel 模式（纯计算，无副作用，可并发）
read_file：sequential 模式（文件 IO，强制串行）
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from minagent.tools.base import AgentTool, ToolResult


# ---------------------------------------------------------------------------
# Calculator：parallel 模式
# ---------------------------------------------------------------------------

class CalculatorParams(BaseModel):
    """计算器参数。"""
    expression: str = Field(..., description="数学表达式，如 '15 * 4 + 12'")


class CalculatorTool(AgentTool):
    """四则运算计算器。

    纯计算无副作用，execution_mode=parallel。
    """
    name = "calculator"
    description = "四则运算计算器。输入数学表达式，返回计算结果。支持 + - * / ( )。"
    params_schema = CalculatorParams
    execution_mode = "parallel"

    async def execute(
        self,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update=None,
    ) -> ToolResult:
        expr = params["expression"]
        # 安全：限制字符集后 eval
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expr):
            return ToolResult.error(f"invalid expression: {expr}")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return ToolResult.text(f"{expr} = {result}", details={"result": result})
        except Exception as e:
            return ToolResult.error(str(e))


# ---------------------------------------------------------------------------
# ReadFile：sequential 模式
# ---------------------------------------------------------------------------

class ReadFileParams(BaseModel):
    """读文件参数。"""
    path: str = Field(..., description="文件路径")
    max_lines: int = Field(default=100, description="最大读取行数")


class ReadFileTool(AgentTool):
    """读取文件内容。

    文件 IO，execution_mode=sequential（强制串行，避免并发打开太多文件）。
    """
    name = "read_file"
    description = "读取文本文件内容。返回前 N 行。"
    params_schema = ReadFileParams
    execution_mode = "sequential"

    async def execute(
        self,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update=None,
    ) -> ToolResult:
        path = Path(params["path"])
        max_lines = params.get("max_lines", 100)
        if not path.exists():
            return ToolResult.error(f"file not found: {path}")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[:max_lines]
            content = "\n".join(lines)
            return ToolResult.text(content, details={"line_count": len(lines)})
        except Exception as e:
            return ToolResult.error(str(e))
