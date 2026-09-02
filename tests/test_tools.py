"""tools 层测试：registry + executor + builtin。"""

import asyncio
import json

import pytest
from pydantic import BaseModel, Field

from minagent.messages import ToolCall
from minagent.tools import (
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    AfterToolCallContext,
    AfterToolCallResult,
    ToolExecutor,
    ToolNotFoundError,
    ToolArgumentError,
    ToolRegistry,
    ToolResult,
)
from minagent.tools.builtin import CalculatorTool, ReadFileTool


# ---------------------------------------------------------------------------
# 测试用工具
# ---------------------------------------------------------------------------

class EchoParams(BaseModel):
    text: str = Field(..., description="要回显的文本")


class EchoTool(AgentTool):
    """回显工具，用于测试。"""
    name = "echo"
    description = "回显输入文本"
    params_schema = EchoParams
    execution_mode = "parallel"

    async def execute(self, params, signal=None, on_update=None) -> ToolResult:
        return ToolResult.text(params["text"], details={"echoed": params["text"]})


class SlowParams(BaseModel):
    delay: float = Field(default=0.1, description="延迟秒数")


class SlowTool(AgentTool):
    """慢工具，用于测试并发。"""
    name = "slow"
    description = "sleep 后返回"
    params_schema = SlowParams
    execution_mode = "parallel"

    async def execute(self, params, signal=None, on_update=None) -> ToolResult:
        await asyncio.sleep(params["delay"])
        return ToolResult.text(f"slept {params['delay']}s")


class SequentialToolImpl(AgentTool):
    """sequential 模式工具。"""
    name = "seq_tool"
    description = "sequential 测试"
    params_schema = EchoParams
    execution_mode = "sequential"

    async def execute(self, params, signal=None, on_update=None) -> ToolResult:
        return ToolResult.text(f"seq:{params['text']}")


def make_tool_call(name: str, arguments: dict) -> ToolCall:
    """构造 ToolCall。"""
    return ToolCall(id=f"call_{name}", name=name, arguments=arguments)


# ---------------------------------------------------------------------------
# ToolRegistry 测试
# ---------------------------------------------------------------------------

def test_registry_register_and_get():
    """注册和查找工具。"""
    reg = ToolRegistry()
    tool = EchoTool()
    reg.register(tool)

    assert reg.has("echo")
    assert reg.get("echo") is tool
    assert not reg.has("nonexistent")


def test_registry_get_not_found():
    """查找不存在的工具抛异常。"""
    reg = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        reg.get("nonexistent")


def test_registry_to_openai_schemas():
    """生成 OpenAI schema。"""
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(CalculatorTool())

    schemas = reg.to_openai_schemas()
    assert len(schemas) == 2
    names = [s["function"]["name"] for s in schemas]
    assert "echo" in names
    assert "calculator" in names

    # 验证 schema 结构
    echo_schema = [s for s in schemas if s["function"]["name"] == "echo"][0]
    assert echo_schema["type"] == "function"
    assert "text" in echo_schema["function"]["parameters"]["properties"]
    assert "text" in echo_schema["function"]["parameters"]["required"]


def test_registry_validate_success():
    """参数校验成功。"""
    reg = ToolRegistry()
    reg.register(CalculatorTool())

    tc = make_tool_call("calculator", {"expression": "1+1"})
    tool, params = reg.validate(tc)
    assert tool.name == "calculator"
    assert params["expression"] == "1+1"


def test_registry_validate_bad_args():
    """参数校验失败。"""
    reg = ToolRegistry()
    reg.register(CalculatorTool())

    tc = make_tool_call("calculator", {"wrong_field": "1+1"})
    with pytest.raises(ToolArgumentError):
        reg.validate(tc)


def test_registry_validate_unknown_tool():
    """校验未知工具。"""
    reg = ToolRegistry()
    tc = make_tool_call("nonexistent", {})
    with pytest.raises(ToolNotFoundError):
        reg.validate(tc)


# ---------------------------------------------------------------------------
# ToolExecutor 测试：sequential 模式
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_executor_sequential():
    """sequential 模式：工具按顺序执行。"""
    reg = ToolRegistry()
    reg.register(EchoTool())
    executor = ToolExecutor(reg, default_execution_mode="sequential")

    tool_calls = [
        make_tool_call("echo", {"text": "first"}),
        make_tool_call("echo", {"text": "second"}),
    ]
    batch = await executor.execute(tool_calls, stream=None)

    assert len(batch.executed) == 2
    assert batch.executed[0].result.content[0].text == "first"
    assert batch.executed[1].result.content[0].text == "second"


@pytest.mark.asyncio
async def test_executor_parallel_order():
    """parallel 模式：结果按原序返回（即使并发完成顺序不同）。"""
    reg = ToolRegistry()
    reg.register(SlowTool())
    reg.register(EchoTool())
    executor = ToolExecutor(reg, default_execution_mode="parallel")

    # slow(delay=0.3) 和 echo 混合
    # 即使 slow 后完成，结果也应按原序
    tool_calls = [
        make_tool_call("slow", {"delay": 0.3}),
        make_tool_call("echo", {"text": "fast"}),
    ]
    batch = await executor.execute(tool_calls, stream=None)

    assert len(batch.executed) == 2
    # 结果按原序
    assert batch.executed[0].tool_call.name == "slow"
    assert batch.executed[1].tool_call.name == "echo"


@pytest.mark.asyncio
async def test_executor_parallel_runs_concurrently():
    """parallel 模式：两个慢工具并发执行，总耗时应小于串行。

    注意：Windows 上 asyncio 事件循环开销较大，delay 要足够大才能体现并发优势。
    只验证并发确实发生了（总时间 < 串行时间 * 1.5），不依赖绝对时间。
    """
    import time
    reg = ToolRegistry()
    reg.register(SlowTool())
    executor_seq = ToolExecutor(reg, default_execution_mode="sequential")
    executor_par = ToolExecutor(reg, default_execution_mode="parallel")

    tool_calls = [
        make_tool_call("slow", {"delay": 0.3}),
        make_tool_call("slow", {"delay": 0.3}),
    ]

    # sequential 应该 ~0.6s
    t0 = time.time()
    batch_seq = await executor_seq.execute(tool_calls, stream=None)
    seq_time = time.time() - t0

    # parallel 应该 ~0.3s（并发）
    t0 = time.time()
    batch_par = await executor_par.execute(tool_calls, stream=None)
    par_time = time.time() - t0

    # 并发确实发生了：parallel 总时间应小于 sequential
    # 宽松断言，避免 CI flaky
    assert par_time < seq_time
    # 两个结果都正确
    assert len(batch_par.executed) == 2
    assert "slept" in batch_par.executed[0].result.content[0].text
    assert "slept" in batch_par.executed[1].result.content[0].text


@pytest.mark.asyncio
async def test_executor_per_tool_sequential_override():
    """per-tool execution_mode=sequential 强制整批串行。"""
    reg = ToolRegistry()
    reg.register(SequentialToolImpl())
    reg.register(EchoTool())
    # 默认 parallel，但 seq_tool 是 sequential，所以整批应该串行
    executor = ToolExecutor(reg, default_execution_mode="parallel")

    tool_calls = [
        make_tool_call("seq_tool", {"text": "a"}),
        make_tool_call("echo", {"text": "b"}),
    ]
    batch = await executor.execute(tool_calls, stream=None)

    assert len(batch.executed) == 2
    assert batch.executed[0].result.content[0].text == "seq:a"
    assert batch.executed[1].result.content[0].text == "b"


# ---------------------------------------------------------------------------
# before/after 钩子测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_before_tool_call_block():
    """before_tool_call 返回 block=True 阻止执行。"""
    reg = ToolRegistry()
    reg.register(EchoTool())

    def block_hook(ctx: BeforeToolCallContext):
        if "blocked" in ctx.params["text"]:
            return BeforeToolCallResult(block=True, reason="blocked by hook")
        return None

    executor = ToolExecutor(reg, before_tool_call=block_hook)

    # 被阻断的工具
    tc = make_tool_call("echo", {"text": "blocked content"})
    batch = await executor.execute([tc], stream=None)
    assert batch.executed[0].result.is_error

    # 正常工具
    tc2 = make_tool_call("echo", {"text": "allowed"})
    batch2 = await executor.execute([tc2], stream=None)
    assert not batch2.executed[0].result.is_error
    assert batch2.executed[0].result.content[0].text == "allowed"


@pytest.mark.asyncio
async def test_after_tool_call_override():
    """after_tool_call 覆盖结果。"""
    reg = ToolRegistry()
    reg.register(EchoTool())

    def override_hook(ctx: AfterToolCallContext):
        return AfterToolCallResult(
            content=[__import__("minagent.messages", fromlist=["TextContent"]).TextContent(text="OVERRIDDEN")],
        )

    executor = ToolExecutor(reg, after_tool_call=override_hook)

    tc = make_tool_call("echo", {"text": "original"})
    batch = await executor.execute([tc], stream=None)
    assert batch.executed[0].result.content[0].text == "OVERRIDDEN"


# ---------------------------------------------------------------------------
# Calculator 工具测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calculator_addition():
    """计算器加法。"""
    tool = CalculatorTool()
    result = await tool.execute({"expression": "15 * 4 + 12"})
    assert not result.is_error
    assert "72" in result.content[0].text


@pytest.mark.asyncio
async def test_calculator_invalid_chars():
    """计算器拒绝非法字符。"""
    tool = CalculatorTool()
    result = await tool.execute({"expression": "__import__('os')"})
    assert result.is_error


@pytest.mark.asyncio
async def test_calculator_division():
    """计算器除法。"""
    tool = CalculatorTool()
    result = await tool.execute({"expression": "10 / 4"})
    assert "2.5" in result.content[0].text


@pytest.mark.asyncio
async def test_calculator_details():
    """计算器返回 details。"""
    tool = CalculatorTool()
    result = await tool.execute({"expression": "1+1"})
    assert result.details == {"result": 2}


# ---------------------------------------------------------------------------
# 集成测试：ToolExecutor + loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_with_tool_executor():
    """loop 用 ToolExecutor 执行工具（端到端）。"""
    import sys
    from typing import AsyncIterator
    from minagent.llm import LLMProvider, ModelConfig, StreamChunk
    from minagent.loop import AgentContext, AgentLoopConfig, agent_loop

    class MockProvider(LLMProvider):
        def __init__(self):
            self._call = 0

        async def stream(self, messages, tools=None, config=None) -> AsyncIterator[StreamChunk]:
            self._call += 1
            if self._call == 1:
                # turn 1: 调用 calculator
                yield StreamChunk(tool_call_id="c1", tool_call_name="calculator")
                yield StreamChunk(tool_call_arguments_delta=json.dumps({"expression": "6*7"}))
                yield StreamChunk(finish_reason="tool_calls")
            else:
                # turn 2: 给出答案
                yield StreamChunk(delta_text="6 乘以 7 等于 42。")
                yield StreamChunk(finish_reason="stop")

    reg = ToolRegistry()
    reg.register(CalculatorTool())
    executor = ToolExecutor(reg)

    context = AgentContext(system_prompt="你是助手", tools=reg.to_openai_schemas())
    config = AgentLoopConfig(
        model_config=ModelConfig(model="mock"),
        max_turns=10,
        tool_executor=executor,
    )

    stream = await agent_loop("算 6*7", context, config, MockProvider())
    events = []
    final_msgs = []
    async for e in stream:
        events.append(e)
        if e.type == "agent_end":
            final_msgs = e.messages

    # 验证工具执行
    tool_starts = [e for e in events if e.type == "tool_execution_start"]
    assert len(tool_starts) == 1
    assert tool_starts[0].tool_name == "calculator"

    # 验证结果包含 42
    tool_results = [e for e in events if e.type == "tool_result_message"]
    assert len(tool_results) == 1
    assert "42" in tool_results[0].message.content[0].content[0].text
