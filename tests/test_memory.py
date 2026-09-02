"""P4 记忆层测试：EpisodicMemory + LongTermMemory + HybridRetriever + loop 集成。"""

import asyncio
import json
import os
import tempfile
import time
from typing import AsyncIterator

import pytest

from minagent.events import EventStream
from minagent.llm import LLMProvider, ModelConfig, StreamChunk
from minagent.loop import AgentContext, AgentLoopConfig, agent_loop
from minagent.memory import (
    EpisodicMemory,
    HybridRetriever,
    LongTermMemory,
    MemoryEntry,
    SimpleBM25,
    SimpleVectorRetriever,
)
from minagent.messages import AssistantMessage, TextContent, UserMessage


# ---------------------------------------------------------------------------
# EpisodicMemory 测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_episodic_memory_add_retrieve():
    """短期记忆添加和检索。"""
    mem = EpisodicMemory(max_entries=10)
    await mem.add_text("用户问了天气", type="transcript")
    await mem.add_text("用户问了股票", type="transcript")
    await mem.add_text("助手回答了天气预报", type="transcript")

    results = await mem.retrieve("天气", top_k=2)
    assert len(results) >= 1
    # 天气相关的应排在前面
    assert "天气" in results[0].content or "天气" in results[1].content


@pytest.mark.asyncio
async def test_episodic_memory_clear():
    """清空短期记忆。"""
    mem = EpisodicMemory()
    await mem.add_text("测试")
    assert len(mem.all_entries()) == 1
    await mem.clear()
    assert len(mem.all_entries()) == 0


@pytest.mark.asyncio
async def test_episodic_memory_max_entries():
    """超过 max_entries 自动丢弃最旧的。"""
    mem = EpisodicMemory(max_entries=3)
    for i in range(5):
        await mem.add_text(f"记忆{i}")
    assert len(mem.all_entries()) == 3
    # 最旧的被丢弃
    contents = [e.content for e in mem.all_entries()]
    assert "记忆0" not in contents
    assert "记忆4" in contents


# ---------------------------------------------------------------------------
# LongTermMemory 测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_long_term_memory_persistence():
    """长期记忆 jsonl 持久化。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "memory.jsonl")

        # 写入
        mem1 = LongTermMemory(file_path=path)
        await mem1.add_text("Python 是一门编程语言", type="fact")
        await mem1.add_text("Java 也是一门编程语言", type="fact")

        # 验证文件存在
        assert os.path.exists(path)

        # 重新加载
        mem2 = LongTermMemory(file_path=path)
        assert len(mem2.all_entries()) == 2


@pytest.mark.asyncio
async def test_long_term_memory_tfidf_retrieve():
    """长期记忆 TF-IDF 检索。"""
    mem = LongTermMemory()
    await mem.add_text("Python 是一门解释型编程语言")
    await mem.add_text("Java 是一门编译型编程语言")
    await mem.add_text("今天的天气很好")

    results = await mem.retrieve("编程语言", top_k=2)
    assert len(results) >= 1
    # 编程相关的应排在前面
    assert "编程语言" in results[0].content


@pytest.mark.asyncio
async def test_long_term_memory_with_hybrid_retriever():
    """长期记忆 + HybridRetriever。"""
    mem = LongTermMemory()
    retriever = HybridRetriever()
    mem.set_retriever(retriever)

    await mem.add_text("Python 是一门解释型编程语言")
    await mem.add_text("Java 是一门编译型编程语言")
    await mem.add_text("今天的天气很好")

    results = await mem.retrieve("编程语言", top_k=2)
    assert len(results) >= 1
    assert "编程语言" in results[0].content


# ---------------------------------------------------------------------------
# SimpleBM25 测试
# ---------------------------------------------------------------------------

def test_bm25_search():
    """BM25 检索。"""
    bm25 = SimpleBM25()
    bm25.add(MemoryEntry(content="Python 编程语言教程"))
    bm25.add(MemoryEntry(content="Java 编程语言入门"))
    bm25.add(MemoryEntry(content="今天天气不错"))

    results = bm25.search("编程语言", top_k=2)
    assert len(results) >= 1
    # 编程相关的应排前面
    assert "编程语言" in results[0][1].content


def test_bm25_no_match():
    """BM25 无匹配返回空。"""
    bm25 = SimpleBM25()
    bm25.add(MemoryEntry(content="Python"))
    results = bm25.search("Java", top_k=5)
    assert len(results) == 0


# ---------------------------------------------------------------------------
# SimpleVectorRetriever 测试
# ---------------------------------------------------------------------------

def test_vector_retriever_search():
    """向量检索（TF-IDF）。"""
    vr = SimpleVectorRetriever()
    vr.add(MemoryEntry(content="Python 编程语言"))
    vr.add(MemoryEntry(content="Java 编程语言"))
    vr.add(MemoryEntry(content="天气晴朗"))

    results = vr.search("编程", top_k=2)
    assert len(results) >= 1
    assert "编程" in results[0][1].content


# ---------------------------------------------------------------------------
# HybridRetriever 测试（BM25 + Vector RRF 融合）
# ---------------------------------------------------------------------------

def test_hybrid_retriever_basic():
    """Hybrid 检索：BM25 + Vector 融合。"""
    retriever = HybridRetriever()
    retriever.add(MemoryEntry(content="Python 是一门解释型编程语言"))
    retriever.add(MemoryEntry(content="Java 是一门编译型编程语言"))
    retriever.add(MemoryEntry(content="今天天气很好"))

    results = retriever.retrieve("编程语言", top_k=2)
    assert len(results) >= 1
    # 编程相关的应排前面
    assert "编程语言" in results[0].content


def test_hybrid_retriever_exact_match():
    """Hybrid 检索：精确匹配（BM25 强项）。"""
    retriever = HybridRetriever()
    retriever.add(MemoryEntry(content="iPhone 15 的价格是 5999"))
    retriever.add(MemoryEntry(content="iPhone 14 的价格是 4999"))
    retriever.add(MemoryEntry(content="Android 手机的价格范围"))

    results = retriever.retrieve("iPhone 15", top_k=1)
    assert len(results) == 1
    # 精确匹配 iPhone 15 应排第一
    assert "iPhone 15" in results[0].content


def test_hybrid_retriever_empty():
    """空检索器返回空。"""
    retriever = HybridRetriever()
    results = retriever.retrieve("anything", top_k=5)
    assert len(results) == 0


def test_hybrid_retriever_rrf_fusion():
    """RRF 融合：两个检索器都命中的条目分数更高。"""
    retriever = HybridRetriever(rrf_k=60)
    # 添加多个相似条目
    for i in range(5):
        retriever.add(MemoryEntry(content=f"Python 编程教程 第{i}章"))

    results = retriever.retrieve("Python 编程", top_k=3)
    assert len(results) == 3
    # 所有结果都应包含 Python
    for r in results:
        assert "Python" in r.content


# ---------------------------------------------------------------------------
# loop 集成测试：记忆注入 + 保存
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_memory_injection_and_save():
    """loop 注入长期记忆 + 保存 assistant 响应。"""
    from typing import AsyncIterator

    class MockProvider(LLMProvider):
        def __init__(self):
            self._call = 0

        async def stream(self, messages, tools=None, config=None) -> AsyncIterator[StreamChunk]:
            self._call += 1
            if self._call == 1:
                yield StreamChunk(delta_text="根据记忆，Python 是解释型语言。", finish_reason="stop")
            else:
                yield StreamChunk(delta_text="结束", finish_reason="stop")

    # 预先存入长期记忆
    mem = LongTermMemory()
    await mem.add_text("Python 是一门解释型编程语言", type="fact")
    initial_count = len(mem.all_entries())

    context = AgentContext(system_prompt="你是助手")
    config = AgentLoopConfig(
        model_config=ModelConfig(model="mock"),
        max_turns=10,
        long_term_memory=mem,
    )

    stream = await agent_loop("Python 是什么", context, config, MockProvider())
    events = []
    final_msgs = []
    async for event in stream:
        events.append(event)
        if event.type == "agent_end":
            final_msgs = event.messages

    # 验证记忆被注入（应有 [相关记忆] 的 system 消息）
    system_msgs = [m for m in final_msgs if hasattr(m, 'content') and isinstance(m.content, str) and "相关记忆" in m.content]
    assert len(system_msgs) >= 1

    # 验证 assistant 响应被保存到长期记忆
    after_count = len(mem.all_entries())
    assert after_count > initial_count
    # 新增的记忆应包含 "Python"
    last_entry = mem.all_entries()[-1]
    assert "Python" in last_entry.content


@pytest.mark.asyncio
async def test_loop_no_memory_no_injection():
    """不配置长期记忆时不注入。"""
    class MockProvider(LLMProvider):
        async def stream(self, messages, tools=None, config=None) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(delta_text="回答", finish_reason="stop")

    context = AgentContext(system_prompt="你是助手")
    config = AgentLoopConfig(model_config=ModelConfig(model="mock"))

    stream = await agent_loop("你好", context, config, MockProvider())
    events = []
    final_msgs = []
    async for event in stream:
        events.append(event)
        if event.type == "agent_end":
            final_msgs = event.messages

    # 不应有 [相关记忆] 消息
    system_msgs = [m for m in final_msgs if hasattr(m, 'content') and isinstance(m.content, str) and "相关记忆" in m.content]
    assert len(system_msgs) == 0
