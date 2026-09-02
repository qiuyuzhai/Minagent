"""记忆示例：长期记忆 + hybrid 检索 + 跨会话持久化。

演示 P4 记忆层：
1. 第一次会话：用户问 Python，agent 回答，记忆存到 jsonl
2. 第二次会话：用户问 Python，agent 先检索到之前的记忆并注入 context

运行：
    python -m examples.memory_demo
"""

import asyncio
import os
import tempfile

from minagent.events import AgentEvent, EventStream
from minagent.llm import ModelConfig, OpenAICompatibleProvider
from minagent.loop import AgentContext, AgentLoopConfig, agent_loop
from minagent.memory import LongTermMemory, HybridRetriever
from minagent.messages import UserMessage


def _print_event(event: AgentEvent) -> None:
    t = event.type
    if t == "turn_start":
        print(f"\n--- turn {event.turn_index} ---")
    elif t == "message_update":
        print(event.delta_text, end="", flush=True)
    elif t == "agent_end":
        print(f"\n=== agent end ===")


async def run_session(
    prompt: str,
    memory: LongTermMemory,
    provider: OpenAICompatibleProvider,
    session_name: str,
) -> None:
    """运行一次会话。"""
    print(f"\n{'='*50}")
    print(f"会话: {session_name}")
    print(f"用户: {prompt}")
    print(f"{'='*50}")

    context = AgentContext(system_prompt="你是助手。")
    config = AgentLoopConfig(
        model_config=provider.config,
        max_turns=5,
        long_term_memory=memory,
        memory_top_k=3,
    )

    stream = await agent_loop(prompt, context, config, provider)
    async for event in stream:
        _print_event(event)

    # 显示记忆条目数
    entries = memory.all_entries()
    print(f"\n[记忆库当前 {len(entries)} 条]")


async def main() -> None:
    model_config = ModelConfig(
        model=os.getenv("LLM_MODEL_ID") or "gpt-4o-mini",
        provider="auto",
    )
    provider = OpenAICompatibleProvider(model_config)
    print(f"Provider: {provider.provider_name}, Model: {provider.model}")

    # 用临时文件做长期记忆持久化
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = os.path.join(tmpdir, "memory.jsonl")

        # 第一次会话：注入空记忆，agent 回答后保存
        mem1 = LongTermMemory(file_path=memory_path)
        # 加 hybrid retriever 提升检索质量
        mem1.set_retriever(HybridRetriever())

        await run_session("Python 是什么语言？", mem1, provider, "第一次会话")

        # 第二次会话：重新加载记忆，检索到第一次的内容
        print(f"\n>>> 重新加载记忆文件: {memory_path}")
        mem2 = LongTermMemory(file_path=memory_path)
        mem2.set_retriever(HybridRetriever())

        await run_session("Python 是什么语言？", mem2, provider, "第二次会话（跨会话记忆）")

        # 验证：第二次会话应注入第一次的记忆
        print("\n>>> 第二次会话的记忆库内容:")
        for e in mem2.all_entries():
            print(f"  - {e.content[:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
