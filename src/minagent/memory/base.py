"""记忆基类：Memory 抽象 + MemoryEntry。

参考 pi 的记忆设计：
- 短期（episodic）：会话内 transcript，存最近 N 轮对话
- 长期（long_term）：跨会话持久化，存重要事实/决策/失败轨迹
- hybrid 检索：BM25（精确词面）+ vector（语义）融合，RRF 排序

MemoryEntry 是记忆的基本单元：content（文本）+ metadata（来源/类型/时间戳）。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """记忆条目。"""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata 常用字段：
    #   type: "fact" | "decision" | "failure" | "preference" | "transcript"
    #   source: "user" | "assistant" | "tool"
    #   session_id: 会话 ID
    #   timestamp: 创建时间
    entry_id: str = ""
    timestamp: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        if not self.entry_id:
            # 用时间戳 + content 哈希生成 ID
            self.entry_id = f"mem_{int(self.timestamp * 1000)}_{abs(hash(self.content)) % 10000}"


class Memory(ABC):
    """记忆抽象基类。

    子类需要实现：
    - add(entry): 添加记忆
    - retrieve(query, top_k): 检索相关记忆
    - clear(): 清空（短期记忆用，长期不清）
    """

    @abstractmethod
    async def add(self, entry: MemoryEntry) -> None:
        """添加一条记忆。"""
        ...

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """检索相关记忆。返回按相关性降序排列。"""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """清空记忆。"""
        ...

    async def add_text(self, content: str, **metadata: Any) -> MemoryEntry:
        """便捷方法：直接添加文本记忆。"""
        entry = MemoryEntry(content=content, metadata=metadata)
        await self.add(entry)
        return entry
