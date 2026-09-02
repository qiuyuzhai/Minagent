"""短期记忆：会话内 transcript。

存最近 N 条对话，用简单的子串匹配 + 时间衰减检索。
会话结束后 clear()，不持久化。

检索策略：
- 子串匹配（query 的关键词出现在 content 中）
- 时间衰减（越近的记忆权重越高）
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from minagent.memory.base import Memory, MemoryEntry


@dataclass
class EpisodicMemory(Memory):
    """短期记忆：会话内 transcript。

    用 deque(maxsize) 自动丢弃最旧的条目。
    """
    max_entries: int = 100
    _entries: deque[MemoryEntry] = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self) -> None:
        self._entries = deque(maxlen=self.max_entries)

    async def add(self, entry: MemoryEntry) -> None:
        """添加记忆。超过 max_entries 自动丢弃最旧的。"""
        self._entries.append(entry)

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """检索：子串匹配 + 时间衰减。

        时间衰减：越近的记忆权重越高，权重 = 1 / (1 + age_in_seconds / 3600)
        """
        if not self._entries:
            return []

        query_lower = query.lower()
        query_terms = [t for t in query_lower.split() if len(t) > 1]

        scored: list[tuple[float, MemoryEntry]] = []
        now = entry_time = 0.0
        import time
        now = time.time()

        for entry in self._entries:
            content_lower = entry.content.lower()
            # 子串匹配得分
            match_score = 0.0
            for term in query_terms:
                if term in content_lower:
                    match_score += 1.0
            # 完整 query 子串匹配加分
            if query_lower and query_lower in content_lower:
                match_score += 2.0

            if match_score == 0:
                continue

            # 时间衰减
            age = now - entry.timestamp
            time_weight = 1.0 / (1.0 + age / 3600.0)  # 1 小时半衰期

            final_score = match_score * time_weight
            scored.append((final_score, entry))

        # 按分数降序
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def clear(self) -> None:
        """清空短期记忆。"""
        self._entries.clear()

    def all_entries(self) -> list[MemoryEntry]:
        """返回所有条目（按时间序）。"""
        return list(self._entries)
