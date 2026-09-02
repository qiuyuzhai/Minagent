"""长期记忆：跨会话持久化（jsonl）。

参考 pi 的 M_I/M_E 设计：jsonl append-only，跨 session 不丢。
每条记忆是一行 JSON，加载时全部读入内存做检索。

检索委托给 HybridRetriever（BM25 + vector 融合），
但为了不依赖外部库（qdrant/rank_bm25）也能跑，
默认降级为子串匹配 + 简单 TF-IDF。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

from minagent.memory.base import Memory, MemoryEntry


def _tokenize(text: str) -> list[str]:
    """简单分词：英文按空格分，中文按字符拆。

    因为中文没有空格，连续中文字符需要逐字拆分才能做 TF-IDF/BM25 检索。
    """
    tokens: list[str] = []
    for chunk in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", text.lower()):
        if not chunk:
            continue
        # 英文/数字块整体保留
        if re.match(r"^[a-z0-9]+$", chunk):
            tokens.append(chunk)
        else:
            # 混合或纯中文：逐字符拆分
            tokens.extend(c for c in chunk)
    return tokens


@dataclass
class LongTermMemory(Memory):
    """长期记忆：jsonl 持久化 + 简单 TF-IDF 检索。

    检索策略：
    - 默认：TF-IDF（纯 Python，无外部依赖）
    - 可选：注入 HybridRetriever 做 BM25 + vector 融合（见 retrieval.py）
    """
    file_path: str = ""
    _entries: list[MemoryEntry] = field(default_factory=list)
    _tf_cache: dict[str, Counter] = field(default_factory=dict)
    _idf_cache: dict[str, float] = field(default_factory=dict)
    _retriever: object | None = None  # HybridRetriever，可选

    def __post_init__(self) -> None:
        if self.file_path:
            self._load()

    def set_retriever(self, retriever: object) -> None:
        """注入 HybridRetriever，启用 BM25 + vector 融合检索。"""
        self._retriever = retriever
        # 把已有条目同步给 retriever
        for entry in self._entries:
            retriever.add(entry)  # type: ignore[attr-defined]

    async def add(self, entry: MemoryEntry) -> None:
        """添加记忆并持久化。"""
        self._entries.append(entry)
        self._tf_cache[entry.entry_id] = Counter(_tokenize(entry.content))
        self._rebuild_idf()
        if self._retriever:
            self._retriever.add(entry)  # type: ignore[attr-defined]
        if self.file_path:
            self._append_to_file(entry)

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """检索：有 retriever 用 hybrid，否则用 TF-IDF。"""
        if not self._entries:
            return []

        if self._retriever:
            return self._retriever.retrieve(query, top_k)  # type: ignore[attr-defined]

        # TF-IDF 检索
        return self._tfidf_retrieve(query, top_k)

    async def clear(self) -> None:
        """清空内存（不删文件）。"""
        self._entries.clear()
        self._tf_cache.clear()
        self._idf_cache.clear()

    def all_entries(self) -> list[MemoryEntry]:
        """返回所有条目。"""
        return list(self._entries)

    # ------------------------------------------------------------------
    # TF-IDF 检索（默认，无外部依赖）
    # ------------------------------------------------------------------

    def _tfidf_retrieve(self, query: str, top_k: int) -> list[MemoryEntry]:
        """TF-IDF 余弦相似度检索。"""
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        # query TF
        query_tf = Counter(query_terms)
        query_vec = {
            term: (tf / len(query_terms)) * self._idf_cache.get(term, math.log(1 + len(self._entries)))
            for term, tf in query_tf.items()
        }

        # 余弦相似度
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries:
            entry_tf = self._tf_cache.get(entry.entry_id, Counter())
            if not entry_tf:
                continue
            entry_len = sum(entry_tf.values())
            entry_vec = {
                term: (tf / entry_len) * self._idf_cache.get(term, math.log(1 + len(self._entries)))
                for term, tf in entry_tf.items()
            }

            # 点积
            dot = sum(query_vec.get(t, 0) * entry_vec.get(t, 0) for t in query_vec if t in entry_vec)
            # 模长
            q_norm = math.sqrt(sum(v * v for v in query_vec.values()))
            e_norm = math.sqrt(sum(v * v for v in entry_vec.values()))
            if q_norm == 0 or e_norm == 0:
                continue
            similarity = dot / (q_norm * e_norm)
            if similarity > 0:
                scored.append((similarity, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def _rebuild_idf(self) -> None:
        """重建 IDF 缓存。"""
        n = len(self._entries)
        if n == 0:
            return
        df: dict[str, int] = {}
        for entry in self._entries:
            terms = set(self._tf_cache.get(entry.entry_id, Counter()).keys())
            for term in terms:
                df[term] = df.get(term, 0) + 1
        self._idf_cache = {term: math.log(1 + n / (1 + freq)) for term, freq in df.items()}

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _append_to_file(self, entry: MemoryEntry) -> None:
        """append-only 写入 jsonl。"""
        path = Path(self.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "entry_id": entry.entry_id,
                "content": entry.content,
                "metadata": entry.metadata,
                "timestamp": entry.timestamp,
            }, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        """从 jsonl 加载所有条目。"""
        path = Path(self.file_path)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = MemoryEntry(
                        content=data["content"],
                        metadata=data.get("metadata", {}),
                        entry_id=data.get("entry_id", ""),
                        timestamp=data.get("timestamp", 0.0),
                    )
                    self._entries.append(entry)
                    self._tf_cache[entry.entry_id] = Counter(_tokenize(entry.content))
                except (json.JSONDecodeError, KeyError):
                    continue
        self._rebuild_idf()
