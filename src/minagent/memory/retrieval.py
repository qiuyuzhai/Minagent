"""Hybrid 检索：BM25（精确词面）+ Vector（语义）融合，RRF 排序。

参考 pi 的 hybridScore 设计：
- BM25：精确匹配，抓关键词（产品名、代码标识符、专有名词）
- Vector：语义匹配，抓近义词和概念关联
- RRF（Reciprocal Rank Fusion）：合并两个排序，1 / (k + rank)

为什么不只 用向量：
- 向量会漏精确匹配（产品名 "iPhone 15" 查 "iPhone 15" 可能排第 5）
- BM25 对精确词面敏感，弥补向量这个弱点

可选依赖：
- rank_bm25：BM25 实现（pip install rank-bm25）
- numpy：向量计算（pip install numpy）
没有这些库时降级为纯 BM25（用内置实现）+ TF-IDF 向量。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from minagent.memory.base import MemoryEntry


def _tokenize(text: str) -> list[str]:
    """简单分词：英文按空格分，中文按字符拆。"""
    tokens: list[str] = []
    for chunk in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", text.lower()):
        if not chunk:
            continue
        if re.match(r"^[a-z0-9]+$", chunk):
            tokens.append(chunk)
        else:
            tokens.extend(c for c in chunk)
    return tokens


# ---------------------------------------------------------------------------
# 内置 BM25（不依赖 rank_bm25 库）
# ---------------------------------------------------------------------------

class SimpleBM25:
    """简单 BM25 实现（不依赖外部库）。

    BM25 公式：
        score(q, d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._entries: list[MemoryEntry] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0

    def add(self, entry: MemoryEntry) -> None:
        """添加文档。"""
        tokens = _tokenize(entry.content)
        self._docs.append(tokens)
        self._entries.append(entry)
        # 更新 df
        for term in set(tokens):
            self._df[term] = self._df.get(term, 0) + 1
        # 更新 avgdl
        self._avgdl = sum(len(d) for d in self._docs) / max(len(self._docs), 1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, MemoryEntry]]:
        """检索，返回 (score, entry) 列表，按分数降序。"""
        if not self._entries:
            return []
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        n = len(self._entries)
        scored: list[tuple[float, MemoryEntry]] = []

        for i, doc in enumerate(self._docs):
            score = 0.0
            doc_len = len(doc)
            tf_counter = Counter(doc)

            for term in query_terms:
                if term not in tf_counter:
                    continue
                tf = tf_counter[term]
                # IDF
                df = self._df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                # BM25 score
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
                score += idf * numerator / denominator

            if score > 0:
                scored.append((score, self._entries[i]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 简单向量检索（TF-IDF 向量 + 余弦相似度，不依赖 numpy）
# ---------------------------------------------------------------------------

class SimpleVectorRetriever:
    """TF-IDF 向量检索（纯 Python，不依赖 numpy/qdrant）。

    生产环境可替换为 qdrant + 真实 embedding 模型。
    """

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []
        self._tfidf_vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    def add(self, entry: MemoryEntry) -> None:
        """添加文档。"""
        self._entries.append(entry)
        self._rebuild_vectors()

    def _rebuild_vectors(self) -> None:
        """重建所有 TF-IDF 向量。"""
        n = len(self._entries)
        if n == 0:
            return
        # IDF
        df: dict[str, int] = {}
        all_tokens = []
        for entry in self._entries:
            tokens = _tokenize(entry.content)
            all_tokens.append(tokens)
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self._idf = {term: math.log(1 + n / (1 + freq)) for term, freq in df.items()}

        # TF-IDF 向量
        self._tfidf_vectors = []
        for tokens in all_tokens:
            tf = Counter(tokens)
            total = len(tokens) or 1
            vec = {term: (freq / total) * self._idf.get(term, 0) for term, freq in tf.items()}
            self._tfidf_vectors.append(vec)

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, MemoryEntry]]:
        """检索，返回 (similarity, entry) 列表。"""
        if not self._entries:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        query_tf = Counter(query_terms)
        query_total = len(query_terms)
        query_vec = {
            term: (freq / query_total) * self._idf.get(term, math.log(1 + len(self._entries)))
            for term, freq in query_tf.items()
        }
        q_norm = math.sqrt(sum(v * v for v in query_vec.values()))

        scored: list[tuple[float, MemoryEntry]] = []
        for i, vec in enumerate(self._tfidf_vectors):
            # 余弦相似度
            dot = sum(query_vec.get(t, 0) * vec.get(t, 0) for t in query_vec if t in vec)
            e_norm = math.sqrt(sum(v * v for v in vec.values()))
            if q_norm == 0 or e_norm == 0:
                continue
            sim = dot / (q_norm * e_norm)
            if sim > 0:
                scored.append((sim, self._entries[i]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# HybridRetriever：BM25 + Vector 融合（RRF）
# ---------------------------------------------------------------------------

@dataclass
class HybridRetriever:
    """混合检索器：BM25 + Vector 融合，RRF 排序。

    RRF (Reciprocal Rank Fusion):
        score(d) = Σ 1 / (k + rank_i(d))
    其中 k 是平滑常数（默认 60），rank_i(d) 是文档 d 在第 i 个检索器中的排名。

    用法：
        retriever = HybridRetriever()
        retriever.add(entry)
        results = retriever.retrieve("query", top_k=5)
    """
    rrf_k: int = 60  # RRF 平滑常数
    _bm25: SimpleBM25 = field(default_factory=SimpleBM25)
    _vector: SimpleVectorRetriever = field(default_factory=SimpleVectorRetriever)
    _all_entries: list[MemoryEntry] = field(default_factory=list)

    def add(self, entry: MemoryEntry) -> None:
        """添加到两个检索器。"""
        self._bm25.add(entry)
        self._vector.add(entry)
        self._all_entries.append(entry)

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """hybrid 检索：BM25 + Vector 各取 top_k*2，RRF 融合后取 top_k。"""
        if not self._all_entries:
            return []

        # 两个检索器各取更多候选
        candidates_k = top_k * 4
        bm25_results = self._bm25.search(query, top_k=candidates_k)
        vec_results = self._vector.search(query, top_k=candidates_k)

        # RRF 融合
        # rank 从 1 开始，第 1 名 score = 1/(k+1)
        rrf_scores: dict[str, float] = {}  # entry_id -> score
        entry_map: dict[str, MemoryEntry] = {}

        for rank, (_, entry) in enumerate(bm25_results, 1):
            rrf_scores[entry.entry_id] = rrf_scores.get(entry.entry_id, 0) + 1.0 / (self.rrf_k + rank)
            entry_map[entry.entry_id] = entry

        for rank, (_, entry) in enumerate(vec_results, 1):
            rrf_scores[entry.entry_id] = rrf_scores.get(entry.entry_id, 0) + 1.0 / (self.rrf_k + rank)
            entry_map[entry.entry_id] = entry

        # 按 RRF 分数降序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda eid: rrf_scores[eid], reverse=True)
        return [entry_map[eid] for eid in sorted_ids[:top_k]]
