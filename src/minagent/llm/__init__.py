"""LLM 层：provider 抽象 + OpenAI 兼容实现。

复用 multi-agent-system 的 provider 自动检测思路，支持：
OpenAI / ModelScope / 智谱 / Ollama / vLLM（都走 OpenAI 兼容协议）。

提供同步与流式两种调用接口，流式接口返回 chunk 迭代器，
每个 chunk 携带 delta_text 或 tool_call 增量。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Provider 配置与自动检测（复用 multi-agent-system 思路）
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """模型配置。"""
    model: str
    provider: str = "auto"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: int = 60


def _auto_detect_provider(api_key: str | None, base_url: str | None) -> str:
    """根据环境变量与 base_url 自动检测 provider。"""
    if os.getenv("MODELSCOPE_API_KEY"):
        return "modelscope"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ZHIPU_API_KEY"):
        return "zhipu"

    actual_base_url = base_url or os.getenv("LLM_BASE_URL", "")
    url_lower = actual_base_url.lower()
    if "api-inference.modelscope.cn" in url_lower:
        return "modelscope"
    if "open.bigmodel.cn" in url_lower:
        return "zhipu"
    if ":11434" in url_lower:
        return "ollama"
    if ":8000" in url_lower:
        return "vllm"

    actual_api_key = api_key or os.getenv("LLM_API_KEY", "")
    if actual_api_key.startswith("ms-"):
        return "modelscope"
    if actual_api_key.startswith("sk-"):
        return "openai"
    return "auto"


def _resolve_credentials(
    provider: str, api_key: str | None, base_url: str | None
) -> tuple[str | None, str | None]:
    """根据 provider 解析 api_key 和 base_url。"""
    defaults: dict[str, tuple[str, str]] = {
        "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
        "modelscope": ("MODELSCOPE_API_KEY", "https://api-inference.modelscope.cn/v1/"),
        "zhipu": ("ZHIPU_API_KEY", "https://open.bigmodel.cn/api/paas/v4/"),
        "ollama": ("", "http://localhost:11434/v1"),
        "vllm": ("", "http://localhost:8000/v1"),
    }
    if provider in defaults:
        env_key, default_url = defaults[provider]
        resolved_key = api_key or (os.getenv(env_key) if env_key else None) or os.getenv("LLM_API_KEY")
        resolved_url = base_url or os.getenv("LLM_BASE_URL") or default_url
        return resolved_key, resolved_url
    # auto / local / 其他
    return (
        api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url or os.getenv("LLM_BASE_URL"),
    )


def _default_model(provider: str) -> str:
    """根据 provider 返回默认模型。"""
    defaults = {
        "openai": "gpt-4o-mini",
        "modelscope": "Qwen/Qwen2.5-72B-Instruct",
        "zhipu": "glm-4",
        "ollama": "llama3",
        "vllm": "Qwen/Qwen1.5-0.5B-Chat",
    }
    return defaults.get(provider, "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Stream chunk：流式响应的增量
# ---------------------------------------------------------------------------

@dataclass
class StreamChunk:
    """流式响应的一个 chunk。"""
    delta_text: str = ""
    # 工具调用增量（流式累积）
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_arguments_delta: str = ""
    # 结束信息
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# LLMProvider 抽象
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """LLM provider 抽象基类。"""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        config: ModelConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式调用 LLM，返回 chunk 迭代器。"""
        ...
        # 这里的 ... 只是占位，实际由子类实现
        # 异步生成器不能用 raise StopAsyncIteration 终止，用 return 即可
        yield  # type: ignore[unreachable]


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider：覆盖 OpenAI/ModelScope/智谱/Ollama/vLLM
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(LLMProvider):
    """OpenAI 兼容协议的 provider 实现。"""

    def __init__(self, config: ModelConfig):
        # 确定 provider
        provider = config.provider
        if provider == "auto":
            provider = _auto_detect_provider(config.api_key, config.base_url)

        # 解析凭证
        api_key, base_url = _resolve_credentials(provider, config.api_key, config.base_url)
        if not api_key:
            api_key = "ollama"  # ollama/vllm 可能不需要 key

        # 解析模型
        model = config.model or _default_model(provider)

        self.provider_name = provider
        self.model = model
        self.config = ModelConfig(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=config.timeout,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        config: ModelConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式调用，逐 chunk yield。"""
        cfg = config or self.config
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "stream": True,
        }
        if cfg.max_tokens:
            kwargs["max_tokens"] = cfg.max_tokens
        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            # 文本增量
            if delta.content:
                yield StreamChunk(delta_text=delta.content)

            # 工具调用增量
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield StreamChunk(
                        tool_call_id=tc.id,
                        tool_call_name=tc.function.name if tc.function else None,
                        tool_call_arguments_delta=tc.function.arguments if tc.function and tc.function.arguments else "",
                    )

            # 结束
            if choice.finish_reason:
                yield StreamChunk(finish_reason=choice.finish_reason)
