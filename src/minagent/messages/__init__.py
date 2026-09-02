"""消息层：双消息模型。

参考 pi 的设计，将消息分为两层：
- AgentMessage（应用层）：transcript 里可以存 UI 通知、artifact、记忆注入标记等非 LLM 消息
- LLMMessage（模型层）：只有 user / assistant / tool_result，是 LLM 能看到的

通过 convert_to_llm 在调用 LLM 前将 AgentMessage[] 转为 LLMMessage[]，
过滤掉非 LLM 消息，避免污染 context window。
"""

from __future__ import annotations

import time
from abc import ABC
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Content blocks：消息内容的积木块
# ---------------------------------------------------------------------------

class ContentBlock(BaseModel):
    """内容块基类。"""
    type: str


class TextContent(ContentBlock):
    """文本内容。"""
    type: Literal["text"] = "text"
    text: str


class ToolCall(ContentBlock):
    """工具调用请求（出现在 assistant 消息里）。"""
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultContent(ContentBlock):
    """工具执行结果（出现在 tool_result 消息里）。"""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: list[TextContent] = Field(default_factory=list)
    is_error: bool = False


# ---------------------------------------------------------------------------
# AgentMessage：应用层消息（transcript 里的任何一条）
# ---------------------------------------------------------------------------

class AgentMessage(BaseModel, ABC):
    """应用层消息基类。

    transcript 里可以存 LLM 可见的消息，也可以存 UI 通知、artifact 等
    非 LLM 消息。后者在 convert_to_llm 时被过滤掉。
    """
    role: str
    timestamp: float = Field(default_factory=lambda: time.time())


class UserMessage(AgentMessage):
    """用户消息。"""
    role: Literal["user"] = "user"
    content: list[TextContent] = Field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> UserMessage:
        return cls(content=[TextContent(text=text)])


class AssistantMessage(AgentMessage):
    """助手消息。可能包含文本和工具调用。"""
    role: Literal["assistant"] = "assistant"
    content: list[Union[TextContent, ToolCall]] = Field(default_factory=list)
    stop_reason: Literal["stop", "tool_use", "error", "aborted"] = "stop"
    error_message: str | None = None

    @property
    def text(self) -> str:
        """拼接所有文本块。"""
        return "".join(c.text for c in self.content if isinstance(c, TextContent))

    @property
    def tool_calls(self) -> list[ToolCall]:
        """所有工具调用块。"""
        return [c for c in self.content if isinstance(c, ToolCall)]


class ToolResultMessage(AgentMessage):
    """工具执行结果消息。"""
    role: Literal["tool_result"] = "tool_result"
    content: list[ToolResultContent] = Field(default_factory=list)


class SystemMessage(AgentMessage):
    """系统消息（system prompt）。"""
    role: Literal["system"] = "system"
    content: str = ""


# ---------------------------------------------------------------------------
# CustomMessage：应用层自定义消息（LLM 不可见）
# ---------------------------------------------------------------------------

class CustomMessage(AgentMessage):
    """自定义消息基类。

    用于在 transcript 里记录 UI 通知、artifact、记忆注入标记等 LLM 不需要
    看到的信息。convert_to_llm 会过滤掉这类消息。
    """
    role: Literal["custom"] = "custom"
    custom_type: str = ""


class NotificationMessage(CustomMessage):
    """UI 通知消息，例如'用户切换了 tab'。"""
    custom_type: Literal["notification"] = "notification"
    text: str = ""


class ArtifactMessage(CustomMessage):
    """artifact 消息，记录 agent 生成的产物（文件、图表等）。"""
    custom_type: Literal["artifact"] = "artifact"
    artifact_type: str = ""
    data: Any = None


# LLM 可见的消息类型
LLMMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage]

# 所有 AgentMessage 类型
AnyAgentMessage = Union[
    SystemMessage, UserMessage, AssistantMessage, ToolResultMessage,
    NotificationMessage, ArtifactMessage,
]


# ---------------------------------------------------------------------------
# convert_to_llm：AgentMessage[] -> LLMMessage[]
# ---------------------------------------------------------------------------

def convert_to_llm(messages: list[AnyAgentMessage]) -> list[dict[str, Any]]:
    """将应用层消息列表转为 LLM 可见的格式。

    1. 过滤掉 CustomMessage（LLM 不需要看到 UI 通知、artifact 等）
    2. 将剩余消息转为 OpenAI 兼容的 dict 格式
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, UserMessage):
            result.append({
                "role": "user",
                "content": "".join(c.text for c in msg.content),
            })
        elif isinstance(msg, AssistantMessage):
            # assistant 消息：文本 + tool_calls
            entry: dict[str, Any] = {"role": "assistant"}
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            if text_parts:
                entry["content"] = "".join(text_parts)
            tool_calls = [c for c in msg.content if isinstance(c, ToolCall)]
            if tool_calls:
                import json
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ]
            result.append(entry)
        elif isinstance(msg, ToolResultMessage):
            # tool_result 消息：每个结果作为一条 tool 消息
            for trc in msg.content:
                result.append({
                    "role": "tool",
                    "tool_call_id": trc.tool_call_id,
                    "content": "".join(c.text for c in trc.content),
                })
        # CustomMessage 被过滤掉，不加入 result
    return result
