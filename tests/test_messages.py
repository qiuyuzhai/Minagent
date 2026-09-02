"""messages 双消息模型测试。"""

import pytest

from minagent.messages import (
    ArtifactMessage,
    AssistantMessage,
    NotificationMessage,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultContent,
    ToolResultMessage,
    UserMessage,
    convert_to_llm,
)


def test_user_message_from_text():
    """UserMessage.from_text 正确构造。"""
    msg = UserMessage.from_text("hello")
    assert msg.role == "user"
    assert len(msg.content) == 1
    assert msg.content[0].text == "hello"


def test_assistant_message_text_and_tool_calls():
    """AssistantMessage 正确分离 text 和 tool_calls。"""
    msg = AssistantMessage(content=[
        TextContent(text="让我算一下"),
        ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"}),
    ])
    assert msg.text == "让我算一下"
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "calculator"


def test_convert_to_llm_filters_custom_messages():
    """convert_to_llm 过滤掉 CustomMessage。"""
    messages = [
        SystemMessage(content="你是助手"),
        UserMessage.from_text("你好"),
        NotificationMessage(text="用户切换了 tab"),  # 应被过滤
        ArtifactMessage(artifact_type="file", data={"path": "/tmp/x"}),  # 应被过滤
        AssistantMessage(content=[TextContent(text="你好")]),
    ]
    llm_msgs = convert_to_llm(messages)

    # system + user + assistant = 3 条
    assert len(llm_msgs) == 3
    assert llm_msgs[0]["role"] == "system"
    assert llm_msgs[1]["role"] == "user"
    assert llm_msgs[2]["role"] == "assistant"

    # 不应包含 custom / notification / artifact
    for m in llm_msgs:
        assert m["role"] not in ("custom", "notification", "artifact")


def test_convert_to_llm_with_tool_calls():
    """convert_to_llm 正确转换工具调用。"""
    import json

    messages = [
        UserMessage.from_text("算 1+1"),
        AssistantMessage(content=[
            TextContent(text="好的"),
            ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"}),
        ]),
        ToolResultMessage(content=[
            ToolResultContent(
                tool_call_id="call_1",
                content=[TextContent(text="1+1 = 2")],
            ),
        ]),
    ]
    llm_msgs = convert_to_llm(messages)

    assert len(llm_msgs) == 3
    # assistant 消息有 tool_calls
    assert llm_msgs[1]["role"] == "assistant"
    assert "tool_calls" in llm_msgs[1]
    tc = llm_msgs[1]["tool_calls"][0]
    assert tc["function"]["name"] == "calculator"
    args = json.loads(tc["function"]["arguments"])
    assert args == {"expression": "1+1"}

    # tool_result 消息
    assert llm_msgs[2]["role"] == "tool"
    assert llm_msgs[2]["tool_call_id"] == "call_1"
    assert "2" in llm_msgs[2]["content"]


def test_custom_message_inheritance():
    """CustomMessage 正确继承。"""
    notif = NotificationMessage(text="test")
    assert notif.role == "custom"
    assert notif.custom_type == "notification"

    artifact = ArtifactMessage(artifact_type="file", data={"x": 1})
    assert artifact.role == "custom"
    assert artifact.custom_type == "artifact"
