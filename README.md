# minagent

参考开源 agent（pi、Claude Code 等）的设计，用 Python 复现 agent runtime 的核心机制。

## 目的

学习 agent 底层原理，非生产级框架。复现的核心机制：

- **双消息模型**：AgentMessage（应用层）vs LLMMessage（模型层），通过 `convertToLlm` 在调用 LLM 前转换，避免 UI 通知、artifact 等非 LLM 消息污染 context window
- **事件流**：全程发事件（`agent_start/end`、`turn_start/end`、`message_delta`、`tool_execution_*`），UI 与后端订阅同一套事件流
- **turn 循环**：一次 assistant 响应 + 它的工具调用为一个 turn，通过 `shouldStopAfterTurn` / `prepareNextTurn` 等钩子控制停止与切换
- **steering / follow-up 队列**：agent 运行中用户可插话（steering），agent 停止后若有排队消息则续跑（follow-up）
- **工具系统**：sequential / parallel 双模式 + per-tool executionMode + before/after 钩子 + content/details 结果分离
- **记忆 / RAG**：短期（会话内 transcript）+ 长期（跨会话持久）+ hybrid 检索（BM25 + vector 融合）

## 开发阶段

- [x] P1：messages + events + loop + llm（最小 ReAct 循环）
- [x] P2：tools（ToolRegistry + ToolExecutor + pydantic schema + sequential/parallel 双模式 + before/after 钩子）
- [x] P3：control（AbortController + SteeringQueue + FollowUpQueue + TimeoutManager）
- [ ] P4：memory（episodic + long_term + hybrid retrieval）

## 技术栈

Python 3.12+、asyncio、pydantic v2、openai SDK、tiktoken

## License

MIT
