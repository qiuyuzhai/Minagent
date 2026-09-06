# minagent

参考开源 agent（[pi](https://github.com/badlogic/pi-mono)、Claude Code 等）的设计，用 Python 复现 agent runtime 的核心机制。

> **定位**：学习型项目，目的是理解 agent 底层原理，而非生产级框架。
> 复现的是设计思想而非代码——pi 是 TypeScript 实现，本项目用 Python 重新实现其核心机制，并做了适合 Python 生态的取舍。

## 为什么做这个项目

市面上 agent 框架（LangChain、CrewAI 等）大多把细节封装起来，用起来方便但出问题很难排查。本项目的做法是反过来的：**从最底层的 runtime 写起**，把 agent 循环、消息转换、工具调度、中断控制、记忆检索这些"框架替你做了"的事情亲手实现一遍。写完之后，对"agent 到底是怎么跑起来的"这个问题就有了第一手的答案。

复现过程中重点吸收了 pi 的几个设计哲学：

1. **双消息模型**——应用层消息和 LLM 可见消息分离
2. **全程事件流**——不是"调一次拿结果"，而是每一步都发事件
3. **turn 抽象**——一个 turn = 一次 assistant 响应 + 它的工具调用
4. **steering 队列**——agent 运行中可以被打断插话，但不停止循环

## 核心机制

### 1. 双消息模型（messages/）

transcript 里允许存两类消息：

- **LLM 可见消息**：`SystemMessage` / `UserMessage` / `AssistantMessage` / `ToolResultMessage`
- **LLM 不可见消息**：`NotificationMessage`（UI 通知）、`ArtifactMessage`（生成的产物）等 `CustomMessage`

调用 LLM 前由 `convert_to_llm()` 统一转换：过滤 CustomMessage、把剩余消息转为 OpenAI 兼容格式、拆解 tool_calls / tool_result。

```python
from minagent.messages import NotificationMessage, convert_to_llm

messages = [user_msg, NotificationMessage(text="用户切换了 tab"), assistant_msg]
llm_input = convert_to_llm(messages)  # NotificationMessage 被过滤掉
```

**解决的问题**：UI 通知、artifact 等信息需要记录在 transcript 里（供 UI 渲染），但塞给 LLM 只会浪费 context window 甚至干扰模型。

### 2. 事件流（events/）

agent 运行的每一步都发事件，UI / 日志 / 后端逻辑订阅同一套流：

```
agent_start
 └─ turn_start
     ├─ message_start / message_update(delta) / message_end
     └─ tool_execution_start / tool_execution_end / tool_result_message
 └─ turn_end
agent_end (携带全部新增消息)
```

实现是 `asyncio.Queue` + 异步生成器，生产者（agent loop）与消费者（打印 UI）完全解耦：

```python
stream = await agent_loop(prompt, context, config, provider)
async for event in stream:
    if event.type == "message_update":
        print(event.delta_text, end="")     # 流式打印
    elif event.type == "tool_execution_start":
        print(f"调用工具 {event.tool_name}")
```

### 3. turn 循环与生命周期钩子（loop/）

```
外层循环：follow-up 续跑（agent 停止后队列有消息则重启）
 └─ 内层循环：turn 循环
     ├─ transform_context钩子：裁剪/注入 context
     ├─ stream assistant 响应（逐 token 发事件）
     ├─ 执行工具（ToolExecutor）
     ├─ steering 注入：turn 间隙 drain 用户插话
     ├─ should_stop_after_turn 钩子：优雅停止
     └─ prepare_next_turn 钩子：动态调整下一轮
```

三层停止机制防止死循环：

| 机制 | 触发条件 |
|---|---|
| 自然停止 | assistant 不再调用工具 |
| `should_stop_after_turn` | 钩子返回 True（如 context 快满） |
| `max_turns` | 硬上限（默认 20） |

### 4. 工具系统（tools/）

```python
from pydantic import BaseModel, Field
from minagent.tools import AgentTool, ToolResult, ToolRegistry, ToolExecutor

class CalculatorParams(BaseModel):
    expression: str = Field(..., description="数学表达式")

class CalculatorTool(AgentTool):
    name = "calculator"
    description = "四则运算计算器"
    params_schema = CalculatorParams
    execution_mode = "parallel"          # 或 "sequential"

    async def execute(self, params, signal=None, on_update=None) -> ToolResult:
        return ToolResult.text(f"{params['expression']} = 72", details={"result": 72})
```

关键设计：

- **pydantic schema**：参数声明即校验，`to_openai_schema()` 自动生成 function calling 格式
- **content / details 分离**：`content` 给 LLM 看，`details` 给 UI/日志看
- **sequential / parallel 双模式**：
  - parallel：预备阶段串行（校验 + before 钩子）→ 执行阶段 `asyncio.gather` 真并发 → **结果按 assistant 原始顺序返回**（保证 LLM 看到的顺序稳定）
  - per-tool `execution_mode = "sequential"` 可强制整批串行（如写文件工具）
- **before / after 钩子**：before 可阻断（权限拦截），after 可覆盖结果

```python
executor = ToolExecutor(
    registry,
    before_tool_call=lambda ctx: BeforeToolCallResult(block=True) if is_dangerous(ctx) else None,
    default_execution_mode="parallel",
)
```

### 5. 中断控制（control/）

| 组件 | 作用 |
|---|---|
| `AbortController` | `asyncio.Event` 信号传播，abort 后工具与 loop 逐层优雅退出 |
| `SteeringQueue` | agent 运行中用户插话：消息入队，**turn 结束后** drain 注入 context，循环不停止 |
| `FollowUpQueue` | agent 自然停止后检查队列，有消息则续跑（外层循环） |
| `TimeoutManager` | 工具级 / turn 级超时（`asyncio.wait_for`），turn 超时自动触发 abort |

steering 与 follow-up 的区别：

```
Steering：  [turn 1] ──(插话入队)──> [drain 注入] ──> [turn 2]     # 不停
FollowUp：  [agent 停止] ──(队列有消息?)──> [注入并重启循环]        # 续跑
```

### 6. 记忆与检索（memory/）

```
Memory (抽象)
├── EpisodicMemory   短期：会话内 transcript，deque 自动淘汰，子串匹配 + 时间衰减
└── LongTermMemory   长期：jsonl append-only 持久化，跨会话不丢
    └── HybridRetriever（可选注入）
        ├── SimpleBM25            精确词面匹配（内置实现，k1=1.5, b=0.75）
        ├── SimpleVectorRetriever TF-IDF 向量 + 余弦相似度
        └── RRF 融合              score(d) = Σ 1/(k + rank_i(d))，k=60
```

**为什么 hybrid 而不是纯向量**：向量检索会漏精确匹配——查 "iPhone 15" 时语义相近的 "iPhone 14" 可能排前面。BM25 对精确词面敏感，两者 RRF 融合互补。

接入 loop 的方式：

- **注入**：`agent_loop` 启动时用最后一条 user 消息检索长期记忆，以 `[相关记忆]` system 消息注入 context
- **保存**：agent 结束时把有实质内容的 assistant 响应写回长期记忆

```python
memory = LongTermMemory(file_path="memory.jsonl")
memory.set_retriever(HybridRetriever())
config = AgentLoopConfig(..., long_term_memory=memory, memory_top_k=3)
```

## 架构总览

```
src/minagent/
├── messages/     双消息模型 + convert_to_llm      ← Context 管理考点
├── events/       EventStream + 10 种事件类型      ← 流式考点
├── loop/         agent_loop 双层循环 + 生命周期钩子 ← Agent 循环考点
├── llm/          OpenAICompatibleProvider（5 种 provider 自动检测）
├── tools/        AgentTool + ToolRegistry + ToolExecutor ← 工具系统考点
│   └── builtin.py  CalculatorTool (parallel) / ReadFileTool (sequential)
├── control/      AbortController + Steering/FollowUp + Timeout ← 中断控制考点
└── memory/       Episodic + LongTerm + HybridRetriever ← 记忆/RAG 考点
```

依赖方向：`loop` 是中枢，向下依赖其余各层；各层之间互不依赖。

## 安装

```bash
git clone https://github.com/qiuyuzhai/minagent.git
cd minagent
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"         # 或手动装: pip install pydantic openai tiktoken pytest pytest-asyncio
```

要求 Python 3.12+。

## 快速开始

### 1. 跑测试（不需要 LLM API）

```bash
$env:PYTHONPATH = "src"        # Windows PowerShell
export PYTHONPATH=src          # Linux/macOS
pytest tests/ -v               # 55 个测试，全部基于 mock provider
```

### 2. 跑示例（需要 LLM API）

配置任一 provider 的环境变量：

```bash
# OpenAI
set OPENAI_API_KEY=sk-xxx
# 智谱
set ZHIPU_API_KEY=xxx
# ModelScope
set MODELSCOPE_API_KEY=ms-xxx
# Ollama 本地
set LLM_BASE_URL=http://localhost:11434/v1
```

三个递进的示例：

```bash
# 最小 ReAct 循环：LLM 调用计算器工具，观察完整事件流
python -m examples.basic_react

# 中断控制：agent 运行中被 steering 插话、被 follow-up 续跑
python -m examples.control_demo

# 跨会话记忆：两次会话共享 jsonl 记忆库，第二次会话自动检索注入
python -m examples.memory_demo
```

### 3. 最小用法

```python
import asyncio
from minagent.llm import ModelConfig, OpenAICompatibleProvider
from minagent.loop import AgentContext, AgentLoopConfig, agent_loop

async def main():
    provider = OpenAICompatibleProvider(ModelConfig(model="gpt-4o-mini"))

    context = AgentContext(system_prompt="你是一个简洁的助手。")
    config = AgentLoopConfig(model_config=provider.config, max_turns=5)

    stream = await agent_loop("用一句话介绍 Python", context, config, provider)
    async for event in stream:
        if event.type == "message_update":
            print(event.delta_text, end="", flush=True)

asyncio.run(main())
```

## 测试

55 个单元测试，按层组织：

| 测试文件 | 覆盖 | 数量 |
|---|---|---|
| `test_messages.py` | 双消息模型、convert_to_llm 过滤 | 4 |
| `test_events.py` | EventStream 生产消费、end 语义 | 3 |
| `test_loop.py` | turn 循环、max_turns 防护、钩子停止 | 4 |
| `test_tools.py` | registry、双模式、并发、before/after 钩子 | 17 |
| `test_control.py` | abort、steering、follow-up、timeout | 11 |
| `test_memory.py` | 短期/长期记忆、BM25/Vector/RRF、loop 集成 | 15 |

所有测试用 `MockProvider`（按脚本回放 chunk），不依赖真实 LLM，可离线跑、可进 CI。

其中并发正确性测试值得一提：`test_executor_parallel_runs_concurrently` 用两个 0.3s 的慢工具验证 parallel 总耗时 < sequential（~0.6s），防止并发退化为串行（开发过程中真实出现过这个 bug，被该测试抓住后修复）。

## 设计取舍记录

| pi 的做法 | 本项目的做法 | 原因 |
|---|---|---|
| TypeBox schema | pydantic v2 | Python 生态标准，声明即校验 |
| pi-ai 多 provider 抽象 | OpenAI 兼容协议一把梭 | OpenAI/ModelScope/智谱/Ollama/vLLM 都兼容同一协议，抽象成本不值 |
| WebWorker 有状态内核 | 未复现（计划中） | 涉及子进程管理，单独一个阶段做 |
| qdrant + embedding | 内置 TF-IDF/BM25 | 零依赖可跑；接口已留好，可无缝替换为真实向量库 |
| TypeScript declaration merging | dataclass / Union | 语言特性差异 |

## 技术栈

- Python 3.12+ / asyncio
- pydantic v2（schema 与校验）
- openai SDK（OpenAI 兼容协议，覆盖 5 种 provider）
- pytest + pytest-asyncio（55 个测试）

## Roadmap

- [ ] Context compaction：token 计数 + 历史摘要压缩（tiktoken 已引入未接入）
- [ ] 有状态 Python 内核：参考 pi 的 worker 设计，常驻进程 + AST 拆分末行回显
- [ ] 多角色编排：6 角色状态机 + CONTINUE/REFINE/PIVOT 检查点
- [ ] 真实向量库接入：qdrant-client 后端替换 SimpleVectorRetriever

## License

MIT
