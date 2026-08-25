# 岗位匹配 Agent 项目说明

## 项目简介

本项目实现了一个基于 LangChain、LangGraph 和 RAG 的岗位匹配 Agent。系统可以读取岗位 JD，抽取结构化岗位信息，结合候选人简历技能、本地职业知识库和用户记忆，生成岗位匹配结果、签证风险提示和下一步行动建议。

项目已从单文件 demo 重构为真实项目结构，并额外提供 FastAPI 后端接口，方便前端或外部系统调用。

## 核心能力

- 岗位信息抽取：使用结构化输出得到岗位标题、技能要求、经验要求、签证风险和 entry-level 友好度。
- RAG 检索：从 `career_knowledge_base/` 中检索职业建议、技能信号和签证风险知识。
- 技能匹配评分：通过 `calculate_fit_score` 计算简历技能和岗位技能的匹配度。
- LangGraph workflow：使用 State、Node、Conditional Edge、Checkpoint、Interrupt/Resume 编排完整业务流程。
- 人工审核：当岗位存在高签证风险时，Graph 会暂停并等待人工确认。
- 行动建议：根据匹配分数、签证风险、entry-level 适配度和缺失技能生成 `next_action`。
- FastAPI 服务：提供中文 API 文档和可调用接口。

## 主要目录结构

```text
G:\agent_learning
├── api.py                         # FastAPI 后端入口
├── main.py                        # 命令行 demo 入口
├── messages_agent_demo.py         # MessagesState + ToolNode 的 ReAct 学习 demo
├── requirements.txt               # FastAPI 运行依赖
├── SUBMISSION.md                  # 项目提交说明
├── career_knowledge_base/         # 本地 RAG Markdown 知识库
└── job_match_agent/
    ├── chat_agent.py              # 外层 Career Chat Agent
    ├── workflow.py                # 内层 LangGraph Job Match workflow
    ├── memory.py                  # 用户业务记忆和历史压缩
    ├── rag.py                     # Markdown RAG、本地向量检索
    ├── chains.py                  # LangChain chains 和 structured output
    ├── models.py                  # Pydantic 结构化模型
    ├── prompts.py                 # Prompt 模板
    ├── tools.py                   # 技能匹配工具
    ├── llm.py                     # LLM 创建函数
    └── config.py                  # .env 配置读取
```

## LangGraph 工作流

当前核心 workflow 位于 `job_match_agent/workflow.py`。

流程如下：

```text
START
  -> extract_job_info
  -> retrieve_context
  -> calculate_score
  -> route_authorization_risk

如果 authorization_risk == high:
  -> visa_warning
  -> interrupt 等待人工确认
  -> recommend_next_action

如果 authorization_risk != high:
  -> recommend_next_action

recommend_next_action
  -> route_after_next_action
  -> generate_summary 或 skip_application_summary
  -> END
```

其中：

- `visa_warning_node` 是人工审核暂停点。
- `resume_job_match_after_human_review` 使用 `Command(resume=...)` 恢复流程。
- `recommend_next_action_node` 是业务决策节点，根据 state 生成下一步行动建议。

## FastAPI 接口

启动服务后访问：

```text
http://127.0.0.1:8000/docs
```

接口包括：

```text
GET  /health
POST /job-match/analyze
POST /job-match/resume
GET  /job-match/status/{thread_id}
```

### POST /job-match/analyze

提交岗位 JD 和简历技能，启动岗位匹配分析。

如果没有高风险，会直接返回：

```json
{
  "status": "completed",
  "thread_id": "...",
  "job_info": {},
  "match_result": {},
  "next_action": "apply",
  "recommended_steps": []
}
```

如果存在高签证风险，会返回：

```json
{
  "status": "pending_human_review",
  "thread_id": "...",
  "review": [
    {
      "reason": "high_authorization_risk",
      "question": "This job is high authorization risk. Continue generating the final summary?"
    }
  ]
}
```

### POST /job-match/resume

当前端或人工审核者确认后，提交：

```json
{
  "thread_id": "...",
  "approved": true,
  "feedback": "Continue, but make the visa risk clear."
}
```

后端会从 checkpoint 恢复 LangGraph workflow，并返回最终结果。

## 运行方式

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置 `.env`：

```text
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
```

3. 启动 FastAPI：

```bash
uvicorn api:app --reload
```

4. 打开接口文档：

```text
http://127.0.0.1:8000/docs
```

## 当前限制和后续改进

- 当前 checkpoint 使用 `MemorySaver`，服务重启后状态会丢失。
- 当前用户记忆保存在进程内字典 `MEMORIES` 中，生产环境应换成数据库。
- 当前 RAG 使用本地 hash embedding，适合学习演示，真实项目可换成更强的 embedding 模型。
- 后续可加入 SQLite checkpoint、用户登录、前端审核页面和 LangSmith 追踪。

## 学习重点

这个项目展示了从学习 demo 到真实服务的完整演进：

```text
单文件 Agent
  -> 项目结构拆分
  -> LangGraph 业务 workflow
  -> RAG + Memory
  -> interrupt/resume 人工审核
  -> FastAPI 后端接口
```

它的重点不是单个语法点，而是理解 Agent 系统如何被拆成可维护、可扩展、可接入前端的真实工程结构。
