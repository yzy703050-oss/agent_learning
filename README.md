# 岗位匹配 Agent

一个用于学习和演示的岗位匹配项目。它使用 DeepSeek 分析岗位描述，通过 LangGraph 串联技能匹配、简历证据检索、岗位硬性条件检查和建议生成，并提供 FastAPI 接口。

## 主要功能

- 从岗位描述中提取岗位名称、必需与优选技能、经验要求及其他硬性条件。
- 对候选人技能进行标准化和加权匹配，并从简历文本中查找技能证据。
- 检查工作年限、学历、地点、工作许可和证书等明确的硬性条件。信息不足时标记为未知；明确不满足时暂停工作流，等待人工审核。
- 从 `career_knowledge_base/` 的 Markdown 文件中检索相关职业建议，结合业务记忆生成匹配结论和后续步骤。
- 通过 FastAPI 发起分析、恢复暂停的流程及查询状态；`main.py` 另提供命令行演示。

## 快速开始

需要 Python 3.10 或更新版本，以及可用的 DeepSeek API Key。在项目根目录创建并激活虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`，macOS/Linux 使用 `source .venv/bin/activate`。随后安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install langchain langchain-core langchain-deepseek langgraph langsmith
```

当前 `requirements.txt` 只列出 API 层依赖，因此第二条安装命令用于补齐 Agent 运行依赖。

将 `.env.example` 复制为 `.env`，至少填写：

```dotenv
DEEPSEEK_API_KEY=你的_API_Key
DEEPSEEK_MODEL=deepseek-chat
```

`.env` 已被 Git 忽略，不要将真实密钥提交到仓库。然后启动服务：

```bash
python -m uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000/docs>，可直接在交互文档中调用接口。旧入口 `python -m uvicorn api:app --reload` 也可使用。运行命令行演示可执行 `python main.py`。

## API 使用

默认前缀为 `/api/v1`，可通过 `.env` 中的 `API_V1_STR` 修改。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 检查服务状态 |
| `POST` | `/api/v1/job-match/analyze` | 分析岗位与候选人资料 |
| `POST` | `/api/v1/job-match/resume` | 提交人工审核结果并恢复流程 |
| `GET` | `/api/v1/job-match/status/{thread_id}` | 查询流程状态 |

在 `/docs` 的 `POST /api/v1/job-match/analyze` 中，可以提交以下 JSON：

```json
{
  "job_description": "招聘初级数据分析师，要求熟悉 Python 和 SQL，Tableau 优先。",
  "candidate_profile": {
    "skills": ["Python", "SQL"],
    "resume_text": "使用 Python 和 SQL 完成销售数据分析项目。",
    "years_experience": 1,
    "education_level": "本科",
    "location": "上海",
    "work_authorization": "已获工作许可",
    "certifications": []
  },
  "target_role": "数据分析师"
}
```

`candidate_profile` 是推荐的候选人输入方式。为兼容简化调用，也可传入 `resume_skills` 和 `resume_text`。`thread_id` 可选；省略时服务会生成并返回一个。请保存该 ID，以便查询或恢复同一流程。

正常完成时，响应的 `status` 为 `completed`，包含结构化 `job_info`、`match_result`、`final_answer`、`next_action`、`next_action_reason` 和 `recommended_steps` 等字段。`match_result` 中包含分数、已匹配与缺失技能、硬性条件检查结果及简历证据。

如果候选人**明确不满足**岗位硬性条件，分析会返回 `pending_human_review`、`thread_id` 和 `review`。审核后调用 `POST /api/v1/job-match/resume`：

```json
{
  "thread_id": "分析接口返回的 thread_id",
  "approved": true,
  "feedback": "继续分析，但明确说明硬性条件差距。"
}
```

`approved: false` 会生成跳过申请的建议；`approved: true` 会继续生成谨慎申请的建议。仅凭缺失的候选人信息无法判断条件时，流程不会因此暂停，但会在建议中提示补充确认。

## 项目结构

```text
app/                    FastAPI 应用、接口模型、配置和服务层
job_match_agent/        LangGraph 工作流、模型、工具、RAG 和业务记忆
career_knowledge_base/  本地 Markdown 职业知识库
tests/                  接口与工作流测试
main.py                 命令行演示入口
api.py                  兼容旧命令的 API 入口
SUBMISSION.md           早期项目提交说明
```

## 当前限制

本项目使用进程内 `MemorySaver` 保存工作流状态，并在内存中保存用户业务记忆。服务重启后，这些状态会丢失；同一个 `thread_id` 的审核恢复需要由同一运行中的服务实例处理。知识库检索使用本地 hash 向量实现，适合演示 RAG 流程，不能视为生产级语义检索。岗位分析由大模型生成，匹配结果和建议应由人核对，不应作为自动筛选或签证判断的唯一依据。

可运行 `python -m unittest discover -s tests -v` 执行现有测试。
