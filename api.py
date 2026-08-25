from uuid import uuid4

from fastapi import FastAPI, HTTPException, Path
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from job_match_agent.memory import JobMatchMemory
from job_match_agent.workflow import (
    analyze_job_with_langgraph,
    build_langgraph_thread_config,
    create_job_match_graph,
    resume_job_match_after_human_review,
)


API_DESCRIPTION = """
这是一个基于 LangGraph 的岗位匹配 Agent 后端示例。

它演示了真实项目中常见的三段能力：

1. 分析岗位 JD，并结合简历技能、RAG 知识库和业务记忆生成匹配结果。
2. 当岗位存在高签证风险时，通过 LangGraph interrupt 暂停流程，等待人工确认。
3. 人工确认后使用 thread_id 和 checkpoint 恢复流程，继续生成下一步行动建议。

当前版本使用内存中的 MemorySaver 和 JobMatchMemory，适合学习和课堂提交；生产环境应换成 SQLite 或数据库持久化。
"""

OPENAPI_TAGS = [
    {
        "name": "系统状态",
        "description": "检查后端服务是否正常运行。",
    },
    {
        "name": "岗位匹配",
        "description": "启动岗位分析、提交人工审核结果、查询工作流状态。",
    },
]


app = FastAPI(
    title="岗位匹配 Agent 后端 API",
    description=API_DESCRIPTION,
    version="0.1.0",
    openapi_tags=OPENAPI_TAGS,
)

CHECKPOINTER = MemorySaver()
MEMORIES: dict[str, JobMatchMemory] = {}


class AnalyzeJobRequest(BaseModel):
    job_description: str = Field(
        description="岗位 JD 原文。",
        examples=[
            (
                "We are looking for a Data Analyst with experience in Python, "
                "SQL, Tableau, and A/B testing. No sponsorship is available."
            )
        ],
    )
    thread_id: str | None = Field(
        default=None,
        description=(
            "工作流线程 ID。第一次请求可以不传，后端会自动生成；"
            "如果需要后续 resume，前端必须保存返回的 thread_id。"
        ),
        examples=["job-match-demo-user-1"],
    )
    resume_skills: list[str] = Field(
        default_factory=list,
        description="候选人简历技能列表。",
        examples=[["Python", "SQL", "Tableau", "Excel"]],
    )
    visa_preference: str | None = Field(
        default=None,
        description="用户对签证、工卡或 sponsorship 的偏好说明。",
        examples=["I need OPT, CPT, H-1B sponsorship, or visa-friendly roles."],
    )
    target_role: str | None = Field(
        default=None,
        description="用户目标岗位。",
        examples=["Entry-level data analyst"],
    )


class ResumeJobRequest(BaseModel):
    thread_id: str = Field(
        description="需要恢复的 LangGraph 工作流线程 ID。",
        examples=["job-match-demo-user-1"],
    )
    approved: bool = Field(
        description="人工审核是否同意继续分析高风险岗位。",
        examples=[True],
    )
    feedback: str = Field(
        default="",
        description="人工审核意见，会写回 LangGraph state。",
        examples=["Continue, but make the visa risk clear."],
    )


def get_memory(thread_id: str) -> JobMatchMemory:
    if thread_id not in MEMORIES:
        MEMORIES[thread_id] = JobMatchMemory()

    return MEMORIES[thread_id]


def serialize_interrupts(interrupts) -> list[dict]:
    serialized = []
    for item in interrupts:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            serialized.append(value)
        else:
            serialized.append({"value": str(value)})

    return serialized


def serialize_completed_result(result: dict) -> dict:
    return {
        "status": "completed",
        "thread_id": result["thread_id"],
        "job_info": result["job_info"].model_dump(),
        "retrieved_context": result["retrieved_context"],
        "visa_warning": result["visa_warning"],
        "match_result": result["match_result"].model_dump(),
        "final_answer": result["final_answer"],
        "next_action": result["next_action"],
        "next_action_reason": result["next_action_reason"],
        "recommended_steps": result["recommended_steps"],
    }


def serialize_interrupted_result(result: dict) -> dict:
    return {
        "status": "pending_human_review",
        "thread_id": result["thread_id"],
        "review": serialize_interrupts(result["interrupts"]),
    }


@app.get(
    "/health",
    tags=["系统状态"],
    summary="健康检查",
    description="用于确认 FastAPI 服务是否已经启动。",
)
def health_check() -> dict:
    return {"status": "ok"}


@app.post(
    "/job-match/analyze",
    tags=["岗位匹配"],
    summary="开始岗位匹配分析",
    description=(
        "提交岗位 JD 和简历技能，启动 LangGraph 工作流。"
        "如果岗位存在高签证风险，接口会返回 pending_human_review，等待人工确认。"
    ),
)
def analyze_job(request: AnalyzeJobRequest) -> dict:
    thread_id = request.thread_id or f"job-match-{uuid4()}"
    memory = get_memory(thread_id)

    resume_skills = request.resume_skills or None
    try:
        result = analyze_job_with_langgraph(
            jd=request.job_description,
            resume_skills=resume_skills,
            memory=memory,
            checkpointer=CHECKPOINTER,
            thread_id=thread_id,
            visa_preference=request.visa_preference,
            target_role=request.target_role,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if result["interrupted"]:
        return serialize_interrupted_result(result)

    return serialize_completed_result(result)


@app.post(
    "/job-match/resume",
    tags=["岗位匹配"],
    summary="提交人工审核并恢复流程",
    description=(
        "当前一次分析因为 interrupt 暂停后，前端调用该接口提交 approved/feedback，"
        "后端会使用 thread_id 从 checkpoint 恢复 LangGraph 工作流。"
    ),
)
def resume_job(request: ResumeJobRequest) -> dict:
    memory = get_memory(request.thread_id)

    try:
        result = resume_job_match_after_human_review(
            checkpointer=CHECKPOINTER,
            thread_id=request.thread_id,
            approved=request.approved,
            feedback=request.feedback,
            memory=memory,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail="没有找到该 thread_id 对应的暂停工作流。",
        ) from error

    return serialize_completed_result(result)


@app.get(
    "/job-match/status/{thread_id}",
    tags=["岗位匹配"],
    summary="查询工作流状态",
    description=(
        "根据 thread_id 查询当前 LangGraph checkpoint 状态，"
        "用于判断任务是已完成、等待人工审核，还是处于部分运行状态。"
    ),
)
def get_job_match_status(
    thread_id: str = Path(description="要查询的工作流线程 ID。"),
) -> dict:
    graph = create_job_match_graph(checkpointer=CHECKPOINTER)
    graph_config = build_langgraph_thread_config(thread_id)
    checkpoint_state = graph.get_state(graph_config)

    if not checkpoint_state.values:
        raise HTTPException(status_code=404, detail="未知的 thread_id。")

    values = checkpoint_state.values
    if "__interrupt__" in values:
        return {
            "status": "pending_human_review",
            "thread_id": thread_id,
            "review": serialize_interrupts(values["__interrupt__"]),
        }

    if "match_result" in values:
        return {
            "status": "completed",
            "thread_id": thread_id,
            "next_action": values.get("next_action"),
            "next_action_reason": values.get("next_action_reason"),
            "recommended_steps": values.get("recommended_steps", []),
        }

    return {
        "status": "running_or_partial",
        "thread_id": thread_id,
        "state_keys": sorted(values.keys()),
    }
