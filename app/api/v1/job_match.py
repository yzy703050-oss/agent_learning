"""岗位匹配 HTTP 路由。

本模块只负责把 HTTP 请求转换为 Job Match Agent 的函数调用，并把运行结果
转换为稳定的 API 响应；Agent 的图定义和业务逻辑仍留在 job_match_agent 包中。
"""

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path
from langgraph.checkpoint.memory import MemorySaver

from app.schemas.job_match import (
    AnalyzeJobRequest,
    CompletedJobResponse,
    CompletedStatusResponse,
    PartialStatusResponse,
    PendingReviewResponse,
    ResumeJobRequest,
)
from job_match_agent.memory import JobMatchMemory
from job_match_agent.workflow import (
    analyze_job_with_langgraph,
    build_langgraph_thread_config,
    create_job_match_graph,
    resume_job_match_after_human_review,
)


router = APIRouter()

# 第一阶段仍使用进程内状态。以后切换数据库 checkpointer 时，只需替换这一层。
CHECKPOINTER = MemorySaver()
MEMORIES: dict[str, JobMatchMemory] = {}


def get_memory(thread_id: str) -> JobMatchMemory:
    if thread_id not in MEMORIES:
        MEMORIES[thread_id] = JobMatchMemory()
    return MEMORIES[thread_id]


def serialize_interrupts(interrupts: Any) -> list[dict[str, Any]]:
    serialized = []
    for item in interrupts:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            serialized.append(value)
        else:
            serialized.append({"value": str(value)})
    return serialized


def serialize_completed_result(result: dict[str, Any]) -> dict[str, Any]:
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


@router.post(
    "/analyze",
    response_model=CompletedJobResponse | PendingReviewResponse,
    summary="开始岗位匹配分析",
    description=(
        "提交岗位 JD 和简历技能，启动 LangGraph 工作流。若岗位存在高签证风险，"
        "接口会返回 pending_human_review，等待人工确认。"
    ),
)
def analyze_job(request: AnalyzeJobRequest):
    thread_id = request.thread_id or f"job-match-{uuid4()}"
    memory = get_memory(thread_id)

    try:
        result = analyze_job_with_langgraph(
            jd=request.job_description,
            resume_skills=request.resume_skills or None,
            memory=memory,
            checkpointer=CHECKPOINTER,
            thread_id=thread_id,
            visa_preference=request.visa_preference,
            target_role=request.target_role,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if result["interrupted"]:
        return {
            "status": "pending_human_review",
            "thread_id": result["thread_id"],
            "review": serialize_interrupts(result["interrupts"]),
        }
    return serialize_completed_result(result)


@router.post(
    "/resume",
    response_model=CompletedJobResponse,
    summary="提交人工审核并恢复流程",
)
def resume_job(request: ResumeJobRequest):
    try:
        result = resume_job_match_after_human_review(
            checkpointer=CHECKPOINTER,
            thread_id=request.thread_id,
            approved=request.approved,
            feedback=request.feedback,
            memory=get_memory(request.thread_id),
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail="没有找到该 thread_id 对应的暂停工作流。",
        ) from error
    return serialize_completed_result(result)


@router.get(
    "/status/{thread_id}",
    response_model=(
        CompletedStatusResponse | PendingReviewResponse | PartialStatusResponse
    ),
    summary="查询工作流状态",
)
def get_job_match_status(
    thread_id: str = Path(description="要查询的工作流线程 ID。"),
):
    graph = create_job_match_graph(checkpointer=CHECKPOINTER)
    checkpoint_state = graph.get_state(build_langgraph_thread_config(thread_id))

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
