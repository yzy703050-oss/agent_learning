"""岗位匹配 HTTP 路由。"""

from fastapi import APIRouter, HTTPException, Path

from app.schemas.job_match import (
    AnalyzeJobRequest,
    CompletedJobResponse,
    CompletedStatusResponse,
    PartialStatusResponse,
    PendingReviewResponse,
    ResumeJobRequest,
)
from app.services.job_match import job_match_service


router = APIRouter()


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
    try:
        return job_match_service.analyze(
            job_description=request.job_description,
            resume_skills=request.resume_skills or None,
            thread_id=request.thread_id,
            visa_preference=request.visa_preference,
            target_role=request.target_role,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post(
    "/resume",
    response_model=CompletedJobResponse,
    summary="提交人工审核并恢复流程",
)
def resume_job(request: ResumeJobRequest):
    try:
        return job_match_service.resume(
            thread_id=request.thread_id,
            approved=request.approved,
            feedback=request.feedback,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail="没有找到该 thread_id 对应的暂停工作流。",
        ) from error


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
    try:
        return job_match_service.get_status(thread_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="未知的 thread_id。") from error
