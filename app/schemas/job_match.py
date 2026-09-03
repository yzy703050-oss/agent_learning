"""岗位匹配 API 的请求与响应数据契约。"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from job_match_agent.models import JobInfo, JobMatchResult


class AnalyzeJobRequest(BaseModel):
    """启动一次岗位匹配分析所需的数据。"""

    job_description: str = Field(
        min_length=1,
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
            "工作流线程 ID。首次请求可省略，由后端生成；如需恢复流程，"
            "客户端必须保存响应中的 thread_id。"
        ),
        examples=["job-match-demo-user-1"],
    )
    resume_skills: list[str] = Field(
        default_factory=list,
        description="候选人的简历技能列表。",
        examples=[["Python", "SQL", "Tableau", "Excel"]],
    )
    visa_preference: str | None = Field(
        default=None,
        description="用户对签证、工卡或 sponsorship 的偏好说明。",
        examples=["I need OPT, CPT, H-1B sponsorship, or visa-friendly roles."],
    )
    target_role: str | None = Field(
        default=None,
        description="用户的目标岗位。",
        examples=["Entry-level data analyst"],
    )


class ResumeJobRequest(BaseModel):
    """人工审核后恢复工作流所需的数据。"""

    thread_id: str = Field(
        min_length=1,
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


class PendingReviewResponse(BaseModel):
    status: Literal["pending_human_review"]
    thread_id: str
    review: list[dict[str, Any]]


class CompletedJobResponse(BaseModel):
    status: Literal["completed"]
    thread_id: str
    job_info: JobInfo
    retrieved_context: str
    visa_warning: str
    match_result: JobMatchResult
    final_answer: str
    next_action: str
    next_action_reason: str
    recommended_steps: list[str]


class CompletedStatusResponse(BaseModel):
    status: Literal["completed"]
    thread_id: str
    next_action: str | None = None
    next_action_reason: str | None = None
    recommended_steps: list[str] = Field(default_factory=list)


class PartialStatusResponse(BaseModel):
    status: Literal["running_or_partial"]
    thread_id: str
    state_keys: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
