"""岗位匹配 API 的请求与响应契约。"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from job_match_agent.models import CandidateProfile, JobInfo, JobMatchResult


class AnalyzeJobRequest(BaseModel):
    job_description: str = Field(min_length=1, description="岗位 JD 原文。")
    thread_id: str | None = Field(default=None, description="LangGraph 工作流线程 ID。")
    candidate_profile: CandidateProfile | None = Field(
        default=None,
        description="结构化候选人资料；新调用推荐使用此字段。",
    )
    resume_skills: list[str] = Field(
        default_factory=list,
        description="兼容旧调用的简历技能列表；candidate_profile 优先。",
    )
    resume_text: str = Field(
        default="",
        description="兼容简化调用的简历文本，用于检索技能证据。",
    )
    target_role: str | None = Field(default=None, description="候选人的目标岗位。")


class ResumeJobRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool = Field(description="是否在明确硬性条件差距下继续分析。")
    feedback: str = Field(default="", description="人工审核意见。")


class PendingReviewResponse(BaseModel):
    status: Literal["pending_human_review"]
    thread_id: str
    review: list[dict[str, Any]]


class CompletedJobResponse(BaseModel):
    status: Literal["completed"]
    thread_id: str
    job_info: JobInfo
    candidate_profile: CandidateProfile
    retrieved_context: str
    hard_constraint_warning: str
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
