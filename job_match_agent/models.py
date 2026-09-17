from typing import Any, Literal

from pydantic import BaseModel, Field


class HardConstraint(BaseModel):
    """A job condition that must be checked independently from skill matching."""

    field: Literal[
        "years_experience",
        "education_level",
        "location",
        "work_authorization",
        "certifications",
    ]
    operator: Literal["gte", "equals", "in", "contains"]
    value: str | float | list[str]
    description: str


class JobInfo(BaseModel):
    job_title: str = Field(
        description="The explicit job title. Use 'Unknown' if not stated."
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description="Skills explicitly described as required or must-have.",
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="Skills explicitly described as preferred or nice-to-have.",
    )
    experience_level: str = Field(
        description="The explicit experience requirement. Use 'Unknown' if not stated."
    )
    hard_constraints: list[HardConstraint] = Field(
        default_factory=list,
        description=(
            "Explicit non-skill must-have conditions such as minimum years, location, "
            "education, work authorization, or certification requirements."
        ),
    )
    entry_level_fit: Literal["Yes", "No", "Maybe"] = Field(
        description=(
            "Whether this role is suitable for an entry-level candidate. "
            "Use Yes for 0-2 years, junior, new grad, or entry-level roles; "
            "No for senior roles or 5+ years; otherwise Maybe."
        )
    )


class CandidateProfile(BaseModel):
    """Structured candidate facts used by deterministic matching tools."""

    skills: list[str] = Field(default_factory=list)
    resume_text: str = ""
    years_experience: float | None = Field(default=None, ge=0)
    education_level: str | None = None
    location: str | None = None
    work_authorization: str | None = None
    certifications: list[str] = Field(default_factory=list)


class ConstraintCheck(BaseModel):
    constraint: HardConstraint
    status: Literal["satisfied", "unsatisfied", "unknown"]
    candidate_value: Any = None
    reason: str


class ResumeEvidence(BaseModel):
    skill: str
    excerpts: list[str] = Field(default_factory=list)


class JobMatchResult(BaseModel):
    job_title: str
    fit_score: int = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    matched_required_skills: list[str] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    matched_preferred_skills: list[str] = Field(default_factory=list)
    missing_preferred_skills: list[str] = Field(default_factory=list)
    experience_level: str
    entry_level_fit: Literal["Yes", "No", "Maybe"]
    constraint_checks: list[ConstraintCheck] = Field(default_factory=list)
    resume_evidence: list[ResumeEvidence] = Field(default_factory=list)


class JobMatchResponse(BaseModel):
    match_result: JobMatchResult
    final_answer: str


def ensure_job_info(value: JobInfo | dict) -> JobInfo:
    return value if isinstance(value, JobInfo) else JobInfo.model_validate(value)


def ensure_candidate_profile(value: CandidateProfile | dict) -> CandidateProfile:
    if isinstance(value, CandidateProfile):
        return value
    return CandidateProfile.model_validate(value)


def ensure_job_match_result(value: JobMatchResult | dict) -> JobMatchResult:
    if isinstance(value, JobMatchResult):
        return value
    return JobMatchResult.model_validate(value)


def ensure_job_match_response(value: JobMatchResponse | dict) -> JobMatchResponse:
    if isinstance(value, JobMatchResponse):
        return value
    return JobMatchResponse.model_validate(value)
