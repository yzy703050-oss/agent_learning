from typing import Literal

from pydantic import BaseModel, Field


class JobInfo(BaseModel):
    job_title: str = Field(
        description="The explicit job title. Use 'Unknown' if not stated."
    )
    required_skills: list[str] = Field(
        description="Only explicit required or preferred skills mentioned in the job description."
    )
    experience_level: str = Field(
        description="The explicit experience requirement. Use 'Unknown' if not stated."
    )
    authorization_risk: Literal["low", "medium", "high", "unknown"] = Field(
        description=(
            "Visa, sponsorship, or work authorization risk. "
            "Use high if the job says no sponsorship, no visa sponsorship, "
            "US citizen required, green card required, or permanent resident required. "
            "Use medium if the job says must be authorized to work in the US, "
            "but does not clearly say no sponsorship. "
            "Use low if OPT, CPT, sponsorship, visa support, or work visa support is clearly allowed. "
            "Use unknown if work authorization is not mentioned."
        )
    )
    entry_level_fit: Literal["Yes", "No", "Maybe"] = Field(
        description=(
            "Whether this role is suitable for an entry-level candidate. "
            "Use Yes for 0-2 years, junior, new grad, or entry-level roles. "
            "Use No for senior roles or 5+ years of experience. "
            "Use Maybe if unclear."
        )
    )


class JobMatchResult(BaseModel):
    job_title: str = Field(description="The job title from JobInfo.")
    fit_score: int = Field(description="The 0 to 100 score returned by calculate_fit_score.")
    matched_skills: list[str] = Field(description="Skills found in both the job and resume.")
    missing_skills: list[str] = Field(description="Required job skills missing from the resume.")
    experience_level: str = Field(description="The experience level from JobInfo.")
    authorization_risk: Literal["low", "medium", "high", "unknown"] = Field(
        description="The authorization risk from JobInfo."
    )
    entry_level_fit: Literal["Yes", "No", "Maybe"] = Field(
        description="The entry-level fit from JobInfo."
    )


class JobMatchResponse(BaseModel):
    match_result: JobMatchResult = Field(
        description="Structured job matching data for programs, UI, and storage."
    )
    final_answer: str = Field(description="A concise Chinese summary for the user.")


def ensure_job_info(value: JobInfo | dict) -> JobInfo:
    if isinstance(value, JobInfo):
        return value

    return JobInfo.model_validate(value)


def ensure_job_match_result(value: JobMatchResult | dict) -> JobMatchResult:
    if isinstance(value, JobMatchResult):
        return value

    return JobMatchResult.model_validate(value)


def ensure_job_match_response(value: JobMatchResponse | dict) -> JobMatchResponse:
    if isinstance(value, JobMatchResponse):
        return value

    return JobMatchResponse.model_validate(value)
