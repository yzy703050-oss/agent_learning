import re
from typing import Any

from langchain.tools import tool

from .models import CandidateProfile, ConstraintCheck, HardConstraint, ResumeEvidence


SKILL_ALIASES = {
    "amazon web services": "aws",
    "aws cloud": "aws",
    "google cloud platform": "gcp",
    "microsoft azure": "azure",
    "postgre sql": "postgresql",
    "postgres": "postgresql",
    "powerbi": "power bi",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "py torch": "pytorch",
    "node js": "node.js",
    "react js": "react",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "machine learning": "machine learning",
    "ml": "machine learning",
    "natural language processing": "nlp",
}


def _normalize_skill_name(skill: str) -> str:
    value = skill.casefold().strip()
    value = re.sub(r"[_/]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" ,;:.")
    return SKILL_ALIASES.get(value, value)


def _normalize_skill_list(skills: list[str]) -> list[str]:
    return sorted({_normalize_skill_name(skill) for skill in skills if skill.strip()})


@tool
def normalize_skills(skills: list[str]) -> dict:
    """Normalize skill names, common aliases, casing, whitespace, and duplicates."""
    return {"normalized_skills": _normalize_skill_list(skills)}


@tool
def calculate_weighted_match(
    required_skills: list[str],
    preferred_skills: list[str],
    candidate_skills: list[str],
    required_weight: float = 2.0,
    preferred_weight: float = 1.0,
) -> dict:
    """Calculate a deterministic weighted match for required and preferred skills."""
    if required_weight <= 0 or preferred_weight < 0:
        raise ValueError("required_weight must be > 0 and preferred_weight must be >= 0")

    required = set(_normalize_skill_list(required_skills))
    preferred = set(_normalize_skill_list(preferred_skills)) - required
    candidate = set(_normalize_skill_list(candidate_skills))

    matched_required = required & candidate
    missing_required = required - candidate
    matched_preferred = preferred & candidate
    missing_preferred = preferred - candidate
    denominator = len(required) * required_weight + len(preferred) * preferred_weight
    numerator = (
        len(matched_required) * required_weight
        + len(matched_preferred) * preferred_weight
    )
    score = round(numerator / denominator * 100) if denominator else 0

    return {
        "fit_score": score,
        "matched_skills": sorted(matched_required | matched_preferred),
        "missing_skills": sorted(missing_required | missing_preferred),
        "matched_required_skills": sorted(matched_required),
        "missing_required_skills": sorted(missing_required),
        "matched_preferred_skills": sorted(matched_preferred),
        "missing_preferred_skills": sorted(missing_preferred),
        "weights": {"required": required_weight, "preferred": preferred_weight},
    }


def _casefold_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.casefold().strip()
    if isinstance(value, list):
        return [_casefold_value(item) for item in value]
    return value


def _evaluate_constraint(constraint: HardConstraint, candidate_value: Any) -> bool:
    expected = constraint.value
    if constraint.operator == "gte":
        return float(candidate_value) >= float(expected)
    if constraint.operator == "equals":
        return _casefold_value(candidate_value) == _casefold_value(expected)
    if constraint.operator == "in":
        expected_values = expected if isinstance(expected, list) else [expected]
        return _casefold_value(candidate_value) in _casefold_value(expected_values)
    if constraint.operator == "contains":
        actual_values = candidate_value if isinstance(candidate_value, list) else [candidate_value]
        expected_values = expected if isinstance(expected, list) else [expected]
        actual = set(_casefold_value(actual_values))
        return set(_casefold_value(expected_values)).issubset(actual)
    return False


@tool
def detect_hard_constraints(
    hard_constraints: list[dict], candidate_profile: dict
) -> dict:
    """Check explicit job must-haves against known candidate facts without guessing."""
    profile = CandidateProfile.model_validate(candidate_profile)
    checks: list[ConstraintCheck] = []
    for raw_constraint in hard_constraints:
        constraint = HardConstraint.model_validate(raw_constraint)
        candidate_value = getattr(profile, constraint.field)
        if candidate_value is None or candidate_value == [] or candidate_value == "":
            status = "unknown"
            reason = f"Candidate value for {constraint.field} was not provided."
        else:
            satisfied = _evaluate_constraint(constraint, candidate_value)
            status = "satisfied" if satisfied else "unsatisfied"
            reason = (
                "Candidate value satisfies the explicit condition."
                if satisfied
                else "Candidate value does not satisfy the explicit condition."
            )
        checks.append(
            ConstraintCheck(
                constraint=constraint,
                status=status,
                candidate_value=candidate_value,
                reason=reason,
            )
        )

    serialized = [check.model_dump() for check in checks]
    return {
        "checks": serialized,
        "has_blocking_violation": any(
            check.status == "unsatisfied" for check in checks
        ),
        "has_unknowns": any(check.status == "unknown" for check in checks),
    }


@tool
def retrieve_resume_evidence(
    target_skills: list[str], resume_text: str, max_excerpts_per_skill: int = 2
) -> dict:
    """Find short resume excerpts that explicitly mention each target skill."""
    if max_excerpts_per_skill < 1:
        raise ValueError("max_excerpts_per_skill must be at least 1")
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", resume_text)
        if segment.strip()
    ]
    evidence = []
    for skill in _normalize_skill_list(target_skills):
        aliases = {skill}
        aliases.update(alias for alias, canonical in SKILL_ALIASES.items() if canonical == skill)
        excerpts = [
            segment
            for segment in segments
            if any(alias in _normalize_skill_name(segment) for alias in aliases)
        ][:max_excerpts_per_skill]
        evidence.append(ResumeEvidence(skill=skill, excerpts=excerpts).model_dump())
    return {"evidence": evidence}
