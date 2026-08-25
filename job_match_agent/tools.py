from langchain.tools import tool


@tool
def calculate_fit_score(required_skills: list[str], resume_skills: list[str]) -> dict:
    """Calculate how well resume skills match the job's required skills."""
    required_set = {skill.lower().strip() for skill in required_skills}
    resume_set = {skill.lower().strip() for skill in resume_skills}

    matched = required_set & resume_set
    missing = required_set - resume_set

    if len(required_set) == 0:
        score = 0
    else:
        score = round(len(matched) / len(required_set) * 100)

    return {
        "fit_score": score,
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
    }

