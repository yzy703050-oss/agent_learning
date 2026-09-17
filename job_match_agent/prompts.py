from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate


JOB_INFO_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You extract structured facts from job descriptions. "
                "Separate required skills from preferred or nice-to-have skills. "
                "Create hard_constraints only for explicit non-skill must-have conditions. "
                "Represent each condition with a candidate field, operator, expected value, "
                "and the original concise description. Do not invent constraints. "
                "Work authorization is only one possible constraint and must not receive "
                "special treatment. Use 'Unknown' for unstated title or experience. "
                "For entry_level_fit use Yes for 0-2 years, junior, new grad, or entry-level; "
                "No for senior or 5+ years; otherwise Maybe. Return only the schema."
            ),
        ),
        ("user", "Extract structured job information from:\n{job_description}"),
    ]
)


JOB_MATCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            """
Analyze Job Info against Candidate Skills.

Job Info:
{job_info}

Candidate Skills:
{resume_skills}

Memory Context:
{memory_context}

Retrieved Context:
{retrieved_context}

Call calculate_weighted_match. Required skills weigh twice as much as preferred skills.
Return a concise Chinese summary and the required structured response.
""",
        )
    ]
)


JOB_MATCH_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a job matching assistant. Return a structured JobMatchResponse. "
                "Copy every factual field in Match Data exactly into match_result. "
                "Write only a concise Chinese explanation in final_answer. "
                "Explain required versus preferred skill gaps and hard-condition status. "
                "Do not let general career context override deterministic match data."
            ),
        ),
        (
            "user",
            """
Job Info:
{job_info}

Candidate Profile:
{candidate_profile}

Match Data:
{match_data}

Hard Constraint Warning:
{hard_constraint_warning}

Memory Context:
{memory_context}

Retrieved Context:
{retrieved_context}
""",
        ),
    ]
)


def build_job_match_system_message() -> SystemMessage:
    return SystemMessage(
        content=(
            "You are a job matching assistant. Call calculate_weighted_match, then return "
            "a structured JobMatchResponse with a concise Chinese final_answer."
        )
    )


def build_career_chat_system_message() -> SystemMessage:
    return SystemMessage(
        content=(
            "You are a career assistant. For job analysis call analyze_job_posting. "
            "The inner workflow handles weighted skills, resume evidence, generic hard "
            "constraints, and recommendations. Answer general career questions directly."
        )
    )
