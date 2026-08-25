from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate


JOB_INFO_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You extract structured information from job descriptions. "
                "Extract only facts explicitly stated or strongly implied. "
                "Use 'Unknown' when job title or experience level is not stated. "
                "For authorization_risk, use high if the job description says no sponsorship, "
                "no visa sponsorship, US citizen required, green card required, "
                "or permanent resident required. "
                "Use medium if the job says must be authorized to work in the US, "
                "but does not clearly say no sponsorship. "
                "Use low if OPT, CPT, sponsorship, visa support, or work visa support "
                "is clearly allowed. "
                "Use unknown if work authorization is not mentioned. "
                "For entry_level_fit, use Yes for 0-2 years, junior, new grad, "
                "or entry-level roles. "
                "Use No for senior or 5+ years roles. "
                "Use Maybe if unclear. "
                "Return the structured response using the required schema. "
                "Do not add commentary."
            ),
        ),
        (
            "user",
            "Extract structured job information from this job description:\n{job_description}",
        ),
    ]
)


JOB_MATCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            """
Analyze this structured job information against my resume skills.

Job Info:
{job_info}

Resume Skills:
{resume_skills}

Memory Context:
{memory_context}

Retrieved Context:
{retrieved_context}

Requirements:
1. Use the calculate_fit_score tool with Job Info.required_skills and Resume Skills.
2. Give a concise Chinese summary with job title, matched skills, missing skills, and fit score from 0 to 100.
3. Mention experience level.
4. Mention authorization risk.
5. Mention whether the role is entry-level friendly.
6. Use Memory Context only for stable user preferences or prior analysis background.
7. Use Retrieved Context as external career advice, but do not override the structured Job Info.
""",
        )
    ]
)


JOB_MATCH_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a job matching assistant. "
                "Create a structured JobMatchResponse. "
                "Put factual matching fields inside match_result. "
                "Use the provided fit_score, matched_skills, and missing_skills exactly. "
                "Use the provided Job Info fields exactly. "
                "Put the concise Chinese user-facing summary only in final_answer."
            ),
        ),
        (
            "user",
            """
Job Info:
{job_info}

Resume Skills:
{resume_skills}

Fit Score:
{fit_score}

Matched Skills:
{matched_skills}

Missing Skills:
{missing_skills}

Memory Context:
{memory_context}

Retrieved Context:
{retrieved_context}

Visa Warning:
{visa_warning}

Requirements:
1. Mention job title, matched skills, missing skills, and fit score.
2. Mention experience level.
3. Mention authorization risk.
4. Mention whether the role is entry-level friendly.
5. If Visa Warning is not empty, include it clearly in final_answer.
6. Use Memory Context only for stable user preferences or prior analysis background.
7. Use Retrieved Context as external career advice, but do not override Job Info or Fit Score.
""",
        ),
    ]
)


def build_job_match_system_message() -> SystemMessage:
    return SystemMessage(
        content=(
            "You are a job matching assistant. "
            "You will receive structured job information and resume skills. "
            "Call calculate_fit_score using the required_skills from the job information. "
            "After the tool returns, create a structured JobMatchResponse response. "
            "Put factual matching fields inside match_result. "
            "Use the tool result for fit_score, matched_skills, and missing_skills. "
            "Use the job information for job_title, experience_level, authorization_risk, "
            "and entry_level_fit. "
            "Put the concise Chinese user-facing summary only in final_answer."
        )
    )


def build_career_chat_system_message() -> SystemMessage:
    return SystemMessage(
        content=(
            "You are a career assistant chat agent. "
            "You are the conversational entry point, not the whole job matching system. "
            "When the user asks you to analyze, evaluate, match, or review a job posting, "
            "call the analyze_job_posting tool. "
            "The tool runs the inner Job Match LangGraph workflow, including RAG, scoring, "
            "authorization-risk review, and final structured summary. "
            "After the tool returns, explain the result to the user in concise Chinese. "
            "If the user is only asking a general career question without a job description, "
            "answer directly and do not call the tool."
        )
    )
