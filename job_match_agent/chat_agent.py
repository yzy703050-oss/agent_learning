import json
import os

import langsmith as ls
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers

from .config import load_env_file
from .llm import create_llm
from .memory import JobMatchMemory
from .prompts import build_career_chat_system_message
from .tracing import build_trace_config, get_langsmith_project_name
from .workflow import analyze_job_with_langgraph, resume_job_match_after_human_review


def create_job_match_workflow_tool(
    memory: JobMatchMemory,
    checkpointer,
    thread_id: str,
    tracing: bool = False,
    trace_tokens: bool = False,
):
    @tool
    def analyze_job_posting(
        job_description: str,
        continue_high_risk_analysis: bool = True,
    ) -> str:
        """Analyze a job posting with the Job Match LangGraph workflow."""
        result = analyze_job_with_langgraph(
            job_description,
            resume_skills=None,
            memory=memory,
            checkpointer=checkpointer,
            thread_id=thread_id,
            stream=False,
            tracing=tracing,
            trace_tokens=trace_tokens,
        )

        if result["interrupted"]:
            if not continue_high_risk_analysis:
                result = resume_job_match_after_human_review(
                    checkpointer=checkpointer,
                    thread_id=result["thread_id"],
                    approved=False,
                    feedback="The chat agent chose not to continue high-risk analysis.",
                    memory=memory,
                    tracing=tracing,
                )
            else:
                result = resume_job_match_after_human_review(
                    checkpointer=checkpointer,
                    thread_id=result["thread_id"],
                    approved=True,
                    feedback="Continue, but make the visa risk clear.",
                    memory=memory,
                    tracing=tracing,
                )

        return json.dumps(
            {
                "job_title": result["job_info"].job_title,
                "fit_score": result["match_result"].fit_score,
                "matched_skills": result["match_result"].matched_skills,
                "missing_skills": result["match_result"].missing_skills,
                "authorization_risk": result["match_result"].authorization_risk,
                "entry_level_fit": result["match_result"].entry_level_fit,
                "visa_warning": result["visa_warning"],
                "final_answer": result["final_answer"],
                "next_action": result["next_action"],
                "next_action_reason": result["next_action_reason"],
                "recommended_steps": result["recommended_steps"],
            },
            ensure_ascii=False,
        )

    return analyze_job_posting


def create_career_chat_agent(
    memory: JobMatchMemory,
    checkpointer,
    thread_id: str,
    streaming: bool = False,
    tracing: bool = False,
    trace_tokens: bool = False,
):
    llm = create_llm(streaming=streaming)
    job_match_tool = create_job_match_workflow_tool(
        memory=memory,
        checkpointer=checkpointer,
        thread_id=thread_id,
        tracing=tracing,
        trace_tokens=trace_tokens,
    )
    return create_agent(
        model=llm,
        tools=[job_match_tool],
        system_prompt=build_career_chat_system_message(),
    )


def run_career_chat_agent(
    user_message: str,
    memory: JobMatchMemory,
    checkpointer,
    thread_id: str = "career-chat-demo-user-1",
    tracing: bool = False,
    trace_tokens: bool = False,
) -> dict:
    load_env_file()

    if tracing and not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("Please set the LANGSMITH_API_KEY environment variable first.")

    project_name = get_langsmith_project_name()
    agent = create_career_chat_agent(
        memory=memory,
        checkpointer=checkpointer,
        thread_id=thread_id,
        streaming=trace_tokens,
        tracing=tracing,
        trace_tokens=trace_tokens,
    )
    trace_config = build_trace_config(tracing, run_name="career-chat-agent")

    try:
        with ls.tracing_context(project_name=project_name, enabled=tracing):
            return agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_message),
                    ]
                },
                config=trace_config,
            )
    finally:
        if tracing:
            wait_for_all_tracers()
