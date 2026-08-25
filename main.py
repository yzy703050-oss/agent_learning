from langgraph.checkpoint.memory import MemorySaver

from job_match_agent.chat_agent import run_career_chat_agent
from job_match_agent.display import print_message_trace
from job_match_agent.memory import JobMatchMemory
from job_match_agent.workflow import (
    analyze_job_with_langgraph,
    resume_job_match_after_human_review,
)


RUN_CHAT_AGENT_DEMO = True
STREAM_OUTPUT = True
TRACE_OUTPUT = False
TRACE_TOKENS = False


def build_demo_memory() -> JobMatchMemory:
    resume_skills = [
        "Python",
        "SQL",
        "Tableau",
        "Power BI",
        "A/B testing",
        "Excel",
        "Pandas",
    ]

    memory = JobMatchMemory()
    memory.update_user_profile(
        resume_skills=resume_skills,
        visa_preference="I need OPT, CPT, H-1B sponsorship, or visa-friendly roles.",
        target_role="Entry-level data analyst",
    )
    return memory


def run_chat_demo(jd: str, memory: JobMatchMemory, checkpointer) -> None:
    chat_result = run_career_chat_agent(
        user_message=(
            "请帮我分析这个岗位是否适合我，并特别注意签证风险：\n"
            f"{jd}"
        ),
        memory=memory,
        checkpointer=checkpointer,
        thread_id="job-match-demo-user-1",
        tracing=TRACE_OUTPUT,
        trace_tokens=TRACE_TOKENS,
    )

    print("Career Chat Agent Message Trace:")
    print_message_trace(chat_result)
    print("\nMemory Context After Chat Run:")
    print(memory.build_context())


def run_inner_graph_demo(jd: str, memory: JobMatchMemory, checkpointer) -> None:
    result = analyze_job_with_langgraph(
        jd,
        resume_skills=None,
        memory=memory,
        checkpointer=checkpointer,
        thread_id="job-match-demo-user-1",
        stream=STREAM_OUTPUT,
        tracing=TRACE_OUTPUT,
        trace_tokens=TRACE_TOKENS,
    )

    if result["interrupted"]:
        print("\nLangGraph interrupted for human review:")
        for item in result["interrupts"]:
            print(item.value)

        result = resume_job_match_after_human_review(
            checkpointer=checkpointer,
            thread_id=result["thread_id"],
            approved=True,
            feedback="Continue, but make the visa risk clear.",
            memory=memory,
            tracing=TRACE_OUTPUT,
        )

    print("Structured Job Info:")
    print(result["job_info"].model_dump_json(indent=2))
    print("\nMemory Context Used:")
    print(result["memory_context"])
    print("\nRetrieved Context:")
    print(result["retrieved_context"])
    print("\nVisa Warning:")
    print(result["visa_warning"] or "No visa warning.")
    print("\nStructured Match Result:")
    print(result["match_result"].model_dump_json(indent=2))
    print("\nFinal Answer:")
    print(result["final_answer"])
    print("\nNext Action:")
    print(result["next_action"])
    print("\nNext Action Reason:")
    print(result["next_action_reason"])
    print("\nRecommended Steps:")
    for step in result["recommended_steps"]:
        print(f"- {step}")
    print("\nGraph State Keys:")
    print(sorted(result["graph_state"].keys()))
    print("\nCheckpoint State Keys:")
    print(sorted(result["checkpoint_state"].values.keys()))
    print("\nMemory Context After This Run:")
    print(memory.build_context())


def main() -> None:
    jd = """
    We are looking for a Data Analyst with experience in Python, SQL, Tableau,
    and A/B testing. 0-2 years of experience preferred. No sponsorship is available.
    """

    memory = build_demo_memory()
    checkpointer = MemorySaver()

    if RUN_CHAT_AGENT_DEMO:
        run_chat_demo(jd, memory, checkpointer)
    else:
        run_inner_graph_demo(jd, memory, checkpointer)


if __name__ == "__main__":
    main()
