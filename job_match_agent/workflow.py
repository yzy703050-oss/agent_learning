import json
from typing import TypedDict

import langsmith as ls
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.core.config import settings

from .chains import (
    build_job_match_variables,
    create_job_match_chain,
    create_job_match_summary_chain,
    extract_job_info_langchain,
    parse_job_match_response,
    stream_job_match_agent,
)
from .memory import JobMatchMemory
from .models import (
    JobInfo,
    JobMatchResult,
    ensure_job_info,
    ensure_job_match_response,
    ensure_job_match_result,
)
from .rag import create_job_context_retriever_chain, retrieve_job_context
from .tools import calculate_fit_score
from .tracing import build_trace_config, get_langsmith_project_name


class JobMatchGraphState(TypedDict, total=False):
    jd: str
    resume_skills: list[str]
    memory_context: str
    tracing: bool
    trace_tokens: bool
    job_info: JobInfo | dict
    retrieved_context: str
    fit_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    visa_warning: str
    human_approved: bool
    human_feedback: str
    interrupted: bool
    match_result: JobMatchResult | dict
    final_answer: str
    next_action: str
    next_action_reason: str
    recommended_steps: list[str]


def extract_job_info_node(state: JobMatchGraphState) -> dict:
    job_info = extract_job_info_langchain(
        state["jd"],
        tracing=state.get("tracing", False),
        trace_tokens=state.get("trace_tokens", False),
    )
    return {"job_info": job_info.model_dump()}


def retrieve_context_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    retrieved_context = retrieve_job_context(
        job_info,
        state["resume_skills"],
    )
    return {"retrieved_context": retrieved_context}


def calculate_score_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    score_result = calculate_fit_score.invoke(
        {
            "required_skills": job_info.required_skills,
            "resume_skills": state["resume_skills"],
        }
    )
    return {
        "fit_score": score_result["fit_score"],
        "matched_skills": score_result["matched_skills"],
        "missing_skills": score_result["missing_skills"],
    }


def route_authorization_risk(state: JobMatchGraphState) -> str:
    job_info = ensure_job_info(state["job_info"])
    if job_info.authorization_risk == "high":
        return "visa_warning"

    return "recommend_next_action"


def visa_warning_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    warning = (
        "This job has high authorization risk because the job description indicates "
        "no sponsorship, citizenship, green card, or permanent-resident-style constraints. "
        "For a candidate who needs OPT, CPT, H-1B sponsorship, or visa support, "
        "this should be treated as an important application risk."
    )
    human_decision = interrupt(
        {
            "reason": "high_authorization_risk",
            "question": "This job is high authorization risk. Continue generating the final summary?",
            "job_title": job_info.job_title,
            "fit_score": state["fit_score"],
            "visa_warning": warning,
            "expected_resume_value": {
                "approved": True,
                "feedback": "Continue, but make the visa risk clear.",
            },
        }
    )

    if isinstance(human_decision, dict):
        approved = bool(human_decision.get("approved", False))
        feedback = str(human_decision.get("feedback", ""))
    else:
        approved = bool(human_decision)
        feedback = ""

    return {
        "visa_warning": warning,
        "human_approved": approved,
        "human_feedback": feedback,
    }


def route_after_human_review(state: JobMatchGraphState) -> str:
    if state.get("human_approved", False):
        return "recommend_next_action"

    return "recommend_next_action"


def skip_application_summary_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    final_answer = (
        "\u8be5\u5c97\u4f4d\u7b7e\u8bc1/\u5de5\u5361\u98ce\u9669\u8f83\u9ad8\uff0c"
        "\u5e76\u4e14\u4eba\u5de5\u5ba1\u6838\u9009\u62e9\u4e0d\u7ee7\u7eed\u751f\u6210\u5b8c\u6574\u7533\u8bf7\u603b\u7ed3\u3002"
        f"\u5f53\u524d\u5339\u914d\u5206\u6570\u4e3a {state['fit_score']}\uff0c"
        "\u5efa\u8bae\u4f18\u5148\u8003\u8651\u7b7e\u8bc1\u66f4\u53cb\u597d\u7684\u5c97\u4f4d\u3002"
    )
    match_result = JobMatchResult(
        job_title=job_info.job_title,
        fit_score=state["fit_score"],
        matched_skills=state["matched_skills"],
        missing_skills=state["missing_skills"],
        experience_level=job_info.experience_level,
        authorization_risk=job_info.authorization_risk,
        entry_level_fit=job_info.entry_level_fit,
    )
    return {
        "match_result": match_result.model_dump(),
        "final_answer": final_answer,
    }


def generate_summary_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    chain = create_job_match_summary_chain(streaming=state.get("trace_tokens", False))
    trace_config = build_trace_config(
        state.get("tracing", False),
        run_name="langgraph-generate-summary",
    )
    response = chain.invoke(
        {
            "job_info": job_info.model_dump_json(indent=2),
            "resume_skills": json.dumps(state["resume_skills"], ensure_ascii=False),
            "fit_score": state["fit_score"],
            "matched_skills": json.dumps(
                state["matched_skills"],
                ensure_ascii=False,
            ),
            "missing_skills": json.dumps(
                state["missing_skills"],
                ensure_ascii=False,
            ),
            "memory_context": state.get(
                "memory_context",
                "No memory context provided.",
            ),
            "retrieved_context": state.get(
                "retrieved_context",
                "No retrieved context provided.",
            ),
            "visa_warning": state.get("visa_warning", ""),
        },
        config=trace_config,
    )
    response = ensure_job_match_response(response)
    match_result = response.match_result

    return {
        "match_result": match_result.model_dump(),
        "final_answer": response.final_answer,
    }


def recommend_next_action_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    fit_score = state["fit_score"]
    missing_skills = state["missing_skills"]

    if state.get("human_approved") is False:
        next_action = "skip"
        reason = "人工审核选择不继续分析该高签证风险岗位。"
        steps = [
            "优先寻找明确支持 OPT、CPT、H-1B 或 visa sponsorship 的岗位。",
            "保留该岗位链接，但暂时不要投入太多申请时间。",
        ]
    elif job_info.authorization_risk == "high":
        next_action = "cautious_apply"
        reason = "岗位存在高签证风险，即使技能匹配也需要先确认工卡或 sponsorship 条件。"
        steps = [
            "投递前先确认招聘方是否接受 OPT/CPT/H-1B 或后续 sponsorship。",
            "如果无法确认签证支持，把该岗位放到低优先级。",
            "简历中突出已匹配技能，减少技能匹配之外的风险。",
        ]
    elif fit_score >= 75 and job_info.entry_level_fit == "Yes":
        next_action = "apply"
        reason = "技能匹配度较高，且岗位对 entry-level 候选人友好。"
        steps = [
            "可以优先投递该岗位。",
            "在简历和 cover letter 中突出 matched skills。",
            "准备一个能证明核心技能的项目案例。",
        ]
    elif fit_score >= 50:
        next_action = "apply_after_tailoring"
        reason = "岗位有一定匹配度，但仍需要针对缺失技能做简历调整。"
        steps = [
            "先把已有项目经历改写得更贴近岗位要求。",
            "针对 missing skills 补充项目、课程或练习证明。",
            "投递前检查岗位是否真的适合 entry-level。",
        ]
    else:
        next_action = "build_skills_first"
        reason = "当前技能匹配度偏低，直接投递的性价比不高。"
        steps = [
            "先选择 1 到 2 个高频 missing skills 做小项目。",
            "补完项目后再重新评估类似岗位。",
            "优先找要求更贴近当前技能栈的 entry-level 岗位。",
        ]

    if missing_skills:
        steps.append(f"优先补齐或证明这些缺失技能：{', '.join(missing_skills[:3])}。")

    return {
        "next_action": next_action,
        "next_action_reason": reason,
        "recommended_steps": steps,
    }


def route_after_next_action(state: JobMatchGraphState) -> str:
    if state["next_action"] == "skip":
        return "skip_application_summary"

    return "generate_summary"


def create_job_match_graph(checkpointer=None):
    graph_builder = StateGraph(JobMatchGraphState)
    graph_builder.add_node("extract_job_info", extract_job_info_node)
    graph_builder.add_node("retrieve_context", retrieve_context_node)
    graph_builder.add_node("calculate_score", calculate_score_node)
    graph_builder.add_node("visa_warning", visa_warning_node)
    graph_builder.add_node("skip_application_summary", skip_application_summary_node)
    graph_builder.add_node("generate_summary", generate_summary_node)
    graph_builder.add_node("recommend_next_action", recommend_next_action_node)

    graph_builder.add_edge(START, "extract_job_info")
    graph_builder.add_edge("extract_job_info", "retrieve_context")
    graph_builder.add_edge("retrieve_context", "calculate_score")
    graph_builder.add_conditional_edges(
        "calculate_score",
        route_authorization_risk,
        {
            "visa_warning": "visa_warning",
            "recommend_next_action": "recommend_next_action",
        },
    )
    graph_builder.add_conditional_edges(
        "visa_warning",
        route_after_human_review,
        {
            "recommend_next_action": "recommend_next_action",
        },
    )
    graph_builder.add_conditional_edges(
        "recommend_next_action",
        route_after_next_action,
        {
            "generate_summary": "generate_summary",
            "skip_application_summary": "skip_application_summary",
        },
    )
    graph_builder.add_edge("skip_application_summary", END)
    graph_builder.add_edge("generate_summary", END)

    return graph_builder.compile(checkpointer=checkpointer)


def build_langgraph_thread_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def analyze_job_with_agent(
    jd: str,
    resume_skills: list[str] | None,
    memory: JobMatchMemory | None = None,
    visa_preference: str | None = None,
    target_role: str | None = None,
    stream: bool = False,
    tracing: bool = False,
    trace_tokens: bool = False,
) -> dict:
    if tracing and settings.LANGSMITH_API_KEY is None:
        raise RuntimeError("Please set the LANGSMITH_API_KEY environment variable first.")

    project_name = get_langsmith_project_name()
    if tracing:
        print(f"LangSmith tracing enabled. Project: {project_name}")

    if memory:
        memory.update_user_profile(
            resume_skills=resume_skills,
            visa_preference=visa_preference,
            target_role=target_role,
        )

    if resume_skills is None:
        if memory and memory.resume_skills:
            resume_skills = memory.resume_skills
        else:
            raise ValueError("resume_skills is required when memory has no saved skills.")

    memory_context = memory.build_context() if memory else "No memory context provided."

    try:
        with ls.tracing_context(project_name=project_name, enabled=tracing):
            job_info = extract_job_info_langchain(
                jd,
                tracing=tracing,
                trace_tokens=trace_tokens,
            )
            retriever_chain = create_job_context_retriever_chain()
            retriever_config = build_trace_config(tracing, run_name="job-context-retriever")
            retrieved_context = retriever_chain.invoke(
                {
                    "job_info": job_info,
                    "resume_skills": resume_skills,
                },
                config=retriever_config,
            )

            if stream:
                result = stream_job_match_agent(
                    job_info,
                    resume_skills,
                    retrieved_context=retrieved_context,
                    memory_context=memory_context,
                    tracing=tracing,
                    trace_tokens=trace_tokens,
                )
            else:
                chain = create_job_match_chain(streaming=trace_tokens)
                trace_config = build_trace_config(tracing, run_name="job-match-chain")
                result = chain.invoke(
                    build_job_match_variables(
                        job_info,
                        resume_skills,
                        retrieved_context,
                        memory_context,
                    ),
                    config=trace_config,
                )

            response = parse_job_match_response(result)
            match_result = response.match_result
            if memory:
                memory.remember_analysis(job_info, match_result)

            return {
                "job_info": job_info,
                "retrieved_context": retrieved_context,
                "memory_context": memory_context,
                "match_result": match_result,
                "final_answer": response.final_answer,
                "agent_result": result,
            }
    finally:
        if tracing:
            wait_for_all_tracers()


def analyze_job_with_langgraph(
    jd: str,
    resume_skills: list[str] | None,
    memory: JobMatchMemory | None = None,
    checkpointer=None,
    thread_id: str = "job-match-demo",
    visa_preference: str | None = None,
    target_role: str | None = None,
    stream: bool = False,
    tracing: bool = False,
    trace_tokens: bool = False,
) -> dict:
    if tracing and settings.LANGSMITH_API_KEY is None:
        raise RuntimeError("Please set the LANGSMITH_API_KEY environment variable first.")

    project_name = get_langsmith_project_name()
    if tracing:
        print(f"LangSmith tracing enabled. Project: {project_name}")

    if memory:
        memory.update_user_profile(
            resume_skills=resume_skills,
            visa_preference=visa_preference,
            target_role=target_role,
        )

    if resume_skills is None:
        if memory and memory.resume_skills:
            resume_skills = memory.resume_skills
        else:
            raise ValueError("resume_skills is required when memory has no saved skills.")

    memory_context = memory.build_context() if memory else "No memory context provided."
    graph = create_job_match_graph(checkpointer=checkpointer)
    graph_config = build_langgraph_thread_config(thread_id)
    graph_input: JobMatchGraphState = {
        "jd": jd,
        "resume_skills": resume_skills,
        "memory_context": memory_context,
        "tracing": tracing,
        "trace_tokens": trace_tokens,
    }

    try:
        with ls.tracing_context(project_name=project_name, enabled=tracing):
            if stream:
                final_state: JobMatchGraphState = graph_input.copy()
                print("\nStreaming LangGraph workflow:")
                for update in graph.stream(
                    graph_input,
                    config=graph_config,
                    stream_mode="updates",
                ):
                    print(json.dumps(update, ensure_ascii=False, default=str, indent=2))
                    for node_update in update.values():
                        final_state.update(node_update)
            else:
                final_state = graph.invoke(graph_input, config=graph_config)

            if "__interrupt__" in final_state:
                checkpoint_state = graph.get_state(graph_config) if checkpointer else None
                return {
                    "interrupted": True,
                    "interrupts": final_state["__interrupt__"],
                    "graph_state": final_state,
                    "checkpoint_state": checkpoint_state,
                    "thread_id": thread_id,
                }

            match_result = ensure_job_match_result(final_state["match_result"])
            job_info = ensure_job_info(final_state["job_info"])
            if memory:
                memory.remember_analysis(job_info, match_result)

            checkpoint_state = None
            if checkpointer:
                checkpoint_state = graph.get_state(graph_config)

            return {
                "job_info": job_info,
                "retrieved_context": final_state["retrieved_context"],
                "memory_context": memory_context,
                "visa_warning": final_state.get("visa_warning", ""),
                "match_result": match_result,
                "final_answer": final_state["final_answer"],
                "next_action": final_state["next_action"],
                "next_action_reason": final_state["next_action_reason"],
                "recommended_steps": final_state["recommended_steps"],
                "graph_state": final_state,
                "checkpoint_state": checkpoint_state,
                "interrupted": False,
                "thread_id": thread_id,
            }
    finally:
        if tracing:
            wait_for_all_tracers()


def resume_job_match_after_human_review(
    checkpointer,
    thread_id: str,
    approved: bool,
    feedback: str = "",
    memory: JobMatchMemory | None = None,
    tracing: bool = False,
) -> dict:
    if tracing and settings.LANGSMITH_API_KEY is None:
        raise RuntimeError("Please set the LANGSMITH_API_KEY environment variable first.")

    project_name = get_langsmith_project_name()
    graph = create_job_match_graph(checkpointer=checkpointer)
    graph_config = build_langgraph_thread_config(thread_id)
    resume_value = {
        "approved": approved,
        "feedback": feedback,
    }

    try:
        with ls.tracing_context(project_name=project_name, enabled=tracing):
            final_state = graph.invoke(
                Command(resume=resume_value),
                config=graph_config,
            )

            match_result = ensure_job_match_result(final_state["match_result"])
            job_info = ensure_job_info(final_state["job_info"])
            if memory:
                memory.remember_analysis(job_info, match_result)

            checkpoint_state = graph.get_state(graph_config)

            return {
                "interrupted": False,
                "job_info": job_info,
                "retrieved_context": final_state["retrieved_context"],
                "memory_context": final_state.get(
                    "memory_context",
                    "No memory context provided.",
                ),
                "visa_warning": final_state.get("visa_warning", ""),
                "human_approved": final_state.get("human_approved", False),
                "human_feedback": final_state.get("human_feedback", ""),
                "match_result": match_result,
                "final_answer": final_state["final_answer"],
                "next_action": final_state["next_action"],
                "next_action_reason": final_state["next_action_reason"],
                "recommended_steps": final_state["recommended_steps"],
                "graph_state": final_state,
                "checkpoint_state": checkpoint_state,
                "thread_id": thread_id,
            }
    finally:
        if tracing:
            wait_for_all_tracers()
