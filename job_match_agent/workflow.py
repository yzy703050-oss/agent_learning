import json
from typing import TypedDict

import langsmith as ls
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.core.config import settings

from .chains import (
    create_job_match_summary_chain,
    extract_job_info_langchain,
)
from .memory import JobMatchMemory
from .models import (
    CandidateProfile,
    JobInfo,
    JobMatchResult,
    ensure_candidate_profile,
    ensure_job_info,
    ensure_job_match_response,
    ensure_job_match_result,
)
from .rag import retrieve_job_context
from .tools import (
    calculate_weighted_match,
    detect_hard_constraints,
    normalize_skills,
    retrieve_resume_evidence,
)
from .tracing import build_trace_config, get_langsmith_project_name


class JobMatchGraphState(TypedDict, total=False):
    jd: str
    candidate_profile: CandidateProfile | dict
    memory_context: str
    tracing: bool
    trace_tokens: bool
    job_info: JobInfo | dict
    retrieved_context: str
    fit_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    matched_required_skills: list[str]
    missing_required_skills: list[str]
    matched_preferred_skills: list[str]
    missing_preferred_skills: list[str]
    constraint_checks: list[dict]
    has_blocking_violation: bool
    has_unknown_constraints: bool
    resume_evidence: list[dict]
    hard_constraint_warning: str
    human_approved: bool
    human_feedback: str
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


def normalize_candidate_node(state: JobMatchGraphState) -> dict:
    profile = ensure_candidate_profile(state["candidate_profile"])
    normalized = normalize_skills.invoke({"skills": profile.skills})
    return {
        "candidate_profile": profile.model_copy(
            update={"skills": normalized["normalized_skills"]}
        ).model_dump()
    }


def retrieve_context_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    profile = ensure_candidate_profile(state["candidate_profile"])
    return {"retrieved_context": retrieve_job_context(job_info, profile.skills)}


def calculate_match_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    profile = ensure_candidate_profile(state["candidate_profile"])
    return calculate_weighted_match.invoke(
        {
            "required_skills": job_info.required_skills,
            "preferred_skills": job_info.preferred_skills,
            "candidate_skills": profile.skills,
        }
    )


def detect_hard_constraints_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    profile = ensure_candidate_profile(state["candidate_profile"])
    result = detect_hard_constraints.invoke(
        {
            "hard_constraints": [
                constraint.model_dump() for constraint in job_info.hard_constraints
            ],
            "candidate_profile": profile.model_dump(),
        }
    )
    return {
        "constraint_checks": result["checks"],
        "has_blocking_violation": result["has_blocking_violation"],
        "has_unknown_constraints": result["has_unknowns"],
    }


def retrieve_resume_evidence_node(state: JobMatchGraphState) -> dict:
    profile = ensure_candidate_profile(state["candidate_profile"])
    result = retrieve_resume_evidence.invoke(
        {
            "target_skills": state["matched_skills"],
            "resume_text": profile.resume_text,
        }
    )
    return {"resume_evidence": result["evidence"]}


def route_hard_constraints(state: JobMatchGraphState) -> str:
    if state.get("has_blocking_violation", False):
        return "hard_constraint_review"
    return "recommend_next_action"


def hard_constraint_review_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    violations = [
        check for check in state.get("constraint_checks", [])
        if check["status"] == "unsatisfied"
    ]
    descriptions = [check["constraint"]["description"] for check in violations]
    warning = "发现明确不满足的岗位硬性条件：" + "；".join(descriptions)
    decision = interrupt(
        {
            "reason": "hard_constraint_violation",
            "question": "候选人明确不满足岗位硬性条件，是否仍继续生成申请建议？",
            "job_title": job_info.job_title,
            "fit_score": state["fit_score"],
            "hard_constraint_warning": warning,
            "violations": violations,
            "expected_resume_value": {
                "approved": True,
                "feedback": "继续分析，但明确提示硬性条件差距。",
            },
        }
    )
    if isinstance(decision, dict):
        approved = bool(decision.get("approved", False))
        feedback = str(decision.get("feedback", ""))
    else:
        approved = bool(decision)
        feedback = ""
    return {
        "hard_constraint_warning": warning,
        "human_approved": approved,
        "human_feedback": feedback,
    }


def recommend_next_action_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    fit_score = state["fit_score"]
    missing_required = state["missing_required_skills"]

    if state.get("human_approved") is False:
        next_action = "skip"
        reason = "候选人明确不满足岗位硬性条件，人工审核选择不继续。"
        steps = ["优先筛选硬性条件符合的岗位。"]
    elif state.get("has_blocking_violation", False):
        next_action = "cautious_apply"
        reason = "技能分数不能抵消明确的硬性条件差距，申请前需先确认例外空间。"
        steps = ["先向招聘方确认硬性条件是否可协商。", "在申请材料中不要隐瞒相关差距。"]
    elif fit_score >= 75 and job_info.entry_level_fit == "Yes":
        next_action = "apply"
        reason = "加权技能匹配度较高，且岗位对初级候选人友好。"
        steps = ["可以优先投递。", "用简历中的具体证据突出已匹配技能。"]
    elif fit_score >= 50:
        next_action = "apply_after_tailoring"
        reason = "岗位具备一定匹配度，但申请材料仍需围绕核心技能调整。"
        steps = ["先改写相关项目经历。", "补充或证明缺失的必需技能。"]
    else:
        next_action = "build_skills_first"
        reason = "当前加权技能匹配度偏低，直接投递的性价比不高。"
        steps = ["优先补齐一到两个缺失的必需技能。", "寻找更贴近当前技能栈的岗位。"]

    if missing_required:
        steps.append(f"优先补齐或证明这些必需技能：{', '.join(missing_required[:3])}。")
    if state.get("has_unknown_constraints", False):
        steps.append("部分硬性条件因候选人信息不足无法判断，投递前请补充确认。")
    return {
        "next_action": next_action,
        "next_action_reason": reason,
        "recommended_steps": steps,
    }


def _build_match_result(state: JobMatchGraphState) -> JobMatchResult:
    job_info = ensure_job_info(state["job_info"])
    return JobMatchResult(
        job_title=job_info.job_title,
        fit_score=state["fit_score"],
        matched_skills=state["matched_skills"],
        missing_skills=state["missing_skills"],
        matched_required_skills=state["matched_required_skills"],
        missing_required_skills=state["missing_required_skills"],
        matched_preferred_skills=state["matched_preferred_skills"],
        missing_preferred_skills=state["missing_preferred_skills"],
        experience_level=job_info.experience_level,
        entry_level_fit=job_info.entry_level_fit,
        constraint_checks=state.get("constraint_checks", []),
        resume_evidence=state.get("resume_evidence", []),
    )


def skip_application_summary_node(state: JobMatchGraphState) -> dict:
    return {
        "match_result": _build_match_result(state).model_dump(),
        "final_answer": (
            f"该岗位加权匹配分为 {state['fit_score']}，但存在明确不满足的硬性条件，"
            "且人工审核选择不继续。建议优先考虑硬性条件符合的岗位。"
        ),
    }


def generate_summary_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    profile = ensure_candidate_profile(state["candidate_profile"])
    chain = create_job_match_summary_chain(streaming=state.get("trace_tokens", False))
    response = chain.invoke(
        {
            "job_info": job_info.model_dump_json(indent=2),
            "candidate_profile": profile.model_dump_json(indent=2),
            "match_data": json.dumps(
                _build_match_result(state).model_dump(), ensure_ascii=False, default=str
            ),
            "memory_context": state.get("memory_context", "No memory context provided."),
            "retrieved_context": state.get("retrieved_context", "No retrieved context provided."),
            "hard_constraint_warning": state.get("hard_constraint_warning", ""),
        },
        config=build_trace_config(
            state.get("tracing", False), run_name="langgraph-generate-summary"
        ),
    )
    response = ensure_job_match_response(response)
    # Deterministic business fields always win over LLM-generated copies.
    return {
        "match_result": _build_match_result(state).model_dump(),
        "final_answer": response.final_answer,
    }


def route_after_next_action(state: JobMatchGraphState) -> str:
    return "skip_application_summary" if state["next_action"] == "skip" else "generate_summary"


def create_job_match_graph(checkpointer=None):
    builder = StateGraph(JobMatchGraphState)
    builder.add_node("extract_job_info", extract_job_info_node)
    builder.add_node("normalize_candidate", normalize_candidate_node)
    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("calculate_match", calculate_match_node)
    builder.add_node("detect_hard_constraints", detect_hard_constraints_node)
    builder.add_node("retrieve_resume_evidence", retrieve_resume_evidence_node)
    builder.add_node("hard_constraint_review", hard_constraint_review_node)
    builder.add_node("recommend_next_action", recommend_next_action_node)
    builder.add_node("skip_application_summary", skip_application_summary_node)
    builder.add_node("generate_summary", generate_summary_node)

    builder.add_edge(START, "extract_job_info")
    builder.add_edge("extract_job_info", "normalize_candidate")
    builder.add_edge("normalize_candidate", "retrieve_context")
    builder.add_edge("retrieve_context", "calculate_match")
    builder.add_edge("calculate_match", "detect_hard_constraints")
    builder.add_edge("detect_hard_constraints", "retrieve_resume_evidence")
    builder.add_conditional_edges(
        "retrieve_resume_evidence",
        route_hard_constraints,
        {
            "hard_constraint_review": "hard_constraint_review",
            "recommend_next_action": "recommend_next_action",
        },
    )
    builder.add_edge("hard_constraint_review", "recommend_next_action")
    builder.add_conditional_edges(
        "recommend_next_action",
        route_after_next_action,
        {
            "generate_summary": "generate_summary",
            "skip_application_summary": "skip_application_summary",
        },
    )
    builder.add_edge("skip_application_summary", END)
    builder.add_edge("generate_summary", END)
    return builder.compile(checkpointer=checkpointer)


def build_langgraph_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _resolve_candidate_profile(
    candidate_profile: CandidateProfile | dict | None,
    resume_skills: list[str] | None,
    resume_text: str,
    memory: JobMatchMemory | None,
) -> CandidateProfile:
    if candidate_profile is not None:
        profile = ensure_candidate_profile(candidate_profile)
    else:
        skills = resume_skills or (memory.resume_skills if memory else [])
        profile = CandidateProfile(skills=skills, resume_text=resume_text)
    if not profile.skills and not profile.resume_text:
        raise ValueError("candidate_profile.skills or resume_skills is required.")
    return profile


def analyze_job_with_langgraph(
    jd: str,
    resume_skills: list[str] | None = None,
    memory: JobMatchMemory | None = None,
    checkpointer=None,
    thread_id: str = "job-match-demo",
    target_role: str | None = None,
    stream: bool = False,
    tracing: bool = False,
    trace_tokens: bool = False,
    candidate_profile: CandidateProfile | dict | None = None,
    resume_text: str = "",
) -> dict:
    if tracing and settings.LANGSMITH_API_KEY is None:
        raise RuntimeError("Please set the LANGSMITH_API_KEY environment variable first.")
    profile = _resolve_candidate_profile(
        candidate_profile, resume_skills, resume_text, memory
    )
    if memory:
        memory.update_user_profile(resume_skills=profile.skills, target_role=target_role)
    memory_context = memory.build_context() if memory else "No memory context provided."
    graph = create_job_match_graph(checkpointer=checkpointer)
    graph_config = build_langgraph_thread_config(thread_id)
    graph_input: JobMatchGraphState = {
        "jd": jd,
        "candidate_profile": profile.model_dump(),
        "memory_context": memory_context,
        "tracing": tracing,
        "trace_tokens": trace_tokens,
    }
    project_name = get_langsmith_project_name()
    try:
        with ls.tracing_context(project_name=project_name, enabled=tracing):
            if stream:
                final_state: dict = graph_input.copy()
                for update in graph.stream(graph_input, config=graph_config, stream_mode="updates"):
                    for node_update in update.values():
                        if isinstance(node_update, dict):
                            final_state.update(node_update)
                checkpoint = graph.get_state(graph_config)
                if checkpoint and checkpoint.values:
                    final_state = checkpoint.values
            else:
                final_state = graph.invoke(graph_input, config=graph_config)
            if "__interrupt__" in final_state:
                return {
                    "interrupted": True,
                    "interrupts": final_state["__interrupt__"],
                    "graph_state": final_state,
                    "thread_id": thread_id,
                }
            return _completed_graph_result(final_state, thread_id, memory, graph, graph_config)
    finally:
        if tracing:
            wait_for_all_tracers()


def _completed_graph_result(final_state, thread_id, memory, graph, graph_config) -> dict:
    match_result = ensure_job_match_result(final_state["match_result"])
    job_info = ensure_job_info(final_state["job_info"])
    if memory:
        memory.remember_analysis(job_info, match_result)
    return {
        "job_info": job_info,
        "candidate_profile": ensure_candidate_profile(final_state["candidate_profile"]),
        "retrieved_context": final_state["retrieved_context"],
        "memory_context": final_state.get("memory_context", "No memory context provided."),
        "hard_constraint_warning": final_state.get("hard_constraint_warning", ""),
        "match_result": match_result,
        "final_answer": final_state["final_answer"],
        "next_action": final_state["next_action"],
        "next_action_reason": final_state["next_action_reason"],
        "recommended_steps": final_state["recommended_steps"],
        "graph_state": final_state,
        "checkpoint_state": graph.get_state(graph_config) if graph else None,
        "interrupted": False,
        "thread_id": thread_id,
    }


def resume_job_match_after_human_review(
    checkpointer,
    thread_id: str,
    approved: bool,
    feedback: str = "",
    memory: JobMatchMemory | None = None,
    tracing: bool = False,
) -> dict:
    graph = create_job_match_graph(checkpointer=checkpointer)
    graph_config = build_langgraph_thread_config(thread_id)
    final_state = graph.invoke(
        Command(resume={"approved": approved, "feedback": feedback}),
        config=graph_config,
    )
    return _completed_graph_result(final_state, thread_id, memory, graph, graph_config)


def analyze_job_with_agent(
    jd: str,
    resume_skills: list[str] | None,
    memory: JobMatchMemory | None = None,
    target_role: str | None = None,
    stream: bool = False,
    tracing: bool = False,
    trace_tokens: bool = False,
) -> dict:
    """Compatibility entry point; the deterministic LangGraph path is now canonical."""
    return analyze_job_with_langgraph(
        jd=jd,
        resume_skills=resume_skills,
        memory=memory,
        target_role=target_role,
        stream=stream,
        tracing=tracing,
        trace_tokens=trace_tokens,
    )
