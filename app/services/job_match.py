"""岗位匹配用例服务。"""

from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver

from job_match_agent.memory import JobMatchMemory
from job_match_agent.models import CandidateProfile
from job_match_agent.workflow import (
    analyze_job_with_langgraph,
    build_langgraph_thread_config,
    create_job_match_graph,
    resume_job_match_after_human_review,
)


class JobMatchService:
    """协调 HTTP 层、业务记忆和 LangGraph 工作流。"""

    def __init__(self, checkpointer=None) -> None:
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self.memories: dict[str, JobMatchMemory] = {}

    def get_memory(self, thread_id: str) -> JobMatchMemory:
        if thread_id not in self.memories:
            self.memories[thread_id] = JobMatchMemory()
        return self.memories[thread_id]

    def analyze(
        self,
        *,
        job_description: str,
        candidate_profile: CandidateProfile | dict | None = None,
        resume_skills: list[str] | None = None,
        resume_text: str = "",
        thread_id: str | None = None,
        target_role: str | None = None,
    ) -> dict[str, Any]:
        resolved_thread_id = thread_id or f"job-match-{uuid4()}"
        result = analyze_job_with_langgraph(
            jd=job_description,
            candidate_profile=candidate_profile,
            resume_skills=resume_skills or None,
            resume_text=resume_text,
            memory=self.get_memory(resolved_thread_id),
            checkpointer=self.checkpointer,
            thread_id=resolved_thread_id,
            target_role=target_role,
        )
        if result["interrupted"]:
            return self._serialize_interrupted_result(result)
        return self._serialize_completed_result(result)

    def resume(self, *, thread_id: str, approved: bool, feedback: str = "") -> dict[str, Any]:
        result = resume_job_match_after_human_review(
            checkpointer=self.checkpointer,
            thread_id=thread_id,
            approved=approved,
            feedback=feedback,
            memory=self.get_memory(thread_id),
        )
        return self._serialize_completed_result(result)

    def get_status(self, thread_id: str) -> dict[str, Any]:
        graph = create_job_match_graph(checkpointer=self.checkpointer)
        values = graph.get_state(build_langgraph_thread_config(thread_id)).values
        if not values:
            raise KeyError(thread_id)
        if "__interrupt__" in values:
            return {
                "status": "pending_human_review",
                "thread_id": thread_id,
                "review": self._serialize_interrupts(values["__interrupt__"]),
            }
        if "match_result" in values:
            return {
                "status": "completed",
                "thread_id": thread_id,
                "next_action": values.get("next_action"),
                "next_action_reason": values.get("next_action_reason"),
                "recommended_steps": values.get("recommended_steps", []),
            }
        return {
            "status": "running_or_partial",
            "thread_id": thread_id,
            "state_keys": sorted(values.keys()),
        }

    @staticmethod
    def _serialize_interrupts(interrupts: Any) -> list[dict[str, Any]]:
        serialized = []
        for item in interrupts:
            value = getattr(item, "value", item)
            serialized.append(value if isinstance(value, dict) else {"value": str(value)})
        return serialized

    @classmethod
    def _serialize_interrupted_result(cls, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "pending_human_review",
            "thread_id": result["thread_id"],
            "review": cls._serialize_interrupts(result["interrupts"]),
        }

    @staticmethod
    def _serialize_completed_result(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "completed",
            "thread_id": result["thread_id"],
            "job_info": result["job_info"].model_dump(),
            "candidate_profile": result["candidate_profile"].model_dump(),
            "retrieved_context": result["retrieved_context"],
            "hard_constraint_warning": result["hard_constraint_warning"],
            "match_result": result["match_result"].model_dump(),
            "final_answer": result["final_answer"],
            "next_action": result["next_action"],
            "next_action_reason": result["next_action_reason"],
            "recommended_steps": result["recommended_steps"],
        }


job_match_service = JobMatchService()
