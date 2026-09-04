"""岗位匹配用例服务。

Service 位于 HTTP 路由与 LangGraph 工作流之间，负责组织一次完整的岗位分析、
人工审核恢复和状态查询。它不处理 HTTP 状态码，也不定义 Graph 节点。
"""

from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver

from job_match_agent.memory import JobMatchMemory
from job_match_agent.workflow import (
    analyze_job_with_langgraph,
    build_langgraph_thread_config,
    create_job_match_graph,
    resume_job_match_after_human_review,
)


class JobMatchService:
    """协调岗位匹配 API 所需的 LangGraph 和内存操作。"""

    def __init__(self, checkpointer=None) -> None:
        # 同一个 Service 实例中的 analyze、resume 和 status 共享这些运行时状态。
        self.checkpointer = (
            checkpointer if checkpointer is not None else MemorySaver()
        )
        self.memories: dict[str, JobMatchMemory] = {}

    def get_memory(self, thread_id: str) -> JobMatchMemory:
        """取得线程的业务记忆；首次访问时创建。"""
        if thread_id not in self.memories:
            self.memories[thread_id] = JobMatchMemory()
        return self.memories[thread_id]

    def analyze(
        self,
        *,
        job_description: str,
        resume_skills: list[str] | None = None,
        thread_id: str | None = None,
        visa_preference: str | None = None,
        target_role: str | None = None,
    ) -> dict[str, Any]:
        """启动岗位分析，并返回完成结果或待人工审核结果。"""
        resolved_thread_id = thread_id or f"job-match-{uuid4()}"
        result = analyze_job_with_langgraph(
            jd=job_description,
            resume_skills=resume_skills or None,
            memory=self.get_memory(resolved_thread_id),
            checkpointer=self.checkpointer,
            thread_id=resolved_thread_id,
            visa_preference=visa_preference,
            target_role=target_role,
        )

        if result["interrupted"]:
            return self._serialize_interrupted_result(result)
        return self._serialize_completed_result(result)

    def resume(
        self,
        *,
        thread_id: str,
        approved: bool,
        feedback: str = "",
    ) -> dict[str, Any]:
        """提交人工审核结果，并从相应 checkpoint 恢复工作流。"""
        result = resume_job_match_after_human_review(
            checkpointer=self.checkpointer,
            thread_id=thread_id,
            approved=approved,
            feedback=feedback,
            memory=self.get_memory(thread_id),
        )
        return self._serialize_completed_result(result)

    def get_status(self, thread_id: str) -> dict[str, Any]:
        """读取指定线程的 LangGraph checkpoint 状态。"""
        graph = create_job_match_graph(checkpointer=self.checkpointer)
        checkpoint_state = graph.get_state(
            build_langgraph_thread_config(thread_id)
        )

        if not checkpoint_state.values:
            raise KeyError(thread_id)

        values = checkpoint_state.values
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
            if isinstance(value, dict):
                serialized.append(value)
            else:
                serialized.append({"value": str(value)})
        return serialized

    @classmethod
    def _serialize_interrupted_result(
        cls,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "pending_human_review",
            "thread_id": result["thread_id"],
            "review": cls._serialize_interrupts(result["interrupts"]),
        }

    @staticmethod
    def _serialize_completed_result(
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "completed",
            "thread_id": result["thread_id"],
            "job_info": result["job_info"].model_dump(),
            "retrieved_context": result["retrieved_context"],
            "visa_warning": result["visa_warning"],
            "match_result": result["match_result"].model_dump(),
            "final_answer": result["final_answer"],
            "next_action": result["next_action"],
            "next_action_reason": result["next_action_reason"],
            "recommended_steps": result["recommended_steps"],
        }


# FastAPI 路由复用这一实例，保证多次请求共享同一个进程内 checkpoint。
job_match_service = JobMatchService()
