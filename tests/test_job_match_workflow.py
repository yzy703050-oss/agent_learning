import json
import unittest
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from job_match_agent.models import JobInfo, JobMatchResponse, JobMatchResult
from job_match_agent.workflow import (
    build_langgraph_thread_config,
    create_job_match_graph,
)


class _SummaryChain:
    def invoke(self, inputs, config=None):
        match_result = JobMatchResult.model_validate(json.loads(inputs["match_data"]))
        return JobMatchResponse(match_result=match_result, final_answer="测试摘要")


class JobMatchWorkflowTests(unittest.TestCase):
    def _invoke_graph(self, job_info, profile, thread_id):
        checkpointer = MemorySaver()
        with (
            patch(
                "job_match_agent.workflow.extract_job_info_langchain",
                return_value=job_info,
            ),
            patch(
                "job_match_agent.workflow.retrieve_job_context",
                return_value="mock career context",
            ),
            patch(
                "job_match_agent.workflow.create_job_match_summary_chain",
                return_value=_SummaryChain(),
            ),
        ):
            graph = create_job_match_graph(checkpointer=checkpointer)
            config = build_langgraph_thread_config(thread_id)
            state = graph.invoke(
                {
                    "jd": "mock jd",
                    "candidate_profile": profile,
                    "memory_context": "none",
                    "tracing": False,
                    "trace_tokens": False,
                },
                config=config,
            )
            yield graph, config, state

    def test_graph_completes_without_constraint_violation(self):
        job_info = JobInfo(
            job_title="Data Analyst",
            required_skills=["Python", "SQL"],
            preferred_skills=["Tableau"],
            experience_level="1-2 years",
            hard_constraints=[],
            entry_level_fit="Yes",
        )
        iterator = self._invoke_graph(
            job_info,
            {
                "skills": ["PYTHON", "SQL"],
                "resume_text": "Built Python services and SQL models.",
            },
            "workflow-complete",
        )
        graph, config, state = next(iterator)
        self.assertNotIn("__interrupt__", state)
        self.assertEqual(state["fit_score"], 80)
        self.assertEqual(state["final_answer"], "测试摘要")
        self.assertEqual(state["resume_evidence"][0]["excerpts"][0], "Built Python services and SQL models.")

    def test_graph_interrupts_and_resumes_for_hard_constraint(self):
        job_info = JobInfo(
            job_title="Senior Analyst",
            required_skills=["Python"],
            preferred_skills=[],
            experience_level="5+ years",
            hard_constraints=[
                {
                    "field": "years_experience",
                    "operator": "gte",
                    "value": 5,
                    "description": "At least five years of experience",
                }
            ],
            entry_level_fit="No",
        )
        iterator = self._invoke_graph(
            job_info,
            {
                "skills": ["Python"],
                "resume_text": "Used Python in production.",
                "years_experience": 2,
            },
            "workflow-interrupt",
        )
        graph, config, state = next(iterator)
        self.assertIn("__interrupt__", state)
        self.assertEqual(state["__interrupt__"][0].value["reason"], "hard_constraint_violation")

        resumed = graph.invoke(
            Command(resume={"approved": True, "feedback": "continue"}),
            config=config,
        )
        self.assertEqual(resumed["next_action"], "cautious_apply")
        self.assertIn("硬性条件", resumed["hard_constraint_warning"])
        self.assertEqual(resumed["final_answer"], "测试摘要")


if __name__ == "__main__":
    unittest.main()
