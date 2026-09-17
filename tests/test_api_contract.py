import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.job_match import AnalyzeJobRequest


class ApiContractTests(unittest.TestCase):
    def test_health_endpoint(self):
        response = TestClient(app).get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_candidate_profile_contract(self):
        request = AnalyzeJobRequest.model_validate(
            {
                "job_description": "Data analyst with Python",
                "candidate_profile": {
                    "skills": ["Python"],
                    "resume_text": "Built Python analytics pipelines.",
                    "years_experience": 2,
                },
            }
        )
        self.assertEqual(request.candidate_profile.skills, ["Python"])

    def test_analyze_endpoint_serializes_new_business_contract(self):
        completed = {
            "status": "completed",
            "thread_id": "api-contract",
            "job_info": {
                "job_title": "Data Analyst",
                "required_skills": ["Python"],
                "preferred_skills": ["Tableau"],
                "experience_level": "1-2 years",
                "hard_constraints": [],
                "entry_level_fit": "Yes",
            },
            "candidate_profile": {
                "skills": ["python"],
                "resume_text": "Built Python pipelines.",
                "years_experience": 1,
                "education_level": None,
                "location": None,
                "work_authorization": None,
                "certifications": [],
            },
            "retrieved_context": "mock context",
            "hard_constraint_warning": "",
            "match_result": {
                "job_title": "Data Analyst",
                "fit_score": 67,
                "matched_skills": ["python"],
                "missing_skills": ["tableau"],
                "matched_required_skills": ["python"],
                "missing_required_skills": [],
                "matched_preferred_skills": [],
                "missing_preferred_skills": ["tableau"],
                "experience_level": "1-2 years",
                "entry_level_fit": "Yes",
                "constraint_checks": [],
                "resume_evidence": [
                    {"skill": "python", "excerpts": ["Built Python pipelines."]}
                ],
            },
            "final_answer": "匹配良好。",
            "next_action": "apply_after_tailoring",
            "next_action_reason": "仍缺少一个优选技能。",
            "recommended_steps": ["突出 Python 证据。"],
        }
        with patch(
            "app.api.v1.job_match.job_match_service.analyze",
            return_value=completed,
        ) as analyze:
            response = TestClient(app).post(
                "/api/v1/job-match/analyze",
                json={
                    "job_description": "Data analyst with Python",
                    "candidate_profile": {
                        "skills": ["Python"],
                        "resume_text": "Built Python pipelines.",
                    },
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["match_result"]["fit_score"], 67)
        self.assertIsNotNone(analyze.call_args.kwargs["candidate_profile"])


if __name__ == "__main__":
    unittest.main()
