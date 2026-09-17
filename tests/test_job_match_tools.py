import unittest

from job_match_agent.tools import (
    calculate_weighted_match,
    detect_hard_constraints,
    normalize_skills,
    retrieve_resume_evidence,
)


class JobMatchToolTests(unittest.TestCase):
    def test_normalize_skills_deduplicates_aliases(self):
        result = normalize_skills.invoke(
            {"skills": [" Python ", "PYTHON", "PowerBI", "Power BI", "sklearn"]}
        )
        self.assertEqual(
            result["normalized_skills"], ["power bi", "python", "scikit-learn"]
        )

    def test_weighted_match_prioritizes_required_skills(self):
        result = calculate_weighted_match.invoke(
            {
                "required_skills": ["Python", "SQL"],
                "preferred_skills": ["Tableau", "PowerBI"],
                "candidate_skills": ["PYTHON", "Power BI"],
            }
        )
        self.assertEqual(result["fit_score"], 50)
        self.assertEqual(result["matched_required_skills"], ["python"])
        self.assertEqual(result["matched_preferred_skills"], ["power bi"])

    def test_detect_hard_constraints_has_three_states(self):
        result = detect_hard_constraints.invoke(
            {
                "hard_constraints": [
                    {
                        "field": "years_experience",
                        "operator": "gte",
                        "value": 3,
                        "description": "At least three years of experience",
                    },
                    {
                        "field": "location",
                        "operator": "equals",
                        "value": "Shanghai",
                        "description": "Must be based in Shanghai",
                    },
                ],
                "candidate_profile": {
                    "skills": ["Python"],
                    "years_experience": 2,
                },
            }
        )
        self.assertTrue(result["has_blocking_violation"])
        self.assertTrue(result["has_unknowns"])
        self.assertEqual(
            [check["status"] for check in result["checks"]],
            ["unsatisfied", "unknown"],
        )

    def test_retrieve_resume_evidence_returns_excerpts(self):
        result = retrieve_resume_evidence.invoke(
            {
                "target_skills": ["Python", "SQL"],
                "resume_text": (
                    "Built a forecasting service in Python.\n"
                    "Designed SQL models for the analytics warehouse."
                ),
            }
        )
        evidence = {item["skill"]: item["excerpts"] for item in result["evidence"]}
        self.assertIn("Python", evidence["python"][0])
        self.assertIn("SQL", evidence["sql"][0])


if __name__ == "__main__":
    unittest.main()
