from typing import Any

from .models import JobInfo, JobMatchResult


class JobMatchMemory:
    """Small explicit memory store for learning how context enters a prompt."""

    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.resume_skills: list[str] = []
        self.user_preferences: dict[str, str] = {}
        self.analysis_history: list[dict[str, Any]] = []
        self.compressed_history: dict[str, Any] = {
            "older_jobs_count": 0,
            "best_fit_score": None,
            "best_fit_job_title": None,
            "missing_skill_counts": {},
        }

    def update_user_profile(
        self,
        resume_skills: list[str] | None = None,
        target_role: str | None = None,
    ) -> None:
        if resume_skills:
            self.resume_skills = resume_skills

        if target_role:
            self.user_preferences["target_role"] = target_role

    def remember_analysis(self, job_info: JobInfo, match_result: JobMatchResult) -> None:
        self.analysis_history.append(
            {
                "job_title": job_info.job_title,
                "fit_score": match_result.fit_score,
                "entry_level_fit": job_info.entry_level_fit,
                "hard_constraint_count": len(job_info.hard_constraints),
                "missing_skills": match_result.missing_skills,
            }
        )
        self.compress_old_history()

    def compress_old_history(self) -> None:
        while len(self.analysis_history) > self.max_history:
            old_item = self.analysis_history.pop(0)
            self.compressed_history["older_jobs_count"] += 1

            best_score = self.compressed_history["best_fit_score"]
            if best_score is None or old_item["fit_score"] > best_score:
                self.compressed_history["best_fit_score"] = old_item["fit_score"]
                self.compressed_history["best_fit_job_title"] = old_item["job_title"]

            missing_skill_counts = self.compressed_history["missing_skill_counts"]
            for skill in old_item["missing_skills"]:
                missing_skill_counts[skill] = missing_skill_counts.get(skill, 0) + 1

    def build_context(self) -> str:
        lines = []

        if self.resume_skills:
            lines.append(f"Remembered resume skills: {', '.join(self.resume_skills)}")

        if self.user_preferences:
            preferences = [
                f"{key}: {value}" for key, value in self.user_preferences.items()
            ]
            lines.append(f"Remembered user preferences: {'; '.join(preferences)}")

        if self.compressed_history["older_jobs_count"]:
            lines.append(
                "Compressed older analyses: "
                f"{self.compressed_history['older_jobs_count']} older jobs."
            )

            if self.compressed_history["best_fit_job_title"]:
                lines.append(
                    "Best older fit: "
                    f"{self.compressed_history['best_fit_job_title']} "
                    f"with score {self.compressed_history['best_fit_score']}."
                )

            missing_skill_counts = self.compressed_history["missing_skill_counts"]
            if missing_skill_counts:
                common_missing_skills = sorted(
                    missing_skill_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
                formatted_skills = [
                    f"{skill} ({count})" for skill, count in common_missing_skills
                ]
                lines.append(
                    "Common missing skills from older analyses: "
                    f"{', '.join(formatted_skills)}"
                )

        if self.analysis_history:
            lines.append("Recent job analyses:")
            for item in self.analysis_history:
                lines.append(
                    "- "
                    f"{item['job_title']}: fit_score={item['fit_score']}, "
                    f"hard_constraints={item['hard_constraint_count']}, "
                    f"entry_level_fit={item['entry_level_fit']}"
                )

        if not lines:
            return "No memory context yet."

        return "\n".join(lines)
