from app.core.config import settings


def get_langsmith_project_name() -> str:
    return settings.LANGSMITH_PROJECT


def build_trace_config(enable_tracing: bool, run_name: str) -> dict | None:
    if not enable_tracing:
        return None

    return {
        "run_name": run_name,
        "tags": ["job-agent-demo"],
        "metadata": {
            "project": get_langsmith_project_name(),
            "demo": "job-agent",
        },
    }
