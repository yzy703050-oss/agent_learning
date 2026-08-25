import os

from .config import load_env_file


def get_langsmith_project_name() -> str:
    load_env_file()
    return os.environ.get("LANGSMITH_PROJECT", "job-agent-demo")


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
