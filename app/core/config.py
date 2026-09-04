"""应用的统一配置入口。"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

# LangSmith 等第三方库仍会直接读取系统环境变量，因此先集中加载一次 .env。
load_dotenv(DEFAULT_ENV_PATH, override=False)


class Settings(BaseSettings):
    """从环境变量和项目根目录的 .env 中读取应用配置。"""

    PROJECT_NAME: str = "岗位匹配 Agent 后端 API"
    APP_VERSION: str = "0.2.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = (
        "这是一个基于 LangGraph 的岗位匹配 Agent 后端示例。\n\n"
        "它支持岗位分析、RAG、业务记忆，以及通过 interrupt/resume 完成人工审核。\n"
        "当前阶段使用进程内 MemorySaver 和 JobMatchMemory，适合学习与本地演示。"
    )

    DEEPSEEK_API_KEY: SecretStr | None = None
    DEEPSEEK_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)

    LANGSMITH_API_KEY: SecretStr | None = None
    LANGSMITH_PROJECT: str = "job-agent-demo"

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("API_V1_STR")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """API 前缀必须形如 /api/v1，并移除结尾的斜杠。"""
        if not value.startswith("/") or value == "/":
            raise ValueError("API_V1_STR must start with '/' and cannot be '/'.")
        return value.rstrip("/")


settings = Settings()


def load_env_file(env_path: Path | str = DEFAULT_ENV_PATH) -> None:
    """兼容旧调用；新代码应直接从 settings 读取配置。"""
    load_dotenv(Path(env_path), override=False)
