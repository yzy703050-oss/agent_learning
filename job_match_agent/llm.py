from langchain_deepseek import ChatDeepSeek

from app.core.config import settings


def create_llm(streaming: bool = False):
    if settings.DEEPSEEK_API_KEY is None:
        raise RuntimeError("Please set the DEEPSEEK_API_KEY environment variable first.")

    return ChatDeepSeek(
        api_key=settings.DEEPSEEK_API_KEY.get_secret_value(),
        model=settings.DEEPSEEK_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        streaming=streaming,
    )
