import os

from langchain_deepseek import ChatDeepSeek

from .config import load_env_file


def create_llm(streaming: bool = False):
    load_env_file()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("Please set the DEEPSEEK_API_KEY environment variable first.")

    return ChatDeepSeek(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0,
        streaming=streaming,
    )
