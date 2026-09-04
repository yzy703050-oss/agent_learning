"""向后兼容的启动入口。

旧命令 ``uvicorn api:app --reload`` 仍然可用；新项目结构推荐使用
``uvicorn app.main:app --reload``。
"""

from app.main import app


__all__ = ["app"]
