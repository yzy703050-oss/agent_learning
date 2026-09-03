"""Job Match Agent FastAPI 应用入口。"""

from fastapi import FastAPI

from app.api.v1.api import api_router


API_DESCRIPTION = """
这是一个基于 LangGraph 的岗位匹配 Agent 后端示例。

它支持岗位分析、RAG、业务记忆，以及通过 interrupt/resume 完成人工审核。
当前阶段使用进程内 MemorySaver 和 JobMatchMemory，适合学习与本地演示。
"""

OPENAPI_TAGS = [
    {"name": "系统状态", "description": "检查后端服务是否正常运行。"},
    {
        "name": "岗位匹配",
        "description": "启动岗位分析、提交人工审核结果、查询工作流状态。",
    },
]

app = FastAPI(
    title="岗位匹配 Agent 后端 API",
    description=API_DESCRIPTION,
    version="0.2.0",
    openapi_tags=OPENAPI_TAGS,
)
app.include_router(api_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": "Job Match Agent API",
        "docs": "/docs",
        "api_version": "v1",
    }
