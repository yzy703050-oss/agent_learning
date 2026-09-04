"""Job Match Agent FastAPI 应用入口。"""

from fastapi import FastAPI

from app.api.v1.api import api_router
from app.core.config import settings


OPENAPI_TAGS = [
    {"name": "系统状态", "description": "检查后端服务是否正常运行。"},
    {
        "name": "岗位匹配",
        "description": "启动岗位分析、提交人工审核结果、查询工作流状态。",
    },
]

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.APP_VERSION,
    openapi_tags=OPENAPI_TAGS,
)
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": settings.PROJECT_NAME,
        "docs": "/docs",
        "api_version": settings.API_V1_STR.rsplit("/", maxsplit=1)[-1],
    }
