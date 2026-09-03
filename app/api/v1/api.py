"""v1 API 路由聚合器。"""

from fastapi import APIRouter

from app.api.v1.job_match import router as job_match_router
from app.schemas.job_match import HealthResponse


api_router = APIRouter()
api_router.include_router(
    job_match_router,
    prefix="/job-match",
    tags=["岗位匹配"],
)


@api_router.get(
    "/health",
    response_model=HealthResponse,
    tags=["系统状态"],
    summary="健康检查",
)
def health_check() -> dict[str, str]:
    return {"status": "ok"}
