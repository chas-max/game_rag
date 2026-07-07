"""Settings route — exposes non-sensitive config to the frontend."""

from fastapi import APIRouter

from app.models import ApiResponse
from config import settings

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
async def get_settings():
    return ApiResponse(
        success=True,
        data={
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k,
            "similarity_threshold": settings.similarity_threshold,
        },
    )
