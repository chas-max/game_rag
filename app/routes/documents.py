"""Document and game listing routes."""

from fastapi import APIRouter, Query

from app import database as db
from app.models import ApiResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def list_documents(game_name: str = Query(..., min_length=1)):
    docs = db.get_documents_by_game(game_name)
    return ApiResponse(success=True, data=docs)


@router.delete("/by-game/{game_name}")
async def delete_documents_by_game(game_name: str):
    count = db.delete_documents_by_game(game_name)
    return ApiResponse(success=True, data={"deleted": count})
