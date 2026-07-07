"""Game listing route."""

from fastapi import APIRouter

from app import database as db
from app.models import ApiResponse

router = APIRouter(prefix="/games", tags=["games"])


@router.get("")
async def list_games():
    games = db.list_games()
    return ApiResponse(success=True, data=games)
