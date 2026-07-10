"""Knowledge base management routes — automatic knowledge acquisition."""

import json

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app import database as db
from app.knowledge_manager import (
    fetch_and_store_game_knowledge,
    process_pending_queries,
    refresh_trending_games,
)
from app.models import ApiResponse
from app.scheduler import trigger_knowledge_cycle_now
from config import settings

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class FetchGameRequest(BaseModel):
    game_name: str = Field(..., min_length=1)
    force: bool = False


@router.get("/status")
async def get_status():
    """Return aggregate knowledge-base statistics."""
    stats = db.get_knowledge_stats()
    return ApiResponse(success=True, data=stats)


@router.get("/pending")
async def list_pending():
    """List user questions the knowledge base could not answer."""
    pending = db.list_pending_queries(limit=100, status="pending")
    return ApiResponse(success=True, data=pending)


@router.delete("/pending/{query_id}")
async def delete_pending(query_id: str):
    deleted = db.delete_pending_query(query_id)
    if not deleted:
        return ApiResponse(success=False, error="Pending query not found")
    return ApiResponse(success=True, data={"deleted": True})


@router.get("/logs")
async def list_logs():
    """Return recent knowledge-acquisition logs."""
    logs = db.list_knowledge_logs(limit=20)
    return ApiResponse(success=True, data=logs)


@router.post("/refresh-trending")
async def refresh_trending():
    """Discover trending games and fetch their knowledge. Runs synchronously."""
    try:
        result = await refresh_trending_games(count=settings.trending_game_count, force=False)
        fetched = len([r for r in result.get("results", []) if r["status"] == "completed"])
        db.add_knowledge_log(
            action="manual_refresh_trending",
            trending_fetched=fetched,
            games_detail=json.dumps(result, ensure_ascii=False),
            message=f"手动获取热门游戏,发现 {result['discovered']} 个,获取 {fetched} 个",
        )
        return ApiResponse(success=True, data=result)
    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.post("/process-pending")
async def process_pending():
    """Process all pending user questions (feedback learning). Runs synchronously."""
    try:
        result = await process_pending_queries()
        db.add_knowledge_log(
            action="manual_process_pending",
            pending_processed=result["processed"],
            games_detail=json.dumps(result, ensure_ascii=False),
            message=f"手动处理待学习问题,涉及 {result['processed']} 个游戏",
        )
        return ApiResponse(success=True, data=result)
    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.post("/games")
async def fetch_game(req: FetchGameRequest):
    """Manually fetch knowledge for a specific game. Runs synchronously."""
    try:
        result = await fetch_and_store_game_knowledge(req.game_name, replace=req.force)
        db.add_knowledge_log(
            action="manual_fetch_game",
            games_detail=json.dumps(result, ensure_ascii=False),
            message=f"手动获取游戏《{req.game_name}》知识,状态: {result['status']}",
        )
        return ApiResponse(success=True, data=result)
    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.post("/cycle")
async def trigger_cycle():
    """Trigger a full knowledge-acquisition cycle in the background."""
    trigger_knowledge_cycle_now()
    return ApiResponse(success=True, data={"status": "triggered"})
