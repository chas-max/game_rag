"""Conversation CRUD routes."""

from fastapi import APIRouter

from app import database as db
from app.models import (
    ApiResponse,
    CreateConversationRequest,
    UpdateConversationRequest,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
async def list_conversations():
    convs = db.list_conversations()
    return ApiResponse(success=True, data=convs)


@router.post("")
async def create_conversation(req: CreateConversationRequest):
    title = req.title or "New Conversation"
    conv = db.create_conversation(game_name=req.game_name, title=title)
    return ApiResponse(success=True, data=conv)


@router.get("/{conv_id}")
async def get_conversation(conv_id: int):
    conv = db.get_conversation(conv_id)
    if conv is None:
        return ApiResponse(success=False, error="Conversation not found")
    return ApiResponse(success=True, data=conv)


@router.put("/{conv_id}")
async def update_conversation(conv_id: int, req: UpdateConversationRequest):
    conv = db.update_conversation(conv_id, req.title)
    if conv is None:
        return ApiResponse(success=False, error="Conversation not found")
    return ApiResponse(success=True, data=conv)


@router.delete("/{conv_id}")
async def delete_conversation(conv_id: int):
    deleted = db.delete_conversation(conv_id)
    if not deleted:
        return ApiResponse(success=False, error="Conversation not found")
    return ApiResponse(success=True, data={"deleted": True})
