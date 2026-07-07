"""Pydantic models for request/response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Generic Response Wrapper ───────────────────────────────────────

class ApiResponse(BaseModel):
    success: bool
    data: dict | list | None = None
    error: str | None = None


# ── Chat ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    conversation_id: int
    game_name: str = Field(..., min_length=1, description="Name of the game to query")
    message: str = Field(..., min_length=1, description="User question")


# ── Conversations ──────────────────────────────────────────────────

class CreateConversationRequest(BaseModel):
    title: str | None = None
    game_name: str = Field(default="", description="Game name for this conversation")


class UpdateConversationRequest(BaseModel):
    title: str = Field(..., min_length=1)


# ── Documents ──────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    id: int
    game_name: str
    title: str | None
    url: str | None
    source_name: str | None
    chunk_index: int
    created_at: str


# ── Scraping ───────────────────────────────────────────────────────

class CreateScrapingTaskRequest(BaseModel):
    game_name: str = Field(..., min_length=1)
    source_name: str = Field(..., min_length=1)
    source_url: str = Field(..., min_length=1)
    interval_hours: int = Field(default=24, ge=1, le=720)


# ── Chat Response Pieces ───────────────────────────────────────────

class SourceInfo(BaseModel):
    title: str | None
    url: str | None
    chunk_index: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    conversation_id: int
