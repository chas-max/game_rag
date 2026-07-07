"""Aggregate all routers for easy registration in main.py."""

from app.routes.chat import router as chat_router
from app.routes.conversations import router as conversations_router
from app.routes.documents import router as documents_router
from app.routes.games import router as games_router
from app.routes.knowledge import router as knowledge_router
from app.routes.settings import router as settings_router

all_routers = [
    chat_router,
    conversations_router,
    documents_router,
    games_router,
    knowledge_router,
    settings_router,
]
