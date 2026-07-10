"""Game RAG Q&A — FastAPI application entry point."""

import sys
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


dotenv.load_dotenv(override=True)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database import init_db
from app.routes import all_routers
from app.scheduler import init_scheduler, shutdown_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Startup
    print("[startup] Initializing database...")
    init_db()
    print("[startup] Initializing scheduler...")
    init_scheduler()
    print("[startup] Application ready.")
    yield
    # Shutdown
    print("[shutdown] Stopping scheduler...")
    shutdown_scheduler()
    print("[shutdown] Application stopped.")


app = FastAPI(
    title="Game RAG Q&A",
    description="RAG-based game information Q&A web application",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routers under /api prefix
for router in all_routers:
    app.include_router(router, prefix="/api")

# Serve static frontend files removed, now using separate Next.js app



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
