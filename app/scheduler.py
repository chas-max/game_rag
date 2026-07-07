"""Task scheduler — manages APScheduler lifecycle for automatic knowledge acquisition."""

import asyncio

from apscheduler.schedulers.background import BackgroundScheduler

from app import database as db
from app.knowledge_manager import run_knowledge_cycle
from config import settings

_scheduler: BackgroundScheduler | None = None


def _run_knowledge_cycle_job() -> None:
    """Synchronous wrapper that runs the async knowledge cycle in a fresh event loop."""
    try:
        print("[scheduler] Knowledge acquisition cycle triggered.")
        asyncio.run(run_knowledge_cycle())
        print("[scheduler] Knowledge acquisition cycle finished.")
    except Exception as e:
        print(f"[scheduler] Knowledge cycle failed: {e}")


def init_scheduler() -> None:
    """Initialize and start the background scheduler.

    Schedules a recurring knowledge-acquisition job that:
      1. Processes pending user questions (feedback learning).
      2. Discovers and fetches knowledge for trending games.
    """
    global _scheduler
    _scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    _scheduler.add_job(
        _run_knowledge_cycle_job,
        trigger="interval",
        hours=settings.knowledge_fetch_interval_hours,
        id="knowledge_cycle",
        replace_existing=True,
        # Run once shortly after startup so the knowledge base isn't empty on a fresh install
        next_run_time=None,  # rely on the interval; manual trigger available via API
    )

    _scheduler.start()
    print(
        f"[scheduler] Background scheduler started. Knowledge cycle runs every "
        f"{settings.knowledge_fetch_interval_hours} hour(s)."
    )


def shutdown_scheduler() -> None:
    """Shut down the background scheduler gracefully."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        print("[scheduler] Background scheduler shut down.")


def trigger_knowledge_cycle_now() -> None:
    """Trigger the knowledge-acquisition cycle to run immediately (in the background)."""
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_knowledge_cycle_job,
        id="knowledge_cycle_manual",
        replace_existing=True,
    )
