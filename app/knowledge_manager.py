"""Knowledge Manager — LLM 驱动的自动知识获取模块。

核心能力:
1. discover_trending_games()  — 用 LLM 发现当下热门游戏
2. generate_game_knowledge()   — 用 LLM 生成单个游戏的百科知识
3. process_pending_queries()   — 处理用户未答上的问题(反馈学习)
4. refresh_trending_games()    — 批量获取热门游戏知识
5. run_knowledge_cycle()       — 完整知识获取周期(pending 优先 + 热门补充)
"""

import json
import re
from collections import defaultdict

import httpx
from openai import AsyncOpenAI

from app import database as db
from app.embedding import encode_batch
from app.scraper import chunk_text
from config import settings

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            http_client=httpx.AsyncClient(timeout=60.0),
        )
    return _client


async def _llm_generate(prompt: str, max_tokens: int = 3000, temperature: float = 0.5) -> str:
    """Call the LLM to generate text. Returns the generated string.

    Retries up to 3 times to handle transient network/proxy issues.
    """
    client = _get_client()
    last_error = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            # Empty content with stop reason — treat as transient, retry
            print(f"[knowledge] LLM returned empty content (attempt {attempt + 1}), retrying...")
        except Exception as e:
            last_error = e
            print(f"[knowledge] LLM call failed (attempt {attempt + 1}): {e}")
        # Brief delay before retry
        import asyncio
        await asyncio.sleep(1.5)
    if last_error:
        raise last_error
    return ""


def _clean_game_name(raw: str) -> str:
    """Strip list markers, quotes, and surrounding whitespace from a game name."""
    name = re.sub(r"^\s*[\d]+[.、\)）]\s*", "", raw)          # "1. " / "1、" / "1)"
    name = re.sub(r"^\s*[•·\-\*]\s*", "", name)                # bullet points
    name = name.strip().strip("\"'“”‘’《》").strip()
    return name


# ── 热门游戏发现 ───────────────────────────────────────────────────

async def discover_trending_games(count: int = 10) -> list[str]:
    """Use the LLM to list currently popular games."""
    prompt = f"""请列出当下 {count} 个最热门、最具讨论度的电子游戏。要求:
1. 每行一个游戏名称,不要编号、不要解释、不要额外文字
2. 涵盖不同类型(开放世界、MOBA、FPS、RPG、动作、独立游戏等)
3. 优先选择玩法丰富、有故事背景、玩家基数大的游戏
4. 包含近期热门新作和长青经典

只输出游戏名称列表:"""
    result = await _llm_generate(prompt, max_tokens=500, temperature=0.6)
    games = []
    for line in result.split("\n"):
        name = _clean_game_name(line)
        if name and len(name) <= 40:
            games.append(name)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for g in games:
        if g not in seen:
            seen.add(g)
            unique.append(g)
    return unique[:count]


# ── 单游戏知识生成 ─────────────────────────────────────────────────

async def generate_game_knowledge(
    game_name: str,
    specific_questions: list[str] | None = None,
) -> str:
    """Generate detailed encyclopedia-style knowledge for a game via the LLM."""
    topics_section = ""
    if specific_questions:
        topics_section = "\n\n用户特别想了解以下问题,请确保知识文本能直接回答它们:\n"
        for i, q in enumerate(specific_questions, 1):
            topics_section += f"{i}. {q}\n"

    prompt = f"""你是游戏百科专家。请为游戏《{game_name}》生成详细、准确的知识文本,用于游戏问答知识库。

请涵盖以下方面:
1. 游戏基本信息(类型、平台、开发商、发行日期)
2. 世界观与背景故事
3. 核心玩法机制(详细说明)
4. 主要角色及其特点
5. 战斗系统/操作机制
6. 重要物品、装备、技能的获取方法
7. 关卡/区域/模式介绍
8. 新手技巧与进阶策略{topics_section}

要求:
- 用结构化的中文输出,使用标题(用【】或数字)和列表
- 内容详细、准确、具体(包含数值、方法、步骤等)
- 信息密度高,适合用于问答检索
- 基于你对该游戏的真实了解,不要编造

请直接输出知识文本:"""
    return await _llm_generate(prompt, max_tokens=4000, temperature=0.4)


async def fetch_and_store_game_knowledge(
    game_name: str,
    specific_questions: list[str] | None = None,
    replace: bool = True,
) -> dict:
    """Generate, chunk, embed, and store knowledge for a single game.

    Args:
        game_name: Target game.
        specific_questions: Optional user questions to focus the generation on.
        replace: If True, delete existing documents for this game first.

    Returns:
        dict with status, chunks, error.
    """
    try:
        content = await generate_game_knowledge(game_name, specific_questions)
        if not content or len(content) < 100:
            return {"status": "failed", "error": "生成内容为空或过短", "chunks": 0, "game": game_name}

        # Chunk the generated text
        chunks = chunk_text(content)
        if not chunks:
            return {"status": "failed", "error": "分块失败", "chunks": 0, "game": game_name}

        # Batch-embed all chunks (run in executor to avoid blocking the event loop)
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, encode_batch, chunks)

        # Replace old knowledge for this game
        if replace:
            db.delete_documents_by_game(game_name)

        # Store all chunks along with semantic chunks
        await db.store_documents_with_semantic_chunks(
            game_name=game_name,
            chunks=chunks,
            embeddings=embeddings,
            title=f"{game_name} - 百科知识",
            url="",
            source_name="LLM生成",
        )

        return {
            "status": "completed",
            "error": None,
            "chunks": len(chunks),
            "game": game_name,
        }
    except Exception as e:
        return {"status": "failed", "error": str(e), "chunks": 0, "game": game_name}


# ── 反馈学习:处理用户未答上的问题 ─────────────────────────────────

async def process_pending_queries() -> dict:
    """Process all pending user questions: group by game and regenerate knowledge.

    For each game that has pending questions, the LLM is asked to generate
    knowledge that specifically answers those questions.
    """
    pending = db.list_pending_queries(limit=100, status="pending")
    if not pending:
        return {"processed": 0, "games": [], "total_questions": 0}

    # Group pending questions by game
    by_game: dict[str, list[str]] = defaultdict(list)
    for q in pending:
        by_game[q["game_name"]].append(q["question"])

    processed_games = []
    for game_name, questions in by_game.items():
        result = await fetch_and_store_game_knowledge(
            game_name, specific_questions=questions, replace=True
        )
        if result["status"] == "completed":
            # Mark all pending questions for this game as resolved
            db.resolve_pending_queries_by_game(game_name)
            processed_games.append({
                "game": game_name,
                "chunks": result["chunks"],
                "questions_resolved": len(questions),
            })
        else:
            processed_games.append({
                "game": game_name,
                "chunks": 0,
                "questions_resolved": 0,
                "error": result.get("error"),
            })

    return {
        "processed": len(processed_games),
        "games": processed_games,
        "total_questions": len(pending),
    }


# ── 热门游戏批量获取 ───────────────────────────────────────────────

async def refresh_trending_games(count: int = 10, force: bool = False) -> dict:
    """Discover trending games and fetch knowledge for each.

    Args:
        count: Number of trending games to discover.
        force: If True, re-fetch even for games that already have knowledge.
    """
    games = await discover_trending_games(count)
    if not games:
        return {"discovered": 0, "results": []}

    results = []
    for game_name in games:
        existing = db.get_document_count_for_game(game_name)
        if existing > 0 and not force:
            results.append({
                "game": game_name,
                "status": "skipped",
                "chunks": existing,
                "reason": "已有知识",
            })
            continue
        result = await fetch_and_store_game_knowledge(game_name, replace=True)
        results.append({
            "game": game_name,
            "status": result["status"],
            "chunks": result["chunks"],
            "error": result.get("error"),
        })
    return {"discovered": len(games), "results": results}


# ── 完整知识获取周期 ───────────────────────────────────────────────

async def run_knowledge_cycle() -> dict:
    """Full knowledge acquisition cycle — called by the scheduler.

    Step 1: Process pending user questions (feedback learning, priority).
    Step 2: Refresh trending games (fill in new popular games).
    """
    print("[knowledge] Starting knowledge acquisition cycle...")

    # Step 1: Process pending queries first (priority)
    pending_result = await process_pending_queries()
    print(f"[knowledge] Pending queries processed: {pending_result['processed']} games")

    # Step 2: Refresh trending games (don't force — skip games we already cover)
    trending_result = await refresh_trending_games(
        count=settings.trending_game_count, force=False
    )
    fetched = len([r for r in trending_result.get("results", []) if r["status"] == "completed"])
    print(f"[knowledge] Trending games fetched: {fetched}")

    # Log the cycle
    db.add_knowledge_log(
        action="knowledge_cycle",
        pending_processed=pending_result["processed"],
        trending_fetched=fetched,
        games_detail=json.dumps(
            {
                "pending_games": pending_result["games"],
                "trending_results": trending_result.get("results", []),
            },
            ensure_ascii=False,
        ),
        message=f"处理 {pending_result['processed']} 个游戏(反馈学习),"
                f"获取 {fetched} 个新热门游戏知识",
    )

    return {
        "pending": pending_result,
        "trending": trending_result,
    }
