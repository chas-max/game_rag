"""构建 RAG 评测测试集 - 从知识库自动生成「问题 + 参考答案 + 相关文档」。

流程: 对每个游戏按 chunk 顺序分层抽样若干 parent chunk,用 LLM 基于该 chunk 生成
一个能被其直接回答的问题与参考答案,以该 chunk 的 document id 作为 ground truth
相关文档(relevant_doc_ids)。生成一次后保存为 JSON 固定复用,保证不同版本用同一份题公平对比。

CLI:
    python -m eval.benchmark --build [--per-game 5] [--games 原神,塞尔达传说] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime

# Windows 控制台 UTF-8,避免中文 mojibake(与 main.py 一致)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import dotenv

# 必须在导入 app/config 之前加载 .env: config.Settings 是冻结 dataclass,
# 在模块定义时一次性读取 os.getenv,若此时 .env 未加载则 OPENAI_API_KEY 为空 -> 401。
dotenv.load_dotenv(override=True)

from app import database as db
from app.rag_pipeline import _llm_chat
from eval.metrics import _extract_json_block

DEFAULT_BENCHMARK_PATH = os.path.join("eval", "benchmark.json")
MIN_CHUNK_LEN = 100          # 过短的 chunk 难以生成好问题,跳过
LLM_CONCURRENCY = 4          # 并发生成问题,避免压垮 DeepSeek


def _select_chunks(chunks: list[dict], per_game: int) -> list[dict]:
    """在按 chunk_index 排序的 chunk 列表上等距分层抽样 per_game 个,跳过过短 chunk。"""
    usable = [c for c in chunks if len(c.get("content") or "") >= MIN_CHUNK_LEN]
    if len(usable) <= per_game:
        return usable
    step = len(usable) / per_game
    idxs = [round(i * step) for i in range(per_game)]
    # 去重保序
    seen = set()
    picked = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            picked.append(usable[i])
    return picked


def _build_question_prompt(game_name: str, content: str) -> str:
    return f"""阅读以下游戏《{game_name}》的知识片段,生成一个能被该片段直接回答的具体问题,以及基于该片段的参考答案。

【知识片段】
{content}

要求:
1. 问题必须能被该片段直接回答,不要问片段外的内容
2. 问题要具体、有信息量(避免"这个游戏是什么"之类过于宽泛)
3. 参考答案简洁准确,仅基于片段内容,不超过 150 字

只输出一行 JSON,不要任何解释,格式示例:
{{"question": "...", "reference_answer": "..."}}"""


async def _gen_qa(game_name: str, content: str) -> dict | None:
    """对单个 chunk 调 LLM 生成 {question, reference_answer},失败返回 None。"""
    raw = await _llm_chat(_build_question_prompt(game_name, content), max_tokens=300, temperature=0.4)
    data = _extract_json_block(raw) or {}
    q = (data.get("question") or "").strip()
    a = (data.get("reference_answer") or "").strip()
    if not q or not a:
        return None
    return {"question": q, "reference_answer": a}


async def build_benchmark(
    games: list[str] | None = None,
    per_game: int = 5,
    out_path: str = DEFAULT_BENCHMARK_PATH,
    force: bool = False,
) -> str:
    """构建测试集并保存为 JSON,返回保存路径。

    Args:
        games: 指定游戏名列表;None 表示全部游戏。
        per_game: 每个游戏抽样多少个 chunk 生成问题。
        out_path: 输出 JSON 路径。
        force: 已存在时是否覆盖(默认不覆盖,保护版本对比的题集稳定性)。
    """
    if os.path.exists(out_path) and not force:
        raise SystemExit(
            f"[benchmark] {out_path} 已存在。版本对比需保持题集不变;如确需重建请加 --force。"
        )

    db.init_db()
    all_games = db.list_games()
    if not all_games:
        raise SystemExit("[benchmark] 知识库为空,请先抓取知识后再构建测试集。")

    target = []
    for g in all_games:
        name = g["game_name"]
        if games and name not in games:
            continue
        target.append(name)
    if not target:
        raise SystemExit(f"[benchmark] 未找到指定游戏: {games}")

    print(f"[benchmark] 目标游戏 {len(target)} 个,每个抽样 {per_game} 个 chunk ...")

    # 收集候选 chunk
    candidates: list[dict] = []  # {game_name, source_doc_id, content}
    for name in target:
        chunks = db.get_documents_by_game(name)  # 已按 chunk_index 排序
        picked = _select_chunks(chunks, per_game)
        for c in picked:
            candidates.append({
                "game_name": name,
                "source_doc_id": c["id"],
                "content": c["content"],
            })
    print(f"[benchmark] 候选 chunk {len(candidates)} 个,开始用 LLM 生成问题 ...")

    # 并发生成
    sem = asyncio.Semaphore(LLM_CONCURRENCY)

    async def _bounded(cand: dict) -> dict | None:
        async with sem:
            qa = await _gen_qa(cand["game_name"], cand["content"])
            if qa is None:
                return None
            return {
                "id": f"{cand['game_name']}#{cand['source_doc_id']}",
                "game_name": cand["game_name"],
                "question": qa["question"],
                "reference_answer": qa["reference_answer"],
                "source_doc_id": cand["source_doc_id"],
                "relevant_doc_ids": [cand["source_doc_id"]],
                "source_preview": cand["content"][:120].replace("\n", " "),
            }

    results = await asyncio.gather(*[_bounded(c) for c in candidates])
    items = [r for r in results if r is not None]
    failed = len(candidates) - len(items)
    print(f"[benchmark] 生成成功 {len(items)} 条,失败 {failed} 条。")

    if not items:
        raise SystemExit("[benchmark] 全部生成失败,请检查 LLM 配置(.env)与网络。")

    benchmark_id = hashlib.sha256(
        json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    payload = {
        "benchmark_id": benchmark_id,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": _embedding_model_name(),
        "per_game": per_game,
        "n_items": len(items),
        "items": items,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[benchmark] 已保存 -> {out_path}  (benchmark_id={benchmark_id})")
    return out_path


def _embedding_model_name() -> str:
    try:
        from config import settings

        return settings.embedding_model
    except Exception:
        return ""


def load_benchmark(path: str = DEFAULT_BENCHMARK_PATH) -> dict:
    """加载测试集 JSON。"""
    if not os.path.exists(path):
        raise SystemExit(
            f"[eval] 测试集不存在: {path}\n请先构建: python -m eval.benchmark --build"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="构建 RAG 评测测试集")
    p.add_argument("--build", action="store_true", help="构建测试集")
    p.add_argument("--per-game", type=int, default=5, help="每个游戏抽样 chunk 数(默认 5)")
    p.add_argument("--games", type=str, default="", help="指定游戏,逗号分隔;留空=全部")
    p.add_argument("--out", type=str, default=DEFAULT_BENCHMARK_PATH, help="输出 JSON 路径")
    p.add_argument("--force", action="store_true", help="已存在时覆盖")
    args = p.parse_args()

    if not args.build:
        p.print_help()
        return

    games = [g.strip() for g in args.games.split(",") if g.strip()] or None
    asyncio.run(build_benchmark(games=games, per_game=args.per_game, out_path=args.out, force=args.force))


if __name__ == "__main__":
    main()
