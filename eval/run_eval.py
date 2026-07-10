"""RAG 评测入口 - 跑测试集、算指标、追加到统一 Excel。

每次运行以追加形式写入 eval/rag_metrics.xlsx 的 runs/per_query 两个 sheet,
并带版本标记(run_id/version_label/git_commit/config_snapshot 等),
便于区分不同项目版本(基线 vs 优化后)的指标。

CLI:
    python -m eval.run_eval --version v1-baseline [--note "..."]
        [--benchmark eval/benchmark.json] [--out eval/rag_metrics.xlsx]
        [--with-generation] [--games 原神,塞尔达传说] [--limit N] [--concurrency 4]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from statistics import mean

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import dotenv

dotenv.load_dotenv(override=True)

from app import database as db
from app.rag_pipeline import PIPELINE_FEATURES, generate_final_answer, retrieve_documents
from config import settings
from eval.benchmark import DEFAULT_BENCHMARK_PATH, load_benchmark
from eval.excel_logger import log_run
from eval.metrics import (
    judge_generation,
    recall_at_k,
)

DEFAULT_OUT = os.path.join("eval", "rag_metrics.xlsx")


# ── 版本标记采集 ────────────────────────────────────────────────────

def _git_info() -> tuple[str, bool]:
    """返回 (短 commit, 是否有未提交改动)。非 git 仓库或无 git 时返回 ("", False)。"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = ""
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
        )
    except Exception:
        dirty = False
    return commit, dirty


def _config_snapshot() -> dict:
    """采集当前配置 + 管线特性,作为版本标记的一部分。"""
    return {
        "top_k": settings.top_k,
        "similarity_threshold": settings.similarity_threshold,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        **PIPELINE_FEATURES,
    }


# ── 单条评测 ────────────────────────────────────────────────────────

def _retrieval_metrics(retrieved: list[dict], relevant_ids: list[int], topk: int) -> dict:
    """计算单条问题的检索指标。"""
    rids = [str(d["id"]) for d in retrieved]
    return {
        "recall": recall_at_k(rids, relevant_ids, topk),
    }


async def _eval_one(item: dict, topk: int, with_generation: bool) -> dict:
    """评测单条问题: 检索 (+可选生成),返回含全部指标的明细行。"""
    game = item["game_name"]
    question = item["question"]
    relevant = [str(x) for x in item.get("relevant_doc_ids", [])]

    t0 = time.perf_counter()
    retrieved = await retrieve_documents(game, question, progress_callback=None, verbose=False)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

    row = {
        "query_id": item.get("id", question[:20]),
        "game_name": game,
        "question": question,
        **_retrieval_metrics(retrieved, relevant, topk),
    }

    if with_generation and retrieved:
        context = "\n\n".join(d.get("content", "") for d in retrieved)
        t1 = time.perf_counter()
        answer = await generate_final_answer(question, retrieved, history_msgs=[])
        gen_ms = round((time.perf_counter() - t1) * 1000, 1)
        judged = await judge_generation(question, answer, item.get("reference_answer", ""), context)
        row.update({
            "accuracy": judged["correctness"],
            "faithfulness": judged["faithfulness"],
            "relevance": judged["relevance"],
            "latency_ms": round(retrieval_ms + gen_ms, 1),
        })
    else:
        row.update({
            "accuracy": None,
            "faithfulness": None,
            "relevance": None,
            "latency_ms": retrieval_ms,
        })

    return row


# ── 聚合 ────────────────────────────────────────────────────────────

def _mean_skip_none(values: list) -> float | None:
    nums = [v for v in values if v is not None]
    return round(mean(nums), 4) if nums else None


def _aggregate(rows: list[dict], with_generation: bool, total_elapsed: float) -> dict:
    """把单条明细聚合为 runs sheet 的汇总指标。"""
    n = len(rows)
    qps = round(n / total_elapsed, 2) if total_elapsed > 0 else 0.0
    summary = {
        "n_queries": n,
        "recall": _mean_skip_none([r.get("recall") for r in rows]),
        "throughput": qps,
    }
    if with_generation:
        summary.update({
            "accuracy": _mean_skip_none([r.get("accuracy") for r in rows]),
            "faithfulness": _mean_skip_none([r.get("faithfulness") for r in rows]),
            "relevance": _mean_skip_none([r.get("relevance") for r in rows]),
        })
    else:
        summary.update({
            "accuracy": None,
            "faithfulness": None,
            "relevance": None,
        })
    return summary


# ── 主流程 ──────────────────────────────────────────────────────────

async def run(args) -> None:
    db.init_db()
    bench = load_benchmark(args.benchmark)
    items = bench["items"]

    # 过滤
    if args.games:
        wanted = {g.strip() for g in args.games.split(",") if g.strip()}
        items = [it for it in items if it["game_name"] in wanted]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("[eval] 过滤后无评测问题,检查 --games/--limit 与测试集。")

    topk = settings.top_k
    print(f"[eval] 版本: {args.version}")
    print(f"[eval] 测试集: {args.benchmark} (benchmark_id={bench.get('benchmark_id')}, {len(items)} 条)")
    print(f"[eval] 配置: top_k={topk}, threshold={settings.similarity_threshold}, "
          f"emb={settings.embedding_model}, hyde={PIPELINE_FEATURES['hyde']}")
    print(f"[eval] 生成指标: {'开启(--with-generation)' if args.with_generation else '关闭'}")
    print(f"[eval] 并发: {args.concurrency}  开始评测 ...\n")

    sem = asyncio.Semaphore(args.concurrency)

    async def _bounded(item):
        async with sem:
            return await _eval_one(item, topk, args.with_generation)

    t_start = time.perf_counter()
    rows = await asyncio.gather(*[_bounded(it) for it in items])
    t_end = time.perf_counter()

    summary = _aggregate(rows, args.with_generation, t_end - t_start)

    # 版本标记
    commit, dirty = _git_info()
    run_meta = {
        "run_id": uuid.uuid4().hex[:8],
        "version_label": args.version,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit,
        "git_dirty": dirty,
        "benchmark_id": bench.get("benchmark_id", ""),
        "config_snapshot": _config_snapshot(),
        "note": args.note or "",
        "with_generation": args.with_generation,
    }

    log_run(args.out, run_meta, summary, rows)
    _print_summary(summary, run_meta, args.out)


def _print_summary(summary: dict, run_meta: dict, out_path: str) -> None:
    """控制台打印关键指标汇总表。"""
    print("=" * 56)
    print(f"评测完成  version={run_meta['version_label']}  run_id={run_meta['run_id']}")
    print(f"git={run_meta['git_commit']}{'(dirty)' if run_meta['git_dirty'] else ''}  "
          f"benchmark={run_meta['benchmark_id']}")
    print("-" * 56)
    print(f"{'指标':<24}{'值':>14}")
    print("-" * 56)
    for key in [
        "n_queries", "accuracy", "recall", "throughput", "faithfulness", "relevance",
    ]:
        if key in summary and summary[key] is not None:
            val = summary[key]
            shown = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"{key:<24}{shown:>14}")
    print("-" * 56)
    print(f"已追加 -> {out_path}  (runs / per_query 两个 sheet)\n")


def main() -> None:
    p = argparse.ArgumentParser(description="跑 RAG 评测并把指标追加到统一 Excel")
    p.add_argument("--version", default="unlabeled", help="版本标记,如 v1-baseline / v2-hyde")
    p.add_argument("--note", default="", help="本次评测备注")
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK_PATH, help="测试集 JSON 路径")
    p.add_argument("--out", default=DEFAULT_OUT, help="输出 Excel 路径")
    p.add_argument("--with-generation", action="store_true", help="额外跑 LLM 裁判生成指标(耗 token)")
    p.add_argument("--games", default="", help="只评测指定游戏,逗号分隔;留空=全部")
    p.add_argument("--limit", type=int, default=0, help="只评测前 N 条(0=不限)")
    p.add_argument("--concurrency", type=int, default=4, help="并发数(默认 4)")
    args = p.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
