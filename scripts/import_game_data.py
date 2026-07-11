"""导入 game_rag_data 文件夹中的游戏数据到 ChromaDB。

数据来源:
1. D:/OneDrive/桌面/game_rag_data/output_rag/全部游戏_知识库.jsonl (结构化数据)
2. D:/OneDrive/桌面/game_rag_data/{game}/ 子文件夹中的 .txt 文件 (详细角色/剧情数据)

流程:
1. 读取 JSONL 文件和子文件夹中的 txt 文件
2. 使用 chunk_text 分块(复用 scraper.py 的分块逻辑)
3. 使用 sentence-transformers 生成 embedding
4. 存入 ChromaDB 的 documents 和 semantic_chunks collection

用法:
    python scripts/import_game_data.py [--dry-run] [--game 原神]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

# 项目根目录加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Windows 控制台 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import dotenv
dotenv.load_dotenv(override=True)

from app.embedding import encode_batch
from app.scraper import chunk_text, parse_chunk_sentences
from app import vector_store as vs
from app import database as db

# 数据文件路径
DATA_ROOT = r"D:/OneDrive/桌面/game_rag_data"
DATA_FILE = os.path.join(DATA_ROOT, "output_rag/全部游戏_知识库.jsonl")

# 游戏文件夹名 -> 系统游戏名映射
GAME_FOLDER_MAP = {
    "genshin": "原神",
    "star_rail": "崩坏星穹铁道",
    "zzz": "绝区零",
}

# 游戏名映射(确保与系统中一致)
GAME_NAME_MAP = {
    "原神": "原神",
    "崩坏星穹铁道": "崩坏星穹铁道",
    "绝区零": "绝区零",
}


def load_jsonl(path: str, game_filter: str | None = None) -> list[dict]:
    """读取 JSONL 文件,可选按游戏名过滤。"""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                game = obj.get("game", "")
                if game_filter and game != game_filter:
                    continue
                entries.append(obj)
            except json.JSONDecodeError:
                continue
    return entries


def load_txt_files(game_filter: str | None = None) -> list[dict]:
    """读取子文件夹中的 txt 文件,返回与 JSONL 格式兼容的条目列表。

    目录结构: DATA_ROOT/{genshin,star_rail,zzz}/{characters,items,locations,lore,story}/*.txt
    跳过 *_合并.txt 文件(这些是合并文件,内容过大)。
    """
    entries = []

    for folder_name, game_name in GAME_FOLDER_MAP.items():
        if game_filter and game_name != game_filter:
            continue

        game_dir = os.path.join(DATA_ROOT, folder_name)
        if not os.path.isdir(game_dir):
            continue

        # 遍历子目录
        for category in os.listdir(game_dir):
            category_path = os.path.join(game_dir, category)
            if not os.path.isdir(category_path):
                continue

            for filename in os.listdir(category_path):
                if not filename.endswith(".txt"):
                    continue
                if filename.endswith("_合并.txt"):
                    continue

                filepath = os.path.join(category_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue

                if len(content) < 50:
                    continue

                title = filename.replace(".txt", "")
                entries.append({
                    "id": f"{folder_name}_{category}_{title}",
                    "title": title,
                    "source": "Bilibili Wiki (中文)",
                    "game": game_name,
                    "category": category,
                    "content": content,
                    "metadata": {"char_count": len(content)},
                })

    return entries


def import_entries(
    entries: list[dict],
    dry_run: bool = False,
    replace: bool = True,
) -> dict:
    """将 JSONL 条目导入 ChromaDB。

    Args:
        entries: JSONL 条目列表
        dry_run: 如果 True,只分析不写入
        replace: 如果 True,先删除同名游戏的旧数据

    Returns:
        统计信息 dict
    """
    # 按游戏分组
    by_game: dict[str, list[dict]] = {}
    for entry in entries:
        game = entry.get("game", "未知")
        by_game.setdefault(game, []).append(entry)

    stats = {"games": {}, "total_chunks": 0, "total_entries": 0}

    for game_name, game_entries in by_game.items():
        mapped_name = GAME_NAME_MAP.get(game_name, game_name)
        print(f"\n{'='*60}")
        print(f"处理游戏: {game_name} -> {mapped_name} ({len(game_entries)} 条目)")

        all_chunks = []
        all_chunk_meta = []  # 每个 chunk 对应的元数据

        for entry in game_entries:
            content = entry.get("content", "")
            title = entry.get("title", "")
            source = entry.get("source", "")
            category = entry.get("category", "")
            entry_id = entry.get("id", "")

            if not content or len(content) < 50:
                print(f"  跳过过短条目: {title} ({len(content)} chars)")
                continue

            # 分块
            chunks = chunk_text(content)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_chunk_meta.append({
                    "title": title,
                    "source": source,
                    "category": category,
                    "entry_id": entry_id,
                })

        print(f"  总条目: {len(game_entries)}, 有效块: {len(all_chunks)}")

        if dry_run:
            print(f"  [DRY RUN] 跳过写入")
            stats["games"][mapped_name] = {
                "entries": len(game_entries),
                "chunks": len(all_chunks),
            }
            stats["total_chunks"] += len(all_chunks)
            stats["total_entries"] += len(game_entries)
            continue

        # 删除旧数据
        if replace:
            deleted = vs.delete_documents_by_game(mapped_name)
            print(f"  已删除旧数据: {deleted} 文档")

        # 批量生成 embedding
        print(f"  生成 embedding ({len(all_chunks)} 块)...")
        t0 = time.time()
        embeddings = encode_batch(all_chunks)
        t1 = time.time()
        print(f"  embedding 完成: {t1-t0:.1f}s")

        # 写入 documents collection
        print(f"  写入 ChromaDB documents...")
        doc_ids = vs.add_documents(
            game_name=mapped_name,
            chunks=all_chunks,
            embeddings=embeddings,
            title=f"{game_name} - Wiki 数据",
            url="",
            source_name="Bilibili Wiki (中文)",
        )

        # 生成 semantic chunks
        print(f"  生成语义块...")
        all_sentences = []
        for chunk, doc_id in zip(all_chunks, doc_ids):
            for s in parse_chunk_sentences(chunk):
                s["document_id"] = doc_id
                all_sentences.append(s)

        if all_sentences:
            print(f"  生成 semantic embedding ({len(all_sentences)} 句)...")
            sentence_texts = [s["content"] for s in all_sentences]
            sentence_embs = encode_batch(sentence_texts)
            vs.add_semantics(mapped_name, all_sentences, sentence_embs)

        t2 = time.time()
        print(f"  完成: {len(doc_ids)} 文档, {len(all_sentences)} 语义块, 总耗时 {t2-t0:.1f}s")

        stats["games"][mapped_name] = {
            "entries": len(game_entries),
            "chunks": len(doc_ids),
            "semantic_chunks": len(all_sentences),
        }
        stats["total_chunks"] += len(doc_ids)
        stats["total_entries"] += len(game_entries)

    return stats


def verify_import(game_names: list[str] | None = None) -> None:
    """验证导入结果。"""
    print(f"\n{'='*60}")
    print("验证导入结果")
    print(f"{'='*60}")

    games = vs.list_games()
    if not games:
        print("  ChromaDB 中没有游戏数据!")
        return

    for g in games:
        name = g["game_name"]
        count = g["document_count"]
        if game_names and name not in game_names:
            continue
        print(f"  {name}: {count} 文档")

    # 测试检索
    from app.embedding import encode_text
    print(f"\n测试检索:")
    test_queries = [
        ("原神", "钟离的技能是什么"),
        ("崩坏星穹铁道", "星穹列车的开拓者"),
        ("绝区零", "新艾利都有什么区域"),
    ]
    for game, query in test_queries:
        if game_names and game not in game_names:
            continue
        q_emb = encode_text(query)
        results = vs.search_similar_semantic(q_emb, game, top_k=3, threshold=0.2)
        print(f"\n  [{game}] 查询: {query}")
        if results:
            for r in results:
                content_preview = r["content"][:80].replace("\n", " ")
                print(f"    sim={r['similarity']:.3f} | {content_preview}...")
        else:
            print(f"    无结果!")


def main():
    parser = argparse.ArgumentParser(description="导入游戏数据到 ChromaDB")
    parser.add_argument("--dry-run", action="store_true", help="只分析不写入")
    parser.add_argument("--game", type=str, help="只导入指定游戏(如 '原神')")
    parser.add_argument("--no-replace", action="store_true", help="不删除旧数据")
    parser.add_argument("--verify-only", action="store_true", help="只验证不导入")
    parser.add_argument("--data-file", type=str, default=DATA_FILE, help="JSONL 数据文件路径")
    args = parser.parse_args()

    if args.verify_only:
        verify_import([args.game] if args.game else None)
        return

    print(f"数据文件: {args.data_file}")
    print(f"模式: {'DRY RUN' if args.dry_run else '正式导入'}")
    if args.game:
        print(f"游戏过滤: {args.game}")

    # 加载数据
    entries = load_jsonl(args.data_file, game_filter=args.game)
    print(f"JSONL 条目: {len(entries)}")

    # 加载子文件夹中的 txt 文件
    txt_entries = load_txt_files(game_filter=args.game)
    print(f"TXT 文件条目: {len(txt_entries)}")

    entries.extend(txt_entries)
    print(f"总条目: {len(entries)}")

    if not entries:
        print("没有找到数据!")
        return

    # 导入
    stats = import_entries(
        entries,
        dry_run=args.dry_run,
        replace=not args.no_replace,
    )

    # 输出统计
    print(f"\n{'='*60}")
    print("导入统计")
    print(f"{'='*60}")
    for game, info in stats["games"].items():
        print(f"  {game}: {info['entries']} 条目 -> {info['chunks']} 文档, {info.get('semantic_chunks', 0)} 语义块")
    print(f"  总计: {stats['total_entries']} 条目, {stats['total_chunks']} 文档")

    # 验证
    if not args.dry_run:
        verify_import([args.game] if args.game else None)


if __name__ == "__main__":
    main()
