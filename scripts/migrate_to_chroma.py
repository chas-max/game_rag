"""迁移脚本: 把 SQLite 中 documents / semantic_chunks 表(含 embedding BLOB)导入 ChromaDB。

迁移后向量与正文的存储/检索完全由 ChromaDB 接管,SQLite 仅保留关系数据
(conversations/messages/scraping_tasks/pending_queries/knowledge_logs)。
旧表保留作备份,不删除。

用法:
    python scripts/migrate_to_chroma.py

注意:迁移只搬运已有向量(不重新调 LLM / 不加载 embedding 模型),速度快。
迁移前若 ChromaDB 已有数据会先清空两个 collection(reset_collections)再写入。
"""

from __future__ import annotations

import os
import sqlite3
import sys

# 把项目根目录加入 sys.path(以脚本路径运行时,默认只加 scripts/ 到 path)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Windows 控制台 UTF-8 + 必须在导入 app/config 前加载 .env
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import dotenv

dotenv.load_dotenv(override=True)

import numpy as np

from app import vector_store as vs
from config import settings

BATCH = 500


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _blob_to_list(blob) -> list[float] | None:
    """把 SQLite embedding BLOB 还原为 float 列表;非法/空则返回 None。"""
    if not blob or len(blob) < 4:
        return None
    return np.frombuffer(blob, dtype=np.float32).tolist()


def _empty(v) -> str:
    return v if v is not None else ""


def _table_exists(conn, name) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def migrate() -> None:
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row

    if not _table_exists(conn, "documents") or not _table_exists(conn, "semantic_chunks"):
        print("[migrate] SQLite 中未找到 documents/semantic_chunks 表,无需迁移(可能已迁移过)。")
        conn.close()
        return

    n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    n_sems = conn.execute("SELECT COUNT(*) FROM semantic_chunks").fetchone()[0]
    print(f"[migrate] SQLite 源: documents={n_docs}, semantic_chunks={n_sems}")

    # 清场后重建两个 collection,确保干净
    vs.reset_collections()
    docs_coll = vs._docs_collection()
    sems_coll = vs._sem_collection()

    # ── documents ──────────────────────────────────────────────────
    done = skipped = 0
    rows = conn.execute(
        "SELECT id, game_name, title, url, source_name, content, embedding, chunk_index, created_at "
        "FROM documents"
    ).fetchall()
    for batch in _chunked(rows, BATCH):
        ids, embs, docs, metas = [], [], [], []
        for r in batch:
            emb = _blob_to_list(r["embedding"])
            if emb is None:
                skipped += 1
                continue
            ids.append(f"doc_{r['id']}")
            embs.append(emb)
            docs.append(r["content"])
            metas.append({
                "game_name": r["game_name"],
                "title": _empty(r["title"]),
                "url": _empty(r["url"]),
                "source_name": _empty(r["source_name"]),
                "chunk_index": int(r["chunk_index"]) if r["chunk_index"] is not None else 0,
                "created_at": _empty(r["created_at"]),
            })
        if ids:
            docs_coll.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        done += len(batch)
        print(f"[migrate] documents: {done}/{n_docs} (跳过无效 {skipped})")
    print(f"[migrate] documents 写入完成, ChromaDB count = {docs_coll.count()}")

    # ── semantic_chunks ────────────────────────────────────────────
    done = skipped = 0
    rows = conn.execute(
        "SELECT id, document_id, game_name, content, embedding, tag, created_at FROM semantic_chunks"
    ).fetchall()
    for batch in _chunked(rows, BATCH):
        ids, embs, docs, metas = [], [], [], []
        for r in batch:
            emb = _blob_to_list(r["embedding"])
            if emb is None:
                skipped += 1
                continue
            ids.append(f"sem_{r['id']}")
            embs.append(emb)
            docs.append(r["content"])
            metas.append({
                "document_id": f"doc_{r['document_id']}",
                "game_name": r["game_name"],
                "tag": _empty(r["tag"]),
                "created_at": _empty(r["created_at"]),
            })
        if ids:
            sems_coll.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        done += len(batch)
        print(f"[migrate] semantic_chunks: {done}/{n_sems} (跳过无效 {skipped})")
    print(f"[migrate] semantic_chunks 写入完成, ChromaDB count = {sems_coll.count()}")

    # ── 验证 ───────────────────────────────────────────────────────
    cd_docs = docs_coll.count()
    cd_sems = sems_coll.count()
    print("=" * 56)
    print(f"验证 documents:       SQLite {n_docs} -> ChromaDB {cd_docs}  "
          f"{'OK' if n_docs - skipped == cd_docs else 'MISMATCH!'}")
    print(f"验证 semantic_chunks: SQLite {n_sems} -> ChromaDB {cd_sems}  "
          f"{'OK' if n_sems - skipped == cd_sems else 'MISMATCH!'}")
    print("迁移完成。旧 SQLite 表已保留作备份,可手动 DROP TABLE documents/semantic_chunks。")
    conn.close()


if __name__ == "__main__":
    migrate()
