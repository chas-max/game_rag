"""Persistence layer.

关系数据(conversations / messages / scraping_tasks / pending_queries / knowledge_logs)
存 SQLite;向量数据(documents / semantic_chunks 的 embedding 与相似度检索)存 ChromaDB,
委托 app.vector_store。对外接口与返回结构保持不变,上层无感。
"""

import json
import os
import sqlite3
from typing import Any

import numpy as np

from app import vector_store as vs
from config import settings

_db: sqlite3.Connection | None = None

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT 'New Conversation',
    game_name TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT (datetime('now','localtime')),
    updated_at TIMESTAMP DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
    content TEXT NOT NULL,
    sources TEXT,
    created_at TIMESTAMP DEFAULT (datetime('now','localtime')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scraping_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_name TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_url TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending','running','completed','failed')),
    interval_hours INTEGER DEFAULT 24,
    last_run TIMESTAMP,
    next_run TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT (datetime('now','localtime'))
);

-- 用户提问但知识库未命中的问题,下次知识获取时优先处理
CREATE TABLE IF NOT EXISTS pending_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_name TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending','resolved')),
    created_at TIMESTAMP DEFAULT (datetime('now','localtime')),
    resolved_at TIMESTAMP
);

-- 知识获取任务执行日志
CREATE TABLE IF NOT EXISTS knowledge_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    pending_processed INTEGER DEFAULT 0,
    trending_fetched INTEGER DEFAULT 0,
    games_detail TEXT,
    message TEXT,
    created_at TIMESTAMP DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_scraping_tasks_game ON scraping_tasks(game_name);
CREATE INDEX IF NOT EXISTS idx_pending_queries_status ON pending_queries(status, created_at);
CREATE INDEX IF NOT EXISTS idx_pending_queries_game ON pending_queries(game_name);

-- 注: documents / semantic_chunks 的向量与正文已迁至 ChromaDB(app.vector_store)。
-- 旧 SQLite 库中若仍存在这两张表,不影响新流程(保留作备份,可手动 DROP)。
"""


def get_db() -> sqlite3.Connection:
    """Return a thread-local database connection, initializing it on first access."""
    global _db
    if _db is None:
        os.makedirs(os.path.dirname(settings.database_path), exist_ok=True)
        _db = sqlite3.connect(settings.database_path, check_same_thread=False)
        _db.row_factory = sqlite3.Row
        _db.executescript(DDL)
    return _db


def init_db() -> None:
    """Explicitly initialize the database (called at startup)."""
    get_db()


# ── Conversation CRUD ──────────────────────────────────────────────

def create_conversation(game_name: str, title: str = "New Conversation") -> dict:
    db = get_db()
    cur = db.execute(
        "INSERT INTO conversations (title, game_name) VALUES (?, ?)",
        (title, game_name),
    )
    db.commit()
    row = db.execute("SELECT * FROM conversations WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def list_conversations() -> list[dict]:
    db = get_db()
    rows = db.execute(
        """
        SELECT c.*, COUNT(m.id) AS message_count
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: int) -> dict | None:
    db = get_db()
    row = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    msgs = db.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conv_id,),
    ).fetchall()
    result["messages"] = [dict(m) for m in msgs]
    return result


def update_conversation(conv_id: int, title: str) -> dict | None:
    db = get_db()
    db.execute(
        "UPDATE conversations SET title = ?, updated_at = datetime('now','localtime') WHERE id = ?",
        (title, conv_id),
    )
    db.commit()
    row = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    return dict(row) if row else None


def update_conversation_game_name(conv_id: int, game_name: str) -> None:
    db = get_db()
    db.execute(
        "UPDATE conversations SET game_name = ?, updated_at = datetime('now','localtime') WHERE id = ?",
        (game_name, conv_id),
    )
    db.commit()


def update_conversation_timestamp(conv_id: int) -> None:
    db = get_db()
    db.execute(
        "UPDATE conversations SET updated_at = datetime('now','localtime') WHERE id = ?",
        (conv_id,),
    )
    db.commit()


def update_conversation_game(conv_id: int, game_name: str) -> None:
    db = get_db()
    db.execute(
        "UPDATE conversations SET game_name = ?, updated_at = datetime('now','localtime') WHERE id = ?",
        (game_name, conv_id),
    )
    db.commit()


def delete_conversation(conv_id: int) -> bool:
    db = get_db()
    cur = db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    db.commit()
    return cur.rowcount > 0


# ── Message CRUD ───────────────────────────────────────────────────

def save_message(conv_id: int, role: str, content: str, sources: str | None = None) -> dict:
    db = get_db()
    cur = db.execute(
        "INSERT INTO messages (conversation_id, role, content, sources) VALUES (?, ?, ?, ?)",
        (conv_id, role, content, sources),
    )
    db.commit()
    row = db.execute("SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def get_messages(conv_id: int, limit: int = 10) -> list[dict]:
    db = get_db()
    rows = db.execute(
        """
        SELECT * FROM (
            SELECT * FROM messages WHERE conversation_id = ?
            ORDER BY created_at DESC LIMIT ?
        ) ORDER BY created_at ASC
        """,
        (conv_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Document CRUD (委托 ChromaDB) ──────────────────────────────────

def insert_document(
    game_name: str,
    content: str,
    embedding: np.ndarray,
    title: str | None = None,
    url: str | None = None,
    source_name: str | None = None,
    chunk_index: int = 0,
) -> str:
    """存储单个 parent chunk 到 ChromaDB,返回 doc id(字符串)。"""
    return vs.insert_document(
        game_name, content, embedding, title=title, url=url, source_name=source_name, chunk_index=chunk_index
    )


async def store_documents_with_semantic_chunks(
    game_name: str,
    chunks: list[str],
    embeddings: list[np.ndarray],
    title: str | None = None,
    url: str | None = None,
    source_name: str | None = None,
) -> int:
    """
    批量存储固定长度分块(documents)及其对应的所有语义句切分块(semantic_chunks)到 ChromaDB。
    在内部对所有切分的单句进行批量向量嵌入,提升导入性能。
    """
    import asyncio
    from app.embedding import encode_batch
    from app.scraper import parse_chunk_sentences

    # 1. 批量存储 Parent Chunks,拿到 doc id 列表(chunk_index = 位置序号)
    doc_ids = vs.add_documents(
        game_name, chunks, embeddings, title=title, url=url, source_name=source_name
    )

    # 2. 从每个 Parent Chunk 解析语义句子,绑定其 doc id
    all_sentences = []
    for chunk, doc_id in zip(chunks, doc_ids):
        for s in parse_chunk_sentences(chunk):
            s["document_id"] = doc_id
            all_sentences.append(s)

    # 3. 批量生成语义句 Embedding 并写入 semantic_chunks collection
    if all_sentences:
        loop = asyncio.get_running_loop()
        sentence_texts = [s["content"] for s in all_sentences]
        sentence_embs = await loop.run_in_executor(None, encode_batch, sentence_texts)
        vs.add_semantics(game_name, all_sentences, sentence_embs)

    return len(chunks)


def get_documents_by_game(game_name: str) -> list[dict]:
    """Return metadata-only document rows (no embedding) for a game."""
    return vs.get_documents_by_game(game_name)


def delete_documents_by_game(game_name: str) -> int:
    return vs.delete_documents_by_game(game_name)


def delete_documents_by_url(url: str) -> int:
    return vs.delete_documents_by_url(url)


def list_games() -> list[dict]:
    return vs.list_games()


def get_document_count_for_game(game_name: str) -> int:
    """Return the number of document chunks stored for a game."""
    return vs.get_document_count_for_game(game_name)


def get_knowledge_stats() -> dict:
    """Return aggregate statistics about the knowledge base."""
    games = vs.list_games()
    total_docs = vs.count_documents()
    db = get_db()
    pending = db.execute(
        "SELECT COUNT(*) AS cnt FROM pending_queries WHERE status = 'pending'"
    ).fetchone()["cnt"]
    return {
        "total_games": len(games),
        "total_documents": total_docs,
        "pending_queries": pending,
        "games": games,
    }


# ── Scraping Task CRUD ─────────────────────────────────────────────

def create_scraping_task(
    game_name: str,
    source_name: str,
    source_url: str,
    interval_hours: int = 24,
) -> dict:
    db = get_db()
    cur = db.execute(
        """INSERT INTO scraping_tasks (game_name, source_name, source_url, interval_hours)
           VALUES (?, ?, ?, ?)""",
        (game_name, source_name, source_url, interval_hours),
    )
    db.commit()
    row = db.execute("SELECT * FROM scraping_tasks WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def list_scraping_tasks() -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT * FROM scraping_tasks ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_scraping_task(task_id: int) -> dict | None:
    db = get_db()
    row = db.execute("SELECT * FROM scraping_tasks WHERE id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


def update_scraping_task_status(
    task_id: int,
    status: str,
    error_message: str | None = None,
    next_run: str | None = None,
) -> None:
    db = get_db()
    db.execute(
        """UPDATE scraping_tasks
           SET status = ?, last_run = datetime('now','localtime'),
               error_message = ?, next_run = ?
           WHERE id = ?""",
        (status, error_message, next_run, task_id),
    )
    db.commit()


def delete_scraping_task(task_id: int) -> bool:
    db = get_db()
    cur = db.execute("DELETE FROM scraping_tasks WHERE id = ?", (task_id,))
    db.commit()
    return cur.rowcount > 0


# ── Pending Queries CRUD (用户未答上的问题) ────────────────────────

def add_pending_query(game_name: str, question: str) -> int:
    """Record a user question that the knowledge base could not answer."""
    db = get_db()
    # Avoid duplicate pending questions for the same game
    existing = db.execute(
        "SELECT id FROM pending_queries WHERE game_name = ? AND question = ? AND status = 'pending'",
        (game_name, question),
    ).fetchone()
    if existing:
        return existing["id"]
    cur = db.execute(
        "INSERT INTO pending_queries (game_name, question) VALUES (?, ?)",
        (game_name, question),
    )
    db.commit()
    return cur.lastrowid


def list_pending_queries(limit: int = 100, status: str = "pending") -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT * FROM pending_queries WHERE status = ? ORDER BY created_at DESC LIMIT ?",
        (status, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def list_pending_queries_by_game(game_name: str, status: str = "pending") -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT * FROM pending_queries WHERE game_name = ? AND status = ? ORDER BY created_at ASC",
        (game_name, status),
    ).fetchall()
    return [dict(r) for r in rows]


def resolve_pending_queries_by_game(game_name: str) -> int:
    """Mark all pending questions for a game as resolved."""
    db = get_db()
    cur = db.execute(
        "UPDATE pending_queries SET status = 'resolved', resolved_at = datetime('now','localtime') "
        "WHERE game_name = ? AND status = 'pending'",
        (game_name,),
    )
    db.commit()
    return cur.rowcount


def delete_pending_query(query_id: int) -> bool:
    db = get_db()
    cur = db.execute("DELETE FROM pending_queries WHERE id = ?", (query_id,))
    db.commit()
    return cur.rowcount > 0


# ── Knowledge Logs CRUD (知识获取日志) ─────────────────────────────

def add_knowledge_log(
    action: str,
    pending_processed: int = 0,
    trending_fetched: int = 0,
    games_detail: str | None = None,
    message: str | None = None,
) -> int:
    db = get_db()
    cur = db.execute(
        """INSERT INTO knowledge_logs
           (action, pending_processed, trending_fetched, games_detail, message)
           VALUES (?, ?, ?, ?, ?)""",
        (action, pending_processed, trending_fetched, games_detail, message),
    )
    db.commit()
    return cur.lastrowid


def list_knowledge_logs(limit: int = 20) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT * FROM knowledge_logs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# ── Semantic Chunks: 向量检索与上下文窗口 (委托 ChromaDB) ──────────

def insert_semantic_chunk(
    document_id: str,
    game_name: str,
    content: str,
    embedding: np.ndarray,
    tag: str | None = None,
) -> str:
    """存储单条语义句到 ChromaDB,返回 sem id。document_id 为 parent doc 的字符串 id。"""
    return vs.insert_semantic_chunk(document_id, game_name, content, embedding, tag=tag)


def search_similar_semantic(
    query_embedding: np.ndarray,
    game_name: str,
    top_k: int = 5,
    threshold: float | None = None,
) -> list[dict]:
    """
    在 semantic_chunks collection 中按向量相似度检索,filtered by game_name。
    返回 [{id, document_id, content, tag, similarity}],与旧 numpy 实现结构一致。
    """
    return vs.search_similar_semantic(query_embedding, game_name, top_k=top_k, threshold=threshold)


def get_document_with_context(doc_id) -> list[dict]:
    """
    Retrieve the specified parent document along with its immediate surrounding context
    (previous and next chunks of the same game based on chunk_index).
    """
    return vs.get_document_with_context(doc_id)
