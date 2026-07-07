"""SQLite persistence layer — schema, CRUD, and vector similarity search."""

import json
import os
import sqlite3
from typing import Any

import numpy as np

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

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_name TEXT NOT NULL,
    title TEXT,
    url TEXT,
    source_name TEXT,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT (datetime('now','localtime'))
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
CREATE INDEX IF NOT EXISTS idx_documents_game ON documents(game_name);
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
CREATE INDEX IF NOT EXISTS idx_scraping_tasks_game ON scraping_tasks(game_name);
CREATE INDEX IF NOT EXISTS idx_pending_queries_status ON pending_queries(status, created_at);
CREATE INDEX IF NOT EXISTS idx_pending_queries_game ON pending_queries(game_name);

-- 语义句子切割表，用于细粒度向量相似度匹配
CREATE TABLE IF NOT EXISTS semantic_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    game_name TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    tag TEXT,
    created_at TIMESTAMP DEFAULT (datetime('now','localtime')),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_semantic_chunks_doc ON semantic_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_semantic_chunks_game ON semantic_chunks(game_name);
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


def update_conversation_timestamp(conv_id: int) -> None:
    db = get_db()
    db.execute(
        "UPDATE conversations SET updated_at = datetime('now','localtime') WHERE id = ?",
        (conv_id,),
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


# ── Document CRUD ──────────────────────────────────────────────────

def insert_document(
    game_name: str,
    content: str,
    embedding: np.ndarray,
    title: str | None = None,
    url: str | None = None,
    source_name: str | None = None,
    chunk_index: int = 0,
) -> int:
    db = get_db()
    cur = db.execute(
        """INSERT INTO documents (game_name, title, url, source_name, content, embedding, chunk_index)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (game_name, title, url, source_name, content, embedding.astype(np.float32).tobytes(), chunk_index),
    )
    db.commit()
    return cur.lastrowid


async def store_documents_with_semantic_chunks(
    game_name: str,
    chunks: list[str],
    embeddings: list[np.ndarray],
    title: str | None = None,
    url: str | None = None,
    source_name: str | None = None,
) -> int:
    """
    批量存储固定长度分块(documents)及其对应的所有语义句切分块(semantic_chunks)。
    在内部对所有切分的单句进行批量向量嵌入，提升导入性能。
    """
    import asyncio
    from app.embedding import encode_batch
    from app.scraper import parse_chunk_sentences

    # 1. 存储 Parent Chunks (documents)
    doc_ids = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        doc_id = insert_document(
            game_name=game_name,
            content=chunk,
            embedding=emb,
            title=title,
            url=url,
            source_name=source_name,
            chunk_index=i,
        )
        doc_ids.append(doc_id)

    # 2. 从每一个 Parent Chunk 中解析出语义句子
    all_sentences = []
    for chunk, doc_id in zip(chunks, doc_ids):
        sentences = parse_chunk_sentences(chunk)
        for s in sentences:
            s["document_id"] = doc_id
            all_sentences.append(s)

    # 3. 批量生成语义句子的 Embedding 并写入 semantic_chunks
    if all_sentences:
        loop = asyncio.get_running_loop()
        sentence_texts = [s["content"] for s in all_sentences]
        sentence_embs = await loop.run_in_executor(None, encode_batch, sentence_texts)

        for s, s_emb in zip(all_sentences, sentence_embs):
            insert_semantic_chunk(
                document_id=s["document_id"],
                game_name=game_name,
                content=s["content"],
                embedding=s_emb,
                tag=s["tag"],
            )

    return len(chunks)


def get_documents_by_game(game_name: str) -> list[dict]:
    """Return metadata-only document rows (no embedding blob) for a game."""
    db = get_db()
    rows = db.execute(
        "SELECT id, game_name, title, url, source_name, content, chunk_index, created_at "
        "FROM documents WHERE game_name = ? ORDER BY chunk_index",
        (game_name,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_documents_by_game(game_name: str) -> int:
    db = get_db()
    cur = db.execute("DELETE FROM documents WHERE game_name = ?", (game_name,))
    db.commit()
    return cur.rowcount


def delete_documents_by_url(url: str) -> int:
    db = get_db()
    cur = db.execute("DELETE FROM documents WHERE url = ?", (url,))
    db.commit()
    return cur.rowcount


def list_games() -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT game_name, COUNT(*) AS document_count FROM documents "
        "GROUP BY game_name ORDER BY game_name"
    ).fetchall()
    return [dict(r) for r in rows]


def get_document_count_for_game(game_name: str) -> int:
    """Return the number of document chunks stored for a game."""
    db = get_db()
    row = db.execute(
        "SELECT COUNT(*) AS cnt FROM documents WHERE game_name = ?", (game_name,)
    ).fetchone()
    return row["cnt"] if row else 0


def get_knowledge_stats() -> dict:
    """Return aggregate statistics about the knowledge base."""
    db = get_db()
    games = db.execute(
        "SELECT game_name, COUNT(*) AS document_count FROM documents "
        "GROUP BY game_name ORDER BY document_count DESC"
    ).fetchall()
    total_docs = db.execute("SELECT COUNT(*) AS cnt FROM documents").fetchone()["cnt"]
    pending = db.execute(
        "SELECT COUNT(*) AS cnt FROM pending_queries WHERE status = 'pending'"
    ).fetchone()["cnt"]
    return {
        "total_games": len(games),
        "total_documents": total_docs,
        "pending_queries": pending,
        "games": [dict(g) for g in games],
    }


# ── Vector Similarity Search ───────────────────────────────────────

def search_similar(
    query_embedding: np.ndarray,
    game_name: str,
    top_k: int = 5,
    threshold: float | None = None,
) -> list[dict]:
    """
    Search for documents most similar to query_embedding, filtered by game_name.

    Uses numpy cosine similarity over all documents for the given game.
    Returns up to top_k results with similarity >= threshold.
    """
    if threshold is None:
        threshold = settings.similarity_threshold

    db = get_db()
    rows = db.execute(
        "SELECT id, content, embedding, title, url, source_name, chunk_index "
        "FROM documents WHERE game_name = ?",
        (game_name,),
    ).fetchall()

    if not rows:
        return []

    # Reconstruct embeddings matrix
    embeddings = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])

    # Cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    doc_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = np.dot(doc_norms, query_norm)

    # Top-k indices sorted by similarity descending
    if len(similarities) <= top_k:
        top_indices = np.argsort(similarities)[::-1]
    else:
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    # Filter by threshold
    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim < threshold:
            continue
        row = rows[idx]
        results.append({
            "id": row["id"],
            "content": row["content"],
            "title": row["title"],
            "url": row["url"],
            "source_name": row["source_name"],
            "similarity": round(sim, 4),
            "chunk_index": row["chunk_index"],
        })

    return results


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


# ── Semantic Chunks CRUD & Similarity Search ───────────────────────

def insert_semantic_chunk(
    document_id: int,
    game_name: str,
    content: str,
    embedding: np.ndarray,
    tag: str | None = None,
) -> int:
    db = get_db()
    cur = db.execute(
        """INSERT INTO semantic_chunks (document_id, game_name, content, embedding, tag)
           VALUES (?, ?, ?, ?, ?)""",
        (document_id, game_name, content, embedding.astype(np.float32).tobytes(), tag),
    )
    db.commit()
    return cur.lastrowid


def search_similar_semantic(
    query_embedding: np.ndarray,
    game_name: str,
    top_k: int = 5,
    threshold: float | None = None,
) -> list[dict]:
    """
    Search for semantic sentences most similar to query_embedding, filtered by game_name.
    Uses numpy cosine similarity over all semantic chunks for the given game.
    """
    if threshold is None:
        threshold = settings.similarity_threshold

    db = get_db()
    rows = db.execute(
        "SELECT id, document_id, content, embedding, tag "
        "FROM semantic_chunks WHERE game_name = ?",
        (game_name,),
    ).fetchall()

    if not rows:
        return []

    # Reconstruct embeddings matrix
    embeddings = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])

    # Cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    doc_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = np.dot(doc_norms, query_norm)

    # Top-k indices sorted by similarity descending
    if len(similarities) <= top_k:
        top_indices = np.argsort(similarities)[::-1]
    else:
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim < threshold:
            continue
        row = rows[idx]
        results.append({
            "id": row["id"],
            "document_id": row["document_id"],
            "content": row["content"],
            "tag": row["tag"],
            "similarity": round(sim, 4),
        })

    return results


def get_document_with_context(doc_id: int) -> list[dict]:
    """
    Retrieve the specified parent document along with its immediate surrounding context
    (previous and next chunks of the same game based on chunk_index).
    """
    db = get_db()
    target = db.execute(
        "SELECT game_name, chunk_index, url, source_name, title FROM documents WHERE id = ?",
        (doc_id,)
    ).fetchone()
    
    if not target:
        return []

    game_name = target["game_name"]
    chunk_index = target["chunk_index"]

    # Retrieve chunks with index in [chunk_index - 1, chunk_index + 1]
    rows = db.execute(
        """SELECT id, content, title, url, source_name, chunk_index
           FROM documents
           WHERE game_name = ? AND chunk_index BETWEEN ? AND ?
           ORDER BY chunk_index""",
        (game_name, chunk_index - 1, chunk_index + 1)
    ).fetchall()

    return [dict(r) for r in rows]

