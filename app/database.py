"""Persistence layer backed purely by ChromaDB.

Replaces SQLite entirely. All data is stored in ChromaDB collections.
"""

import json
import uuid
import numpy as np
from datetime import datetime

from app import vector_store as vs
from config import settings


def init_db() -> None:
    """Initialize collections."""
    pass # Vector store initializes lazily


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _to_list(vec) -> list[float]:
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)

# ── Conversation CRUD ──────────────────────────────────────────────

def create_conversation(game_name: str, title: str = "New Conversation") -> dict:
    coll = vs.get_collection(vs.CONV_COLLECTION)
    conv_id = str(uuid.uuid4())
    now = _now()
    meta = {"title": title, "game_name": game_name, "created_at": now, "updated_at": now}
    coll.add(ids=[conv_id], metadatas=[meta], documents=[""], embeddings=[[0.0]])
    return {"id": conv_id, **meta}


def list_conversations() -> list[dict]:
    coll = vs.get_collection(vs.CONV_COLLECTION)
    res = coll.get(include=["metadatas"])
    
    msg_coll = vs.get_collection(vs.MSG_COLLECTION)
    
    out = []
    for i, cid in enumerate(res["ids"]):
        meta = res["metadatas"][i]
        
        # We can't efficiently join, so we fetch count for each or just set to 0. 
        # For simplicity, we just get count for each (can be slow for many, but ok for now)
        msg_res = msg_coll.get(where={"conversation_id": cid}, include=[])
        
        out.append({
            "id": cid,
            "title": meta.get("title", ""),
            "game_name": meta.get("game_name", ""),
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
            "message_count": len(msg_res["ids"])
        })
    out.sort(key=lambda x: x["updated_at"], reverse=True)
    return out


def get_conversation(conv_id: str) -> dict | None:
    coll = vs.get_collection(vs.CONV_COLLECTION)
    res = coll.get(ids=[conv_id], include=["metadatas"])
    if not res["ids"]:
        return None
    
    meta = res["metadatas"][0]
    result = {"id": conv_id, **meta}
    
    # Get messages
    msg_coll = vs.get_collection(vs.MSG_COLLECTION)
    msg_res = msg_coll.get(where={"conversation_id": conv_id}, include=["metadatas", "documents"])
    
    msgs = []
    for i, mid in enumerate(msg_res["ids"]):
        m_meta = msg_res["metadatas"][i]
        m_doc = msg_res["documents"][i]
        msgs.append({
            "id": mid,
            "conversation_id": conv_id,
            "role": m_meta.get("role", ""),
            "content": m_doc,
            "sources": m_meta.get("sources", ""),
            "created_at": m_meta.get("created_at", "")
        })
    msgs.sort(key=lambda x: x["created_at"])
    result["messages"] = msgs
    return result


def update_conversation(conv_id: str, title: str) -> dict | None:
    coll = vs.get_collection(vs.CONV_COLLECTION)
    res = coll.get(ids=[conv_id], include=["metadatas"])
    if not res["ids"]:
        return None
    meta = res["metadatas"][0]
    meta["title"] = title
    meta["updated_at"] = _now()
    coll.update(ids=[conv_id], metadatas=[meta])
    return {"id": conv_id, **meta}




def update_conversation_game(conv_id: str, game_name: str) -> None:
    coll = vs.get_collection(vs.CONV_COLLECTION)
    res = coll.get(ids=[conv_id], include=["metadatas"])
    if res["ids"]:
        meta = res["metadatas"][0]
        meta["game_name"] = game_name
        meta["updated_at"] = _now()
        coll.update(ids=[conv_id], metadatas=[meta])

def update_conversation_game_name(conv_id: str, game_name: str) -> None:
    update_conversation_game(conv_id, game_name)

def update_conversation_timestamp(conv_id: str) -> None:
    coll = vs.get_collection(vs.CONV_COLLECTION)
    res = coll.get(ids=[conv_id], include=["metadatas"])
    if res["ids"]:
        meta = res["metadatas"][0]
        meta["updated_at"] = _now()
        coll.update(ids=[conv_id], metadatas=[meta])


def delete_conversation(conv_id: str) -> bool:
    coll = vs.get_collection(vs.CONV_COLLECTION)
    before = coll.get(ids=[conv_id], include=[])["ids"]
    if before:
        coll.delete(ids=[conv_id])
        # delete messages
        msg_coll = vs.get_collection(vs.MSG_COLLECTION)
        msg_res = msg_coll.get(where={"conversation_id": conv_id}, include=[])
        if msg_res["ids"]:
            msg_coll.delete(ids=msg_res["ids"])
        return True
    return False


# ── Message CRUD ───────────────────────────────────────────────────

def save_message(conv_id: str, role: str, content: str, sources: str | None = None) -> dict:
    coll = vs.get_collection(vs.MSG_COLLECTION)
    mid = str(uuid.uuid4())
    now = _now()
    meta = {
        "conversation_id": conv_id,
        "role": role,
        "sources": sources or "",
        "created_at": now
    }
    coll.add(ids=[mid], metadatas=[meta], documents=[content], embeddings=[[0.0]])
    return {"id": mid, "content": content, **meta}


def get_messages(conv_id: str, limit: int = 10) -> list[dict]:
    coll = vs.get_collection(vs.MSG_COLLECTION)
    res = coll.get(where={"conversation_id": conv_id}, include=["metadatas", "documents"])
    msgs = []
    for i, mid in enumerate(res["ids"]):
        meta = res["metadatas"][i]
        msgs.append({
            "id": mid,
            "conversation_id": conv_id,
            "role": meta.get("role", ""),
            "content": res["documents"][i],
            "sources": meta.get("sources", ""),
            "created_at": meta.get("created_at", "")
        })
    msgs.sort(key=lambda x: x["created_at"], reverse=True)
    return list(reversed(msgs[:limit]))


# ── Document CRUD (Delegated to vs) ──────────────────────────────────

def insert_document(*args, **kwargs) -> str:
    return vs.insert_document(*args, **kwargs)

async def store_documents_with_semantic_chunks(
    game_name: str,
    chunks: list[str],
    embeddings: list[np.ndarray],
    title: str | None = None,
    url: str | None = None,
    source_name: str | None = None,
) -> int:
    import asyncio
    from app.embedding import encode_batch
    from app.scraper import parse_chunk_sentences

    doc_ids = vs.add_documents(
        game_name, chunks, embeddings, title=title, url=url, source_name=source_name
    )

    all_sentences = []
    for chunk, doc_id in zip(chunks, doc_ids):
        for s in parse_chunk_sentences(chunk):
            s["document_id"] = doc_id
            all_sentences.append(s)

    if all_sentences:
        loop = asyncio.get_running_loop()
        sentence_texts = [s["content"] for s in all_sentences]
        sentence_embs = await loop.run_in_executor(None, encode_batch, sentence_texts)
        vs.add_semantics(game_name, all_sentences, sentence_embs)

    return len(chunks)

def get_documents_by_game(game_name: str) -> list[dict]:
    return vs.get_documents_by_game(game_name)

def delete_documents_by_game(game_name: str) -> int:
    return vs.delete_documents_by_game(game_name)

def delete_documents_by_url(url: str) -> int:
    return vs.delete_documents_by_url(url)

def list_games() -> list[dict]:
    return vs.list_games()

def get_document_count_for_game(game_name: str) -> int:
    return vs.get_document_count_for_game(game_name)

def get_knowledge_stats() -> dict:
    games = vs.list_games()
    total_docs = vs.count_documents()
    
    pending_coll = vs.get_collection(vs.PENDING_COLLECTION)
    pending_res = pending_coll.get(where={"status": "pending"}, include=[])
    pending = len(pending_res["ids"])
    
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
    coll = vs.get_collection(vs.TASKS_COLLECTION)
    tid = str(uuid.uuid4())
    now = _now()
    meta = {
        "game_name": game_name,
        "source_name": source_name,
        "source_url": source_url,
        "status": "pending",
        "interval_hours": interval_hours,
        "last_run": "",
        "next_run": "",
        "error_message": "",
        "created_at": now
    }
    coll.add(ids=[tid], metadatas=[meta], documents=[""], embeddings=[[0.0]])
    return {"id": tid, **meta}

def list_scraping_tasks() -> list[dict]:
    coll = vs.get_collection(vs.TASKS_COLLECTION)
    res = coll.get(include=["metadatas"])
    tasks = []
    for i, tid in enumerate(res["ids"]):
        tasks.append({"id": tid, **res["metadatas"][i]})
    tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return tasks

def get_scraping_task(task_id: str) -> dict | None:
    coll = vs.get_collection(vs.TASKS_COLLECTION)
    res = coll.get(ids=[task_id], include=["metadatas"])
    if not res["ids"]:
        return None
    return {"id": task_id, **res["metadatas"][0]}

def update_scraping_task_status(
    task_id: str,
    status: str,
    error_message: str | None = None,
    next_run: str | None = None,
) -> None:
    coll = vs.get_collection(vs.TASKS_COLLECTION)
    res = coll.get(ids=[task_id], include=["metadatas"])
    if res["ids"]:
        meta = res["metadatas"][0]
        meta["status"] = status
        meta["last_run"] = _now()
        meta["error_message"] = error_message or ""
        meta["next_run"] = next_run or ""
        coll.update(ids=[task_id], metadatas=[meta])

def delete_scraping_task(task_id: str) -> bool:
    coll = vs.get_collection(vs.TASKS_COLLECTION)
    if coll.get(ids=[task_id], include=[])["ids"]:
        coll.delete(ids=[task_id])
        return True
    return False

# ── Pending Queries CRUD ───────────────────────────────────────────

def add_pending_query(game_name: str, question: str) -> str:
    coll = vs.get_collection(vs.PENDING_COLLECTION)
    res = coll.get(where={"$and": [{"game_name": game_name}, {"status": "pending"}]}, include=["documents"])
    if question in res["documents"]:
        idx = res["documents"].index(question)
        return res["ids"][idx]
        
    qid = str(uuid.uuid4())
    meta = {
        "game_name": game_name,
        "status": "pending",
        "created_at": _now(),
        "resolved_at": ""
    }
    coll.add(ids=[qid], metadatas=[meta], documents=[question], embeddings=[[0.0]])
    return qid

def list_pending_queries(limit: int = 100, status: str = "pending") -> list[dict]:
    coll = vs.get_collection(vs.PENDING_COLLECTION)
    res = coll.get(where={"status": status}, include=["metadatas", "documents"], limit=limit)
    queries = []
    for i, qid in enumerate(res["ids"]):
        meta = res["metadatas"][i]
        queries.append({
            "id": qid,
            "game_name": meta.get("game_name", ""),
            "question": res["documents"][i],
            "status": meta.get("status", ""),
            "created_at": meta.get("created_at", ""),
            "resolved_at": meta.get("resolved_at", "")
        })
    queries.sort(key=lambda x: x["created_at"], reverse=True)
    return queries

def list_pending_queries_by_game(game_name: str, status: str = "pending") -> list[dict]:
    coll = vs.get_collection(vs.PENDING_COLLECTION)
    res = coll.get(where={"$and": [{"game_name": game_name}, {"status": status}]}, include=["metadatas", "documents"])
    queries = []
    for i, qid in enumerate(res["ids"]):
        meta = res["metadatas"][i]
        queries.append({
            "id": qid,
            "game_name": meta.get("game_name", ""),
            "question": res["documents"][i],
            "status": meta.get("status", ""),
            "created_at": meta.get("created_at", ""),
            "resolved_at": meta.get("resolved_at", "")
        })
    queries.sort(key=lambda x: x["created_at"])
    return queries

def resolve_pending_queries_by_game(game_name: str) -> int:
    coll = vs.get_collection(vs.PENDING_COLLECTION)
    res = coll.get(where={"$and": [{"game_name": game_name}, {"status": "pending"}]}, include=["metadatas"])
    if not res["ids"]:
        return 0
    now = _now()
    for meta in res["metadatas"]:
        meta["status"] = "resolved"
        meta["resolved_at"] = now
    coll.update(ids=res["ids"], metadatas=res["metadatas"])
    return len(res["ids"])

def delete_pending_query(query_id: str) -> bool:
    coll = vs.get_collection(vs.PENDING_COLLECTION)
    if coll.get(ids=[query_id], include=[])["ids"]:
        coll.delete(ids=[query_id])
        return True
    return False

# ── Knowledge Logs CRUD ────────────────────────────────────────────

def add_knowledge_log(
    action: str,
    pending_processed: int = 0,
    trending_fetched: int = 0,
    games_detail: str | None = None,
    message: str | None = None,
) -> str:
    coll = vs.get_collection(vs.LOGS_COLLECTION)
    lid = str(uuid.uuid4())
    meta = {
        "action": action,
        "pending_processed": pending_processed,
        "trending_fetched": trending_fetched,
        "games_detail": games_detail or "",
        "created_at": _now()
    }
    coll.add(ids=[lid], metadatas=[meta], documents=[message or ""], embeddings=[[0.0]])
    return lid

def list_knowledge_logs(limit: int = 20) -> list[dict]:
    coll = vs.get_collection(vs.LOGS_COLLECTION)
    res = coll.get(include=["metadatas", "documents"])
    logs = []
    for i, lid in enumerate(res["ids"]):
        meta = res["metadatas"][i]
        logs.append({
            "id": lid,
            "action": meta.get("action", ""),
            "pending_processed": meta.get("pending_processed", 0),
            "trending_fetched": meta.get("trending_fetched", 0),
            "games_detail": meta.get("games_detail", ""),
            "message": res["documents"][i],
            "created_at": meta.get("created_at", "")
        })
    logs.sort(key=lambda x: x["created_at"], reverse=True)
    return logs[:limit]

# ── Cache CRUD ─────────────────────────────────────────────────────

def set_query_cache(query_hash: str, game_name: str, response: str, query_text: str, query_embedding: list[float] | None = None):
    coll = vs.get_collection(vs.CACHE_COLLECTION)
    cid = f"cache_{query_hash}"
    meta = {"query_hash": query_hash, "game_name": game_name, "query_text": query_text, "created_at": _now()}
    
    kwargs = {
        "ids": [cid],
        "metadatas": [meta],
        "documents": [response]
    }
    if query_embedding is not None:
        kwargs["embeddings"] = [_to_list(query_embedding)]
    else:
        # We must provide dummy embedding for Chroma if embedding_function=None
        zeros = [0.0]*384
        kwargs["embeddings"] = [zeros]
        
    try:
        coll.add(**kwargs)
    except Exception as e:
        print(f"Error caching query: {e}")

def get_exact_query_cache(query_text: str, game_name: str) -> str | None:
    coll = vs.get_collection(vs.CACHE_COLLECTION)
    res = coll.get(where={"$and": [{"query_text": query_text}, {"game_name": game_name}]}, include=["documents"])
    if res["ids"]:
        return res["documents"][0]
    return None

def get_semantic_query_cache(query_embedding: list[float], game_name: str, threshold: float = 0.85) -> str | None:
    coll = vs.get_collection(vs.CACHE_COLLECTION)
    try:
        res = coll.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=1,
            where={"game_name": game_name},
            include=["documents", "distances"]
        )
        if res["ids"] and res["ids"][0]:
            sim = 1.0 - float(res["distances"][0][0])
            if sim >= threshold:
                return res["documents"][0][0]
    except Exception:
        pass
    return None

# ── User Memory CRUD ───────────────────────────────────────────────

def add_user_memory(fact: str, embedding: list[float]):
    coll = vs.get_collection(vs.MEMORY_COLLECTION)
    mid = str(uuid.uuid4())
    coll.add(
        ids=[mid],
        embeddings=[_to_list(embedding)],
        documents=[fact],
        metadatas=[{"created_at": _now()}]
    )

def get_user_memories(query_embedding: list[float] | None = None, top_k: int = 5) -> list[str]:
    coll = vs.get_collection(vs.MEMORY_COLLECTION)
    if query_embedding is None:
        # return all or recent
        res = coll.get(include=["documents"], limit=top_k)
        return res["documents"]
    
    try:
        res = coll.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=top_k,
            include=["documents"]
        )
        if res["ids"] and res["ids"][0]:
            return res["documents"][0]
    except Exception:
        pass
    return []

# ── Semantic Chunks ──────────────────────────────────────────

def insert_semantic_chunk(*args, **kwargs) -> str:
    return vs.insert_semantic_chunk(*args, **kwargs)

def search_similar_semantic(*args, **kwargs) -> list[dict]:
    return vs.search_similar_semantic(*args, **kwargs)

def get_document_with_context(*args, **kwargs) -> list[dict]:
    return vs.get_document_with_context(*args, **kwargs)
