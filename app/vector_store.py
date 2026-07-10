"""ChromaDB 向量存储 - documents 与 semantic_chunks 两个 collection。

承接原 SQLite 中 documents/semantic_chunks 表的向量存储与相似度检索职责。
- cosine 距离空间(归一化向量下等价于原 numpy 余弦相似度行为)
- embedding_function=None: 继续用 sentence-transformers 预计算向量传入,
  保持与 paraphrase-multilingual-MiniLM-L12-v2 兼容
- id 为字符串(迁移用 "doc_{旧id}"/"sem_{旧id}",新插入用 uuid)
- 返回结构与旧 database.py 向量函数保持一致,上层(rag_pipeline/knowledge_manager/
  scraper/routes)无感

metadata 限制:ChromaDB 的 metadata 值不能为 None,故写入时 None->"",
读取时把 title/tag 的 "" 转回 None(与旧 SQLite NULL 行为一致)。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

# 关闭 ChromaDB 遥测(避免 posthog 版本不匹配的报错噪声),须在 import chromadb 前设置
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb

from config import settings

DOCS_COLLECTION = "documents"
SEM_COLLECTION = "semantic_chunks"
# cosine 距离 = 1 - cosine_similarity(归一化向量),与原 numpy 余弦相似度一致
_SPACE_META = {"hnsw:space": "cosine"}
_BATCH = 500

_client: chromadb.api.ClientAPI | None = None
_docs_coll = None
_sem_coll = None


def get_chroma() -> chromadb.api.ClientAPI:
    """单例 PersistentClient,首次访问时创建持久化目录。"""
    global _client
    if _client is None:
        os.makedirs(settings.chroma_path, exist_ok=True)
        _client = chromadb.PersistentClient(path=settings.chroma_path)
    return _client


def _docs_collection():
    global _docs_coll
    if _docs_coll is None:
        _docs_coll = get_chroma().get_or_create_collection(
            DOCS_COLLECTION, metadata=_SPACE_META, embedding_function=None
        )
    return _docs_coll


def _sem_collection():
    global _sem_coll
    if _sem_coll is None:
        _sem_coll = get_chroma().get_or_create_collection(
            SEM_COLLECTION, metadata=_SPACE_META, embedding_function=None
        )
    return _sem_coll


def reset_collections() -> None:
    """删除并重建两个 collection(迁移脚本清场用)。"""
    global _docs_coll, _sem_coll
    client = get_chroma()
    for name in (DOCS_COLLECTION, SEM_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass
    _docs_coll = None
    _sem_coll = None
    _docs_collection()
    _sem_collection()


# ── metadata / 行构造 ───────────────────────────────────────────────

def _to_list(vec) -> list[float]:
    """把 numpy 向量或 list 转为 Python float 列表(ChromaDB 要求)。"""
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)


def _doc_meta(game_name, title, url, source_name, chunk_index, created_at) -> dict:
    return {
        "game_name": game_name,
        "title": title or "",
        "url": url or "",
        "source_name": source_name or "",
        "chunk_index": int(chunk_index),
        "created_at": created_at,
    }


def _sem_meta(document_id, game_name, tag, created_at) -> dict:
    return {
        "document_id": str(document_id),
        "game_name": game_name,
        "tag": tag or "",
        "created_at": created_at,
    }


def _doc_row(id_: str, meta: dict, content: str) -> dict:
    return {
        "id": id_,
        "game_name": meta.get("game_name"),
        "title": meta.get("title") or None,
        "url": meta.get("url") or "",
        "source_name": meta.get("source_name") or None,
        "content": content,
        "chunk_index": meta.get("chunk_index", 0),
        "created_at": meta.get("created_at"),
    }


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _batched(seq, size: int = _BATCH):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ── 写入 ────────────────────────────────────────────────────────────

def add_documents(
    game_name: str,
    chunks: list[str],
    embeddings: list,
    title: str | None = None,
    url: str | None = None,
    source_name: str | None = None,
) -> list[str]:
    """批量写入 parent chunks,返回生成的 doc id 列表(chunk_index = 位置序号)。"""
    coll = _docs_collection()
    now = _now()
    doc_ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in chunks]
    idxs = list(range(len(chunks)))
    for b in _batched(idxs):
        coll.add(
            ids=[doc_ids[i] for i in b],
            embeddings=[_to_list(embeddings[i]) for i in b],
            documents=[chunks[i] for i in b],
            metadatas=[_doc_meta(game_name, title, url, source_name, i, now) for i in b],
        )
    return doc_ids


def add_semantics(game_name: str, sentences: list[dict], embeddings: list) -> list[str]:
    """批量写入语义句,sentences 每项含 {content, tag, document_id}。返回 sem id 列表。"""
    coll = _sem_collection()
    now = _now()
    sem_ids = [f"sem_{uuid.uuid4().hex[:8]}" for _ in sentences]
    idxs = list(range(len(sentences)))
    for b in _batched(idxs):
        coll.add(
            ids=[sem_ids[i] for i in b],
            embeddings=[_to_list(embeddings[i]) for i in b],
            documents=[sentences[i]["content"] for i in b],
            metadatas=[
                _sem_meta(sentences[i]["document_id"], game_name, sentences[i].get("tag"), now)
                for i in b
            ],
        )
    return sem_ids


def insert_document(
    game_name: str,
    content: str,
    embedding,
    title: str | None = None,
    url: str | None = None,
    source_name: str | None = None,
    chunk_index: int = 0,
) -> str:
    """单条写入 parent chunk,返回 doc id。"""
    coll = _docs_collection()
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    coll.add(
        ids=[doc_id],
        embeddings=[_to_list(embedding)],
        documents=[content],
        metadatas=[_doc_meta(game_name, title, url, source_name, chunk_index, _now())],
    )
    return doc_id


def insert_semantic_chunk(
    document_id: str,
    game_name: str,
    content: str,
    embedding,
    tag: str | None = None,
) -> str:
    """单条写入语义句,返回 sem id。"""
    coll = _sem_collection()
    sid = f"sem_{uuid.uuid4().hex[:8]}"
    coll.add(
        ids=[sid],
        embeddings=[_to_list(embedding)],
        documents=[content],
        metadatas=[_sem_meta(document_id, game_name, tag, _now())],
    )
    return sid


# ── 读取 / 统计 ─────────────────────────────────────────────────────

def get_documents_by_game(game_name: str) -> list[dict]:
    """返回某游戏的全部文档(按 chunk_index 排序),不含 embedding。"""
    coll = _docs_collection()
    res = coll.get(where={"game_name": game_name}, include=["metadatas", "documents"], limit=100000)
    rows = [
        _doc_row(res["ids"][i], res["metadatas"][i], res["documents"][i])
        for i in range(len(res["ids"]))
    ]
    rows.sort(key=lambda x: x["chunk_index"])
    return rows


def get_document_count_for_game(game_name: str) -> int:
    coll = _docs_collection()
    res = coll.get(where={"game_name": game_name}, include=[], limit=100000)
    return len(res["ids"])


def count_documents() -> int:
    return _docs_collection().count()


def list_games() -> list[dict]:
    """聚合各游戏文档数,按游戏名排序。"""
    coll = _docs_collection()
    res = coll.get(include=["metadatas"], limit=100000)
    counts: dict[str, int] = {}
    for m in res["metadatas"]:
        g = m.get("game_name")
        counts[g] = counts.get(g, 0) + 1
    return [{"game_name": g, "document_count": c} for g, c in sorted(counts.items())]


def get_document_with_context(doc_id) -> list[dict]:
    """取 doc_id 所在段落及其前后相邻段落(同游戏 chunk_index ±1),按 chunk_index 排序。"""
    coll = _docs_collection()
    tgt = coll.get(ids=[str(doc_id)], include=["metadatas"])
    if not tgt["ids"]:
        return []
    m = tgt["metadatas"][0]
    game_name = m.get("game_name")
    ci = int(m.get("chunk_index", 0))
    # ChromaDB where 每个字段只允许一个操作符,范围查询需用 $and 组合两个子条件
    res = coll.get(
        where={
            "$and": [
                {"game_name": game_name},
                {"chunk_index": {"$gte": ci - 1}},
                {"chunk_index": {"$lte": ci + 1}},
            ]
        },
        include=["metadatas", "documents"],
        limit=100000,
    )
    rows = [
        _doc_row(res["ids"][i], res["metadatas"][i], res["documents"][i])
        for i in range(len(res["ids"]))
    ]
    rows.sort(key=lambda x: x["chunk_index"])
    return rows


# ── 检索 ────────────────────────────────────────────────────────────

def search_similar_semantic(
    query_embedding,
    game_name: str,
    top_k: int = 5,
    threshold: float | None = None,
) -> list[dict]:
    """在 semantic_chunks 中按向量相似度检索,返回 [{id, document_id, content, tag, similarity}]。

    similarity = 1 - cosine_distance,过滤 similarity >= threshold(与旧 numpy 行为一致)。
    """
    if threshold is None:
        threshold = settings.similarity_threshold
    coll = _sem_collection()
    where_clause = {"game_name": game_name} if game_name else None
    res = coll.query(
        query_embeddings=[_to_list(query_embedding)],
        n_results=top_k,
        where=where_clause,
        include=["metadatas", "documents", "distances"],
    )
    ids = res["ids"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]
    docs = res["documents"][0]
    out = []
    for i in range(len(ids)):
        sim = round(1.0 - float(dists[i]), 4)
        if sim < threshold:
            continue
        out.append({
            "id": ids[i],
            "document_id": metas[i].get("document_id"),
            "content": docs[i],
            "tag": metas[i].get("tag") or None,
            "similarity": sim,
        })
    return out


# ── 删除 ────────────────────────────────────────────────────────────

def delete_documents_by_game(game_name: str) -> int:
    """删除某游戏的全部文档及语义句,返回删除的文档数。"""
    docs = _docs_collection()
    sems = _sem_collection()
    before = len(docs.get(where={"game_name": game_name}, include=[], limit=100000)["ids"])
    docs.delete(where={"game_name": game_name})
    sems.delete(where={"game_name": game_name})
    return before


def delete_documents_by_url(url: str) -> int:
    """删除某 url 的文档及其关联语义句,返回删除的文档数。"""
    docs = _docs_collection()
    sems = _sem_collection()
    res = docs.get(where={"url": url}, include=[], limit=100000)
    ids = res["ids"]
    if not ids:
        return 0
    docs.delete(ids=ids)
    sems.delete(where={"document_id": {"$in": ids}})
    return len(ids)
