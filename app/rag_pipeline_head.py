"""Core RAG pipeline — HyDE + 混合检索 + LLM 优化生成。

流程:
1. LLM 分析问题,生成假设性回答 (HyDE)
   - 假设回答是陈述句,语义空间更接近知识库文档,能显著提升检索召回率
2. 混合检索: 假设回答 + 原始问题,合并去重
3. 动态阈值: 标准阈值无结果时降低阈值重试,避免误判"知识不足"
4. 仍无结果 → 记录 pending query,返回知识不足
5. LLM 基于检索结果优化生成最终回答
"""

import asyncio
import json

import httpx
from openai import AsyncOpenAI

from app import database as db
from app.embedding import encode_text
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


async def _llm_chat(prompt: str, max_tokens: int = 2000, temperature: float = 0.4) -> str:
    """Call the LLM with retry. Returns empty string on persistent failure."""
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
            print(f"[rag] LLM returned empty content (attempt {attempt + 1})")
        except Exception as e:
            last_error = e
            print(f"[rag] LLM call failed (attempt {attempt + 1}): {e}")
        await asyncio.sleep(1.5)
    if last_error:
        print(f"[rag] LLM persistent failure: {last_error}")
    return ""


SYSTEM_PROMPT = """你是一个专业的游戏信息问答助手。请基于提供的参考上下文回答用户问题。

要求:
- 综合参考资料中的信息,给出详细、准确、结构化的回答
- 引用来源时使用 [1], [2] 等标记
- 可以对检索到的零散信息进行整合、补全、优化,使回答流畅完整
- 如果参考资料确实不包含相关问题答案,回复:"当前数据库知识不足,未能检索到你提问的信息。"
- 不要编造资料中不存在的信息,回答使用中文"""

KNOWLEDGE_INSUFFICIENT = "当前数据库知识不足，未能检索到你提问的信息。"

# 管线特性自描述 - 评测时写入 config_snapshot,便于区分不同版本管线的指标。
# 新增/关闭某项检索优化时请同步修改此处,让 Excel 里的版本标记准确反映管线状态。
PIPELINE_FEATURES = {
    "hyde": True,               # 假设性回答辅助检索 (Step 1)
    "context_window": True,     # 召回命中段落及其前后相邻段落 (retrieve_with_fallback)
    "dynamic_threshold": True,  # 标准阈值无结果时降低阈值重试
}


# ── Step 1: 生成假设性回答 (HyDE) ──────────────────────────────────

async def generate_hypothetical_answer(game_name: str, user_message: str) -> str:
    """LLM 分析问题,生成假设性回答。

    假设性回答是陈述句,语义空间更接近知识库中的百科文档,
    用它检索比用原始问题检索召回率更高(HyDE 原理)。
    即使回答不准确也无妨,只需包含相关术语和概念。
    """
    prompt = f"""用户想了解游戏《{game_name}》的以下问题:
{user_message}

请基于你对这款游戏的了解,生成一个简短的假设性回答(150-300字)。
要求:
1. 即使不确定也要给出可能的答案,绝对不要拒绝或说"我不知道"
2. 包含相关的游戏术语、概念、机制名称、数值、地点、角色名
3. 用陈述句,像百科知识条目一样写作
4. 这个回答仅用于知识库检索辅助,不需要完全准确

假设性回答:"""
    return await _llm_chat(prompt, max_tokens=500, temperature=0.5)


async def extract_game_name(user_message: str, history_msgs: list[dict]) -> str:
    """LLM 从用户输入和历史记录中提取游戏名称。"""
    prompt = f"""请从用户的提问和上下文历史中，提取出用户当前讨论的具体游戏名称。
如果明确提到了某个游戏名称，或者根据上下文能明确判断是哪款游戏，请仅仅输出该游戏名称（例如：塞尔达传说旷野之息、原神），不要包含任何多余的字词、标点或解释。
如果没有提到任何具体的游戏，或者无法确定，请直接输出 "None"。

用户提问: {user_message}"""
    if history_msgs:
        history_text = "\n".join([f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:200]}" for m in history_msgs[-3:]])
        prompt += f"\n\n近期历史记录:\n{history_text}"
        
    extracted = await _llm_chat(prompt, max_tokens=20, temperature=0.0)
    extracted = extracted.strip()
    if not extracted or extracted.lower() in ("none", "null", "不知道", "未指定"):
        return ""
    return extracted


# ── Step 2: 混合检索 ──────────────────────────────────────────────

def merge_and_dedupe(result_lists: list[list[dict]], top_k: int) -> list[dict]:
    """合并多路检索结果,按文档 id 去重,取每篇最高相似度,排序后取 top_k。"""
    seen: dict[int, dict] = {}
    for results in result_lists:
        for r in results:
            doc_id = r["id"]
            if doc_id not in seen or r["similarity"] > seen[doc_id]["similarity"]:
                seen[doc_id] = r
    merged = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)
    return merged[:top_k]


def retrieve_with_fallback(query_vec, game_name: str, top_k: int) -> list[dict]:
    """
    语义优先 + 上下文窗口扩充检索。
    1. 先在 semantic_chunks 表中搜索语义最相近的句子。
    2. 如果以 settings.similarity_threshold 为阈值无结果，则降低阈值重试。
    3. 获取匹配句子所归属的 parent chunk 及其前后的相邻 chunk (Context Window)。
    4. 合并相邻 chunk 的文本，去重，组装成完整且上下文连贯的候选检索结果。
    """
    threshold = settings.similarity_threshold
    semantic_matches = db.search_similar_semantic(
        query_vec, game_name, top_k=top_k, threshold=threshold
    )

    if not semantic_matches:
        lowered = max(0.1, threshold - 0.15)
        print(f"[rag] Semantic search: Standard threshold {threshold} returned nothing, retrying with {lowered}")
        semantic_matches = db.search_similar_semantic(
            query_vec, game_name, top_k=top_k, threshold=lowered
        )

    if not semantic_matches:
        return []

    # 聚合父文档并拉取上下文窗口
    seen_parent_ids = set()
    results = []

    for match in semantic_matches:
        doc_id = match["document_id"]
        if doc_id in seen_parent_ids:
            continue
        seen_parent_ids.add(doc_id)

        # 召回该段落及其前后相邻的段落
        context_docs = db.get_document_with_context(doc_id)
        if not context_docs:
            continue

        # 拼接段落内容
        combined_content = "\n\n".join([doc["content"] for doc in context_docs])

        # 找到被命中的主段落
        main_doc = next((d for d in context_docs if d["id"] == doc_id), context_docs[0])

        # 构造召回条目，并加上标签 (tag) 提供给 LLM
        tag_str = f"【主题：{match['tag']}】\n" if match["tag"] else ""
        results.append({
            "id": main_doc["id"],
            "content": tag_str + combined_content,
            "title": main_doc["title"],
            "url": main_doc["url"],
            "source_name": main_doc["source_name"],
            "similarity": match["similarity"],
            "chunk_index": main_doc["chunk_index"],
        })

    # 按相似度重新排序
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


# ── Step 1-2 封装: HyDE + 混合检索 (供 rag_query 与评测共用) ──────

async def retrieve_documents(
    game_name: str,
    user_message: str,
    progress_callback=None,
    verbose: bool = True,
) -> list[dict]:
    """HyDE + 混合检索 + 合并去重。

    封装原 rag_query 的 Step 1-2(假设回答生成 + 原始问题/假设回答双路检索 + 去重),
    **无 DB 写副作用**,既被 rag_query 调用,也被 eval 评测调用,
    保证评测与生产检索逻辑同源、不会随迭代而分叉。

    Args:
        progress_callback: 可选 async callable(stage: str, message: str[, content])。
            存在时在每个阶段调用,用于推送实时进度(rag_query 传入其 _report 闭包)。
        verbose: 是否打印阶段日志。生产路径保持 True;评测批量跑时传 False 减少噪声。

    Returns: 按相似度降序排列的检索结果列表,每个元素含 id/content/similarity/title/url 等。
    """
    async def _report(stage: str, message: str) -> None:
        if progress_callback is not None:
            try:
                await progress_callback(stage, message)
            except Exception:
                # 回调失败不应影响检索主流程
                pass

    # Step 1: LLM 分析问题,生成假设性回答 (HyDE)
    await _report("analyzing", "正在分析问题，构思假设回答…")
    if verbose:
        print(f"[rag] Step 1: Generating hypothetical answer for 《{game_name}》...")
    hypothetical = await generate_hypothetical_answer(game_name, user_message)
    if verbose:
        if hypothetical:
            print(f"[rag] Hypothetical answer: {hypothetical[:80]}...")
        else:
            print("[rag] Hypothetical answer generation failed, falling back to query-only retrieval")

    # Step 2: 混合检索(假设回答 + 原始问题)
    await _report("retrieving", "正在检索知识库，匹配相关内容…")
    if verbose:
        print("[rag] Step 2: Hybrid retrieval (HyDE + original query)...")
    result_lists = []

    # 2a. 原始问题检索
    query_vec = encode_text(user_message)
    result_lists.append(retrieve_with_fallback(query_vec, game_name, settings.top_k))

    # 2b. 假设回答检索(如果生成成功)
    if hypothetical:
        hyde_vec = encode_text(hypothetical)
        result_lists.append(retrieve_with_fallback(hyde_vec, game_name, settings.top_k))

    # 合并去重
    retrieved = merge_and_dedupe(result_lists, settings.top_k)
    if verbose:
        print(f"[rag] Retrieved {len(retrieved)} unique documents after merge")
    return retrieved


# ── Step 5: LLM 优化生成最终回答 ───────────────────────────────────

async def generate_final_answer(
    user_message: str,
    retrieved: list[dict],
    history_msgs: list[dict],
) -> str:
    """LLM 基于检索结果优化生成最终回答。"""
    # 构建上下文
    context_parts = []
    for i, doc in enumerate(retrieved):
        title = doc.get("title") or "未命名"
        url = doc.get("url") or ""
        context_parts.append(f"[来源{i+1}: {title} ({url})]\n{doc['content']}")

    # 构建历史
    history_parts = []
    for msg in history_msgs:
        role_label = "用户" if msg["role"] == "user" else "助手"
        history_parts.append(f"{role_label}: {msg['content']}")

    # 组装 prompt
    full_prompt = SYSTEM_PROMPT
    if context_parts:
        full_prompt += "\n\n参考上下文:\n" + "\n\n".join(context_parts)
    if history_parts:
        full_prompt += "\n\n对话历史:\n" + "\n".join(history_parts)
    full_prompt += f"\n\n用户问题: {user_message}\n回答(中文,引用来源请标注编号):"

    answer = await _llm_chat(full_prompt, max_tokens=2000, temperature=0.3)
    return answer if answer else KNOWLEDGE_INSUFFICIENT


async def generate_final_answer_stream(
    user_message: str,
    retrieved: list[dict],
    history_msgs: list[dict],
    on_token_callback=None,
) -> str:
    """LLM 基于检索结果优化生成最终回答，支持流式输出。"""
    # 构建上下文
    context_parts = []
    for i, doc in enumerate(retrieved):
        title = doc.get("title") or "未命名"
        url = doc.get("url") or ""
        context_parts.append(f"[来源{i+1}: {title} ({url})]\n{doc['content']}")

    # 构建历史
    history_parts = []
    for msg in history_msgs:
        role_label = "用户" if msg["role"] == "user" else "助手"
        history_parts.append(f"{role_label}: {msg['content']}")

    # 组装 prompt
    full_prompt = SYSTEM_PROMPT
    if context_parts:
        full_prompt += "\n\n参考上下文:\n" + "\n\n".join(context_parts)
    if history_parts:
        full_prompt += "\n\n对话历史:\n" + "\n".join(history_parts)
    full_prompt += f"\n\n用户问题: {user_message}\n回答(中文,引用来源请标注编号):"

    client = _get_client()
    answer_chunks = []
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.3,
                max_tokens=2000,
                stream=True,
            )
            async for chunk in response:
                token = chunk.choices[0].delta.content or ""
                if token:
                    answer_chunks.append(token)
                    if on_token_callback:
                        await on_token_callback(token)
            answer = "".join(answer_chunks).strip()
            if answer:
                return answer
        except Exception as e:
            print(f"[rag] LLM stream call failed (attempt {attempt + 1}): {e}")
            if attempt == 2:
                print("[rag] Streaming failed, falling back to non-stream chat...")
                answer = await _llm_chat(full_prompt, max_tokens=2000, temperature=0.3)
                if answer:
                    if on_token_callback:
                        await on_token_callback(answer)
                    return answer
            await asyncio.sleep(1.5)

    return KNOWLEDGE_INSUFFICIENT


# ── 主流程 ─────────────────────────────────────────────────────────

async def rag_query(
    game_name: str,
    user_message: str,
    conversation_id: int,
    progress_callback=None,
) -> dict:
    """Execute the full HyDE RAG pipeline.

    Args:
        progress_callback: optional async callable(stage: str, message: str).
            当存在时,在每个思考阶段被调用,用于向前端推送实时进度。

    Returns: dict with answer, sources, conversation_id.
    """
    async def _report(stage: str, message: str, content: str = None) -> None:
        if progress_callback is not None:
            try:
                await progress_callback(stage, message, content)
            except Exception:
                # 回调失败不应影响主流程
                pass

    # Step 0: 加载历史 + 保存用户消息 + 提取游戏名
    history_msgs = db.get_messages(conversation_id, limit=settings.max_history_messages)
    db.save_message(conversation_id, "user", user_message, None)

    extracted_game = await extract_game_name(user_message, history_msgs)
    if extracted_game:
        game_name = extracted_game
        db.update_conversation_game(conversation_id, game_name)
    else:
        conv = db.get_conversation(conversation_id)
        if conv and conv.get("game_name"):
            game_name = conv["game_name"]
        else:
            game_name = game_name # fallback to the one passed in (e.g. empty string)

    # Step 1-2: HyDE + 混合检索(封装为 retrieve_documents,与评测共用同一检索逻辑)
    retrieved = await retrieve_documents(game_name, user_message, _report)

    # Step 4: 无结果 → 记录 pending query,返回知识不足
    if not retrieved:
        await _report("insufficient", "未检索到相关知识，已记录问题待学习…")
        db.add_pending_query(game_name, user_message)
        db.save_message(conversation_id, "assistant", KNOWLEDGE_INSUFFICIENT, "[]")
        db.update_conversation_timestamp(conversation_id)
        print("[rag] No documents retrieved — recorded as pending query for feedback learning")
        return {"answer": KNOWLEDGE_INSUFFICIENT, "sources": [], "conversation_id": conversation_id}

    # Step 5: LLM 基于检索结果优化生成最终回答
    await _report("generating", f"已找到 {len(retrieved)} 条相关内容，正在整合生成回答…")
    print("[rag] Step 3: Generating final answer with retrieved context...")
    
    async def _on_token(token: str):
        await _report("generating", f"已找到 {len(retrieved)} 条相关内容，正在整合生成回答…", content=token)

    answer = await generate_final_answer_stream(user_message, retrieved, history_msgs, _on_token)

    # Step 6: 构建来源列表
    sources = [
        {
            "title": doc.get("title"),
            "url": doc.get("url"),
            "chunk_index": doc.get("chunk_index", 0),
            "similarity": doc.get("similarity"),
        }
        for doc in retrieved
    ]

    # Step 7: 保存助手消息 + 更新时间戳 + 自动标题
    db.save_message(conversation_id, "assistant", answer, json.dumps(sources, ensure_ascii=False))
    db.update_conversation_timestamp(conversation_id)

    conv = db.get_conversation(conversation_id)
    if conv and conv.get("title") == "New Conversation":
        new_title = user_message[:30] + ("..." if len(user_message) > 30 else "")
        db.update_conversation(conversation_id, new_title)

    print(f"[rag] Done. Answer length: {len(answer)}")
    return {"answer": answer, "sources": sources, "conversation_id": conversation_id}
