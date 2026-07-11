import asyncio
import json
import hashlib
import uuid

import httpx
from openai import AsyncOpenAI

from app import database as db
from app.embedding import encode_text
from app.tools import execute_tool, TOOLS_SCHEMA
from config import settings

FALLBACK_PROVIDERS = [
    p for p in [
        # 优先使用 .env 中实际配置的 OPENAI_* (默认指向 DeepSeek)，
        # 再回退 mimo / dashscope / deepseek; 空 key 的 provider 会被过滤掉。
        {
            "name": "openai",
            "api_key": settings.openai_api_key,
            "base_url": settings.openai_base_url,
            "model": settings.llm_model,
        },
        {
            "name": "mimo",
            "api_key": settings.mimo_api_key,
            "base_url": settings.mimo_base_url,
            "model": settings.mimo_model,
        },
        {
            "name": "dashscope",
            "api_key": settings.dashscope_api_key,
            "base_url": settings.dashscope_base_url,
            "model": settings.dashscope_model,
        },
        {
            "name": "deepseek",
            "api_key": settings.deepseek_api_key,
            "base_url": settings.deepseek_base_url,
            "model": settings.deepseek_model,
        },
    ]
    if p["api_key"]
]


_clients = {}

def _get_client(api_key: str, base_url: str) -> AsyncOpenAI:
    client_key = f"{api_key}_{base_url}"
    if client_key not in _clients:
        _clients[client_key] = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(timeout=60.0),
        )
    return _clients[client_key]

async def _chat_with_fallback(messages: list, tools: list = None, tool_choice: str = "auto", temperature: float = 0.4, report_callback=None):
    last_error = None
    for provider in FALLBACK_PROVIDERS:
        try:
            if report_callback:
                await report_callback("analyzing", f"正在尝试使用 {provider['name']} 模型思考...")
            
            client = _get_client(provider["api_key"], provider["base_url"])
            response = await client.chat.completions.create(
                model=provider["model"],
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
            )
            return response
        except Exception as e:
            print(f"[agent] Provider {provider['name']} failed: {e}")
            last_error = e
            continue
    raise Exception(f"All fallback models failed. Last error: {last_error}")


async def _chat_with_fallback_stream(
    messages: list,
    tools: list = None,
    tool_choice: str = "auto",
    temperature: float = 0.4,
    report_callback=None,
    on_content=None,
) -> dict:
    """流式版 _chat_with_fallback:逐 token 推送 content,同时拼装 tool_calls。

    遍历 SSE chunk:
    - delta.content -> 累积并经 on_content(piece) 实时推送给前端(逐字显示)。
    - delta.tool_calls -> 按 tc.index 拼装 id / function.name / function.arguments
      (同一工具调用的字段会分多个 chunk 到达,name/id 通常只在首个 chunk 出现)。
    - 记录 finish_reason。

    返回 {"content": str, "tool_calls": list[{id,name,arguments}], "finish_reason": str}。
    当某 provider 在已推送 content 之后失败时,返回 {"content": <已收到的部分>,
    "tool_calls": [], "finish_reason": "error", "error": ...} 而非抛异常,
    避免切换 provider 导致两段输出拼接错乱(降级守卫)。

    前提(OpenAI 兼容 provider 成立):同一轮 content 与 tool_calls 互斥——
    工具调用轮 content 为空,最终回答轮 tool_calls 为空;推理模型的推理 token
    走 reasoning_content,不进 content,故不会污染最终回答流。
    """
    last_error = None
    for provider in FALLBACK_PROVIDERS:
        emitted = False
        content_parts: list[str] = []
        tool_calls_buf: dict[int, dict] = {}
        finish_reason = None
        try:
            if report_callback:
                await report_callback("analyzing", f"正在使用 {provider['name']} 模型思考...")
            client = _get_client(provider["api_key"], provider["base_url"])
            response = await client.chat.completions.create(
                model=provider["model"],
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                stream=True,
            )
            async for chunk in response:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                piece = getattr(delta, "content", None)
                if piece:
                    content_parts.append(piece)
                    emitted = True
                    if on_content:
                        await on_content(piece)

                tcs = getattr(delta, "tool_calls", None)
                if tcs:
                    for tc in tcs:
                        idx = getattr(tc, "index", None)
                        idx = 0 if idx is None else idx
                        slot = tool_calls_buf.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if getattr(tc, "id", None):
                            slot["id"] += tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["name"] += fn.name
                            if getattr(fn, "arguments", None):
                                slot["arguments"] += fn.arguments

            return {
                "content": "".join(content_parts),
                "tool_calls": [tool_calls_buf[i] for i in sorted(tool_calls_buf)],
                "finish_reason": finish_reason,
            }
        except Exception as e:
            print(f"[agent] Provider {provider['name']} stream failed: {e}")
            if emitted:
                # 已向用户推送过 content,不能再换 provider(输出会拼接错乱),
                # 把已收到的部分作为最终结果返回。
                return {
                    "content": "".join(content_parts),
                    "tool_calls": [],
                    "finish_reason": "error",
                    "error": f"{provider['name']} mid-stream: {e}",
                }
            last_error = e
            continue
    raise Exception(f"All fallback models failed. Last error: {last_error}")


async def _chat_non_stream_dict(
    messages: list,
    tools: list = None,
    tool_choice: str = "auto",
    temperature: float = 0.4,
    report_callback=None,
) -> dict:
    """非流式版:复用 _chat_with_fallback(任一 provider 异常即 continue 到下一个,
    返回完整响应或抛错),整理成与 _chat_with_fallback_stream 相同的 dict 形状供
    agent 循环统一消费。

    用于 POST /api/chat(非流式接口, progress_callback=None):该路径未向客户端
    推送任何 token, provider 失败时可安全跨 provider 回退, 不受流式降级守卫(emitted,
    仅在已向客户端推送 content 后阻止换 provider)约束, 行为与改动前的非流式路径一致。
    """
    response = await _chat_with_fallback(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        report_callback=report_callback,
    )
    choice = response.choices[0]
    msg = choice.message
    tool_calls = []
    if msg.tool_calls:
        for i, tc in enumerate(msg.tool_calls):
            tool_calls.append({
                "id": tc.id or f"call_{i}_{uuid.uuid4().hex[:8]}",
                "name": tc.function.name,
                "arguments": tc.function.arguments or "{}",
            })
    return {
        "content": msg.content or "",
        "tool_calls": tool_calls,
        "finish_reason": choice.finish_reason,
    }


def _get_query_hash(query: str, game_name: str) -> str:
    return hashlib.md5(f"{game_name}:{query}".encode("utf-8")).hexdigest()

_client = None
def _get_default_client() -> AsyncOpenAI:

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
    client = _get_default_client()
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
    """LLM 基于检索结果优化生成最终回答，支持流式输出。

    注意:此函数当前未被活跃的 agent 路径(rag_query)调用--agent 循环自身的
    _chat_with_fallback_stream 最后一轮(无 tool_calls)即产出最终回答并逐 token 推送。
    保留此函数供 HyDE 非_agent 管线(见未跟踪的 rag_pipeline_head.py)及评测备用。
    """
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

    client = _get_default_client()
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


SYSTEM_PROMPT_TEMPLATE = """你是一个专业的游戏信息问答助手。你配备了工具，可以检索本地游戏知识库、搜索互联网或保存用户的长期记忆。

【你的核心目标】
准确回答用户的游戏相关问题。

【工具使用说明】
你可以使用以下工具：
1. `search_knowledge_base`: 优先使用此工具在本地数据库查找关于游戏的机制、故事、角色等详情。
2. `search_web`: 如果本地知识库没有提供足够的答案，或者用户询问最新的资讯，请使用此工具搜索互联网。
3. `save_user_preference`: 如果用户提到他们喜欢某类游戏，或者有什么特定的习惯/偏好，请使用此工具记录下来。

【回答要求】
- 当你给出最终回答时，如果你使用了任何工具检索到了信息，请使用 [来源标题](URL) 或 [来源N] 的格式标注引用。
- 请始终用中文回答。
- 如果多次尝试仍未找到信息，请诚实地告诉用户。

【关于当前用户的长期记忆】
以下是你之前了解到的关于当前用户的事实：
{user_memory_context}
请在回答时参考这些偏好。"""

async def rag_query(
    game_name: str,
    user_message: str,
    conversation_id: str,
    progress_callback=None,
) -> dict:
    """Agent loop handling tools, cache, and long-term memory."""
    async def _report(stage: str, message: str, content: str = None) -> None:
        if progress_callback is not None:
            try:
                await progress_callback(stage, message, content)
            except Exception:
                pass

    # 0. 尝试提取游戏名
    history_msgs = db.get_messages(conversation_id, limit=settings.max_history_messages)
    if not game_name or game_name == "":
        extracted_game = await extract_game_name(user_message, history_msgs)
        if extracted_game:
            game_name = extracted_game
            db.update_conversation_game(conversation_id, game_name)
        else:
            conv = db.get_conversation(conversation_id)
            if conv and conv.get("game_name"):
                game_name = conv["game_name"]
            else:
                game_name = ""

    # 1. 检查精确缓存
    query_hash = _get_query_hash(user_message, game_name)
    exact_hit = db.get_exact_query_cache(user_message, game_name)
    if exact_hit:
        await _report("cached", "命中精确缓存，直接返回。")
        db.save_message(conversation_id, "user", user_message, None)
        db.save_message(conversation_id, "assistant", exact_hit, "[]")
        db.update_conversation_timestamp(conversation_id)
        return {"answer": exact_hit, "sources": [], "conversation_id": conversation_id}

    # 2. 检查语义缓存
    query_vec = encode_text(user_message)
    semantic_hit = db.get_semantic_query_cache(query_vec, game_name, threshold=0.85)
    if semantic_hit:
        await _report("cached", "命中语义缓存，直接返回。")
        db.save_message(conversation_id, "user", user_message, None)
        db.save_message(conversation_id, "assistant", semantic_hit, "[]")
        db.update_conversation_timestamp(conversation_id)
        return {"answer": semantic_hit, "sources": [], "conversation_id": conversation_id}

    # 3. 准备 Agent 运行
    db.save_message(conversation_id, "user", user_message, None)
    history_msgs = db.get_messages(conversation_id, limit=settings.max_history_messages)
    
    # 提取长期记忆
    memories = db.get_user_memories(query_vec, top_k=5)
    memory_text = "\n".join([f"- {m}" for m in memories]) if memories else "暂无记录。"
    
    system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{user_memory_context}", memory_text)
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history_msgs:
        if msg["content"] == user_message and msg["role"] == "user":
            continue # Already added as current message
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_message})

    max_loops = 5
    loop_count = 0
    final_answer = ""
    truncated = False       # 回答被截断/异常:不缓存,并向前端透出标志提示用户
    got_real_answer = False # 是否得到完整有效的最终回答(决定是否缓存)

    # 非流式接口(POST /api/chat, progress_callback=None)未向客户端推送 token,
    # 走非流式 _chat_non_stream_dict(任一 provider 失败即回退到下一个,与改动前一致);
    # 流式接口(/api/chat/stream)走 _chat_with_fallback_stream 逐 token 推送。
    streaming = progress_callback is not None

    # 最终回答的逐 token 推送:首个 token 时先发一条无 content 的阶段进度
    # (前端显示"正在生成回答…"),之后每个 token 经 _report(content=token) 实时推送。
    first_token_sent = False

    async def _on_token(token: str) -> None:
        nonlocal first_token_sent
        if not first_token_sent:
            first_token_sent = True
            await _report("generating", "正在生成回答…")
        await _report("generating", "正在生成回答…", content=token)

    while loop_count < max_loops:
        loop_count += 1

        try:
            if streaming:
                result = await _chat_with_fallback_stream(
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.4,
                    report_callback=_report,
                    on_content=_on_token,
                )
            else:
                result = await _chat_non_stream_dict(
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.4,
                    report_callback=_report,
                )
        except Exception as e:
            final_answer = f"大模型调用失败，已尝试降级机制全部失败: {str(e)}"
            truncated = True
            break

        content = result["content"]
        tool_calls = result["tool_calls"]
        finish_reason = result.get("finish_reason")

        if tool_calls:
            # 工具调用轮:解析每个 tool_call 的 id/arguments(对 provider 偶发缺失的
            # 空 id/空 arguments 做兜底,避免回传 API 时 400),拼装 assistant 消息入历史,
            # 执行工具后继续循环。
            resolved = []
            for i, tc in enumerate(tool_calls):
                resolved.append({
                    "id": tc["id"] or f"call_{i}_{uuid.uuid4().hex[:8]}",
                    "name": tc["name"],
                    "arguments": tc["arguments"] or "{}",
                })
            messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {"id": r["id"], "type": "function",
                     "function": {"name": r["name"], "arguments": r["arguments"]}}
                    for r in resolved
                ],
            })
            for r in resolved:
                fn_name = r["name"]
                try:
                    args = json.loads(r["arguments"])
                except json.JSONDecodeError:
                    args = {}

                await _report("tool_call", f"正在使用工具：{fn_name}...")
                print(f"[agent] Calling tool {fn_name} with args {args}")

                tool_result = await execute_tool(fn_name, args)

                messages.append({
                    "tool_call_id": r["id"],
                    "role": "tool",
                    "name": fn_name,
                    "content": tool_result,
                })
        else:
            # 最终回答轮:content 已(流式时)通过 _on_token 逐 token 输出
            final_answer = content or ""
            if result.get("error") or finish_reason in ("length", "content_filter") or not final_answer:
                # 流式中途失败 / token 上限截断 / 内容过滤 / 空回答:视为截断,不缓存
                truncated = True
                if not final_answer:
                    final_answer = "（未生成有效回答，请重试。）"
            else:
                got_real_answer = True
            break

    if loop_count >= max_loops and not final_answer:
        final_answer = "思考过程过长，未能得出最终结论。"
        truncated = True

    # 最终回答已逐 token 流式输出,这里不再整段补发(避免与已推送的 token 重复)。
    # 缓存命中 / 兜底文案等"无 token 流出"的场景,由 done 事件携带 answer 整段显示。

    # 只缓存完整有效的回答(截断/错误/兜底/空内容均不缓存,避免污染后续相同查询)
    if got_real_answer:
        db.set_query_cache(query_hash, game_name, final_answer, user_message, query_vec)

    # Save to db
    db.save_message(conversation_id, "assistant", final_answer, "[]")
    db.update_conversation_timestamp(conversation_id)

    # Auto title for new conversation
    conv = db.get_conversation(conversation_id)
    if conv and conv.get("title") == "New Conversation":
        new_title = user_message[:30] + ("..." if len(user_message) > 30 else "")
        db.update_conversation(conversation_id, new_title)

    return {"answer": final_answer, "sources": [], "conversation_id": conversation_id, "truncated": truncated}
