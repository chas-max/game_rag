"""ReAct Agent loop with tools, memory, and caching."""

import asyncio
import json
import hashlib

import httpx
from openai import AsyncOpenAI

from app import database as db
from app.embedding import encode_text
from app.tools import execute_tool, TOOLS_SCHEMA
from config import settings

FALLBACK_PROVIDERS = [
    {
        "name": "mimo",
        "api_key": settings.mimo_api_key,
        "base_url": settings.mimo_base_url,
        "model": settings.mimo_model,
    },
    {
        "name": "qwen",
        "api_key": settings.qwen_api_key,
        "base_url": settings.qwen_base_url,
        "model": settings.qwen_model,
    },
    {
        "name": "deepseek",
        "api_key": settings.deepseek_api_key,
        "base_url": settings.deepseek_base_url,
        "model": settings.deepseek_model,
    },
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


def _get_query_hash(query: str, game_name: str) -> str:
    return hashlib.md5(f"{game_name}:{query}".encode("utf-8")).hexdigest()


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
    
    while loop_count < max_loops:
        loop_count += 1
        
        try:
            response = await _chat_with_fallback(
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.4,
                report_callback=_report
            )
        except Exception as e:
            final_answer = f"大模型调用失败，已尝试降级机制全部失败: {str(e)}"
            break
            
        choice = response.choices[0]
        msg = choice.message
        
        messages.append(msg)
        
        if msg.tool_calls:
            # Execute tools
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                await _report("tool_call", f"正在使用工具：{fn_name}...")
                print(f"[agent] Calling tool {fn_name} with args {args}")
                
                tool_result = await execute_tool(fn_name, args)
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": tool_result,
                })
        else:
            # Final answer
            final_answer = msg.content or ""
            break

    if loop_count >= max_loops and not final_answer:
        final_answer = "思考过程过长，未能得出最终结论。"

    # Stream the final answer if needed (we'll just report it as a chunk for now)
    await _report("generating", "生成完毕。", content=final_answer)

    # Cache the result
    db.set_query_cache(query_hash, game_name, final_answer, user_message, query_vec)

    # Save to db
    db.save_message(conversation_id, "assistant", final_answer, "[]")
    db.update_conversation_timestamp(conversation_id)

    # Auto title for new conversation
    conv = db.get_conversation(conversation_id)
    if conv and conv.get("title") == "New Conversation":
        new_title = user_message[:30] + ("..." if len(user_message) > 30 else "")
        db.update_conversation(conversation_id, new_title)

    return {"answer": final_answer, "sources": [], "conversation_id": conversation_id}
