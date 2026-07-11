"""Tools for the ReAct Agent."""

import json
from duckduckgo_search import DDGS

from app import database as db
from app.embedding import encode_text
from config import settings


async def search_knowledge_base(query: str, game_name: str) -> str:
    """Search the local knowledge base for game information.
    
    Args:
        query: The question or keyword to search for.
        game_name: The name of the game.
    """
    try:
        query_vec = encode_text(query)
        # We use a similar retrieval logic as before, searching semantic chunks
        threshold = settings.similarity_threshold
        matches = db.search_similar_semantic(query_vec, game_name, top_k=settings.top_k, threshold=threshold)
        
        if not matches:
            # lower threshold fallback
            lowered = max(0.1, threshold - 0.15)
            matches = db.search_similar_semantic(query_vec, game_name, top_k=settings.top_k, threshold=lowered)
            
        if not matches:
            return "当前数据库知识不足，未能检索到相关信息。"
            
        # Build context
        seen_parent_ids = set()
        context_parts = []
        for match in matches:
            doc_id = match["document_id"]
            if doc_id in seen_parent_ids:
                continue
            seen_parent_ids.add(doc_id)
            
            context_docs = db.get_document_with_context(doc_id)
            if not context_docs:
                continue
                
            combined = "\n\n".join([d["content"] for d in context_docs])
            main_doc = next((d for d in context_docs if d["id"] == doc_id), context_docs[0])
            title = main_doc.get("title") or "未命名"
            url = main_doc.get("url") or ""
            
            context_parts.append(f"[来源: {title} ({url})]\n{combined}")
            
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        return f"Tool execution failed: {str(e)}. Please adjust parameters and try again."


async def search_web(query: str) -> str:
    """Search the internet for up-to-date information.

    优先使用 Tavily (配置了 TAVILY_API_KEY 时), 否则回退到 DuckDuckGo。

    Args:
        query: The search query string.
    """
    if settings.tavily_api_key:
        try:
            return await _tavily_search(query)
        except Exception as e:
            # Tavily 失败时回退 DuckDuckGo,保证工具可用
            print(f"[tool] Tavily search failed, falling back to DuckDuckGo: {e}")
    return await _ddg_search(query)


async def _tavily_search(query: str) -> str:
    """Tavily 网络检索。"""
    from tavily import AsyncTavilyClient  # 延迟导入,避免未安装时影响模块加载

    client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    res = await client.search(query, max_results=5, search_depth="basic")
    results = res.get("results") or []
    if not results:
        return "No web search results found."
    parts = []
    for r in results:
        title = r.get("title", "")
        href = r.get("url", "")
        body = r.get("content", "")
        parts.append(f"[{title}]({href}): {body}")
    return "\n\n".join(parts)


async def _ddg_search(query: str) -> str:
    """DuckDuckGo 网络检索 (Tavily 未配置或失败时的回退)。"""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No web search results found."

        parts = []
        for r in results:
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")
            parts.append(f"[{title}]({href}): {body}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Web search failed: {str(e)}. The internet might be unreachable."


async def save_user_preference(fact: str) -> str:
    """Save a user preference or fact to long-term memory.
    
    Args:
        fact: A concise statement of the user's preference or fact (e.g., 'User likes RPG games').
    """
    try:
        emb = encode_text(fact)
        db.add_user_memory(fact, emb)
        return "Successfully saved to long-term memory."
    except Exception as e:
        return f"Failed to save preference: {str(e)}"

# Define standard tool schemas for OpenAI/MIMO/Qwen API
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索本地游戏知识库以获取详尽的游戏内容、机制、设定等信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "具体的搜索关键词或问题",
                    },
                    "game_name": {
                        "type": "string",
                        "description": "游戏名称",
                    },
                },
                "required": ["query", "game_name"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "当本地知识库没有答案时，使用此工具搜索互联网获取最新信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要在搜索引擎中输入的查询语句",
                    },
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_preference",
            "description": "如果用户在对话中透露了关于自己的偏好、习惯或事实，调用此工具将其保存到长期记忆中。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "关于用户的偏好或事实的简练总结（例如：'用户喜欢玩RPG游戏'）",
                    },
                },
                "required": ["fact"],
            },
        }
    }
]

async def execute_tool(name: str, arguments: dict) -> str:
    if name == "search_knowledge_base":
        return await search_knowledge_base(**arguments)
    elif name == "search_web":
        return await search_web(**arguments)
    elif name == "save_user_preference":
        return await save_user_preference(**arguments)
    else:
        return f"Unknown tool: {name}"
