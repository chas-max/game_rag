"""Chat route — the primary RAG query endpoint.

提供两个接口:
- POST /api/chat        : 一次性返回完整结果(保留向后兼容)
- POST /api/chat/stream : SSE 流式接口,实时推送思考阶段 + 最终结果
                         前端据此显示"思考中"转圈动画与当前阶段
"""

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models import ApiResponse, ChatRequest
from app.rag_pipeline import rag_query

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(req: ChatRequest):
    try:
        result = await rag_query(
            game_name=req.game_name,
            user_message=req.message,
            conversation_id=req.conversation_id,
        )
        return ApiResponse(success=True, data=result)
    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """流式聊天接口。

    通过 Server-Sent Events 向前端推送事件:
    - {"type": "progress", "stage": "...", "message": "..."}  思考阶段进度(无 content)
    - {"type": "token",    "content": "..."}                  最终回答的逐 token 文本
    - {"type": "done",     "data": {...}}                     最终回答完成(含 answer/sources)
    - {"type": "error",    "error": "..."}                    异常

    pipeline 通过 progress_callback 把阶段事件与 token 放入队列; event_stream
    从队列读取并逐条 yield,收到 done/error 后结束。
    token 事件刻意精简(只含 content),降低逐 token 推送时的冗余载荷。
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def progress(stage: str, message: str, content: str = None) -> None:
        if content is not None:
            # 最终回答的 token:走精简的 token 事件,前端转成 0:"content" 逐字追加
            await queue.put({"type": "token", "content": content})
        else:
            await queue.put({"type": "progress", "stage": stage, "message": message})

    async def run_pipeline() -> None:
        try:
            result = await rag_query(
                game_name=req.game_name,
                user_message=req.message,
                conversation_id=req.conversation_id,
                progress_callback=progress,
            )
            await queue.put({"type": "done", "data": result})
        except Exception as e:
            await queue.put({"type": "error", "error": str(e)})

    async def event_stream():
        task = asyncio.create_task(run_pipeline())
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
        finally:
            # 确保 pipeline 任务完成(即使客户端中途断开,DB 写入也已落盘)
            await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用反向代理缓冲,保证实时推送
        },
    )
