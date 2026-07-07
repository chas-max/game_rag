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

    通过 Server-Sent Events 向前端推送两类事件:
    - {"type": "progress", "stage": "...", "message": "..."}  思考阶段进度
    - {"type": "done",    "data": {...}}                       最终回答
    - {"type": "error",   "error": "..."}                      异常

    使用 asyncio.Queue 解耦 pipeline 执行与 HTTP 流: pipeline 在后台任务中
    运行,通过 progress_callback 把阶段事件放入队列; event_stream 从队列
    读取并逐条 yield,收到 done/error 后结束。
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def progress(stage: str, message: str, content: str = None) -> None:
        event = {"type": "progress", "stage": stage, "message": message}
        if content is not None:
            event["content"] = content
        await queue.put(event)

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
