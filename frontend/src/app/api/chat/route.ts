export async function POST(req: Request) {
  const { messages, conversationId } = await req.json();
  const lastMessage = messages[messages.length - 1];

  const response = await fetch('http://localhost:8000/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      conversation_id: conversationId || 0, // Fallback to 0 if not provided
      game_name: "", // Empty for global search
      message: lastMessage.content
    })
  });

  if (!response.body) {
    throw new Error('No response body from backend');
  }

  const stream = new ReadableStream({
    async start(controller) {
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      
      let buffer = "";
      let hasStreamedTokens = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        buffer = lines.pop() || "";
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const payload = JSON.parse(line.slice(6));
              if (payload.type === 'progress') {
                if (payload.content) {
                  hasStreamedTokens = true;
                  // Text chunk format: 0:"content"
                  controller.enqueue(new TextEncoder().encode(`0:${JSON.stringify(payload.content)}\n`));
                } else {
                  // Data chunk format: 2:[...] (formerly d:)
                  controller.enqueue(new TextEncoder().encode(`2:${JSON.stringify([{
                    type: 'progress',
                    stage: payload.stage,
                    message: payload.message
                  }])}\n`));
                }
              } else if (payload.type === 'done') {
                // If we didn't stream any tokens (e.g. database insufficient or error fallback),
                // we send the entire answer as a single text chunk now.
                if (!hasStreamedTokens && payload.data && payload.data.answer) {
                  controller.enqueue(new TextEncoder().encode(`0:${JSON.stringify(payload.data.answer)}\n`));
                }
                
                controller.enqueue(new TextEncoder().encode(`2:${JSON.stringify([{ 
                  type: 'done', 
                  sources: payload.data.sources,
                  conversation_id: payload.data.conversation_id
                }])}\n`));
              } else if (payload.type === 'error') {
                // Error chunk format: 3:"message"
                controller.enqueue(new TextEncoder().encode(`3:${JSON.stringify(payload.error)}\n`));
              }
            } catch (err) {
              console.error("Error parsing SSE JSON:", err);
            }
          }
        }
      }
      controller.close();
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'x-vercel-ai-data-stream': 'v1',
    }
  });
}
