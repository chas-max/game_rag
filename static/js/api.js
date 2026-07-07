/* ═══════════════════════════════════════════════════════════════════
   API Client — fetch wrapper with consistent error handling.
   ═══════════════════════════════════════════════════════════════════ */

const API = {
  async get(path, params) {
    let url = path;
    if (params) {
      const qs = new URLSearchParams(params).toString();
      url += "?" + qs;
    }
    const resp = await fetch(url);
    const json = await resp.json();
    if (!json.success) throw new Error(json.error || "Request failed");
    return json.data;
  },

  async post(path, body) {
    const resp = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const json = await resp.json();
    if (!json.success) throw new Error(json.error || "Request failed");
    return json.data;
  },

  async put(path, body) {
    const resp = await fetch(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const json = await resp.json();
    if (!json.success) throw new Error(json.error || "Request failed");
    return json.data;
  },

  async del(path) {
    const resp = await fetch(path, { method: "DELETE" });
    const json = await resp.json();
    if (!json.success) throw new Error(json.error || "Request failed");
    return json.data;
  },

  /**
   * POST 并以 SSE 方式读取流式响应。
   * 回调: onProgress(event) / onDone(data) / onError(message)。
   * 用于 /api/chat/stream,实时接收思考阶段。
   */
  async postStream(path, body, { onProgress, onDone, onError } = {}) {
    let resp;
    try {
      resp = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) {
      if (onError) onError("网络错误: " + e.message);
      return;
    }
    if (!resp.ok || !resp.body) {
      if (onError) onError("请求失败: HTTP " + resp.status);
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE 事件以空行分隔
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const line = chunk.trim();
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        let event;
        try {
          event = JSON.parse(payload);
        } catch {
          continue;
        }
        if (event.type === "progress" && onProgress) onProgress(event);
        else if (event.type === "done" && onDone) onDone(event.data);
        else if (event.type === "error" && onError) onError(event.error);
      }
    }
  },
};
