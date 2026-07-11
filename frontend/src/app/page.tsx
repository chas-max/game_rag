"use client";

import { useChat } from "ai/react";
import { useState, useEffect, useRef } from "react";
import { Database, Plus, Send, X } from "lucide-react";
import { Markdown } from "./components/Markdown";

export default function ChatApp() {
  const [conversations, setConversations] = useState<any[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(null);
  const [showKbModal, setShowKbModal] = useState(false);
  const [kbStats, setKbStats] = useState<any>(null);

  const { messages, input, handleInputChange, handleSubmit, setMessages, isLoading, data, append, setInput, setData } = useChat({
    api: "/api/chat",
    body: {
      conversationId: currentConversationId
    },
    onFinish: (message) => {
      fetchConversations(); 
    }
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, data]);

  const fetchConversations = async (selectFirstIfNone = false) => {
    try {
      const res = await fetch("http://localhost:8000/api/conversations");
      const data = await res.json();
      if (data.success) {
        setConversations(data.data);
        if (selectFirstIfNone && data.data.length > 0 && !currentConversationId) {
          selectConversation(data.data[0].id);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchConversations(true);
  }, []);

  const handleNewConversation = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/conversations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ game_name: "", title: "New Conversation" })
      });
      const data = await res.json();
      if (data.success) {
        setCurrentConversationId(data.data.id);
        setMessages([]);
        setData([]); // 新建对话,清空残留载荷
        fetchConversations();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const selectConversation = async (id: number) => {
    setCurrentConversationId(id);
    try {
      const res = await fetch(`http://localhost:8000/api/conversations/${id}`);
      const data = await res.json();
      if (data.success) {
        const msgs = data.data.messages.map((m: any) => ({
          id: m.id.toString(),
          role: m.role,
          content: m.content
        }));
        setMessages(msgs);
        setData([]); // 切换对话时清空上一对话残留的 progress/done 载荷,避免来源错位
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchKbStats = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/knowledge/stats");
      const data = await res.json();
      if (data.success) {
        setKbStats(data.data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Form submission handler to prevent default behavior when Enter is pressed
  const onFormSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const messageContent = input.trim();
    if (!messageContent) return;

    let targetConvId = currentConversationId;
    if (!targetConvId) {
      try {
        const res = await fetch("http://localhost:8000/api/conversations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ game_name: "", title: messageContent.slice(0, 30) })
        });
        const data = await res.json();
        if (data.success) {
          targetConvId = data.data.id;
          setCurrentConversationId(targetConvId);
          await fetchConversations();
        }
      } catch (err) {
        console.error("Failed to auto-create conversation:", err);
        return;
      }
    }

    setInput(""); // clear input
    append({
      role: "user",
      content: messageContent
    }, {
      options: {
        body: {
          conversationId: targetConvId
        }
      }
    });
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const form = e.currentTarget.form;
      if (form) form.requestSubmit();
    }
  };

  // data 累积了整段会话的 2:[...] 载荷;取最新的 progress / done 用于当前流式消息。
  // ai v3 的 data 是 JSONValue[],这里以 any 取用,避免联合类型属性访问报错。
  const latestProgress: any = data
    ? [...data].reverse().find((d: any) => d?.type === "progress")
    : null;
  const latestDone: any = data
    ? [...data].reverse().find((d: any) => d?.type === "done")
    : null;
  const latestSources = latestDone?.sources ?? [];

  return (
    <>
      <header id="top-bar">
        <h1>游戏信息问答智能体</h1>
      </header>

      <aside id="sidebar">
        <div className="sidebar-header">
          <button id="btn-new-conversation" onClick={handleNewConversation}>
            <Plus size={14} style={{ display: 'inline', verticalAlign: 'text-bottom' }} /> 新建对话
          </button>
        </div>
        <div id="conversation-list">
          {conversations.map(c => (
            <div 
              key={c.id} 
              className={`conversation-item ${currentConversationId === c.id ? 'active' : ''}`}
              onClick={() => selectConversation(c.id)}
            >
              <div className="title" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {c.title}
              </div>
              <div className="meta">{c.message_count} messages</div>
            </div>
          ))}
        </div>
      </aside>

      <main id="chat-area">
        <div id="chat-header">
          <span className="game-name" id="chat-game-name">全局查询模式 (无需指定游戏)</span>
          <button id="btn-manage-sources" onClick={() => { setShowKbModal(true); fetchKbStats(); }}>
            <Database size={14} style={{ display: 'inline', verticalAlign: 'text-bottom' }} /> 知识库管理
          </button>
        </div>

        <div id="message-list">
          {messages.length === 0 && !isLoading && (
            <div className="message system-message">
              欢迎使用全局查询模式！直接输入您想了解的任何游戏相关问题。
            </div>
          )}
          {messages.map((m, index) => {
            const isLast = index === messages.length - 1;
            const isStreamingAssistant = isLast && isLoading && m.role === "assistant";
            const showThinking = isStreamingAssistant && !m.content;
            const showCursor = isStreamingAssistant && !!m.content;
            const isLastAssistant = isLast && m.role === "assistant";
            return (
              <div key={index} className={`message ${m.role === 'user' ? 'user' : 'assistant'}`}>
                <div className="message-role">{m.role === 'user' ? '你' : '助手'}</div>
                <div className="message-content">
                  {m.role === 'assistant' ? (
                    showThinking ? (
                      <div className="thinking-row">
                        <span className="spinner" />
                        <span className="thinking-stage-text">
                          {latestProgress?.message || '正在思考…'}
                        </span>
                      </div>
                    ) : (
                      <>
                        <Markdown content={m.content} />
                        {showCursor && <span className="streaming-cursor-dot" aria-hidden>▋</span>}
                      </>
                    )
                  ) : (
                    m.content
                  )}
                </div>

                {isLastAssistant && !isLoading && latestSources.length > 0 && (
                  <div className="message-sources">
                    {latestSources.map((s: any, idx: number) => (
                      <a
                        key={idx}
                        className="source-badge"
                        href={s.url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {s.title || '未知来源'}
                      </a>
                    ))}
                  </div>
                )}
                {isLastAssistant && !isLoading && latestDone?.truncated && (
                  <div className="message-truncated">⚠ 回答因模型异常被截断，请重试</div>
                )}
              </div>
            );
          })}
          {isLoading && (messages.length === 0 || messages[messages.length - 1]?.role !== 'assistant') && (
            <div className="message assistant">
              <div className="message-role">助手</div>
              <div className="message-content">
                <div className="thinking-row">
                  <span className="spinner" />
                  <span className="thinking-stage-text">
                    {latestProgress?.message || '正在思考…'}
                  </span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div id="chat-input-area">
          <form onSubmit={onFormSubmit} style={{ display: 'flex', width: '100%', gap: '10px' }}>
            <textarea
              id="input-message"
              value={input}
              onChange={handleInputChange}
              onKeyDown={onKeyDown}
              placeholder="输入你的问题，例如：塞尔达传说旷野之息 如何获得大师剑？"
              rows={1}
              style={{ flex: 1 }}
            ></textarea>
            <button type="submit" id="btn-send" disabled={isLoading || !input.trim()}>
              <Send size={14} style={{ display: 'inline', verticalAlign: 'text-bottom' }} /> 发送
            </button>
          </form>
        </div>
      </main>

      {/* Knowledge Base Modal */}
      {showKbModal && (
        <div id="source-modal-overlay" className="modal-overlay" style={{ display: 'flex' }}>
          <div className="modal modal-wide">
            <h2 style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 0 }}>
              <span><Database size={20} style={{ display: 'inline', verticalAlign: 'middle' }} /> 知识库管理</span>
              <button onClick={() => setShowKbModal(false)} style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: 0 }}>
                <X size={24} />
              </button>
            </h2>

            <div id="knowledge-stats" className="stats-row">
              <div className="stat-card">
                <div className="stat-value">{kbStats?.total_games || 0}</div>
                <div className="stat-label">已收录游戏</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{kbStats?.total_documents || 0}</div>
                <div className="stat-label">知识文档块</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{kbStats?.pending_queries || 0}</div>
                <div className="stat-label">待学习问题</div>
              </div>
            </div>

            <div className="kb-actions">
              <button className="kb-btn primary">🔄 获取热门游戏 (开发中)</button>
              <button className="kb-btn">📚 处理待学习问题 (开发中)</button>
            </div>

            <div className="modal-footer">
              <button className="btn-close" onClick={() => setShowKbModal(false)}>关闭</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
