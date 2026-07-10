"use client";

import { useChat } from "ai/react";
import { useState, useEffect, useRef } from "react";
import { Database, Plus, Send, X } from "lucide-react";

export default function ChatApp() {
  const [conversations, setConversations] = useState<any[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(null);
  const [showKbModal, setShowKbModal] = useState(false);
  const [kbStats, setKbStats] = useState<any>(null);

  const { messages, input, handleInputChange, handleSubmit, setMessages, isLoading, data, append, setInput } = useChat({
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
          {messages.map((m, index) => (
            <div key={index} className={`message ${m.role === 'user' ? 'user' : 'assistant'}`}>
              <div className="message-role">{m.role === 'user' ? '你' : '助手'}</div>
              <div className="message-content">{m.content}</div>
              
              {/* Render custom Vercel AI SDK data sources if available on the last assistant message */}
              {m.role === 'assistant' && index === messages.length - 1 && data && (
                <div className="message-sources" style={{marginTop: '10px', padding: '10px', background: '#f5f5f5', borderRadius: '4px', fontSize: '0.85em', color: '#666'}}>
                  {data.map((d: any, i) => (
                    <div key={i}>
                      {d.type === 'progress' && <span><strong>思考阶段:</strong> {d.message}</span>}
                      {d.type === 'done' && d.sources && d.sources.length > 0 && (
                        <div style={{marginTop: '5px'}}>
                          <strong>参考来源:</strong>
                          <ul style={{margin: '5px 0', paddingLeft: '20px'}}>
                            {d.sources.map((s: any, idx: number) => (
                              <li key={idx}>
                                <a href={s.url} target="_blank" rel="noreferrer" style={{color: '#0070f3'}}>{s.title || '未知来源'}</a>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="message assistant">
              <div className="message-role">助手</div>
              <div className="message-content typing-indicator">
                <span></span><span></span><span></span>
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
