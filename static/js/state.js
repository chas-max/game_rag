/* ═══════════════════════════════════════════════════════════════════
   Application State — centralized client-side state management.
   ═══════════════════════════════════════════════════════════════════ */

const AppState = {
  conversations: [],
  currentConversationId: null,
  currentMessages: [],
  isStreaming: false,
  games: [],

  async refreshConversations() {
    this.conversations = await API.get("/api/conversations");
    Sidebar.render();
  },

  async refreshGames() {
    this.games = await API.get("/api/games");
  },

  async loadConversation(convId) {
    const conv = await API.get("/api/conversations/" + convId);
    this.currentConversationId = convId;
    this.currentMessages = conv.messages || [];
    Chat.render();
  },

  async createConversation() {
    const conv = await API.post("/api/conversations", {
      game_name: "",
      title: "New Conversation",
    });
    this.currentConversationId = conv.id;
    this.currentMessages = [];
    await this.refreshConversations();
    Chat.render();
  },

  async sendMessage(message) {
    if (this.isStreaming) return;
    this.isStreaming = true;
    Chat.setInputEnabled(false);

    // If no conversation exists, create one first
    if (!this.currentConversationId) {
      await this.createConversation();
    }

    // Optimistically show user message
    this.currentMessages.push({
      role: "user",
      content: message,
      created_at: new Date().toLocaleString(),
    });
    Chat.render();
    Chat.scrollToBottom();

    // Show thinking indicator — spinner + dynamic stage text streamed from backend
    Chat.showThinking("正在分析问题…");
    Chat.scrollToBottom();

    try {
      let hasStartedContent = false;

      const data = await new Promise((resolve, reject) => {
        API.postStream(
          "/api/chat/stream",
          {
            conversation_id: this.currentConversationId,
            game_name: "",
            message: message,
          },
          {
            onProgress: (ev) => {
              if (ev.content) {
                Chat.hideThinking();
                if (!hasStartedContent) {
                  hasStartedContent = true;
                  AppState.currentMessages.push({
                    role: "assistant",
                    content: ev.content,
                    sources: null,
                    created_at: new Date().toLocaleString(),
                  });
                  Chat.render();
                } else {
                  const lastMsg = AppState.currentMessages[AppState.currentMessages.length - 1];
                  lastMsg.content += ev.content;
                  
                  // Dynamically update DOM for smooth rendering
                  const list = document.getElementById("message-list");
                  const assistantMsgs = list.querySelectorAll(".message.assistant:not(#thinking-indicator)");
                  if (assistantMsgs.length > 0) {
                    const lastBubble = assistantMsgs[assistantMsgs.length - 1];
                    const contentEl = lastBubble.querySelector(".message-content");
                    if (contentEl) {
                      contentEl.innerHTML = Components.renderMarkdown(lastMsg.content);
                    }
                  }
                }
                Chat.scrollToBottom();
              } else {
                Chat.updateThinking(ev.message);
              }
            },
            onDone: (data) => resolve(data),
            onError: (err) => reject(new Error(err)),
          }
        );
      });

      // Add assistant response or finalize streaming response
      const lastMsg = AppState.currentMessages[AppState.currentMessages.length - 1];
      if (lastMsg && lastMsg.role === "assistant") {
        lastMsg.content = data.answer;
        lastMsg.sources = data.sources ? JSON.stringify(data.sources) : null;
      } else {
        AppState.currentMessages.push({
          role: "assistant",
          content: data.answer,
          sources: data.sources ? JSON.stringify(data.sources) : null,
          created_at: new Date().toLocaleString(),
        });
      }

      await this.refreshConversations();
      // Update the active conversation item in sidebar
      document.querySelectorAll(".conversation-item").forEach(el => {
        if (parseInt(el.dataset.id) === this.currentConversationId) {
          el.classList.add("active");
        }
      });
    } catch (err) {
      this.currentMessages.push({
        role: "assistant",
        content: "请求失败: " + err.message,
        sources: null,
        created_at: new Date().toLocaleString(),
      });
    } finally {
      Chat.hideThinking();
      this.isStreaming = false;
      Chat.setInputEnabled(true);
      Chat.render();
      Chat.scrollToBottom();
    }
  },

  clearCurrent() {
    this.currentConversationId = null;
    this.currentMessages = [];
    document.getElementById("input-message").value = "";
    Chat.render();
  },
};
