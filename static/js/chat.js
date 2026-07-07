/* ═══════════════════════════════════════════════════════════════════
   Chat Interface — message rendering and input handling.
   ═══════════════════════════════════════════════════════════════════ */

const Chat = {
  render() {
    const list = document.getElementById("message-list");
    list.innerHTML = "";

    // Update game name display
    const gameDisplay = document.getElementById("chat-game-name");
    if (gameDisplay) {
      gameDisplay.textContent = AppState.currentGameName || "未选择游戏";
    }

    if (!AppState.currentMessages || AppState.currentMessages.length === 0) {
      list.appendChild(Components.emptyState());
      return;
    }

    AppState.currentMessages.forEach((msg) => {
      const bubble = Components.messageBubble(
        msg.role,
        msg.content,
        msg.sources,
        msg.created_at
      );
      list.appendChild(bubble);
    });
  },

  scrollToBottom() {
    const list = document.getElementById("message-list");
    requestAnimationFrame(() => {
      list.scrollTop = list.scrollHeight;
    });
  },

  // ── Thinking indicator (shown while the agent is working) ────────
  showThinking(text) {
    this.hideThinking();
    const list = document.getElementById("message-list");
    list.appendChild(Components.thinkingIndicator(text));
  },

  updateThinking(text) {
    const el = document.querySelector("#thinking-indicator .thinking-stage-text");
    if (el) el.textContent = text;
  },

  hideThinking() {
    const el = document.getElementById("thinking-indicator");
    if (el) el.remove();
  },

  setInputEnabled(enabled) {
    document.getElementById("btn-send").disabled = !enabled;
    document.getElementById("input-message").disabled = !enabled;
    document.getElementById("input-game-name").disabled = !enabled;
  },

  async handleSend() {
    const gameNameInput = document.getElementById("input-game-name");
    const messageInput = document.getElementById("input-message");

    const gameName = gameNameInput.value.trim();
    const message = messageInput.value.trim();

    if (!gameName) {
      alert("请输入游戏名称");
      gameNameInput.focus();
      return;
    }
    if (!message) {
      alert("请输入你的问题");
      messageInput.focus();
      return;
    }

    messageInput.value = "";
    messageInput.style.height = "auto";

    await AppState.sendMessage(gameName, message);
    messageInput.focus();
  },

  init() {
    const messageInput = document.getElementById("input-message");
    const btnSend = document.getElementById("btn-send");

    // Send on button click
    btnSend.addEventListener("click", () => this.handleSend());

    // Send on Enter (Shift+Enter for new line)
    messageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.handleSend();
      }
    });

    // Auto-resize textarea
    messageInput.addEventListener("input", () => {
      messageInput.style.height = "auto";
      messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + "px";
    });
  },
};
