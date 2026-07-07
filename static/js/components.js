/* ═══════════════════════════════════════════════════════════════════
   DOM Components — reusable element builders.
   ═══════════════════════════════════════════════════════════════════ */

const Components = {
  /**
   * Build a message bubble DOM element.
   */
  /**
   * Simple regex-based Markdown to HTML renderer.
   * Handles escaping, paragraphs, headers, bold, italics, inline code, lists, and citation badges.
   */
  renderMarkdown(text) {
    if (!text) return "";

    // Escape HTML first to prevent XSS
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Convert headers: ### Header
    html = html.replace(/^### (.*?)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.*?)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.*?)$/gm, "<h1>$1</h1>");

    // Convert bold: **text**
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Convert italic: *text*
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");

    // Convert inline code: `code`
    html = html.replace(/`(.*?)`/g, "<code>$1</code>");

    // Convert list items: - item or * item
    html = html.replace(/^\s*[-*]\s+(.*?)$/gm, "<li>$1</li>");

    // Convert citation tags [1], [2] to styled reference badges
    html = html.replace(/\[(\d+)\]/g, '<span class="citation-ref">[$1]</span>');

    // Split by double newlines into blocks (paragraphs, lists, headings)
    const blocks = html.split(/\n\n+/);
    const parsedBlocks = blocks.map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return "";

      if (trimmed.startsWith("<h") || trimmed.startsWith("<li")) {
        return trimmed;
      }

      if (trimmed.includes("<li>")) {
        return `<ul>${trimmed}</ul>`;
      }

      return `<p>${trimmed}</p>`;
    });

    return parsedBlocks.join("");
  },

  messageBubble(role, content, sourcesJson, createdAt) {
    const div = document.createElement("div");
    div.className = "message " + role;

    // Role label
    const roleEl = document.createElement("div");
    roleEl.className = "message-role";
    roleEl.textContent = role === "user" ? "你" : "🤖 游戏问答助手";
    div.appendChild(roleEl);

    // Content
    const contentEl = document.createElement("div");
    contentEl.className = "message-content";
    contentEl.innerHTML = this.renderMarkdown(content);
    div.appendChild(contentEl);

    // Sources (assistant only)
    if (role === "assistant" && sourcesJson) {
      try {
        const sources = typeof sourcesJson === "string" ? JSON.parse(sourcesJson) : sourcesJson;
        if (sources && sources.length > 0) {
          const sourcesDiv = document.createElement("div");
          sourcesDiv.className = "message-sources";
          sources.forEach((src, i) => {
            const badge = document.createElement("a");
            badge.className = "source-badge";
            badge.textContent = `[${i + 1}] ${src.title || "来源"}`;
            badge.title = src.url || "";
            if (src.url) {
              badge.href = src.url;
              badge.target = "_blank";
              badge.rel = "noopener";
            }
            sourcesDiv.appendChild(badge);
          });
          div.appendChild(sourcesDiv);
        }
      } catch (e) {
        // ignore parse errors
      }
    }

    // Timestamp
    if (createdAt) {
      const timeEl = document.createElement("div");
      timeEl.className = "message-time";
      timeEl.textContent = createdAt;
      div.appendChild(timeEl);
    }

    return div;
  },

  /**
   * Build a conversation item for the sidebar.
   */
  conversationItem(conv) {
    const div = document.createElement("div");
    div.className = "conversation-item";
    div.dataset.id = conv.id;
    if (conv.id === AppState.currentConversationId) {
      div.classList.add("active");
    }

    div.addEventListener("click", (e) => {
      if (e.target.closest(".btn-delete")) return;
      AppState.loadConversation(conv.id);
    });

    // Title
    const title = document.createElement("div");
    title.className = "conv-title";
    title.textContent = conv.title || "New Conversation";
    div.appendChild(title);

    // Meta row
    const meta = document.createElement("div");
    meta.className = "conv-meta";

    if (conv.game_name) {
      const game = document.createElement("span");
      game.className = "conv-game";
      game.textContent = conv.game_name;
      meta.appendChild(game);
    }

    const count = document.createElement("span");
    count.textContent = (conv.message_count || 0) + " 条消息";
    meta.appendChild(count);

    div.appendChild(meta);

    // Delete button
    const delBtn = document.createElement("button");
    delBtn.className = "btn-delete";
    delBtn.textContent = "×";
    delBtn.title = "删除对话";
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      Sidebar.deleteConversation(conv.id);
    });
    div.appendChild(delBtn);

    return div;
  },

  /**
   * Show a loading spinner in the message area.
   */
  loadingSpinner() {
    const div = document.createElement("div");
    div.className = "loading-indicator";
    div.id = "loading-spinner";
    const spinner = document.createElement("div");
    spinner.className = "spinner";
    div.appendChild(spinner);
    const text = document.createElement("span");
    text.textContent = "正在检索游戏信息...";
    div.appendChild(text);
    return div;
  },

  /**
   * Build a "thinking" indicator bubble — assistant-style message with a
   * spinning circle and dynamic stage text. Not part of currentMessages;
   * appended/removed directly so Chat.render() won't wipe it mid-flight.
   */
  thinkingIndicator(initialText = "正在思考…") {
    const div = document.createElement("div");
    div.className = "message assistant thinking";
    div.id = "thinking-indicator";

    const roleEl = document.createElement("div");
    roleEl.className = "message-role";
    roleEl.textContent = "🤖 游戏问答助手";
    div.appendChild(roleEl);

    const contentEl = document.createElement("div");
    contentEl.className = "message-content thinking-content";

    const row = document.createElement("div");
    row.className = "thinking-row";
    const spinner = document.createElement("div");
    spinner.className = "spinner";
    row.appendChild(spinner);
    const stageText = document.createElement("span");
    stageText.className = "thinking-stage-text";
    stageText.textContent = initialText;
    row.appendChild(stageText);
    contentEl.appendChild(row);

    const detail = document.createElement("div");
    detail.className = "thinking-detail";
    detail.innerHTML =
      '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    contentEl.appendChild(detail);

    div.appendChild(contentEl);
    return div;
  },

  /**
   * Empty state placeholder.
   */
  emptyState() {
    const div = document.createElement("div");
    div.id = "empty-chat";
    div.innerHTML = `
      <div class="icon">🎮</div>
      <div class="text">选择一个游戏，开始提问吧</div>
      <div class="hint">输入游戏名称和你想要了解的信息</div>
    `;
    return div;
  },
};
