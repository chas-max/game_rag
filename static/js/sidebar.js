/* ═══════════════════════════════════════════════════════════════════
   Sidebar — conversation list rendering and management.
   ═══════════════════════════════════════════════════════════════════ */

const Sidebar = {
  render() {
    const list = document.getElementById("conversation-list");
    list.innerHTML = "";

    if (!AppState.conversations || AppState.conversations.length === 0) {
      const empty = document.createElement("div");
      empty.style.cssText = "text-align:center;padding:40px 16px;color:#90a4ae;font-size:13px;";
      empty.textContent = "暂无对话记录\n点击上方按钮新建";
      list.appendChild(empty);
      return;
    }

    AppState.conversations.forEach((conv) => {
      list.appendChild(Components.conversationItem(conv));
    });
  },

  async deleteConversation(convId) {
    if (!confirm("确定要删除此对话吗？")) return;
    try {
      await API.del("/api/conversations/" + convId);
      if (AppState.currentConversationId === convId) {
        AppState.clearCurrent();
      }
      await AppState.refreshConversations();
    } catch (err) {
      alert("删除失败: " + err.message);
    }
  },

  init() {
    document.getElementById("btn-new-conversation").addEventListener("click", () => {
      AppState.clearCurrent();
      document.getElementById("input-message").focus();
    });
  },
};
