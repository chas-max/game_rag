/* ═══════════════════════════════════════════════════════════════════
   App Bootstrap — initialization and event wiring.
   ═══════════════════════════════════════════════════════════════════ */

const App = {
  async init() {
    // Initialize sub-modules
    Chat.init();
    Sidebar.init();
    this.initKnowledgeModal();

    // Load initial data
    try {
      await AppState.refreshConversations();
      await AppState.refreshGames();
    } catch (err) {
      console.error("Failed to load initial data:", err);
    }

    // Render initial empty state
    Chat.render();
  },



  // ── Knowledge Modal ─────────────────────────────────────────────
  initKnowledgeModal() {
    const btnManage = document.getElementById("btn-manage-sources");
    const overlay = document.getElementById("source-modal-overlay");
    const btnClose = overlay.querySelector(".btn-close");

    btnManage.addEventListener("click", () => {
      overlay.classList.remove("hidden");
      this.loadKnowledgeData();
    });

    btnClose.addEventListener("click", () => {
      overlay.classList.add("hidden");
    });

    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.classList.add("hidden");
    });

    // Tab switching
    overlay.querySelectorAll(".kb-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        overlay.querySelectorAll(".kb-tab").forEach((t) => t.classList.remove("active"));
        overlay.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
      });
    });

    // Action buttons
    document.getElementById("btn-refresh-trending").addEventListener("click", () =>
      this.knowledgeAction("/api/knowledge/refresh-trending", "正在获取热门游戏知识,这可能需要1-2分钟...")
    );
    document.getElementById("btn-process-pending").addEventListener("click", () =>
      this.knowledgeAction("/api/knowledge/process-pending", "正在处理待学习问题...")
    );
    document.getElementById("btn-trigger-cycle").addEventListener("click", async () => {
      try {
        await API.post("/api/knowledge/cycle", {});
        this.setKbStatus("后台周期已触发,请稍后刷新查看结果");
      } catch (err) {
        alert("触发失败: " + err.message);
      }
    });

    // Add game manually
    const addInput = document.getElementById("input-add-game");
    const addBtn = document.getElementById("btn-add-game");
    const addGame = async () => {
      const name = addInput.value.trim();
      if (!name) {
        alert("请输入游戏名称");
        return;
      }
      addInput.value = "";
      this.setKbStatus(`正在获取《${name}》的知识,请稍候...`);
      try {
        const result = await API.post("/api/knowledge/games", { game_name: name, force: false });
        this.setKbStatus(`《${name}》获取完成,共 ${result.chunks} 个知识块`);
        await this.loadKnowledgeData();
        await AppState.refreshGames();
      } catch (err) {
        this.setKbStatus(`获取失败: ${err.message}`);
      }
    };
    addBtn.addEventListener("click", addGame);
    addInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") addGame();
    });
  },

  async knowledgeAction(endpoint, statusMsg) {
    this.setKbStatus(statusMsg);
    this.setKbButtonsDisabled(true);
    try {
      const result = await API.post(endpoint, {});
      const summary = this.summarizeKnowledgeResult(result);
      this.setKbStatus(summary);
      await this.loadKnowledgeData();
      await AppState.refreshGames();
    } catch (err) {
      this.setKbStatus("操作失败: " + err.message);
    } finally {
      this.setKbButtonsDisabled(false);
    }
  },

  summarizeKnowledgeResult(result) {
    if (result.pending) {
      const p = result.pending;
      return `反馈学习完成: 处理 ${p.processed} 个游戏, ${p.total_questions} 个问题`;
    }
    if (result.results) {
      const completed = result.results.filter((r) => r.status === "completed").length;
      const skipped = result.results.filter((r) => r.status === "skipped").length;
      const failed = result.results.filter((r) => r.status === "failed").length;
      return `热门游戏获取完成: 发现 ${result.discovered} 个, 新获取 ${completed} 个, 已有 ${skipped} 个, 失败 ${failed} 个`;
    }
    return "操作完成";
  },

  setKbStatus(text) {
    document.getElementById("kb-status-text").textContent = text;
  },

  setKbButtonsDisabled(disabled) {
    document.querySelectorAll(".kb-actions .kb-btn").forEach((b) => (b.disabled = disabled));
  },

  async loadKnowledgeData() {
    await Promise.all([
      this.loadKnowledgeStatus(),
      this.loadPendingQueries(),
      this.loadKnowledgeLogs(),
    ]);
  },

  async loadKnowledgeStatus() {
    try {
      const stats = await API.get("/api/knowledge/status");
      document.getElementById("stat-games").textContent = stats.total_games;
      document.getElementById("stat-docs").textContent = stats.total_documents;
      document.getElementById("stat-pending").textContent = stats.pending_queries;

      // Render games list
      const gamesPane = document.getElementById("tab-games");
      if (!stats.games || stats.games.length === 0) {
        gamesPane.innerHTML = '<div class="kb-empty">知识库为空,点击"获取热门游戏"开始</div>';
        return;
      }
      gamesPane.innerHTML = "";
      stats.games.forEach((g) => {
        const item = document.createElement("div");
        item.className = "kb-game-item";
        item.innerHTML = `
          <div class="game-info">
            <div class="game-name">${this.escapeHtml(g.game_name)}</div>
            <div class="game-docs">${g.document_count} 个知识块</div>
          </div>
          <div class="game-actions">
            <button class="btn-refresh-game" data-game="${this.escapeHtml(g.game_name)}" title="重新获取">🔄</button>
            <button class="btn-del-game" data-game="${this.escapeHtml(g.game_name)}" title="删除">🗑</button>
          </div>
        `;
        item.querySelector(".btn-refresh-game").addEventListener("click", async (e) => {
          const name = e.target.dataset.game;
          this.setKbStatus(`正在重新获取《${name}》...`);
          try {
            await API.post("/api/knowledge/games", { game_name: name, force: true });
            this.setKbStatus(`《${name}》已更新`);
            await this.loadKnowledgeData();
            await AppState.refreshGames();
          } catch (err) {
            this.setKbStatus("更新失败: " + err.message);
          }
        });
        item.querySelector(".btn-del-game").addEventListener("click", async (e) => {
          const name = e.target.dataset.game;
          if (!confirm(`确定要删除《${name}》的全部知识吗?`)) return;
          try {
            await API.del(`/api/documents/by-game/${encodeURIComponent(name)}`);
            await this.loadKnowledgeData();
            await AppState.refreshGames();
          } catch (err) {
            alert("删除失败: " + err.message);
          }
        });
        gamesPane.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load knowledge status:", err);
    }
  },

  async loadPendingQueries() {
    try {
      const pending = await API.get("/api/knowledge/pending");
      const pane = document.getElementById("tab-pending");
      if (!pending || pending.length === 0) {
        pane.innerHTML = '<div class="kb-empty">暂无待学习问题</div>';
        return;
      }
      pane.innerHTML = "";
      pending.forEach((q) => {
        const item = document.createElement("div");
        item.className = "kb-pending-item";
        item.innerHTML = `
          <div style="flex:1;">
            <div class="pending-game">🎮 ${this.escapeHtml(q.game_name)}</div>
            <div class="pending-question">${this.escapeHtml(q.question)}</div>
            <div class="pending-time">${q.created_at || ""}</div>
          </div>
          <button class="btn-del-pending" data-id="${q.id}" title="删除">×</button>
        `;
        item.querySelector(".btn-del-pending").addEventListener("click", async (e) => {
          const id = e.target.dataset.id;
          try {
            await API.del(`/api/knowledge/pending/${id}`);
            await this.loadKnowledgeData();
          } catch (err) {
            alert("删除失败: " + err.message);
          }
        });
        pane.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load pending queries:", err);
    }
  },

  async loadKnowledgeLogs() {
    try {
      const logs = await API.get("/api/knowledge/logs");
      const pane = document.getElementById("tab-logs");
      if (!logs || logs.length === 0) {
        pane.innerHTML = '<div class="kb-empty">暂无获取日志</div>';
        return;
      }
      pane.innerHTML = "";
      logs.forEach((log) => {
        const item = document.createElement("div");
        item.className = "kb-log-item";
        const actionMap = {
          knowledge_cycle: "自动周期",
          manual_refresh_trending: "手动获取热门",
          manual_process_pending: "手动处理待学习",
          manual_fetch_game: "手动获取游戏",
        };
        item.innerHTML = `
          <div class="log-header">
            <span class="log-action">${actionMap[log.action] || log.action}</span>
            <span class="log-time">${log.created_at || ""}</span>
          </div>
          <div class="log-message">${this.escapeHtml(log.message || "")}</div>
        `;
        pane.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load knowledge logs:", err);
    }
  },

  escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  },
};

// ── Bootstrap ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => App.init());
