"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Frozen settings dataclass — all values loaded from env with sensible defaults."""

    # LLM Configuration (OpenAI-compatible API)
    active_provider: str = os.getenv("ACTIVE_PROVIDER", "mimo")  # We'll use fallback MIMO -> DashScope -> DeepSeek
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    mimo_api_key: str = os.getenv("MIMO_API_KEY", "")
    mimo_base_url: str = os.getenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    mimo_model: str = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

    # DashScope (阿里云百炼 MaaS, 兼容 OpenAI 接口; 同时承载 qwen/deepseek/glm/kimi 等多模型)
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    dashscope_model: str = os.getenv("DASHSCOPE_MODEL", "qwen3.7-max")

    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Tavily (agent search_web 工具的互联网检索; 未配置则回退 DuckDuckGo)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Embedding Model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    # Database (关系数据: 对话/消息/任务/日志)
    database_path: str = os.getenv("DATABASE_PATH", "data/game_rag.db")

    # Vector Store (ChromaDB) - RAG 向量存储与相似度检索
    chroma_path: str = os.getenv("CHROMA_PATH", "data/chroma")

    # RAG Settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

    # Knowledge Acquisition (自动知识获取)
    knowledge_fetch_interval_hours: int = int(os.getenv("KNOWLEDGE_FETCH_INTERVAL_HOURS", "12"))
    trending_game_count: int = int(os.getenv("TRENDING_GAME_COUNT", "10"))


# Singleton settings instance
settings = Settings()
