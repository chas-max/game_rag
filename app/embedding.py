"""Embedding service — singleton wrapper around sentence-transformers."""

import os

# 模型已本地缓存；HuggingFace 在本机网络下不可达，开启离线模式跳过每次加载时的
# HEAD 更新检查（否则每个文件重试 5 次退避，单次请求耗时数分钟甚至超时）。
# 用 setdefault 以便需要联网更新模型时可在环境变量里覆盖为 0。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the singleton embedding model, loading it lazily on first access."""
    global _model
    if _model is None:
        print(f"[embedding] Loading model: {settings.embedding_model} ...")
        _model = SentenceTransformer(settings.embedding_model)
        print("[embedding] Model loaded successfully.")
    return _model


def encode_text(text: str) -> np.ndarray:
    """Encode a single text string to a normalized embedding vector (384,)."""
    model = get_embedding_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


def encode_batch(texts: list[str]) -> np.ndarray:
    """Encode a batch of text strings to embedding vectors (N, 384)."""
    model = get_embedding_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.astype(np.float32)
