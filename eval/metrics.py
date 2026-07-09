"""RAG 评测指标 - 检索指标 + 生成指标(LLM 裁判) + 廉价 token F1。

检索指标为纯函数,输入按相似度降序排列的 retrieved_ids 与 relevant_ids 集合。
生成指标 faithfulness/relevance/correctness 由 LLM 裁判打分(0.0-1.0),复用
rag_pipeline._llm_chat(继承 DeepSeek 客户端与 httpx timeout 修复)。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Iterable

# 检索指标阈值/排名类指标均假设 retrieved_ids 已按相似度降序排列(检索器保证)。


def _to_set(ids: Iterable) -> set:
    # ChromaDB 文档 id 为字符串,统一转 str 比较(int/str id 均兼容)
    return {str(x) for x in ids}


def recall_at_k(retrieved_ids: list, relevant_ids: Iterable, k: int) -> float:
    """Recall@K(召回率@K): 前 K 条结果中命中的相关文档数 / 全部相关文档数。

    直觉: 所有该召回的文档,实际在前 K 条里召回了多少比例。越高越好。
    """
    rel = _to_set(relevant_ids)
    if not rel:
        return 0.0
    topk = {str(x) for x in retrieved_ids[:k]}
    return len(rel & topk) / len(rel)


def precision_at_k(retrieved_ids: list, relevant_ids: Iterable, k: int) -> float:
    """Precision@K(精确率@K): 前 K 条结果中相关文档数 / K。"""
    if k <= 0:
        return 0.0
    rel = _to_set(relevant_ids)
    topk = {str(x) for x in retrieved_ids[:k]}
    return len(rel & topk) / k


def hit_rate_at_k(retrieved_ids: list, relevant_ids: Iterable, k: int) -> float:
    """HitRate@K(命中率@K): 前 K 条中是否至少命中一个相关文档(1.0/0.0)。

    单相关文档时等价于 Recall@K。衡量"有没有召回到"。
    """
    rel = _to_set(relevant_ids)
    topk = {str(x) for x in retrieved_ids[:k]}
    return 1.0 if (rel & topk) else 0.0


def reciprocal_rank(retrieved_ids: list, relevant_ids: Iterable) -> float:
    """MRR 的单查询分量: 1 / 第一个相关文档的排名(无命中则 0)。

    越靠前命中得分越高,对排序质量敏感。
    """
    rel = _to_set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if str(rid) in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list, relevant_ids: Iterable, k: int) -> float:
    """NDCG@K(归一化折损累积增益): 排序质量,越靠前命中得分越高并折损。

    二值相关性: DCG = Σ rel_i / log2(i+1); IDCG 为理想排序下的 DCG; NDCG = DCG/IDCG ∈ [0,1]。
    """
    rel = _to_set(relevant_ids)
    if not rel:
        return 0.0
    topk = [str(x) for x in retrieved_ids[:k]]
    dcg = sum((1.0 if rid in rel else 0.0) / math.log2(i + 1) for i, rid in enumerate(topk, start=1))
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved_ids: list, relevant_ids: Iterable) -> float:
    """AP(平均精度): 在每个命中位置计算 precision 并取平均,再除以相关文档数。

    多相关文档时 MAP = mean(AP);单相关文档时 AP = 1/rank(命中)或 0(未命中)。
    """
    rel = _to_set(relevant_ids)
    if not rel:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        if str(rid) in rel:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(rel)


# ── 生成指标 ────────────────────────────────────────────────────────

def _extract_json_block(text: str) -> dict | None:
    """从 LLM 输出中防御式提取首个 JSON 对象(容忍 ```json 围栏与多余文字)。"""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()
    # 直接解析
    try:
        return json.loads(s)
    except Exception:
        pass
    # 退而求其次: 抓首个 {...} 片段
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _clamp01(v) -> float | None:
    """把裁判返回值规整为 [0,1] 浮点,失败返回 None。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f < 0:
        f = 0.0
    elif f > 1:
        f = 1.0
    return f


async def judge_generation(
    question: str,
    answer: str,
    reference_answer: str,
    context: str,
) -> dict:
    """LLM 裁判对生成回答打三个维度分(0.0-1.0)。

    单次 LLM 调用同时返回 faithfulness/relevance/correctness 以省 token。
    任意维度解析失败记为 None,聚合时不计入均值。

    Returns: {"faithfulness": float|None, "relevance": float|None, "correctness": float|None}
    """
    # 延迟导入,避免仅用检索指标时也加载 rag_pipeline 全链路依赖。
    from app.rag_pipeline import _llm_chat

    prompt = f"""你是一个严格的 RAG 问答评测裁判。请根据以下信息对「待评回答」打分,分值 0.0-1.0(可保留一位小数)。

【问题】
{question}

【参考答案】
{reference_answer}

【检索上下文】
{context}

【待评回答】
{answer}

请从三个维度打分:
1. faithfulness(忠实度): 待评回答中的陈述是否都能被检索上下文支持,有无编造。1.0=完全有据,0.0=大量编造/编造关键信息。
2. relevance(相关性): 待评回答是否切题回答了问题。1.0=完全切题,0.0=答非所问。
3. correctness(正确性): 待评回答与参考答案的语义一致度。1.0=完全一致,0.0=完全不符。

只输出一行 JSON,不要任何解释,格式示例:
{{"faithfulness": 0.8, "relevance": 1.0, "correctness": 0.6}}"""

    raw = await _llm_chat(prompt, max_tokens=200, temperature=0.0)
    data = _extract_json_block(raw) or {}
    return {
        "faithfulness": _clamp01(data.get("faithfulness")),
        "relevance": _clamp01(data.get("relevance")),
        "correctness": _clamp01(data.get("correctness")),
    }


def _tokenize(text: str) -> list[str]:
    """中文字符级 + 拉丁词级混合分词,用于 token F1。

    每个 CJK 字符作为一个 token(适合中文短答),拉丁字母数字串按词切分。
    """
    if not text:
        return []
    tokens: list[str] = []
    # 拉丁词(字母/数字)
    for m in re.findall(r"[A-Za-z0-9]+", text):
        tokens.append(m.lower())
    # CJK 单字(中日韩统一表意文字 + 常见扩展)
    for ch in text:
        if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿":
            tokens.append(ch)
    return tokens


def token_f1(answer: str, reference: str) -> float:
    """字符/词级 token F1(不调 LLM),作为正确性的廉价补充信号。∈ [0,1]。"""
    a = _tokenize(answer)
    r = _tokenize(reference)
    if not a or not r:
        return 0.0
    ca, cr = Counter(a), Counter(r)
    overlap = sum((ca & cr).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)
