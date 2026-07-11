"""测试 RAG 数据导入后的检索效果。

测试内容:
1. ChromaDB 数据完整性
2. 多游戏语义检索准确性
3. 边界情况处理

用法:
    python scripts/test_rag_import.py
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import dotenv
dotenv.load_dotenv(override=True)

from app.embedding import encode_text
from app import vector_store as vs


def test_data_integrity():
    """测试数据完整性。"""
    print("=" * 60)
    print("1. 数据完整性测试")
    print("=" * 60)

    games = vs.list_games()
    print(f"  游戏总数: {len(games)}")

    target_games = {"原神", "崩坏星穹铁道", "绝区零"}
    found_games = {g["game_name"] for g in games}

    for game in target_games:
        if game in found_games:
            count = next(g["document_count"] for g in games if g["game_name"] == game)
            print(f"  [PASS] {game}: {count} 文档")
        else:
            print(f"  [FAIL] {game}: 未找到!")

    total = vs.count_documents()
    print(f"  文档总数: {total}")
    return target_games.issubset(found_games)


def test_semantic_search():
    """测试语义检索准确性。"""
    print(f"\n{'='*60}")
    print("2. 语义检索测试")
    print("=" * 60)

    test_cases = [
        # (game, query, expected_keywords_in_top_results)
        ("原神", "钟离是谁", ["钟离", "岩", "璃月"]),
        ("原神", "派蒙的介绍", ["派蒙", "旅行者"]),
        ("原神", "璃月地区", ["璃月", "港"]),
        ("崩坏星穹铁道", "星穹列车开拓者", ["开拓者", "星穹列车"]),
        ("崩坏星穹铁道", "布洛妮娅", ["布洛妮娅", "贝洛伯格"]),
        ("崩坏星穹铁道", "仙舟罗浮", ["仙舟", "罗浮"]),
        ("绝区零", "新艾利都", ["新艾利都", "空洞"]),
        ("绝区零", "绳匠", ["绳匠"]),
    ]

    passed = 0
    for game, query, keywords in test_cases:
        q_emb = encode_text(query)
        results = vs.search_similar_semantic(q_emb, game, top_k=5, threshold=0.2)

        if not results:
            print(f"  [FAIL] [{game}] '{query}' -> 无结果")
            continue

        # 检查 top-3 结果中是否包含期望关键词
        top3_text = " ".join(r["content"][:200] for r in results[:3])
        found_keywords = [kw for kw in keywords if kw in top3_text]

        if found_keywords:
            sim = results[0]["similarity"]
            print(f"  [PASS] [{game}] '{query}' -> sim={sim:.3f}, 关键词: {found_keywords}")
            passed += 1
        else:
            sim = results[0]["similarity"]
            preview = results[0]["content"][:60].replace("\n", " ")
            print(f"  [WARN] [{game}] '{query}' -> sim={sim:.3f}, 未匹配关键词 {keywords}")
            print(f"         top1: {preview}...")
            passed += 0.5  # 部分通过(有结果但关键词不匹配)

    print(f"\n  通过: {passed}/{len(test_cases)}")
    return passed >= len(test_cases) * 0.6


def test_cross_game_isolation():
    """测试游戏间数据隔离。"""
    print(f"\n{'='*60}")
    print("3. 游戏隔离测试")
    print("=" * 60)

    # 在原神中搜索星穹铁道的内容,应该找不到
    q_emb = encode_text("星穹列车")
    genshin_results = vs.search_similar_semantic(q_emb, "原神", top_k=3, threshold=0.3)
    star_rail_results = vs.search_similar_semantic(q_emb, "崩坏星穹铁道", top_k=3, threshold=0.3)

    print(f"  '星穹列车' 在原神中: {len(genshin_results)} 结果")
    print(f"  '星穹列车' 在崩坏星穹铁道中: {len(star_rail_results)} 结果")

    if len(star_rail_results) > len(genshin_results):
        print(f"  [PASS] 游戏隔离正常: 星穹铁道结果多于原神")
        return True
    else:
        print(f"  [WARN] 游戏隔离可能有问题")
        return False


def test_document_retrieval():
    """测试文档上下文检索。"""
    print(f"\n{'='*60}")
    print("4. 文档上下文检索测试")
    print("=" * 60)

    q_emb = encode_text("钟离")
    results = vs.search_similar_semantic(q_emb, "原神", top_k=1, threshold=0.2)

    if not results:
        print(f"  [FAIL] 无检索结果")
        return False

    doc_id = results[0]["document_id"]
    context = vs.get_document_with_context(doc_id)

    if context:
        print(f"  [PASS] 获取到 {len(context)} 个上下文文档块")
        for i, doc in enumerate(context[:3]):
            preview = doc["content"][:80].replace("\n", " ")
            print(f"    [{i}] chunk_index={doc['chunk_index']}: {preview}...")
        return True
    else:
        print(f"  [FAIL] 无法获取上下文")
        return False


def main():
    print("RAG 数据导入验证测试")
    print("=" * 60)

    results = {
        "数据完整性": test_data_integrity(),
        "语义检索": test_semantic_search(),
        "游戏隔离": test_cross_game_isolation(),
        "文档上下文": test_document_retrieval(),
    }

    print(f"\n{'='*60}")
    print("测试总结")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n所有测试通过! RAG 数据导入成功。")
    else:
        print(f"\n部分测试未完全通过,但数据已导入。")


if __name__ == "__main__":
    main()
