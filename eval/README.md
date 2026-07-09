# RAG 评测工具 (eval/)

为游戏问答 RAG 项目提供可复用的评测与指标记录:对检索/生成质量打分,把每次评测的指标
**追加**到同一个 Excel 文件 (`eval/rag_metrics.xlsx`),并用「版本标记」区分不同项目版本
(基线 vs 优化后),便于横向对比优化效果。

## 快速开始

```bash
# 1. 构建测试集(从知识库自动生成「问题+参考答案+相关文档」,生成一次后固定复用)
python -m eval.benchmark --build                      # 全部游戏,每游戏 5 条
python -m eval.benchmark --build --per-game 3 --games 原神,塞尔达传说   # 小规模试跑

# 2. 跑评测,指标追加到 Excel
python -m eval.run_eval --version v1-baseline         # 检索指标(默认,快、不花钱)
python -m eval.run_eval --version v1-full --with-generation   # 额外跑 LLM 裁判生成指标

# 3. 优化项目后(如调阈值/加 HyDE)再跑一次,新行追加到同一个 Excel
python -m eval.run_eval --version v2-hyde --note "HyDE + 阈值 0.3->0.15 fallback"
```

结果在 `eval/rag_metrics.xlsx`,两个 sheet:
- **runs**:每次评测一行(汇总指标)
- **per_query**:每条问题一行(明细,便于定位哪些问题召回失败)

> 版本对比前提:**测试集 `benchmark.json` 必须保持不变**。`benchmark_id`(题集内容哈希)
> 会写入 Excel,题集不一致时一眼可辨。需要换题集时用 `--force` 重建,但新旧结果不宜直接对比。

## 命令参数

`python -m eval.run_eval`:
| 参数 | 说明 |
|---|---|
| `--version` | 版本标记,如 `v1-baseline` / `v2-hyde`(默认 `unlabeled`) |
| `--note` | 本次评测备注 |
| `--benchmark` | 测试集 JSON 路径(默认 `eval/benchmark.json`) |
| `--out` | 输出 Excel 路径(默认 `eval/rag_metrics.xlsx`) |
| `--with-generation` | 额外跑 LLM 裁判生成指标(每条多 2 次 LLM 调用,耗 token) |
| `--games` | 只评测指定游戏,逗号分隔;留空=全部 |
| `--limit` | 只评测前 N 条(0=不限) |
| `--concurrency` | 并发数(默认 4) |

`python -m eval.benchmark --build`:`--per-game`、`--games`、`--out`、`--force`。

## 版本标记列(两 sheet 均有,用于区分版本)

| 列 | 说明 |
|---|---|
| `run_id` | 本次运行的短 uuid |
| `version_label` | 用户传入的版本标记 |
| `run_time` | 本地运行时间 |
| `git_commit` / `git_dirty` | 提交 SHA / 是否有未提交改动 |
| `benchmark_id` | 测试集内容哈希(题集变了会不同) |
| `config_snapshot` | 当时的配置 JSON(top_k/threshold/chunk_size/embedding_model/llm_model/hyde/...) |
| `note` | 备注 |
| `with_generation` | 是否含生成指标 |

> 即使忘了写 `--version`,`config_snapshot` + `git_commit` 也能区分不同版本配置。

## 指标说明

### 检索指标(默认,衡量「召回到没有、排序好不好」)
| 指标 | 含义 | 越高越好 |
|---|---|:---:|
| `recall@topk` 召回率@K | 前 K 条里召回了多少比例的相关文档 | ✅ |
| `precision@topk` 精确率@K | 前 K 条里相关文档占比 | ✅ |
| `hitrate@topk` 命中率@K | 前 K 条是否至少命中一个相关文档 | ✅ |
| `mrr` 平均倒数排名 | 第一个相关文档排名的倒数(越靠前越高) | ✅ |
| `ndcg@topk` 归一化折损累积增益 | 排序质量,越靠前命中分越高并折损 | ✅ |
| `map` 平均精度均值 | 各命中位置的 precision 平均 | ✅ |
| `recall@5` / `hitrate@5` / `ndcg@5` | 固定 K=5 的同款指标,**便于 top_k 变化时跨版本对比** | ✅ |
| `recall@1` | 前 1 条命中率 | ✅ |
| `avg_top1_sim` / `avg_sim` | 平均最高相似度 / 平均相似度 | ✅ |
| `insufficient_rate` 拒答率 | 检索为空的比例 | ❌ 越低越好 |

> 主指标用生产 `top_k`(= `config_snapshot.top_k`)。另含固定 K=5/1 的列,
> 这样即使你把 `top_k` 从 5 调到 8,仍能用 `recall@5` 公平对比。

### 生成指标(仅 `--with-generation`,LLM 当裁判,0-1)
| 指标 | 含义 |
|---|---|
| `faithfulness_avg` 忠实度 | 回答陈述被检索上下文支持的比例(不编造) |
| `answer_relevance_avg` 相关性 | 回答是否切题 |
| `answer_correctness_avg` 正确性 | 回答与参考答案语义一致度 |
| `token_f1_avg` | 字符/词级 F1(**不调 LLM**,正确性的廉价补充信号) |
| `generation_latency_avg_ms` | 生成平均耗时 |

### 系统指标
`retrieval_latency_avg_ms`、`total_latency_avg_ms`。

## 工作原理

- 评测直接调用 `app/rag_pipeline.retrieve_documents`(HyDE + 混合检索 + 去重),与生产
  `rag_query` **共用同一检索逻辑**,无 DB 写副作用,保证评测忠实反映线上检索质量。
- 复用 `rag_pipeline._get_client` / `_llm_chat`,继承 DeepSeek 客户端与 httpx timeout 修复
  (避免 AsyncOpenAI 代理返回空内容的坑)。
- 测试集 ground truth:从知识库 chunk 用 LLM 生成问题,该 chunk 的 document id 即为
  `relevant_doc_ids`(严格定义:命中 = 源 chunk 出现在检索结果中)。

## 常见问题

- **Excel 写入报错「被占用」**:文件在 Excel 中打开会锁文件,关闭后重试。
- **生成指标耗时长/token 多**:`--with-generation` 每条问题多 2 次 LLM 调用(生成 + 裁判)。
  日常迭代用默认检索指标即可,里程碑再跑全套。
- **想换测试集**:`python -m eval.benchmark --build --force`,但新题集与旧结果不可直接对比
  (看 `benchmark_id` 区分)。
