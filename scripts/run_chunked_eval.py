"""
多模型接力 Ragas 评估脚本。

解决的核心问题：
  单个模型的免费额度（100 万 token）不够跑完全部 210 条 Ragas 评估（需 ~250 万）。
  本脚本把测试集切成 N 块，每块用不同模型跑，最后合并为一份完整报告。

前置条件：
  - data/test_query/eval_answers_210.json 已存在且 210/210 完整（答案缓存）
  - data/test_query/eval_set_210.json 已存在（测试集，含 ground_truth）
  - .env 已配置 LLM_API_KEY / LLM_BASE_URL 指向腾讯云 TokenHub

用法：
  # 默认：用 3 个模型各跑 ~70 条
  python scripts/run_chunked_eval.py

  # 指定模型列表（按顺序分配给各块）
  python scripts/run_chunked_eval.py --models deepseek-v4-flash deepseek-v4-pro glm-5.1

  # 指定块大小（每块条数，默认均分）
  python scripts/run_chunked_eval.py --chunk-size 50

  # 只做合并（之前各块已跑完，重新合并最终报告）
  python scripts/run_chunked_eval.py --merge-only

输出：
  data/test_query/eval_final_merged.json    — 合并后的完整报告（含逐条分数 + 平均分）
  data/test_query/eval_final_merged.csv     — CSV 版本
  data/test_query/chunks/                   — 各块的原始 JSON（保留用于排查）
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANSWERS_PATH = ROOT / "data/test_query" / "eval_answers_210.json"
TESTSET_PATH = ROOT / "data/test_query" / "eval_set_210.json"
CHUNKS_DIR = ROOT / "data/test_query" / "chunks"
FINAL_JSON = ROOT / "data/test_query" / "eval_final_merged.json"
FINAL_CSV = ROOT / "data/test_query" / "eval_final_merged.csv"

# 默认候选模型（腾讯云 TokenHub 免费体验，每个 100 万 token）
DEFAULT_MODELS = [
    "deepseek-v4-flash",   # 已验证可用（test_judge_models.py 通过）
    "deepseek-v4-pro",
    "glm-5.1",
]

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_data():
    """加载答案缓存与测试集，返回 (answers_list, testset_dict)。"""
    if not ANSWERS_PATH.exists():
        print(f"[FATAL] 答案缓存不存在: {ANSWERS_PATH}")
        print("        请先运行 evaluate_rag.py 生成答案（或已有 --reuse-answers 缓存）")
        sys.exit(1)
    if not TESTSET_PATH.exists():
        print(f"[FATAL] 测试集不存在: {TESTSET_PATH}")
        sys.exit(1)

    answers = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    gt_map = {x["question"]: x.get("reference_answer", "") for x in testset}
    print(f"答案缓存: {len(answers)} 条 | 测试集: {len(testset)} 条 | GT 覆盖: {len(gt_map)}")
    return answers, gt_map


def build_chunks(items, n_models: int, chunk_size: int = 0):
    """将 items 切成 n_models 块，返回 list[list]。

    chunk_size > 0 时优先按 size 切；否则均分（最后一块可能多几个）。
    """
    total = len(items)
    if chunk_size > 0:
        n_chunks = max(1, (total + chunk_size - 1) // chunk_size)
        sizes = [chunk_size] * (n_chunks - 1)
        sizes.append(total - sum(sizes))
    else:
        base, rem = divmod(total, n_models)
        sizes = [base + (1 if i < rem else 0) for i in range(n_models)]

    chunks, offset = [], 0
    for s in sizes:
        chunks.append(items[offset:offset + s])
        offset += s
    return chunks


def run_one_chunk(chunk_items: list, model: str, chunk_idx: int,
                  out_path: Path) -> dict:
    """用指定模型跑一块数据，返回评估结果 dict。"""
    print(f"\n{'='*60}")
    print(f"  块 #{chunk_idx + 1} | 模型: {model} | 样本数: {len(chunk_items)}")
    print(f"{'='*60}")

    os.environ["LLM_MODEL_NAME"] = model

    from src.config.settings import Config
    from src.online_service.rag_evaluator import RAGEvaluator
    from src.offline_pipeline.embedding_provider import BGEEmbeddingProvider

    config = Config()
    config.LLM_MODEL_NAME = model  # 显式覆盖

    provider = BGEEmbeddingProvider(config)
    evaluator = RAGEvaluator(config, embedding_provider=provider)

    t0 = time.time()
    result = evaluator.evaluate_dataset(chunk_items, show_progress=True)
    elapsed = time.time() - t0

    result["_meta"] = {
        "model": model,
        "chunk_idx": chunk_idx,
        "n_samples": len(chunk_items),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    avg = result.get("average", {})
    scores_str = " / ".join(f"{k}={avg.get(k, 0):.3f}" for k in METRICS if k in avg)
    print(f"\n  [{model}] 完成！耗时 {elapsed:.0f}s | {scores_str}")
    print(f"  结果已存: {out_path}")
    return result


def merge_results(results: list[dict]) -> dict:
    """合并多块结果为一份完整报告。

    规则：
    - average: 所有逐条分数的算术平均（跨块统一）
    - scores: 按原始顺序拼接
    - engine / metrics / has_ground_truth 取第一块的值（应全部一致）
    """
    all_scores = []
    meta_list = []
    for r in results:
        all_scores.extend(r.get("scores", []))
        meta_list.append(r.get("_meta", {}))

    if not all_scores:
        return {"error": "无有效结果可合并"}

    # 跨块统一计算平均分
    avg = {}
    metric_keys = results[0].get("metrics", [])
    for k in metric_keys:
        vals = [s[k] for s in all_scores if k in s
                and isinstance(s[k], (int, float)) and __import__("math").isfinite(s[k])]
        avg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0

    merged = {
        "engine": results[0].get("engine", "unknown"),
        "total": len(all_scores),
        "metrics": metric_keys,
        "has_ground_truth": results[0].get("has_ground_truth", False),
        "average": avg,
        "scores": all_scores,
        "_chunks_meta": meta_list,
    }
    return merged


def save_final(result: dict):
    """保存合并后的最终报告（JSON + CSV）。"""
    FINAL_JSON.parent.mkdir(parents=True, exist_ok=True)

    FINAL_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n最终 JSON 报告: {FINAL_JSON}")

    # CSV
    try:
        import pandas as pd
        metric_keys = result.get("metrics", [])
        rows = []
        for s in result["scores"]:
            rows.append({k: s.get(k, 0.0) for k in metric_keys} | {"question": s.get("question", "")})
        avg_row = {k: result["average"].get(k, 0.0) for k in metric_keys} | {"question": "__average__"}
        rows.append(avg_row)
        df = pd.DataFrame(rows, columns=["question"] + metric_keys)
        df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
        print(f"最终 CSV 报告: {FINAL_CSV}")
    except Exception as e:
        print(f"CSV 保存失败（不影响 JSON）: {e}")


def print_summary(result: dict):
    """打印最终汇总表。"""
    avg = result.get("average", {})
    meta = result.get("_chunks_meta", [])
    metric_keys = result.get("metrics", [])

    print(f"\n{'='*70}")
    print(f"  最终评估结果（{result.get('total', 0)} 条样本，{len(meta)} 个模型接力）")
    print(f"{'='*70}")

    # 各块明细
    for m in meta:
        model = m.get("model", "?")
        n = m.get("n_samples", 0)
        elapsed = m.get("elapsed_seconds", 0)
        print(f"  {model:<24}  {n:>3} 条  {elapsed:>5.0f}s")

    print(f"  {'-'*50}")
    for k in metric_keys:
        v = avg.get(k, 0)
        bar = "█" * int(v * 40) + "░" * (40 - int(v * 40))
        print(f"  {k:<22}  {v:.4f}  {bar}")

    # 加权综合分（等权）
    if metric_keys:
        weights = {k: 1.0 for k in metric_keys}
        weighted = sum(avg.get(k, 0) * w for k, w in weights.items()) / sum(weights.values())
        print(f"  {'-'*50}")
        print(f"  {'等权加权综合':<22}  {weighted:.4f}")

    print(f"{'='*70}")


def main():
    ap = argparse.ArgumentParser(description="多模型接力 Ragas 评估")
    ap.add_argument("--models", nargs="*", default=None,
                    help=f"模型列表（默认: {DEFAULT_MODELS}）")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="每块条数（默认: 均分到各模型）")
    ap.add_argument("--merge-only", action="store_true",
                    help="跳过评估，只合并已有的块结果")
    args = ap.parse_args()

    models = args.models or DEFAULT_MODELS
    CHUNKS_DIR.mkdir(exist_ok=True)

    # ---- 合并模式 ----
    if args.merge_only:
        chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.json"))
        if not chunk_files:
            print(f"[FATAL] 无块结果可合并: {CHUNKS_DIR}/chunk_*.json")
            sys.exit(1)
        results = [json.loads(f.read_text(encoding="utf-8")) for f in chunk_files]
        merged = merge_results(results)
        save_final(merged)
        print_summary(merged)
        return

    # ---- 正常评估 ----
    answers, gt_map = load_data()

    # 构建完整评估项列表（与 evaluate_rag.py 格式一致）
    items = []
    for a in answers:
        q = a["question"]
        items.append({
            "question": q,
            "answer": a.get("answer", ""),
            "contexts": a.get("contexts", []),
            "ground_truth": gt_map.get(q, ""),
        })

    chunks = build_chunks(items, len(models), args.chunk_size)
    if len(chunks) > len(models):
        models = models * ((len(chunks) // len(models)) + 1)
        print(f"  注意: 切成了 {len(chunks)} 块，但只给了 {len(args.models or DEFAULT_MODELS)} 个模型，"
              f"部分模型会重复使用")

    print(f"\n总样本: {len(items)} | 分成 {len(chunks)} 块:")
    for i, (c, m) in enumerate(zip(chunks, models)):
        print(f"  块 #{i+1}: {len(c)} 条 → {m}")

    results = []
    for i, (chunk, model) in enumerate(zip(chunks, models)):
        out = CHUNKS_DIR / f"chunk_{i+1:02d}_{model}.json"
        # 如果该块结果已存在且非空，询问是否跳过
        if out.exists():
            existing = json.loads(out.read_text(encoding="utf-8"))
            if existing.get("total", 0) > 0:
                print(f"\n[SKIP] 块 #{i+1} 已有结果: {out} ({existing['total']} 条)")
                results.append(existing)
                continue

        res = run_one_chunk(chunk, model, i, out)
        results.append(res)

    # 合并
    print("\n\n正在合并所有块 ...")
    merged = merge_results(results)
    save_final(merged)
    print_summary(merged)

    # 额外产出：token 预估 vs 实际（供后续优化参考）
    print(f"\n所有块原始结果保存在: {CHUNKS_DIR}/")
    print(f"最终合并报告: {FINAL_JSON} / {FINAL_CSV}")


if __name__ == "__main__":
    main()
