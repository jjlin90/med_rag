"""
多模型接力 Ragas 评估 v2。

v1 的问题（实测教训）：
  1. 单条样本实测消耗 ~3.5 万 token（不是估算的 1 万），单模型 100 万额度只够 ~28 条。
     v1 按 70 条/块切，导致每块后 60% 样本因 402 额度耗尽变全零。
  2. 全零样本被计入平均分，污染结果（块1 均值被拉到 0.168，真实有效均值 0.564）。
  3. 未验证模型直接上生产块，整个块可能全部返零。

v2 改进：
  - 小块化：默认每块 25 条（留安全余量，避免烧穿）
  - 有效值校验：保留真实全零样本，缺失、越界或非数值项继续补评
  - 按题续跑：已拿到有效分的样本直接跳过，不重复烧额度
  - 失败追补：某块因额度耗尽失败，自动用下一个模型补跑剩余样本

前置条件：
  - data/test_query/eval_answers_210.json（答案缓存，210/210 完整）
  - data/test_query/eval_set_210.json（测试集，含 ground_truth）
  - .env 已配置腾讯云 TokenHub 的 LLM_API_KEY / LLM_BASE_URL

用法：
  # 续跑模式（默认）：自动跳过已有有效分的样本，用后续模型补跑剩余
  python scripts/run_chunked_eval_v2.py --models deepseek-v4-pro glm-5.2 minimax-m3 hy3

  # 指定每块条数
  python scripts/run_chunked_eval_v2.py --models m1 m2 --chunk-size 20

  # 只合并已有块结果
  python scripts/run_chunked_eval_v2.py --merge-only

输出：
  data/test_query/eval_final_merged.json   — 合并报告（仅统计有效样本）
  data/test_query/eval_final_merged.csv    — CSV 版本
  data/test_query/chunks/                  — 各块原始 JSON
"""

import argparse
import json
import math
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

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# 已通过 test_judge_models.py 验证可用于 Ragas 的模型
# （2026-08-29 探测结果：glm-5.3-flash / kimi-k3 不可用，返回结构解析失败）
VERIFIED_MODELS = [
    "deepseek-v4-flash",   # 已耗尽
    "deepseek-v4-pro",
    "glm-5.2",
    "minimax-m3",
    "hy3",
]


def is_valid(score_row: dict) -> bool:
    """四项均为有限的 0~1 分数才完整；真实零分保留，缺失项需补评。"""
    return all(isinstance(score_row.get(k), (int, float))
               and not isinstance(score_row[k], bool)
               and math.isfinite(score_row[k]) and 0 <= score_row[k] <= 1 for k in METRICS)


def load_collected(chunks_dir: Path) -> dict:
    """扫描已有块结果，返回 {question: score_row}（只保留有效分）。"""
    collected = {}
    if not chunks_dir.exists():
        return collected
    for f in sorted(chunks_dir.glob("chunk_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get('engine') != 'ragas':
            continue
        for row in data.get("scores", []):
            q = row.get("question")
            if q and is_valid(row) and q not in collected:
                collected[q] = {k: row.get(k, 0) for k in METRICS}
    return collected


def build_items(answers, gt_map):
    items = []
    for a in answers:
        q = a["question"]
        items.append({
            "question": q,
            "answer": a.get("answer", ""),
            "contexts": a.get("contexts", []),
            "ground_truth": gt_map.get(q, ""),
        })
    return items


def run_one_chunk(chunk_items: list, model: str, chunk_idx: int,
                  out_path: Path) -> dict:
    """用指定模型跑一块数据。"""
    print(f"\n{'='*64}")
    print(f"  块 #{chunk_idx} | 模型: {model} | 样本数: {len(chunk_items)}")
    print(f"{'='*64}")

    os.environ["LLM_MODEL_NAME"] = model

    from src.config.settings import Config
    from src.online_service.rag_evaluator import RAGEvaluator
    from src.offline_pipeline.embedding_provider import BGEEmbeddingProvider

    config = Config()
    config.LLM_MODEL_NAME = model

    provider = BGEEmbeddingProvider(config)
    evaluator = RAGEvaluator(config, embedding_provider=provider)

    t0 = time.time()
    result = evaluator.evaluate_dataset(chunk_items, show_progress=True)
    elapsed = time.time() - t0

    scores = result.get("scores", [])
    n_valid = sum(1 for s in scores if is_valid(s))
    result["_meta"] = {
        "model": model,
        "chunk_idx": chunk_idx,
        "n_samples": len(chunk_items),
        "n_valid": n_valid,
        "n_failed": len(scores) - n_valid,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # 只在有效样本上打印均值
    valid_rows = [s for s in scores if is_valid(s)]
    if valid_rows:
        parts = []
        for k in METRICS:
            v = sum(s.get(k, 0) for s in valid_rows) / len(valid_rows)
            parts.append(f"{k}={v:.3f}")
        print(f"\n  [{model}] 耗时 {elapsed:.0f}s | 有效 {n_valid}/{len(scores)}")
        print(f"  有效样本均值: {' / '.join(parts)}")
    else:
        print(f"\n  [{model}] 耗时 {elapsed:.0f}s | 有效 0/{len(scores)} ← 模型不可用或额度已耗尽")
    print(f"  结果已存: {out_path.name}")
    return result


def merge_results(collected: dict, meta_list: list) -> dict:
    """从收集到的有效样本合并最终报告。"""
    all_scores = [{"question": q, **v} for q, v in collected.items()]
    if not all_scores:
        return {"error": "无有效结果"}

    avg = {}
    for k in METRICS:
        vals = [s[k] for s in all_scores
                if isinstance(s.get(k), (int, float)) and not isinstance(s[k], bool)
                and math.isfinite(s[k]) and 0 <= s[k] <= 1]
        avg[k] = sum(vals) / len(vals) if vals else None

    weighted = (sum(avg[k] for k in METRICS) / len(METRICS)
                if all(avg[k] is not None for k in METRICS) else None)
    return {
        "engine": "ragas",
        "total": len(all_scores),
        "metrics": METRICS,
        "has_ground_truth": True,
        "average": {k: round(v, 4) if v is not None else None for k, v in avg.items()},
        "weighted_composite": round(weighted, 4) if weighted is not None else None,
        "scores": all_scores,
        "_chunks_meta": meta_list,
    }


def save_final(result: dict):
    FINAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    FINAL_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n最终 JSON: {FINAL_JSON}")
    if result.get("error"):
        return
    try:
        import pandas as pd
        rows = [{k: s.get(k) for k in METRICS} | {"question": s.get("question", "")}
                for s in result["scores"]]
        avg_row = {k: result["average"].get(k) for k in METRICS} | {"question": "__average__"}
        rows.append(avg_row)
        pd.DataFrame(rows, columns=["question"] + METRICS).to_csv(
            FINAL_CSV, index=False, encoding="utf-8-sig")
        print(f"最终 CSV : {FINAL_CSV}")
    except Exception as e:
        print(f"CSV 保存失败（不影响 JSON）: {e}")


def print_summary(result: dict):
    if result.get("error"):
        print(f"\n无法汇总：{result['error']}")
        return
    avg = result.get("average", {})
    meta = result.get("_chunks_meta", [])
    print(f"\n{'='*72}")
    print(f"  最终评估结果 — {result.get('total', 0)} 条样本")
    print(f"{'='*72}")
    for m in meta:
        print(f"  {m.get('model', '?'):<24} "
              f"有效 {m.get('n_valid', 0):>3}/{m.get('n_samples', 0):<3} "
              f"{m.get('elapsed_seconds', 0):>5.0f}s")
    print(f"  {'-'*56}")
    for k in METRICS:
        v = avg.get(k)
        if v is None:
            print(f"  {k:<22}  缺失")
            continue
        bar = "█" * int(v * 36) + "░" * (36 - int(v * 36))
        print(f"  {k:<22}  {v:.4f}  {bar}")
    print(f"  {'-'*56}")
    composite = result.get('weighted_composite')
    value = f"{composite:.4f}" if composite is not None else "缺失"
    print(f"  {'等权加权综合':<22}  {value}")
    print(f"{'='*72}")


def main():
    ap = argparse.ArgumentParser(description="多模型接力 Ragas 评估 v2")
    ap.add_argument("--models", nargs="*", default=None,
                    help=f"模型列表（默认: {VERIFIED_MODELS}）")
    ap.add_argument("--chunk-size", type=int, default=25,
                    help="每块条数（默认 25，留安全余量避免额度烧穿）")
    ap.add_argument("--target", type=int, default=210, help="目标样本数（默认 210）")
    ap.add_argument("--merge-only", action="store_true", help="只合并已有块")
    args = ap.parse_args()

    CHUNKS_DIR.mkdir(exist_ok=True)

    if args.merge_only:
        collected = load_collected(CHUNKS_DIR)
        meta = []
        for f in sorted(CHUNKS_DIR.glob("chunk_*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                if "_meta" in d:
                    meta.append(d["_meta"])
            except Exception:
                pass
        merged = merge_results(collected, meta)
        save_final(merged)
        print_summary(merged)
        return

    answers = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    gt_map = {x["question"]: x.get("reference_answer", "") for x in testset}
    items = build_items(answers, gt_map)

    # 已收集的有效样本
    collected = load_collected(CHUNKS_DIR)
    todo = [it for it in items if it["question"] not in collected]

    print(f"总样本: {len(items)} | 已有效: {len(collected)} | 待跑: {len(todo)}")

    if not todo:
        print("所有样本已有效评分，直接合并。")
    else:
        models = args.models or VERIFIED_MODELS
        # 每个模型能承载的块数（按 chunk_size 切）
        n_needed = math.ceil(len(todo) / args.chunk_size)
        print(f"需要 {n_needed} 块 | 可用模型 {len(models)} 个")

        chunk_idx = 0
        # 扫描已有块编号，避免文件名冲突
        existing = list(CHUNKS_DIR.glob("chunk_*.json"))
        chunk_idx = len(existing)

        for mi, model in enumerate(models):
            if not todo:
                break
            # 该模型连续跑多块，直到额度烧穿（有效数明显下降）或跑完
            while todo:
                chunk = todo[:args.chunk_size]
                out = CHUNKS_DIR / f"chunk_{chunk_idx:02d}_{model}.json"
                res = run_one_chunk(chunk, model, chunk_idx, out)
                chunk_idx += 1

                # 只把本块拿到有效分的样本从 todo 移除；
                # 失败的样本留给下一个模型补跑
                scored = {s["question"] for s in res.get("scores", []) if is_valid(s)}
                before = len(todo)
                todo = [it for it in todo if it["question"] not in scored]
                print(f"  本块新增有效 {before - len(todo)} 条，剩余待跑 {len(todo)}")

                # 若该块有效率 < 50%，说明这个模型额度已耗尽，换下一个
                valid_rate = len(scored) / max(1, len(chunk))
                if valid_rate < 0.5:
                    print(f"  [{model}] 有效率 {valid_rate:.0%} < 50%，判定额度耗尽，切换模型")
                    break
                if len(todo) == 0:
                    break

    # 最终合并
    collected = load_collected(CHUNKS_DIR)
    meta = []
    for f in sorted(CHUNKS_DIR.glob("chunk_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if "_meta" in d:
                meta.append(d["_meta"])
        except Exception:
            pass
    merged = merge_results(collected, meta)
    save_final(merged)
    print_summary(merged)


if __name__ == "__main__":
    main()
