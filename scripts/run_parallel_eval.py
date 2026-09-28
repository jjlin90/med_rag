"""
多模型并行 Ragas 评估启动器。

背景：run_chunked_eval_v2 是串行的——一个模型跑完才轮下一个，5 个模型里只有 1 个在干活。
但每个模型有独立的 100 万免费额度，完全可以同时开多个进程各跑一段。

本脚本把"剩余待评估样本"按 round-robin 切成 N 份，每份交给一个独立 python 子进程（不同模型）
并行评估，最后统一合并。

用法：
  # 并行模式（默认）：3 路并行，把剩余样本分给 3 个模型
  python scripts/run_parallel_eval.py --models glm-5.2 minimax-m3 hy3

  # 只合并已有块（含历史 chunk_*.json 与本次 chunk_par_*.json）
  python scripts/run_parallel_eval.py --merge-only

  # 单 worker 模式（内部被子进程调用，一般不直接用）
  python scripts/run_parallel_eval.py --worker --model X --questions-file Q.json --out O.json

注意：
  - 每个子进程都会独立加载 BGE-M3（CPU，~2GB）。并行数越多越吃内存，笔记本建议 ≤3。
  - 合并时只统计有效样本（全零=额度耗尽/解析失败，已过滤）。
"""

import argparse
import json
import math
import os
import subprocess
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


def is_valid(score_row: dict) -> bool:
    return all(isinstance(score_row.get(k), (int, float))
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


def run_worker(model: str, questions: list, out_path: Path):
    """子进程入口：用 model 评估 questions 列表里的样本，写出单块 JSON。"""
    from src.config.settings import Config
    from src.online_service.rag_evaluator import RAGEvaluator
    from src.offline_pipeline.embedding_provider import BGEEmbeddingProvider

    config = Config()
    config.LLM_MODEL_NAME = model
    os.environ["LLM_MODEL_NAME"] = model

    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    gt_map = {x["question"]: x.get("reference_answer", "") for x in testset}
    answers = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    all_items = build_items(answers, gt_map)
    qset = set(questions)
    items = [it for it in all_items if it["question"] in qset]

    provider = BGEEmbeddingProvider(config)
    evaluator = RAGEvaluator(config, embedding_provider=provider)

    print(f"[worker {model}] 加载完成，开始评估 {len(items)} 条", flush=True)
    t0 = time.time()
    result = evaluator.evaluate_dataset(items, show_progress=False)
    elapsed = time.time() - t0

    scores = result.get("scores", [])
    n_valid = sum(1 for s in scores if is_valid(s))
    result["_meta"] = {
        "model": model,
        "chunk_idx": "par",
        "n_samples": len(items),
        "n_valid": n_valid,
        "n_failed": len(scores) - n_valid,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[worker {model}] 完成 {n_valid}/{len(scores)} 有效，耗时 {elapsed:.0f}s -> {out_path.name}", flush=True)


def merge_results(collected: dict, meta_list: list) -> dict:
    all_scores = [{"question": q, **v} for q, v in collected.items()]
    if not all_scores:
        return {"error": "无有效结果"}
    avg = {}
    for k in METRICS:
        vals = [s[k] for s in all_scores
                if isinstance(s.get(k), (int, float)) and math.isfinite(s[k])]
        avg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    weighted = sum(avg[k] for k in METRICS) / len(METRICS)
    return {
        "engine": "ragas",
        "total": len(all_scores),
        "metrics": METRICS,
        "has_ground_truth": True,
        "average": avg,
        "weighted_composite": round(weighted, 4),
        "scores": all_scores,
        "_chunks_meta": meta_list,
    }


def save_final(result: dict):
    FINAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    FINAL_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n最终 JSON: {FINAL_JSON}")
    try:
        import pandas as pd
        rows = [{k: s.get(k, 0.0) for k in METRICS} | {"question": s.get("question", "")}
                for s in result["scores"]]
        avg_row = {k: result["average"].get(k, 0.0) for k in METRICS} | {"question": "__average__"}
        rows.append(avg_row)
        pd.DataFrame(rows, columns=["question"] + METRICS).to_csv(
            FINAL_CSV, index=False, encoding="utf-8-sig")
        print(f"最终 CSV : {FINAL_CSV}")
    except Exception as e:
        print(f"CSV 保存失败（不影响 JSON）: {e}")


def print_summary(result: dict):
    avg = result.get("average", {})
    meta = result.get("_chunks_meta", [])
    print(f"\n{'='*72}")
    print(f"  最终评估结果 — {result.get('total', 0)} 条有效样本")
    print(f"{'='*72}")
    for m in meta:
        print(f"  {str(m.get('model', '?')):<26} "
              f"有效 {m.get('n_valid', 0):>3}/{m.get('n_samples', 0):<3} "
              f"{m.get('elapsed_seconds', 0):>5.0f}s")
    print(f"  {'-'*56}")
    for k in METRICS:
        v = avg.get(k, 0)
        bar = "█" * int(v * 36) + "░" * (36 - int(v * 36))
        print(f"  {k:<22}  {v:.4f}  {bar}")
    print(f"  {'-'*56}")
    print(f"  {'等权加权综合':<22}  {result.get('weighted_composite', 0):.4f}")
    print(f"{'='*72}")


def main():
    ap = argparse.ArgumentParser(description="多模型并行 Ragas 评估")
    ap.add_argument("--models", nargs="*", required=False,
                    help="并行模型列表（每个模型一个进程）")
    ap.add_argument("--merge-only", action="store_true", help="只合并已有块")
    ap.add_argument("--worker", action="store_true", help="内部子进程模式")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--questions-file", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    # 独立目录参数：重测时避免与历史 chunk_*.json 混合
    ap.add_argument("--answers", type=str, default=None,
                    help="答案文件（默认 eval_answers_210.json）")
    ap.add_argument("--chunks-dir", type=str, default=None,
                    help="块结果目录（默认 data/test_query/chunks）")
    ap.add_argument("--final", type=str, default=None,
                    help="最终合并 JSON 输出路径")
    args = ap.parse_args()

    # 应用独立目录覆写（必须在读取全局常量之前）
    global ANSWERS_PATH, CHUNKS_DIR, FINAL_JSON, FINAL_CSV, TESTSET_PATH
    if args.answers:
        ANSWERS_PATH = Path(args.answers)
    if args.chunks_dir:
        CHUNKS_DIR = Path(args.chunks_dir)
    if args.final:
        FINAL_JSON = Path(args.final)
        FINAL_CSV = FINAL_JSON.with_suffix(".csv")

    CHUNKS_DIR.mkdir(exist_ok=True)

    # 子进程 worker
    if args.worker:
        if not (args.model and args.questions_file and args.out):
            print("worker 模式需要 --model --questions-file --out")
            sys.exit(1)
        questions = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
        run_worker(args.model, questions, Path(args.out))
        return

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

    if not args.models:
        print("需指定 --models")
        sys.exit(1)

    # 计算剩余待评估样本
    answers = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    gt_map = {x["question"]: x.get("reference_answer", "") for x in testset}
    items = build_items(answers, gt_map)
    all_questions = [it["question"] for it in items]

    collected = load_collected(CHUNKS_DIR)
    todo = [q for q in all_questions if q not in collected]

    print(f"总样本: {len(all_questions)} | 已有效: {len(collected)} | 待跑: {len(todo)}")
    print(f"并行模型数: {len(args.models)}")

    if not todo:
        print("全部已有效，直接合并。")
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

    n = len(args.models)
    # round-robin 切分，保证各模型负载均衡
    slices = [todo[i::n] for i in range(n)]
    for i, sl in enumerate(slices):
        print(f"  模型 {args.models[i]:<22} 分到 {len(sl):>2} 条")

    procs = []
    for i, model in enumerate(args.models):
        sl = slices[i]
        qfile = CHUNKS_DIR / f"_slice_{i}.json"
        qfile.write_text(json.dumps(sl), encoding="utf-8")
        out = CHUNKS_DIR / f"chunk_par_{model}_{i}.json"
        log = CHUNKS_DIR / f"_par_{model}_{i}.log"
        p = subprocess.Popen(
            [sys.executable, str(Path(__file__)), "--worker",
             "--model", model, "--questions-file", str(qfile), "--out", str(out),
             "--answers", str(ANSWERS_PATH)],
            stdout=open(log, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        procs.append((model, p, log))
        print(f"  启动子进程 [{model}] pid={p.pid} -> {out.name}", flush=True)

    print(f"\n{len(procs)} 路并行评估中...（完成后会自动合并）\n")
    for model, p, log in procs:
        rc = p.wait()
        # 读取子进程日志尾巴
        try:
            tail = log.read_text(encoding="utf-8").strip().splitlines()[-3:]
        except Exception:
            tail = []
        print(f"  [{model}] 退出码={rc}")
        for line in tail:
            print(f"     {line}")

    # 合并所有块（含历史 chunk_01~03 与本次并行块）
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
