"""
RAG 评估脚本（离线）

两种模式：
1. 实时管线评估（默认）：从 data/test_query/test_qa.json 读取问题集，用真实 RAG 管线
   （RAGSystem._handle_query）对每题生成答案与检索上下文，再调用 RAGEvaluator（Ragas）
   做四项指标打分。最贴近线上真实质量。
       python scripts/evaluate_rag.py
       python scripts/evaluate_rag.py --test-set data/test_query/test_qa.json

2. 静态评估（--static）：直接读取一份 Ragas 格式的评测集
   （question / context / answer / ground_truth，与 EduRag 的 ragas_evaluate.py 对齐），
   不跑检索管线，直接对给定的 问答/上下文 打分。适合评估固定的 QA 对（如 FAQ 答案）。
       python scripts/evaluate_rag.py --static data/test_query/rag_evaluate_data.json

说明：
- 四项指标均为 0~1：faithfulness / answer_relevancy / context_precision / context_recall。
- ground_truth 缺失时自动跳过 context_precision / context_recall。
- 结果同时保存 JSON（eval_report.json，含逐条明细）与 CSV（ragas_evaluation_results.csv）。
- 依赖 LLM（DashScope）与本地 BGE-M3 embedding 可用性。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import Config
from src.online_service.rag_evaluator import RAGEvaluator
from src.offline_pipeline.embedding_provider import BGEEmbeddingProvider


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("评测集应为 JSON 数组")
    return data


def print_and_save(result: Dict[str, Any], out_json: str, out_csv: str) -> None:
    avg = result.get("average", {})
    metrics = result.get("metrics", [])

    print("\n==================== 评估结果（平均分，0~1） ====================")
    print(f"  评估引擎 engine : {result.get('engine')}")
    print(f"  样本数 total    : {result.get('total')}")
    print(f"  含标准答案 gt   : {result.get('has_ground_truth')}")
    for k in metrics:
        print(f"  {k:<18}: {avg.get(k, 0):.3f}")
    print(f"  成功评估/总数   : {result.get('total', 0)}/{result.get('total', 0)}")
    print("================================================================")

    # 逐条明细
    scores = result.get("scores", [])
    print("\n逐条明细：")
    for i, s in enumerate(scores, 1):
        parts = "  ".join(f"{k}={s.get(k, 0):.3f}" for k in metrics)
        q = s.get("question", "")[:42]
        print(f"  [{i}] {q:<44} {parts}")

    # 保存 JSON（逐条 + 平均分）
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存: {out_json}")

    # 保存 CSV（与 EduRag ragas_evaluate.py 对齐的产出物）：每行为一条样本的分数，
    # 末尾追加一行平均分（question 列标记为 __average__）。
    try:
        import pandas as pd
        rows = []
        for s in scores:
            rows.append({k: s.get(k, 0.0) for k in metrics})
            rows[-1]["question"] = s.get("question", "")
        avg_row = {k: avg.get(k, 0.0) for k in metrics}
        avg_row["question"] = "__average__"
        rows.append(avg_row)
        df = pd.DataFrame(rows, columns=["question"] + metrics)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"CSV 结果已保存: {out_csv}")
    except Exception as e:
        print(f"CSV 保存失败（不影响 JSON）: {e}")


def run_live_eval(config: Config, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """实时管线评估：构建完整 RAGSystem，对每题跑管线再打分。"""
    from src.online_service.main_api import RAGWebAPI

    print("加载配置与 RAG 系统（首次会加载 BERT / BGE-M3、连接 Milvus / MySQL，请稍候）...")
    t0 = time.time()
    rag = RAGWebAPI(config)
    print(f"RAG 系统就绪，耗时 {time.time() - t0:.1f}s")

    eval_items: List[Dict[str, Any]] = []
    for idx, it in enumerate(items, 1):
        q = it.get("question", "")
        if not q:
            continue
        print(f"[{idx}/{len(items)}] 生成答案: {q[:40]} ...")
        try:
            resp = rag._handle_query(
                q,
                source_filter=None,
                use_cache=False,
                history=None,
                strategy=None,
                session_id=f"eval-{int(time.time())}",
            )
            contexts = [s.get("content", "") for s in (resp.sources or [])]
            answer = resp.answer or ""
        except Exception as e:
            print(f"    生成失败: {e}")
            answer = ""
            contexts = []

        eval_items.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": it.get("reference_answer", ""),
        })

    print("\n开始 Ragas 评估 ...")
    return rag.evaluator.evaluate_dataset(eval_items, show_progress=True)


def run_static_eval(config: Config, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """静态评估：直接对给定的 问答/上下文 打分，无需 Milvus / MySQL。"""
    # 规整字段：参考脚本用 context（单段）或 contexts（列表），统一为 contexts 列表
    norm = []
    for it in items:
        ctx = it.get("contexts") or it.get("context")
        if isinstance(ctx, str):
            ctx = [ctx]
        elif ctx is None:
            ctx = []
        norm.append({
            "question": it.get("question", ""),
            "answer": it.get("answer", ""),
            "contexts": [str(c) for c in ctx],
            "ground_truth": it.get("ground_truth") or it.get("reference") or "",
        })

    print("加载本地 BGE-M3 用于语义打分（首次约需数十秒）...")
    t0 = time.time()
    provider = BGEEmbeddingProvider(config)
    evaluator = RAGEvaluator(config, embedding_provider=provider)
    print(f"BGE-M3 就绪，耗时 {time.time() - t0:.1f}s")

    print("\n开始 Ragas 评估 ...")
    return evaluator.evaluate_dataset(norm, show_progress=True)


def main():
    parser = argparse.ArgumentParser(description="MedRAG RAG 评估脚本（基于 Ragas）")
    parser.add_argument("--test-set", default=str(ROOT / "data/test_query/test_qa.json"),
                        help="实时管线评估用的测试问题集（含 question / reference_answer）")
    parser.add_argument("--static", default=None,
                        help="静态评估：直接读取 Ragas 格式的评测集 "
                             "(question/context/answer/ground_truth)，不跑检索管线")
    parser.add_argument("--out-json", default=str(ROOT / "data/test_query/eval_report.json"))
    parser.add_argument("--out-csv", default=str(ROOT / "data/test_query/ragas_evaluation_results.csv"))
    args = parser.parse_args()

    config = Config()

    if args.static:
        if not os.path.exists(args.static):
            print(f"静态评测集不存在: {args.static}")
            sys.exit(1)
        print(f"静态评估模式：{args.static}\n")
        items = load_json(args.static)
        result = run_static_eval(config, items)
    else:
        if not os.path.exists(args.test_set):
            print(f"测试集不存在: {args.test_set}")
            sys.exit(1)
        print(f"实时管线评估模式：{args.test_set}\n")
        items = load_json(args.test_set)
        print(f"共 {len(items)} 条评测问题\n")
        result = run_live_eval(config, items)

    print_and_save(result, args.out_json, args.out_csv)


if __name__ == "__main__":
    main()
