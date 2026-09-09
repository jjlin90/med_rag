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

断点续跑（2026-08-29 补）：
- 答案生成阶段是最耗时的一环（CPU 环境下约 15~20s/条），200+ 条要跑近 1 小时。
  中途挂掉就全丢，因此生成阶段会边跑边把 answer/contexts 落盘到
  `--answers-file`（默认 <out-json>.answers.json），每 10 条刷一次。
- 重跑时加 `--reuse-answers`，脚本会按 question 匹配已有答案直接跳过生成，
  只重跑打分阶段。想强制重跑加 `--force-rerun`。
- `--limit N` 只跑前 N 条，用于冒烟。
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
    print(f"  四项完整/总数   : {result.get('complete_count', result.get('total', 0))}/{result.get('total', 0)}")
    if result.get("valid_counts"):
        print(f"  各指标有效数    : {result['valid_counts']}")
    ps = result.get("pipeline_stats")
    if ps:
        print("  ---- 管线侧统计（不计入 Ragas 分数，用于判读）----")
        print(f"  平均上下文数    : {ps.get('avg_contexts')}")
        print(f"  命中降级        : {ps.get('degraded')}")
        print(f"  空上下文        : {ps.get('empty_context')}")
        print(f"  空答案          : {ps.get('empty_answer')}")
    print("================================================================")

    # 逐条明细
    scores = result.get("scores", [])
    print("\n逐条明细：")
    for i, s in enumerate(scores, 1):
        parts = "  ".join(
            f"{k}={s[k]:.3f}" if isinstance(s.get(k), (int, float)) else f"{k}=NA"
            for k in metrics
        )
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


def load_cached_answers(path: Path) -> Dict[str, Dict[str, Any]]:
    """读取已生成的答案缓存，返回 {question: {answer, contexts}}。"""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {d["question"]: d for d in data if d.get("question")}
    except Exception as e:
        print(f"  [WARN] 答案缓存读取失败，按无缓存处理: {e}")
        return {}


def run_live_eval(config: Config, items: List[Dict[str, Any]],
                  answers_file: Path, reuse: bool = True,
                  limit: int = 0, offset: int = 0) -> Dict[str, Any]:
    """实时管线评估：构建完整 RAGSystem，对每题跑管线再打分。

    答案生成阶段支持断点续跑：已生成过的 question 直接从 answers_file 复用，
    避免中途失败后重复烧掉近 1 小时的生成时间。

    offset / limit 用于**分块评估**：单个 LLM 的免费额度撑不住全量 210 条的
    judge 调用时，把测试集切成互不相交的块，每块换一个模型打分，最后合并。
    注意与 limit 的交互顺序——先 offset 后 limit，即取 items[offset:offset+limit]。
    """
    from src.online_service.main_api import RAGWebAPI

    if offset:
        items = items[offset:]
    if limit:
        items = items[:limit]

    cache = load_cached_answers(answers_file) if reuse else {}
    if cache:
        print(f"[续跑] 已缓存 {len(cache)} 条答案，将跳过这些题目的生成阶段")

    todo = [it for it in items if it.get("question") and it["question"] not in cache]

    # 全部命中缓存时无需加载 RAG 系统（省掉 10s 模型加载 + Milvus 连接）
    rag = None
    if todo:
        print("加载配置与 RAG 系统（首次会加载 BERT / BGE-M3、连接 Milvus / MySQL，请稍候）...")
        t0 = time.time()
        rag = RAGWebAPI(config)
        print(f"RAG 系统就绪，耗时 {time.time() - t0:.1f}s")
        print(f"待生成 {len(todo)} 条（共 {len(items)} 条）\n")

    new_items: List[Dict[str, Any]] = []
    t_gen = time.time()
    for idx, it in enumerate(todo, 1):
        q = it["question"]
        print(f"[{idx}/{len(todo)}] 生成答案: {q[:40]} ...")
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
            # 记录降级标记：评估时被降级/拒答的样本应单独统计，
            # 否则 L2 安全拒答会拉低 faithfulness，掩盖真实检索质量问题
            degraded = bool(getattr(resp, "degraded", False))
            degraded_reason = str(getattr(resp, "degraded_reason", "") or "")
        except Exception as e:
            print(f"    生成失败: {e}")
            answer, contexts, degraded, degraded_reason = "", [], True, f"exception:{e}"

        new_items.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": it.get("reference_answer", ""),
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "n_contexts": len(contexts),
        })
        # 边跑边存：每 10 条刷一次，中断不丢已生成部分
        if idx % 10 == 0:
            _flush_answers(answers_file, cache, new_items)

    _flush_answers(answers_file, cache, new_items)
    if todo:
        print(f"\n答案生成完成，{len(todo)} 条耗时 {time.time() - t_gen:.0f}s")

    # 按原始顺序重建 eval_items（保持与测试集一致，便于逐条比对）
    merged = dict(cache)
    for d in new_items:
        merged[d["question"]] = d
    eval_items = []
    for it in items:
        q = it.get("question", "")
        if not q or q not in merged:
            continue
        eval_items.append({
            "question": q,
            "answer": merged[q].get("answer", ""),
            "contexts": merged[q].get("contexts", []),
            "ground_truth": it.get("reference_answer", ""),
        })

    # 额外产出：降级/空上下文统计（不计入 Ragas，但写进报告供人工判读）
    stats = {
        "total": len(eval_items),
        "empty_answer": sum(1 for d in new_items if not d.get("answer")),
        "empty_context": sum(1 for d in new_items if not d.get("contexts")),
        "degraded": sum(1 for q in eval_items
                        if merged.get(q["question"], {}).get("degraded")),
        "avg_contexts": round(
            sum(merged.get(q["question"], {}).get("n_contexts", 0)
                for q in eval_items) / max(1, len(eval_items)), 2),
    }

    if rag is None:
        # 全部复用缓存：evaluator 由静态路径构造
        from src.offline_pipeline.embedding_provider import BGEEmbeddingProvider
        print("加载本地 BGE-M3 用于语义打分 ...")
        provider = BGEEmbeddingProvider(config)
        from src.online_service.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(config, embedding_provider=provider)
    else:
        evaluator = rag.evaluator

    print("\n开始 Ragas 评估 ...")
    result = evaluator.evaluate_dataset(eval_items, show_progress=True)
    result["pipeline_stats"] = stats
    return result


def _flush_answers(path: Path, cache: Dict[str, Dict[str, Any]],
                  new_items: List[Dict[str, Any]]) -> None:
    """把 cache + 本轮新生成的答案合并落盘（含降级标记，供后续分析）。"""
    merged = dict(cache)
    for d in new_items:
        merged[d["question"]] = d
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


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
    result = evaluator.evaluate_dataset(norm, show_progress=True)
    gen_models = sorted({
        str(it.get("_gen_model")) for it in items if it.get("_gen_model")
    })
    result["evaluation_config"] = {
        "judge_model": config.LLM_MODEL_NAME,
        "generation_models": gen_models,
        "self_judge": config.LLM_MODEL_NAME in gen_models,
        "note": ("same-model judge; absolute scores may contain self-preference bias"
                 if config.LLM_MODEL_NAME in gen_models else "independent judge"),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="MedRAG RAG 评估脚本（基于 Ragas）")
    parser.add_argument("--test-set", default=str(ROOT / "data/test_query/test_qa.json"),
                        help="实时管线评估用的测试问题集（含 question / reference_answer）")
    parser.add_argument("--static", default=None,
                        help="静态评估：直接读取 Ragas 格式的评测集 "
                             "(question/context/answer/ground_truth)，不跑检索管线")
    parser.add_argument("--out-json", default=str(ROOT / "data/test_query/eval_report.json"))
    parser.add_argument("--out-csv", default=str(ROOT / "data/test_query/ragas_evaluation_results.csv"))
    parser.add_argument("--answers-file", default=None,
                        help="答案缓存文件（默认 <out-json>.answers.json），支持断点续跑")
    parser.add_argument("--reuse-answers", action="store_true",
                        help="复用已有的答案缓存，跳过已生成过的题目")
    parser.add_argument("--force-rerun", action="store_true",
                        help="忽略已有答案缓存，全部重新生成")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条（冒烟用）")
    parser.add_argument("--offset", type=int, default=0,
                        help="跳过前 N 条（配合 --limit 做分块评估："
                             "取 items[offset:offset+limit]，每块可换不同 judge 模型）")
    parser.add_argument("--indices", default=None,
                        help="仅静态评估：按 1-based 序号选择非连续样本，"
                             "例如 18,86,92；不可与 --offset/--limit 同用")
    args = parser.parse_args()

    config = Config()
    answers_file = (Path(args.answers_file) if args.answers_file
                    else Path(str(args.out_json) + ".answers.json"))

    if args.static:
        if not os.path.exists(args.static):
            print(f"静态评测集不存在: {args.static}")
            sys.exit(1)
        print(f"静态评估模式：{args.static}\n")
        items = load_json(args.static)
        if args.indices:
            if args.offset or args.limit:
                parser.error("--indices 不可与 --offset/--limit 同用")
            try:
                selected = [int(v.strip()) for v in args.indices.split(",") if v.strip()]
            except ValueError:
                parser.error("--indices 必须是逗号分隔的整数")
            invalid = [i for i in selected if i < 1 or i > len(items)]
            if invalid:
                parser.error(f"--indices 越界: {invalid}，数据集共 {len(items)} 条")
            items = [items[i - 1] for i in selected]
            print(f"静态评估序号：{selected}，共 {len(items)} 条\n")
        else:
            if args.offset:
                items = items[args.offset:]
            if args.limit:
                items = items[:args.limit]
        if (args.offset or args.limit) and not args.indices:
            print(
                f"静态评估区间：[{args.offset}, "
                f"{args.offset + len(items)})，共 {len(items)} 条\n"
            )
        result = run_static_eval(config, items)
    else:
        if not os.path.exists(args.test_set):
            print(f"测试集不存在: {args.test_set}")
            sys.exit(1)
        print(f"实时管线评估模式：{args.test_set}\n")
        items = load_json(args.test_set)
        print(f"共 {len(items)} 条评测问题"
              + (f"，本次评估区间 [{args.offset}, {args.offset + args.limit})"
                 if (args.offset or args.limit) else ""))
        print(f"答案缓存：{answers_file}"
              f"{'（复用）' if args.reuse_answers else '（不复用，将覆盖）'}\n")
        result = run_live_eval(config, items, answers_file,
                               reuse=args.reuse_answers and not args.force_rerun,
                               limit=args.limit, offset=args.offset)

    print_and_save(result, args.out_json, args.out_csv)


if __name__ == "__main__":
    main()
