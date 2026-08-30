"""
精简配对验证：用【单个裁判】对 N 条问题分别给「原答案」与「grounding 重生成答案」
打分，求 F/AR/CP/CR 的逐题差。用来在有限额度内确认 grounding 是否抬升 F。

配额的现实约束（2026-08-29）：
- 全量 210 题 4 指标复评需 ~750 万 judge token，剩余免费模型拆 5 路仍超单模型 ~1M 上限，
  故全量复评在当前免费额度下不可行。
- 本脚本只跑 N=12（分层 6 条旧F=0 + 6 条旧F>0.4），单裁判 ~85 万 token，落在单模型免费额度内。
- 配对比较（同裁判给原/新答案打分）抵消裁判校准偏差，Δ 方向可信；绝对分值仍受单裁判风格影响。

用法：
  python scripts/exp_grounding_small.py --n 12 --judge hy3
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

M = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def valid(r):
    return any((r.get(k) or 0) > 0 for k in M)


def load_merged():
    q2 = {}
    for f in sorted(ROOT.glob("data/test_query/chunks/chunk_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        for x in d.get("scores", []):
            if not valid(x):
                continue
            q = x["question"]
            if q not in q2:
                q2[q] = {k: x.get(k, 0) for k in M}
    return q2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--judge", default="hy3")
    ap.add_argument("--cached", default="data/test_query/_exp_grounding_answers.json")
    args = ap.parse_args()

    from src.config.settings import Config
    from src.online_service.rag_evaluator import RAGEvaluator
    from src.offline_pipeline.embedding_provider import BGEEmbeddingProvider

    merged = load_merged()
    grounded_cache = json.load(open(ROOT / args.cached, encoding="utf-8"))
    ans = json.load(open(ROOT / "data/test_query/eval_answers_210.json", encoding="utf-8"))
    ans_map = {a["question"]: a for a in ans}
    testset = json.load(open(ROOT / "data/test_query/eval_set_210.json", encoding="utf-8"))
    gt_map = {x["question"]: x.get("reference_answer", "") for x in testset}

    # 分层抽取 N 条：前 15 为旧F=0 组，后 15 为旧F>0.4 组
    half = args.n // 2
    idx = list(range(half)) + list(range(15, 15 + (args.n - half)))
    sample = [grounded_cache[i]["question"] for i in idx if i < len(grounded_cache)]
    print(f"抽样 {len(sample)} 题（旧F=0: {half}, 旧F>0.4: {args.n - half}），裁判={args.judge}")

    grounded_items, orig_items = [], []
    for q in sample:
        a = ans_map[q]
        ctxs = a.get("contexts", [])
        gt = gt_map.get(q, "")
        g_ans = next((x["answer"] for x in grounded_cache if x["question"] == q), None)
        if not g_ans:
            print(f"  [跳过] 无缓存 grounding 答案: {q[:30]}")
            continue
        grounded_items.append({"question": q, "answer": g_ans, "contexts": ctxs, "ground_truth": gt})
        orig_items.append({"question": q, "answer": a.get("answer", ""), "contexts": ctxs, "ground_truth": gt})

    config = Config()
    config.LLM_MODEL_NAME = args.judge
    os.environ["LLM_JUDGE_TEMPERATURE"] = "1"  # 统一用 temp=1，兼容所有模型
    provider = BGEEmbeddingProvider(config)
    ev = RAGEvaluator(config, embedding_provider=provider)

    print("  · 评估原答案 ...")
    res_o = ev.evaluate_dataset(orig_items, show_progress=False)
    print("  · 评估 grounding 答案 ...")
    res_g = ev.evaluate_dataset(grounded_items, show_progress=False)

    so = {s["question"]: s for s in res_o.get("scores", [])}
    sg = {s["question"]: s for s in res_g.get("scores", [])}

    print("\n=== 配对比较（同裁判，原 vs grounding）===")
    print(f'{"问题":<26}{"ΔF":>8}{"ΔAR":>8}{"ΔCP":>8}{"ΔCR":>8}')
    g_f = g_ar = g_cp = g_cr = 0
    o_f = o_ar = o_cp = o_cr = 0
    for it in grounded_items:
        q = it["question"]
        o = so.get(q, {}); g = sg.get(q, {})
        df = round((g.get("faithfulness", 0) or 0) - (o.get("faithfulness", 0) or 0), 3)
        dar = round((g.get("answer_relevancy", 0) or 0) - (o.get("answer_relevancy", 0) or 0), 3)
        dcp = round((g.get("context_precision", 0) or 0) - (o.get("context_precision", 0) or 0), 3)
        dcr = round((g.get("context_recall", 0) or 0) - (o.get("context_recall", 0) or 0), 3)
        g_f += g.get("faithfulness", 0) or 0; o_f += o.get("faithfulness", 0) or 0
        g_ar += g.get("answer_relevancy", 0) or 0; o_ar += o.get("answer_relevancy", 0) or 0
        g_cp += g.get("context_precision", 0) or 0; o_cp += o.get("context_precision", 0) or 0
        g_cr += g.get("context_recall", 0) or 0; o_cr += o.get("context_recall", 0) or 0
        print(f"{q[:24]:<26}{df:>+8.3f}{dar:>+8.3f}{dcp:>+8.3f}{dcr:>+8.3f}")

    n = len(grounded_items)
    print(f'{"均值":<26}{ (g_f-o_f)/n:>+8.3f}{(g_ar-o_ar)/n:>+8.3f}{(g_cp-o_cp)/n:>+8.3f}{(g_cr-o_cr)/n:>+8.3f}')
    print(f"\n原答案均值   F={o_f/n:.4f} AR={o_ar/n:.4f} CP={o_cp/n:.4f} CR={o_cr/n:.4f}")
    print(f"grounding均值 F={g_f/n:.4f} AR={g_ar/n:.4f} CP={g_cp/n:.4f} CR={g_cr/n:.4f}")

    out = {
        "judge": args.judge, "n": n,
        "orig_mean": {"faithfulness": o_f/n, "answer_relevancy": o_ar/n,
                      "context_precision": o_cp/n, "context_recall": o_cr/n},
        "grounded_mean": {"faithfulness": g_f/n, "answer_relevancy": g_ar/n,
                          "context_precision": g_cp/n, "context_recall": g_cr/n},
    }
    p = ROOT / "data/test_query/_exp_grounding_small.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存: {p}")


if __name__ == "__main__":
    main()
