"""合并分块 Ragas 评估报告（多裁判场景）。

背景：单个 LLM 的免费额度撑不住 210 条题目的 judge 调用（实测全量约需
240 万 token，单个模型通常只有 100 万）。解决办法是把测试集切成互不相交的
若干块，每块换一个模型打分，最后合并。

这样做可让每个模型的开销可预算、可控，避免中途额度耗尽产生大面积 0 分。

重要：各模型评的是互不相交的题块，这叫“多裁判路由”，不是 ensemble。
模型均值差同时混入了题目难度差，不能解释为纯粹的裁判方差。真正的 ensemble
需要每一道题都由相同的一组裁判评分后再聚合。

合并时做三件事：
1. 拼接所有块的逐条明细（按 question 去重，防止块之间有重叠）；
2. 计算全局四项指标均值 + 等权加权综合；
3. 输出**每个裁判负责题块的均值与极差**，仅作运行诊断。

用法：
    python scripts/merge_eval_chunks.py \
        --chunk data/test_query/chunk1.json=deepseek-v4-flash \
        --chunk data/test_query/chunk2.json=glm-5.3-flash \
        --out-json data/test_query/eval_report_merged.json \
        --out-csv  data/test_query/ragas_merged.csv
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def parse_chunk(spec: str):
    """解析 --chunk path=model 形式；model 可省略。"""
    if "=" in spec:
        path, model = spec.rsplit("=", 1)
    else:
        path, model = spec, None
    return Path(path).resolve(), (model or Path(path).stem)


def main():
    ap = argparse.ArgumentParser(description="合并分块评估报告")
    ap.add_argument("--chunk", action="append", required=True,
                    help="分块报告路径，可重复；格式 path=模型名")
    ap.add_argument("--out-json", default=str(ROOT / "data/test_query/eval_report_merged.json"))
    ap.add_argument("--out-csv", default=str(ROOT / "data/test_query/ragas_merged.csv"))
    args = ap.parse_args()

    all_scores = []
    per_judge = []
    seen_questions = set()
    dup = 0

    for spec in args.chunk:
        path, model = parse_chunk(spec)
        if not path.exists():
            print(f"[警告] 分块文件不存在，跳过: {path}")
            continue
        report = json.load(open(path, encoding="utf-8"))
        scores = report.get("scores", [])
        valid = 0
        for s in scores:
            q = s.get("question", "")
            if q and q in seen_questions:
                dup += 1
                continue
            if q:
                seen_questions.add(q)
            s = dict(s)
            s["_judge"] = model
            all_scores.append(s)
            valid += 1

        # 该裁判下的分块均值（只统计非全 0 的有效条，避免额度事故拖低）
        effective = [s for s in scores
                     if any(s.get(m, 0) > 0 for m in METRICS)]
        avg = {m: (statistics.mean([s.get(m, 0) for s in effective])
                   if effective else 0.0) for m in METRICS}
        per_judge.append({
            "judge": model,
            "file": str(path),
            "engine": report.get("engine", "?"),
            "n_scored": valid,
            "n_effective": len(effective),
            "average": {m: round(avg[m], 4) for m in METRICS},
            "composite": round(statistics.mean(avg.values()), 4) if effective else 0.0,
        })

    if not all_scores:
        print("没有可合并的评分，退出。")
        return 1

    overall = {m: statistics.mean([s.get(m, 0) for s in all_scores]) for m in METRICS}
    composite = statistics.mean(overall.values())

    # 路由题块间差异：同时包含裁判口径和题目难度，不得称为裁判方差。
    comps = [j["composite"] for j in per_judge if j["n_effective"] > 0]
    judge_spread = {
        "n_judges": len(comps),
        "composite_min": round(min(comps), 4) if comps else 0.0,
        "composite_max": round(max(comps), 4) if comps else 0.0,
        "composite_range": round(max(comps) - min(comps), 4) if comps else 0.0,
        "composite_stdev": round(statistics.stdev(comps), 4) if len(comps) > 1 else 0.0,
    }

    out = {
        "engine": "ragas",
        "mode": "multi-judge-routed-chunks",
        "aggregation_note": (
            "Each question was scored by one routed judge. This is not a per-item "
            "ensemble; between-chunk spread is confounded by sample difficulty."
        ),
        "n_total": len(all_scores),
        "n_duplicate_dropped": dup,
        "average": {m: round(overall[m], 4) for m in METRICS},
        "composite": round(composite, 4),
        "judge_spread": judge_spread,
        "per_judge": per_judge,
        "scores": all_scores,
    }

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # CSV 逐条明细
    out_csv = Path(args.out_csv).resolve()
    cols = ["question", "_judge"] + METRICS
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in all_scores:
            row = {c: s.get(c, "") for c in cols}
            for m in METRICS:
                row[m] = round(s.get(m, 0), 4)
            w.writerow(row)

    print("=" * 78)
    print(f"合并完成：{len(all_scores)} 条"
          + (f"（丢弃重复 {dup} 条）" if dup else ""))
    print("-" * 78)
    print("全局指标（等权加权综合）:")
    for m in METRICS:
        print(f"  {m:<20}: {overall[m]:.4f}")
    print(f"  {'composite':<20}: {composite:.4f}")
    print("-" * 78)
    print("各裁判负责题块的表现（混入题目难度，仅作运行诊断）:")
    for j in per_judge:
        print(f"  {j['judge']:<24} n={j['n_effective']:<4} "
              f"composite={j['composite']:.4f}  {j['average']}")
    print("-" * 78)
    print(f"路由题块 composite 极差={judge_spread['composite_range']:.4f}  "
          f"标准差={judge_spread['composite_stdev']:.4f}  "
          f"（{judge_spread['n_judges']} 个裁判）")
    print("=" * 78)
    print(f"JSON: {out_json}")
    print(f"CSV : {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
