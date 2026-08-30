"""
小实验：验证「严格 grounding prompt」能否抬升 faithfulness。

设计（相比旧版的关键修正）：
- 生成层加多模型兜底 + 强制 temperature=1（TokenHub 部分模型只接受 temp=1，
  旧版用默认 0.2 直接触发 400 → generate 返回 None → 全部跳过，实验空跑）。
- 评估层做「同裁判配对比较」：对每条问题，用【同一个裁判】分别给
  原始答案 与 grounding 重生成答案 打分，再求差。这样消除了
  "baseline=6裁判均值 vs new=单裁判" 的校准偏差，F 的升降才可信。
- 裁判用一个小面板（默认 glm-5.2 + minimax-m3）交叉验证，降低单裁判抖动。

用法：
  python scripts/exp_grounding.py --n 30
  python scripts/exp_grounding.py --n 30 --judges glm-5.2 minimax-m3
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

M = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# 生成模型兜底链（按可靠性排序；全部用 temp=1 以兼容只接受 1 的模型）
GEN_MODELS = [
    "deepseek-v4-pro-202606",
    "glm-5.2",
    "glm-5",
    "hy3",
    "minimax-m3",
    "kimi-k3",
    "glm-5.3-flash",
]

STRICT_SYS = """你是一个严谨的医疗知识问答助手。你必须【严格只】基于下面【相关知识】中给出的内容来回答用户问题，遵守以下规则：

1. 只能使用【相关知识】中明确包含的信息。绝对不允许引入【相关知识】之外的任何医学知识、常识或个人推断。
2. 如果【相关知识】的内容不足以回答用户问题，或完全不相关，必须明确说明"根据提供的资料，无法回答该问题"，不要尝试用外部知识补充。
3. 回答中的每一条事实陈述都必须能在【相关知识】中找到对应依据。
4. 保持专业、简洁；涉及医疗建议时提醒"仅供参考，不能替代专业医疗建议"。"""

STRICT_USR = """请仅基于以下【相关知识】回答用户问题，不要使用任何额外知识：

【相关知识】
{context}

【用户问题】
{query}

请严格依据上述【相关知识】作答："""


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


def gen_with_fallback(cfg, q, ctxs):
    """按 GEN_MODELS 顺序尝试，返回首个非空答案与所用模型。"""
    from src.online_service.llm_generator import LLMGenerator
    last_err = None
    for m in GEN_MODELS:
        cfg.LLM_MODEL_NAME = m
        cfg.LLM_TEMPERATURE = 1  # 兼容只接受 temp=1 的模型
        try:
            g = LLMGenerator(cfg)
            if not g.client:
                continue
            ans = g.generate(STRICT_USR.format(context="\n\n".join(ctxs), query=q),
                             system_prompt=STRICT_SYS)
            if ans:
                return ans, m
        except Exception as e:
            last_err = e
    if last_err:
        print(f"  [warn] 全部生成模型失败: {str(last_err)[:60]}")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--judges", nargs="*", default=["glm-5.2", "minimax-m3"])
    args = ap.parse_args()

    from src.config.settings import Config
    from src.online_service.rag_evaluator import RAGEvaluator
    from src.offline_pipeline.embedding_provider import BGEEmbeddingProvider

    merged = load_merged()
    ans = json.load(open(ROOT / "data/test_query/eval_answers_210.json", encoding="utf-8"))
    ans_map = {a["question"]: a for a in ans}

    # 分层抽样：15 条旧F=0（最难），15 条旧F>0.4（易）
    zero = [q for q, s in merged.items() if s["faithfulness"] == 0]
    hi = [q for q, s in merged.items() if s["faithfulness"] > 0.4]
    random.seed(42)
    random.shuffle(zero); random.shuffle(hi)
    half = args.n // 2
    sample = zero[:half] + hi[: args.n - half]
    print(f"抽样 {len(sample)} 题（F=0: {min(half, len(zero))}, F>0.4: {args.n - min(half, len(zero))}）")

    config = Config()
    config.LLM_TEMPERATURE = 1

    # 1) 生成 grounding 答案（带兜底）
    grounded_items, orig_items, used_models = [], [], []
    ok = 0
    for q in sample:
        a = ans_map[q]
        ctxs = a.get("contexts", [])
        gt = a.get("ground_truth", "") or a.get("reference_answer", "")
        new_ans, used = gen_with_fallback(config, q, ctxs)
        if not new_ans:
            print(f"  [跳过] 生成彻底失败: {q[:30]}")
            continue
        grounded_items.append({"question": q, "answer": new_ans, "contexts": ctxs, "ground_truth": gt})
        orig_items.append({"question": q, "answer": a.get("answer", ""), "contexts": ctxs, "ground_truth": gt})
        used_models.append(used)
        ok += 1
    print(f"成功重生成 {ok} 题；生成模型分布: {dict((m, used_models.count(m)) for m in set(used_models))}")

    if not grounded_items:
        print("无可用样本，退出")
        return

    # 2) 同裁判配对评估：每位裁判分别给 orig / grounded 打分
    provider = BGEEmbeddingProvider(config)
    # per_q[m][judge] = {metric: score}
    per_q = defaultdict(lambda: defaultdict(dict))
    for judge in args.judges:
        config.LLM_MODEL_NAME = judge
        os.environ["LLM_JUDGE_TEMPERATURE"] = "0"
        ev = RAGEvaluator(config, embedding_provider=provider)
        print(f"  · 裁判 {judge} 评估中 ...")
        res_o = ev.evaluate_dataset(orig_items, show_progress=False)
        res_g = ev.evaluate_dataset(grounded_items, show_progress=False)
        so = {s["question"]: s for s in res_o.get("scores", [])}
        sg = {s["question"]: s for s in res_g.get("scores", [])}
        for it in grounded_items:
            q = it["question"]
            if q not in per_q:
                per_q[q] = {"orig": {}, "grnd": {}}
            per_q[q]["orig"][judge] = {k: (so.get(q, {}).get(k, 0) or 0) for k in M}
            per_q[q]["grnd"][judge] = {k: (sg.get(q, {}).get(k, 0) or 0) for k in M}

    # 3) 跨裁判平均后，求 orig vs grounded 逐题差与全局均值
    # 题目级：先对每个裁判的 orig/grnd 取指标均值，再跨裁判平均
    q_summary = []
    for it in grounded_items:
        q = it["question"]
        o = per_q[q]["orig"]; g = per_q[q]["grnd"]
        row = {"question": q}
        for k in M:
            ov = sum(o[j].get(k, 0) for j in o) / len(o) if o else 0
            gv = sum(g[j].get(k, 0) for j in g) / len(g) if g else 0
            row[k] = round(ov, 4)
            row[k + "_g"] = round(gv, 4)
            row[k + "_delta"] = round(gv - ov, 4)
        q_summary.append(row)

    # 全局均值
    print("\n=== 配对比较（orig vs grounded，跨裁判均值）===")
    print(f'{"指标":<20}{"orig":>10}{"grounded":>12}{"Δ":>10}')
    globalsum = {}
    for k in M:
        b = sum(r[k] for r in q_summary) / len(q_summary)
        nv = sum(r[k + "_g"] for r in q_summary) / len(q_summary)
        globalsum[k] = (b, nv)
        print(f"{k:<20}{b:>10.4f}{nv:>12.4f}{nv - b:>+10.4f}")
    w_b = sum(v[0] for v in globalsum.values()) / 4
    w_n = sum(v[1] for v in globalsum.values()) / 4
    print(f'{"weighted":<20}{w_b:>10.4f}{w_n:>12.4f}{w_n - w_b:>+10.4f}')

    # 保存
    out = {
        "gen_models": GEN_MODELS,
        "judges": args.judges,
        "n": ok,
        "global": {k: {"orig": v[0], "grounded": v[1], "delta": v[1] - v[0]} for k, v in globalsum.items()},
        "weighted": {"orig": w_b, "grounded": w_n, "delta": w_n - w_b},
        "per_question": q_summary,
    }
    out_path = ROOT / "data/test_query/_exp_grounding.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存实验结果到 {out_path}")

    # 顺手保存 grounded 样本，供全量重生成参考
    gsample = ROOT / "data/test_query/_exp_grounding_answers.json"
    gsample.write_text(json.dumps(
        [{"question": it["question"], "answer": it["answer"], "gen_model": used_models[i]}
         for i, it in enumerate(grounded_items)], ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n总耗时 {time.time() - t0:.0f}s")
