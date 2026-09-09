"""
全量 210 题 Faithfulness + AnswerRelevancy 配对复评（同裁判）
==========================================================
用智谱官方 glm-4.7 作为唯一裁判，分别对「原答案」和「grounding 重生成答案」
打分，配对比较 ΔF / ΔAR。CP/CR 不变（contexts 未改），沿用原值。

额度估算：
  生成 ~210 条答案 ≈ 10 万 token
  裁判 210×2 (orig+grounded) × 2指标 ≈ 374 万 token
  总计 ≈ 384 万 < glm-4.7 的 500 万免费额度

用法：
  python scripts/eval_f_ar_paired.py [--n N] [--judge-model MODEL] [--answers FILE]

输出：
  data/test_query/_eval_f_ar_paired.json   每题明细
  data/test_query/_eval_f_ar_paired.log    运行日志
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import numpy as np

# ===================== 常量 =====================
SRC = PROJECT_ROOT / "data/test_query/eval_final_merged.json"
ORIG_ANSWERS = PROJECT_ROOT / "data/test_query/eval_answers_210.json"  # 原始 210 条答案
ZHIPU_BASE = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_JUDGE = "glm-4.5-air"
DEFAULT_GEN_MODELS = ["deepseek-v4-pro-202606", "glm-5"]  # TokenHub 生成链

STRICT_SYS = (
    "你是一个严谨的医疗问答助手。你的回答必须**严格基于**提供的参考上下文。\n"
    "规则：\n"
    "1. 只使用上下文中明确提到的信息作答\n"
    "2. 如果上下文中没有足够信息回答问题，请明确说明\"根据提供的资料无法回答\"\n"
    "3. 不要编造、推断或补充上下文之外的知识\n"
    "4. 引用具体内容时保持原意，不要意译或概括到失真"
)
STRICT_USR = (
    "请根据以下参考上下文回答用户问题：\n\n"
    "【参考上下文】\n{context}\n\n"
    "【用户问题】\n{query}"
)


def get_zhipu_client():
    """创建智谱官方 OpenAI 客户端"""
    from openai import OpenAI
    key = os.getenv("LLM_API_KEY", "")
    if not key:
        raise ValueError(".env 中 LLM_API_KEY 为空")
    return OpenAI(base_url=ZHIPU_BASE, api_key=key)


def generate_grounding_zhipu(client, question, contexts):
    """用智谱官方 API 生成 grounding 答案"""
    ctx_text = "\n\n".join(contexts) if contexts else ""
    prompt = STRICT_USR.format(context=ctx_text, query=question)
    try:
        r = client.chat.completions.create(
            model="glm-4.7",
            messages=[
                {"role": "system", "content": STRICT_SYS},
                {"role": "user", "content": prompt},
            ],
            temperature=1,
            max_tokens=1024,
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"  [生成失败] {question[:30]}...: {str(e)[:80]}")
        return ""


def get_tokenhub_generator():
    """不再使用 TokenHub 生成器，改用智谱 glm-4.7 直接生成"""
    return None, "glm-4.7(zhipu)"


# ===================== Ragas F+AR 评估（单裁判） =====================

def eval_f_ar_ragas(items, judge_model, label=""):
    """
    用 Ragas 库评估一批 (question, answer, contexts) 的 F 和 AR。
    故意不传 ground_truth → Ragas 自动只跑 Faithfulness + AnswerRelevancy。
    返回 dict: {question: {"faithfulness": float, "answer_relevancy": float}}
    """
    from src.online_service.rag_evaluator import RAGEvaluator

    # 临时注入智谱配置
    old_base = os.environ.get("LLM_BASE_URL")
    old_model = os.environ.get("LLM_MODEL_NAME")
    old_temp = os.environ.get("LLM_JUDGE_TEMPERATURE")
    os.environ["LLM_BASE_URL"] = ZHIPU_BASE
    os.environ["LLM_MODEL_NAME"] = judge_model
    os.environ["LLM_JUDGE_TEMPERATURE"] = "1"  # 智谱兼容：配对比较消除了随机性偏差

    from src.config.settings import Config
    cfg = Config()
    evaluator = RAGEvaluator(cfg)
    result = evaluator.evaluate_dataset(items)  # 无 ground_truth → 只跑 F+AR

    # 恢复环境变量
    if old_base is not None:
        os.environ["LLM_BASE_URL"] = old_base
    else:
        os.environ.pop("LLM_BASE_URL", None)
    if old_model is not None:
        os.environ["LLM_MODEL_NAME"] = old_model
    else:
        os.environ.pop("LLM_MODEL_NAME", None)
    if old_temp is not None:
        os.environ["LLM_JUDGE_TEMPERATURE"] = old_temp
    else:
        os.environ.pop("LLM_JUDGE_TEMPERATURE", None)

    scores = {}
    if result.get("scores"):
        for s in result["scores"]:
            q = s.get("question", "")
            scores[q] = {
                "faithfulness": s.get("faithfulness", 0) or 0,
                "answer_relevancy": s.get("answer_relevancy", 0) or 0,
            }
    return scores


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description="F+AR 配对复评")
    parser.add_argument("--n", type=int, default=0, help="只评估前 N 题（0=全部）")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE, help=f"裁判模型（默认 {DEFAULT_JUDGE}）")
    parser.add_argument("--answers", type=str, default="", help="预生成的 grounded 答案 JSON 路径（不提供则在线生成）")
    args = parser.parse_args()

    judge_model = args.judge_model
    print(f"=== F+AR 配对复评 | 裁判={judge_model} ({ZHIPU_BASE}) ===")

    # ---- 加载原始答案（含 answer 字段） ----
    orig_all = json.loads(ORIG_ANSWERS.read_text(encoding="utf-8"))
    orig_map = {it["question"]: it for it in orig_all}
    print(f"原始答案: {len(orig_map)} 条")

    # ---- 加载源评估数据（用于取 contexts / ground_truth） ----
    src_data = json.loads(SRC.read_text(encoding="utf-8"))
    if isinstance(src_data, dict):
        src_scores = src_data.get("scores", [])
    else:
        src_scores = src_data
    src_map = {it["question"]: it for it in src_scores}

    # ---- 合并：以原始答案为准，补上 contexts ----
    items = []
    for q, orig_it in orig_map.items():
        src_it = src_map.get(q, {})
        items.append({
            "question": q,
            "answer": orig_it.get("answer", ""),
            "contexts": orig_it.get("contexts") or src_it.get("contexts", []),
            "ground_truth": orig_it.get("ground_truth") or src_it.get("ground_truth", ""),
        })
    if args.n > 0:
        items = items[:args.n]
    print(f"合并后: {len(items)} 题")

    # ---- 准备 grounded 答案 ----
    grounded_path = Path(args.answers) if args.answers else None
    if grounded_path and grounded_path.exists():
        ga = json.loads(grounded_path.read_text(encoding="utf-8"))
        # 兼容两种 key 名：grounded_answer / answer
        grounded_map = {}
        for g in ga:
            q = g["question"]
            grounded_map[q] = g.get("grounded_answer") or g.get("answer", "")
        print(f"加载预生成 grounded 答案: {len(grounded_map)} 条")
    else:
        grounded_map = {}
        print("无预生成答案，将在线生成（TokenHub 链）...")

    # ---- 构建两批 items（故意不传 ground_truth → Ragas 只跑 F+AR） ----
    orig_items = []
    grnd_items = []

    gen, gen_model = None, "glm-4.7(zhipu)"
    need_gen = []

    for it in items:
        q = it["question"]
        ctxs = it.get("contexts", [])

        # 原答案批次（不传 ground_truth → Ragas 只跑 F+AR）
        orig_items.append({
            "question": q,
            "answer": it.get("answer", ""),
            "contexts": ctxs,
        })

        # Grounded 答案批次
        if q in grounded_map:
            ans_g = grounded_map[q]
        else:
            need_gen.append((q, ctxs))
            ans_g = ""  # 占位

        grnd_items.append({
            "question": q,
            "answer": ans_g,
            "contexts": ctxs,
        })

    # 在线生成缺失的 grounded 答案（用智谱 glm-4.7）
    if need_gen:
        print(f"\n需在线生成 {len(need_gen)} 条 grounded 答案（智谱 glm-4.7）...")
        zhipu_client = get_zhipu_client()
        gen_idx = 0
        for i, (q, ctxs) in enumerate(need_gen):
            ans = generate_grounding_zhipu(zhipu_client, q, ctxs)
            if ans and len(ans.strip()) > 3:
                grnd_items[i]["answer"] = ans
                grounded_map[q] = ans
                gen_idx += 1
            else:
                grnd_items[i]["answer"] = f"[生成失败] {q}"
            if (i + 1) % 20 == 0:
                print(f"  生成进度: {i+1}/{len(need_gen)} (成功 {gen_idx})")
        print(f"生成完成: {gen_idx}/{len(need_gen)} 成功")

        # 保存完整 grounded 答案供后续复用
        out_ans = PROJECT_ROOT / "data/test_query/_paired_grounding_answers.json"
        out_ans.write_text(
            json.dumps([
                {"question": it["question"], "grounded_answer": it["answer"]}
                for it in grnd_items
            ], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"已保存 grounded 答案: {out_ans}")

    # ---- 过滤掉没有有效答案的条目 ----
    valid_orig = [it for it in orig_items if it["answer"] and not it["answer"].startswith("[")]
    valid_grnd = [it for it in grnd_items if it["answer"] and not it["answer"].startswith("[")]
    print(f"\n有效条目: 原={len(valid_orig)}, grounded={len(valid_grnd)}")

    # ---- 第一轮：评估原答案 F+AR ----
    print(f"\n{'='*60}")
    print(f"第 1 轮: 评估原答案 ({len(valid_orig)} 题)")
    print(f"{'='*60}")
    t0 = time.time()
    scores_o = eval_f_ar_ragas(valid_orig, judge_model, label="orig")
    t1 = time.time()
    print(f"原答案评估完成: {len(scores_o)} 条有效分数, 耗时 {t1-t0:.0f}s")

    # ---- 第二轮：评估 grounded 答案 F+AR ----
    print(f"\n{'='*60}")
    print(f"第 2 轮: 评估 grounded 答案 ({len(valid_grnd)} 题)")
    print(f"{'='*60}")
    scores_g = eval_f_ar_ragas(valid_grnd, judge_model, label="grnd")
    t2 = time.time()
    print(f"Grounded 评估完成: {len(scores_g)} 条有效分数, 耗时 {t2-t1:.0f}s")

    # ---- 配对比较 ----
    print(f"\n{'='*60}")
    print(f"配对比较（同裁判 {judge_model}）")
    print(f"{'='*60}")

    deltas = []  # (ΔF, ΔAR)
    per_q = {}

    for it in valid_orig:
        q = it["question"]
        so = scores_o.get(q, {})
        sg = scores_g.get(q, {})
        f_o = so.get("faithfulness", 0) or 0
        ar_o = so.get("answer_relevancy", 0) or 0
        f_g = sg.get("faithfulness", 0) or 0
        ar_g = sg.get("answer_relevancy", 0) or 0

        df = f_g - f_o
        dar = ar_g - ar_o
        deltas.append((df, dar))
        per_q[q] = {
            "orig_F": round(f_o, 4),
            "orig_AR": round(ar_o, 4),
            "grnd_F": round(f_g, 4),
            "grnd_AR": round(ar_g, 4),
            "delta_F": round(df, 4),
            "delta_AR": round(dar, 4),
        }

    # 汇总统计
    d_arr = np.array(deltas)
    mean_df = np.mean(d_arr[:, 0]) if len(deltas) else 0
    mean_dar = np.mean(d_arr[:, 1]) if len(deltas) else 0
    med_df = np.median(d_arr[:, 0]) if len(deltas) else 0
    med_dar = np.median(d_arr[:, 1]) if len(deltas) else 0

    orig_f_mean = np.mean([so.get("faithfulness", 0) or 0 for so in scores_o.values()])
    orig_ar_mean = np.mean([so.get("answer_relevancy", 0) or 0 for so in scores_o.values()])
    grnd_f_mean = np.mean([sg.get("faithfulness", 0) or 0 for sg in scores_g.values()])
    grnd_ar_mean = np.mean([sg.get("answer_relevancy", 0) or 0 for sg in scores_g.values()])

    # 输出表格
    header = f"{'问题':<40} {'ΔF':>7} {'ΔAR':>7} {'原F':>6} {'原AR':>6} {'新F':>6} {'新AR':>6}"
    print(header)
    print("-" * len(header))

    for q, v in sorted(per_q.items(), key=lambda x: -abs(x[1]["delta_F"]))[:30]:
        qlabel = q[:38] + ".." if len(q) > 40 else q
        print(f"{qlabel:<40} {v['delta_F']:>+7.3f} {v['delta_AR']:>+7.3f} "
              f"{v['orig_F']:>6.3f} {v['orig_AR']:>6.3f} "
              f"{v['grnd_F']:>6.3f} {v['grnd_AR']:>6.3f}")

    if len(per_q) > 30:
        print(f"... 共 {len(per_q)} 题，前 30 按 |ΔF| 排序")

    print("-" * len(header))
    print(f"{'均值':<40} {mean_df:>+7.3f} {mean_dar:>+7.3f} "
          f"{orig_f_mean:>6.3f} {orig_ar_mean:>6.3f} "
          f"{grnd_f_mean:>6.3f} {grnd_ar_mean:>6.3f}")
    print(f"{'中位数':<40} {med_df:>+7.3f} {med_dar:>+7.3f}")

    # 综合分（用原 CP/CR）
    cp_old = 0.636
    cr_old = 0.559
    w = 0.25
    old_composite = w * (orig_f_mean + orig_ar_mean + cp_old + cr_old)
    new_composite = w * (grnd_f_mean + grnd_ar_mean + cp_old + cr_old)

    print(f"\n{'='*60}")
    print(f"综合分对比（CP={cp_old}, CR={cr_old} 不变）")
    print(f"{'='*60}")
    print(f"  原综合 = .25×({orig_f_mean:.3f}+{orig_ar_mean:.3f}+{cp_old}+{cr_old}) = {old_composite:.3f}")
    print(f"  新综合 = .25×({grnd_f_mean:.3f}+{grnd_ar_mean:.3f}+{cp_old}+{cr_old}) = {new_composite:.3f}")
    print(f"  Δ综合 = {new_composite - old_composite:+.3f}")

    # ---- 保存结果 ----
    output = {
        "judge_model": judge_model,
        "judge_base_url": ZHIPU_BASE,
        "n_questions": len(per_q),
        "n_valid_orig": len(scores_o),
        "n_valid_grnd": len(scores_g),
        "gen_model_used": gen_model,
        "summary": {
            "orig": {"F": round(orig_f_mean, 4), "AR": round(orig_ar_mean, 4)},
            "grounded": {"F": round(grnd_f_mean, 4), "AR": round(grnd_ar_mean, 4)},
            "delta": {"F": round(mean_df, 4), "AR": round(mean_dar, 4)},
            "composite_old": round(old_composite, 4),
            "composite_new": round(new_composite, 4),
            "composite_delta": round(new_composite - old_composite, 4),
            "context_precision": cp_old,
            "context_recall": cr_old,
        },
        "per_question": per_q,
    }

    out_file = PROJECT_ROOT / "data/test_query/_eval_f_ar_paired.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存: {out_file}")


if __name__ == "__main__":
    main()
