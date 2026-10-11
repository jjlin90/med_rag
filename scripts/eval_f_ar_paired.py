"""
全量 210 题 Faithfulness + AnswerRelevancy 配对复评（同裁判）
==========================================================
使用指定的独立裁判，分别对原答案和重生成答案评分；
只在同题、同上下文且两侧 F/AR 均有效时汇总，不拼接历史 CP/CR。

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

from scripts.eval_score_utils import (valid_score, make_pair_row, summarize_pairs,
                                     validate_frozen_contexts, require_independent_judge)

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
        if r.choices[0].finish_reason != 'stop':
            raise RuntimeError('Generation did not finish normally')
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
    os.environ["LLM_JUDGE_TEMPERATURE"] = "1"  # 两侧保持一致；配对不能消除采样随机性
    try:
        from src.config.settings import Config
        cfg = Config()
        evaluator = RAGEvaluator(cfg)
        result = evaluator.evaluate_dataset(items)  # 无 ground_truth → 只跑 F+AR
        if result.get('engine') != 'ragas':
            raise RuntimeError('Paired Ragas comparison cannot substitute a fallback engine')
    finally:
        for key, previous in (('LLM_BASE_URL', old_base), ('LLM_MODEL_NAME', old_model),
                              ('LLM_JUDGE_TEMPERATURE', old_temp)):
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    scores = {}
    if result.get("scores"):
        for s in result["scores"]:
            q = s.get("question", "")
            scores[q] = {
                "faithfulness": s.get("faithfulness") if valid_score(s.get("faithfulness")) else None,
                "answer_relevancy": s.get("answer_relevancy") if valid_score(s.get("answer_relevancy")) else None,
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
    ga = []
    grounded_path = Path(args.answers) if args.answers else None
    if grounded_path and grounded_path.exists():
        ga = json.loads(grounded_path.read_text(encoding="utf-8"))
        validate_frozen_contexts(items, ga)
        require_independent_judge(judge_model, orig_all, ga)
        # 兼容两种 key 名：grounded_answer / answer
        grounded_map = {}
        for g in ga:
            q = g["question"]
            grounded_map[q] = g.get("grounded_answer") or g.get("answer", "")
        print(f"加载预生成 grounded 答案: {len(grounded_map)} 条")
    else:
        grounded_map = {}
        print("无预生成答案，将在线生成（智谱 glm-4.7）...")

    # ---- 构建两批 items（故意不传 ground_truth → Ragas 只跑 F+AR） ----
    orig_items = []
    grnd_items = []

    gen, gen_model = None, "glm-4.7(zhipu)"
    need_gen = []

    for item_index, it in enumerate(items):
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
            need_gen.append((item_index, q, ctxs))
            ans_g = ""  # 占位

        grnd_items.append({
            "question": q,
            "answer": ans_g,
            "contexts": ctxs,
        })

    # 在线生成缺失的 grounded 答案（用智谱 glm-4.7）
    if need_gen:
        if judge_model == 'glm-4.7':
            parser.error('在线生成使用 glm-4.7，请选择不同的独立裁判')
        require_independent_judge(judge_model, orig_all)
        print(f"\n需在线生成 {len(need_gen)} 条 grounded 答案（智谱 glm-4.7）...")
        zhipu_client = get_zhipu_client()
        gen_idx = 0
        for i, (item_index, q, ctxs) in enumerate(need_gen):
            ans = generate_grounding_zhipu(zhipu_client, q, ctxs)
            if ans and len(ans.strip()) > 3:
                grnd_items[item_index]["answer"] = ans
                grounded_map[q] = ans
                gen_idx += 1
            else:
                grnd_items[item_index]["answer"] = ""
            if (i + 1) % 20 == 0:
                print(f"  生成进度: {i+1}/{len(need_gen)} (成功 {gen_idx})")
        print(f"生成完成: {gen_idx}/{len(need_gen)} 成功")

        # 保存完整 grounded 答案供后续复用
        out_ans = PROJECT_ROOT / "data/test_query/_paired_grounding_answers.json"
        out_ans.parent.mkdir(parents=True, exist_ok=True)
        out_ans.write_text(
            json.dumps([
                {"question": it["question"], "grounded_answer": it["answer"],
                 "contexts": it['contexts'],
                 "_gen_model": 'glm-4.7' if it['question'] in {q for _, q, _ in need_gen}
                               else next((g.get('_gen_model') for g in ga
                                          if g['question'] == it['question']), None)}
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

    # 汇总只使用两侧 F/AR 都有效的同题配对；失败题保留 null 明细。
    rows = []
    for it in items:
        q = it['question']
        so, sg = scores_o.get(q, {}), scores_g.get(q, {})
        rows.append(make_pair_row(q, so.get('faithfulness'), so.get('answer_relevancy'),
                                  sg.get('faithfulness'), sg.get('answer_relevancy')))
    summary, n_complete = summarize_pairs(rows)
    summary['grounded'] = summary.pop('grnd')
    per_q = {r['q']: {'orig_F': r['oF'], 'orig_AR': r['oAR'],
                       'grnd_F': r['gF'], 'grnd_AR': r['gAR'],
                       'delta_F': r['dF'], 'delta_AR': r['dAR'],
                       'failed': r['failed']} for r in rows}
    output = {
        'engine': 'ragas', 'judge_model': judge_model, 'judge_base_url': ZHIPU_BASE,
        'n_questions': len(items), 'n_complete_pairs': n_complete,
        'n_failed_pairs': len(items) - n_complete,
        'n_valid_orig': sum(all(valid_score(s.get(k)) for k in
                               ('faithfulness', 'answer_relevancy')) for s in scores_o.values()),
        'n_valid_grnd': sum(all(valid_score(s.get(k)) for k in
                               ('faithfulness', 'answer_relevancy')) for s in scores_g.values()),
        'aggregation_note': 'All means use the same complete pairs; only F/AR are evaluated.',
        'summary': summary, 'per_question': per_q,
    }
    out_file = PROJECT_ROOT / 'data/test_query/_eval_f_ar_paired.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
                        encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    print(f'完整配对 {n_complete}/{len(items)}；结果已保存: {out_file}')


if __name__ == '__main__':
    main()
