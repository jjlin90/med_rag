"""
全量 F+AR 配对复评（最终版：单线程、短 prompt、一次一题）
======================================================
用法：python scripts/eval_f_ar_direct.py [--n N] [--answers FILE]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI
from scripts.eval_score_utils import (parse_judge_score, make_pair_row, summarize_pairs,
                                     validate_frozen_contexts, require_independent_judge)

ZHIPU_BASE = "https://open.bigmodel.cn/api/paas/v4"
JUDGE = "glm-4.7"
ORIG = PROJECT_ROOT / "data/test_query/eval_answers_210.json"
GRND_D = PROJECT_ROOT / "data/test_query/eval_answers_210_grounded.json"

SYS = "评估员。只输出JSON: {\"score\": 0.0~1.0}。"

F_P = "评估忠实度：答案是否严格基于上下文？\n上下文：{ctx}\n答案：{ans}\nJSON: {\"score\": }"
AR_P = "评估相关性：答案是否回应问题？\n问题：{q}\n答案：{ans}\nJSON: {\"score\": }"

def client():
    k = os.getenv("LLM_API_KEY", "")
    assert k, ".env 缺 LLM_API_KEY"
    return OpenAI(base_url=ZHIPU_BASE, api_key=k)

def build(template, **kw):
    # brace-safe 替换：避免答案/问题文本中的 { } 触发 .format() 异常
    s = template
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v).replace("{", "[").replace("}", "]"))
    return s

def score(c, prompt):
    try:
        r = c.chat.completions.create(model=JUDGE, messages=[{"role":"system","content":SYS},{"role":"user","content":prompt}], temperature=0.01, max_tokens=40, timeout=60)
        if r.choices[0].finish_reason != 'stop':
            raise RuntimeError('Judge response did not finish normally')
        return parse_judge_score(r.choices[0].message.content)
    except Exception as e:
        print(f'    [err] {type(e).__name__}')
        return None


def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=0); p.add_argument("--answers", type=str, default="")
    a = p.parse_args()

    orig = json.loads(ORIG.read_text(encoding="utf-8"))
    om = {it["question"]: it for it in orig}
    gp = Path(a.answers) if a.answers else GRND_D
    gm = {}
    grounded = json.loads(gp.read_text(encoding='utf-8')) if gp.exists() else []
    validate_frozen_contexts(orig, grounded)
    require_independent_judge(JUDGE, orig, grounded)
    if grounded:
        for g in grounded:
            gm[g["question"]] = g.get("answer") or g.get("grounded_answer", "")

    items = [(q, om[q].get("answer",""), gm.get(q,""), om[q].get("contexts",[])) for q in om]
    if a.n: items = items[:a.n]

    print(f"F+AR 配对 | {JUDGE} | {len(items)} 题", flush=True)
    c = client()
    rows, t0 = [], time.time()

    for i, (q, oa, ga, ctx) in enumerate(items):
        ct = "\n".join(ctx) if ctx else "(无)"
        fo = score(c, build(F_P, ctx=ct, ans=oa))
        aro = score(c, build(AR_P, q=q, ans=oa))
        fg = score(c, build(F_P, ctx=ct, ans=ga)) if ga else None
        arg = score(c, build(AR_P, q=q, ans=ga)) if ga else None
        r = make_pair_row(q, fo, aro, fg, arg)
        rows.append(r)
        elapsed = time.time() - t0
        print(f'[{i+1}/{len(items)}] complete={not r["failed"]} ({elapsed:.0f}s)', flush=True)

    summary, n_valid = summarize_pairs(rows)
    out = {'engine': 'custom_direct_llm_judge', 'judge': JUDGE,
           'method': 'direct_openai_single_thread',
           'metric_compatibility': 'not directly comparable to Ragas scores',
           'aggregation_note': 'Only complete F/AR pairs; no cross-engine CP/CR composite.',
           'n': n_valid, 'n_total': len(rows), 'n_failed': len(rows) - n_valid,
           'summary': summary, 'per_question': rows}
    out_file = PROJECT_ROOT / 'data/test_query/_eval_f_ar_direct.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False),
                        encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    print(f'完整配对 {n_valid}/{len(rows)}；结果已保存: {out_file}')


if __name__ == '__main__':
    main()
