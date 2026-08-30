"""
全量 F+AR 配对复评（最终版：单线程、短 prompt、一次一题）
======================================================
用法：python scripts/eval_f_ar_direct.py [--n N] [--answers FILE]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import numpy as np
from openai import OpenAI

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
        m = re.search(r'"score"\s*:\s*(\d+\.?\d*)', r.choices[0].message.content or "")
        return min(1.0, max(0.0, float(m.group(1)))) if m else 0.0
    except Exception as e:
        print(f"    [err] {str(e)[:60]}")
        return 0.0

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=0); p.add_argument("--answers", type=str, default="")
    a = p.parse_args()

    orig = json.loads(ORIG.read_text(encoding="utf-8"))
    om = {it["question"]: it for it in orig}
    gp = Path(a.answers) if a.answers else GRND_D
    gm = {}
    if gp.exists():
        for g in json.loads(gp.read_text(encoding="utf-8")):
            gm[g["question"]] = g.get("answer") or g.get("grounded_answer", "")

    items = [(q, om[q].get("answer",""), gm.get(q,""), om[q].get("contexts",[])) for q in om]
    if a.n: items = items[:a.n]

    print(f"F+AR 配对 | {JUDGE} | {len(items)} 题", flush=True)
    c = client()
    rows, t0 = [], time.time()

    for i, (q, oa, ga, ctx) in enumerate(items):
        ct = "\n".join(ctx[:3]) if ctx else "(无)"
        fo = score(c, build(F_P, ctx=ct, ans=oa))
        aro = score(c, build(AR_P, q=q, ans=oa))
        fg = score(c, build(F_P, ctx=ct, ans=ga)) if ga else 0.0
        arg = score(c, build(AR_P, q=q, ans=ga)) if ga else 0.0
        r = {"q": q, "oF": round(fo,3), "oAR": round(aro,3), "gF": round(fg,3), "gAR": round(arg,3), "dF": round(fg-fo,3), "dAR": round(arg-aro,3)}
        rows.append(r)
        elapsed = time.time() - t0
        rate = (i+1)/elapsed*60 if elapsed > 0 else 0
        print(f"[{i+1}/{len(items)}] ΔF={r['dF']:+.2f} ΔAR={r['dAR']:+.2f} ({rate:.1f}q/min, {elapsed:.0f}s)", flush=True)

    dF=np.array([r["dF"] for r in rows]); dAR=np.array([r["dAR"] for r in rows])
    oF=np.array([r["oF"] for r in rows]); oAR=np.array([r["oAR"] for r in rows])
    gF=np.array([r["gF"] for r in rows]); gAR=np.array([r["gAR"] for r in rows])
    cp, cr, w = 0.636, 0.559, 0.25
    oc, nc = w*(np.mean(oF)+np.mean(oAR)+cp+cr), w*(np.mean(gF)+np.mean(gAR)+cp+cr)

    print(f"\n{'='*55}", flush=True)
    print(f"汇总 ({len(rows)}题, {time.time()-t0:.0f}s)", flush=True)
    print(f"{'='*55}", flush=True)
    for l,ov,nv,dv in [("F",np.mean(oF),np.mean(gF),np.mean(dF)),("AR",np.mean(oAR),np.mean(gAR),np.mean(dAR)),("CP",cp,cp,0),("CR",cr,cr,0)]:
        print(f"{l:<6} {ov:>7.3f} {nv:>7.3f} {dv:>+7.3f}", flush=True)
    print(f"{'-'*35}", flush=True)
    print(f"综合   {oc:>7.3f} {nc:>7.3f} {nc-oc:>+7.3f}", flush=True)

    out = {"judge": JUDGE, "method": "direct_openai_single_thread", "n": len(rows),
           "summary": {"orig":{"F":round(float(np.mean(oF)),4),"AR":round(float(np.mean(oAR)),4)},
                      "grnd":{"F":round(float(np.mean(gF)),4),"AR":round(float(np.mean(gAR)),4)},
                      "delta":{"F":round(float(np.mean(dF)),4),"AR":round(float(np.mean(dAR)),4)},
                      "composite_old":round(oc,4),"composite_new":round(nc,4)},
           "per_question": rows}
    (PROJECT_ROOT/"data/test_query/_eval_f_ar_direct.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存: data/test_query/_eval_f_ar_direct.json")

if __name__ == "__main__":
    main()
