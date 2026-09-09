"""
全量 210 题 F+AR 配对复评（独立单裁判，扁平并发 + 重试退避）
==============================================================
- 单一固定裁判同时对「原答案」和「grounding 答案」打分（配对比较，裁判偏差两次抵消）
- 扁平 ThreadPoolExecutor（max_workers 可控），每题 4 次顺序调用，避免嵌套死锁
- 每次调用失败自动指数退避重试（应对 429/timeout）
用法：python scripts/eval_f_ar_single_judge_210.py [--n N] [--workers W]
"""
import argparse, json, os, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import numpy as np
from openai import OpenAI

ZHIPU_BASE = "https://open.bigmodel.cn/api/paas/v4"
JUDGE = os.getenv("EVAL_JUDGE", "glm-4.5-air")
# thinking 类模型需 >=200，否则推理 token 吃满配额导致 content 为空
MAX_TOKENS = int(os.getenv("EVAL_MAX_TOKENS", "1024"))
ORIG = PROJECT_ROOT / "data/test_query/eval_answers_210.json"
GRND_D = PROJECT_ROOT / "data/test_query/eval_answers_210_grounded.json"
OUT = PROJECT_ROOT / "data/test_query/_eval_210_single_judge.json"

SYS = "你是严格的 RAG 评估员。只输出 JSON：{\"score\": 数字}，不要输出其他内容。"

# 忠实度：核心是"有没有编造"。明确：如实说"上下文没有"= 最高忠实度（1.0），
# 否则拒答会被误判为不忠实，从而系统性低估 grounding 的收益。
F_P = (
    "【任务】评估答案的「忠实度」：答案中的实质信息是否都能在【上下文】中找到依据？\n"
    "【评分标准】\n"
    "1.0 = 答案完全基于上下文，没有编造；\n"
    "      或者答案明确说明上下文未提供相关信息、因而无法回答，且未编造任何内容"
    "（如实拒答 = 最高忠实度）。\n"
    "0.5 = 大部分有依据，但混入少量上下文中没有的信息。\n"
    "0.0 = 答案包含上下文中不存在的实质性事实（数值、剂量、病因、疗法等编造内容）。\n"
    "【注意】不要因为答案简短或没有回答出问题而扣分，那属于相关性问题，不影响忠实度。\n"
    "【上下文】\n{ctx}\n"
    "【答案】\n{ans}\n"
    "【输出】只输出 JSON：{\"score\": 数字}"
)

# 相关性：核心是"有没有回应问题"。拒答在这里应得低分。
AR_P = (
    "【任务】评估答案的「相关性」：答案是否回应了用户的问题？\n"
    "【评分标准】\n"
    "1.0 = 直接、完整地回答了用户的问题。\n"
    "0.5 = 部分回应，或只回答了其中一部分。\n"
    "0.0 = 完全没有回应问题（包括仅回复“无法回答”“资料未提及”这类拒答表述）。\n"
    "【注意】拒答虽然忠实度高，但相关性应判低分；两者要分开判断，互不干扰。\n"
    "【问题】\n{q}\n"
    "【答案】\n{ans}\n"
    "【输出】只输出 JSON：{\"score\": 数字}"
)

def build(template, **kw):
    # brace-safe：答案/问题文本里的 { } 转义，避免 .format 异常
    s = template
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v).replace("{", "[").replace("}", "]"))
    return s

def score(c, prompt, max_retry=3):
    """打分。注意：thinking 类模型（如 glm-4.5-air）会先输出隐藏推理 token，
    max_tokens 过小会导致 content 为空（finish_reason=length），故需 >=200。"""
    delay = 2.0
    for attempt in range(1, max_retry + 1):
        try:
            r = c.chat.completions.create(
                model=JUDGE,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                temperature=0.01, max_tokens=MAX_TOKENS, timeout=45)
            txt = r.choices[0].message.content or ""
            if not txt.strip():
                # 空内容多为额度耗尽或 max_tokens 被推理 token 吃满，显式报错
                raise RuntimeError(
                    f"empty content (finish={r.choices[0].finish_reason}, "
                    f"max_tokens={MAX_TOKENS})")
            m = re.search(r'"score"\s*:\s*(\d+\.?\d*)', txt)
            return min(1.0, max(0.0, float(m.group(1)))) if m else 0.0
        except Exception as e:
            # 1113 是账户/资源包不可用，不是瞬时限流；重试只会制造大量
            # 无效请求。返回 None，由整题失败逻辑显式剔除。
            if "1113" in str(e) or "无可用资源包" in str(e) or "余额不足" in str(e):
                print(f"    [FATAL] judge resource unavailable: {str(e)[:120]}", flush=True)
                return None
            if attempt == max_retry:
                # 不再静默返回 0.0：打印失败，避免整批结果无声失效
                print(f"    [FAIL] {type(e).__name__}: {str(e)[:90]}", flush=True)
                return None
            time.sleep(delay)
            delay *= 2.0
    return None

def eval_one(c, q, oa, ga, ctx):
    ct = "\n".join(ctx[:3]) if ctx else "(无)"
    fo = score(c, build(F_P, ctx=ct, ans=oa))
    aro = score(c, build(AR_P, q=q, ans=oa))
    fg = score(c, build(F_P, ctx=ct, ans=ga)) if ga else 0.0
    arg = score(c, build(AR_P, q=q, ans=ga)) if ga else 0.0
    # 任一打分失败（None）→ 整题作废，不参与均值，避免静默 0 污染结论
    if any(v is None for v in (fo, aro, fg, arg)):
        return {"q": q, "failed": True}
    return {"q": q, "oF": round(fo, 3), "oAR": round(aro, 3),
            "gF": round(fg, 3), "gAR": round(arg, 3),
            "dF": round(fg - fo, 3), "dAR": round(arg - aro, 3),
            "failed": False}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    # live 复测：指定「线上链路生成的答案」与独立输出，避免覆盖离线版结果
    p.add_argument("--grnd", type=str, default=str(GRND_D),
                   help="对照答案文件（默认离线 grounded；live 复测传 eval_answers_210_live.json）")
    p.add_argument("--out", type=str, default=str(OUT))
    p.add_argument(
        "--allow-self-judge", action="store_true",
        help="允许裁判与生成模型相同（仅调试；正式报告不应开启）",
    )
    a = p.parse_args()

    grnd_path = Path(a.grnd)
    out_path = Path(a.out)

    orig = json.loads(ORIG.read_text(encoding="utf-8"))
    om = {it["question"]: it for it in orig}
    gm = {}
    grounded_models = set()
    if grnd_path.exists():
        for g in json.loads(grnd_path.read_text(encoding="utf-8")):
            gm[g["question"]] = g.get("answer") or g.get("grounded_answer", "")
            model = (g.get("_gen_model") or "").strip()
            if model:
                grounded_models.add(model)

    # 2026-08-30 的 live 结果曾误用 glm-4.5-air 同时生成和裁判，造成自评
    # 偏好风险。正式 A/B 默认 fail-fast，不能只靠注释约定不同模型。
    if JUDGE in grounded_models and not a.allow_self_judge:
        models = ", ".join(sorted(grounded_models))
        p.error(
            f"裁判 {JUDGE!r} 与 grounded 生成模型重合（{models}）。"
            "请更换 EVAL_JUDGE；仅调试时才使用 --allow-self-judge。"
        )

    items = [(q, om[q].get("answer", ""), gm.get(q, ""), om[q].get("contexts", [])) for q in om]
    if a.n:
        items = items[:a.n]

    print(f"F+AR 单裁判配对 | {JUDGE} | {len(items)} 题 | workers={a.workers}", flush=True)
    c = OpenAI(base_url=ZHIPU_BASE, api_key=os.getenv("LLM_API_KEY", ""))
    # 在提交 210×4 个并发任务前做一次极小探测。资源包未绑定时立即退出，
    # 避免每个 worker 都走完整重试链。
    try:
        c.chat.completions.create(
            model=JUDGE,
            messages=[{"role": "user", "content": "只回复数字1"}],
            temperature=0.01, max_tokens=8, timeout=20,
        )
    except Exception as e:
        if "1113" in str(e) or "无可用资源包" in str(e) or "余额不足" in str(e):
            p.error(f"裁判 {JUDGE!r} 的余额或资源包不可用：{str(e)[:160]}")
        raise
    t0 = time.time()
    rows = [None] * len(items)
    lock = Lock()
    done = [0]

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(eval_one, c, q, oa, ga, ctx): idx
                for idx, (q, oa, ga, ctx) in enumerate(items)}
        for fut in as_completed(futs):
            idx = futs[fut]
            r = fut.result()
            rows[idx] = r
            done[0] += 1
            with lock:
                elapsed = time.time() - t0
                rate = done[0] / elapsed * 60 if elapsed > 0 else 0
                tag = (f"ΔF={r['dF']:+.2f} ΔAR={r['dAR']:+.2f}"
                       if not r.get("failed") else "SKIP(打分失败)")
                print(f"[{done[0]}/{len(items)}] {tag} "
                      f"({rate:.1f}q/min, {elapsed:.0f}s)", flush=True)

    valid = [r for r in rows if r and not r.get("failed")]
    n_fail = len(rows) - len(valid)
    if not valid:
        print(f"\n!! 全部 {len(rows)} 题打分失败，无有效结果（多为额度耗尽）。", flush=True)
        return
    print(f"\n有效 {len(valid)} 题 | 失败作废 {n_fail} 题", flush=True)

    dF = np.array([r["dF"] for r in valid]); dAR = np.array([r["dAR"] for r in valid])
    oF = np.array([r["oF"] for r in valid]); oAR = np.array([r["oAR"] for r in valid])
    gF = np.array([r["gF"] for r in valid]); gAR = np.array([r["gAR"] for r in valid])
    cp, cr, w = 0.636, 0.559, 0.25
    oc = w * (np.mean(oF) + np.mean(oAR) + cp + cr)
    nc = w * (np.mean(gF) + np.mean(gAR) + cp + cr)

    print(f"\n{'='*55}", flush=True)
    print(f"汇总 ({len(valid)}题有效 / {len(rows)}题, {time.time()-t0:.0f}s)", flush=True)
    print(f"{'='*55}", flush=True)
    for l, ov, nv, dv in [("F", np.mean(oF), np.mean(gF), np.mean(dF)),
                          ("AR", np.mean(oAR), np.mean(gAR), np.mean(dAR)),
                          ("CP", cp, cp, 0), ("CR", cr, cr, 0)]:
        print(f"{l:<6} {ov:>7.3f} {nv:>7.3f} {dv:>+7.3f}", flush=True)
    print(f"{'-'*35}", flush=True)
    print(f"综合   {oc:>7.3f} {nc:>7.3f} {nc-oc:>+7.3f}", flush=True)

    out = {"judge": JUDGE, "method": "custom_direct_llm_judge_flat_concurrent",
           "metric_compatibility": "not directly comparable to Ragas scores",
           "n": len(valid), "n_failed": n_fail, "workers": a.workers,
           "max_tokens": MAX_TOKENS,
           "orig_file": str(ORIG), "grnd_file": str(grnd_path),
           "summary": {"orig": {"F": round(float(np.mean(oF)), 4), "AR": round(float(np.mean(oAR)), 4)},
                       "grnd": {"F": round(float(np.mean(gF)), 4), "AR": round(float(np.mean(gAR)), 4)},
                       "delta": {"F": round(float(np.mean(dF)), 4), "AR": round(float(np.mean(dAR)), 4)},
                       "composite_old": round(oc, 4), "composite_new": round(nc, 4)},
           "per_question": valid}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存: {out_path}", flush=True)

if __name__ == "__main__":
    main()
