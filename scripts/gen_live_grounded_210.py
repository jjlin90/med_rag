#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Live 复测 · 第 1 步：用【线上生成链路】重新生成 210 题答案。

与离线 regen_answers_grounded.py 的区别（关键）：
  - 离线版：脚本里手写 STRICT_SYS/STRICT_USR 常量。
  - 本脚本：调用 LLMGenerator.generate_with_context()，内部走
    _build_system_prompt() / _build_user_prompt()，
    即【线上实际部署的 prompt 路径】，由 LLM_GROUNDING 开关控制。
    → 验证的是"线上部署是否真的产生预期效果"，而非某个脚本常量。

变量隔离设计：
  - 复用已有 contexts（不重新检索），使"prompt 路径"成为唯一变量，
    避免检索波动混入 ΔF 归因。

反自评偏好（关键）：
  - 生成器 glm-4.5-air ≠ 裁判 glm-4.7，避免 judge 给自己生成的答案打高分。

用法:
  python scripts/gen_live_grounded_210.py --n 210 --workers 4
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.config.settings import Config
from src.online_service.llm_generator import LLMGenerator

GEN_MODEL = "glm-4.5-air"          # 生成器（≠ 裁判 glm-4.7，避免自评偏好）
ORIG = ROOT / "data/test_query/eval_answers_210.json"
OUT = ROOT / "data/test_query/eval_answers_210_live.json"

_TLS = {}


def make_gen():
    """每线程独立 LLMGenerator 实例，避免并发共享 client。"""
    import threading
    tid = threading.get_ident()
    if tid not in _TLS:
        cfg = Config()
        cfg.LLM_MODEL_NAME = GEN_MODEL
        cfg.LLM_GROUNDING = True          # 线上严格 grounding prompt
        cfg.LLM_TEMPERATURE = 0.2         # 与线上一致
        _TLS[tid] = LLMGenerator(cfg)
    return _TLS[tid]


def gen_one(item, idx):
    """生成单条答案，失败重试。返回 (idx, answer, model_used)。"""
    q = (item.get("question") or "").strip()
    ctxs = item.get("contexts") or []
    context = "\n\n".join(str(c) for c in ctxs).strip()
    if not q:
        return idx, None, None

    for attempt in range(1, 4):
        try:
            g = make_gen()
            ans = g.generate_with_context(query=q, context=context)
            if ans and ans.strip():
                return idx, ans.strip(), GEN_MODEL
            return idx, None, None       # 空内容不重试（模型 bug 特征）
        except Exception as e:
            msg = f"{type(e).__name__} {str(e)[:60]}"
            if attempt == 3:
                return idx, None, msg
            time.sleep(2 ** attempt)
    return idx, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=210)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    orig = json.load(open(ORIG, encoding="utf-8"))
    items = orig[:args.n]

    # 断点续跑：读取已有结果
    done = {}
    if OUT.exists():
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            for r in prev:
                if (r.get("answer") or "").strip():
                    done[r.get("question", "")] = r
            print(f"断点续跑：已完成 {len(done)} 条")
        except Exception as e:
            print(f"已有结果读取失败，重新生成: {e}")

    todo = [(i, it) for i, it in enumerate(items)
            if (it.get("question") or "").strip() not in done]
    print(f"生成器={GEN_MODEL}（裁判=glm-4.7，不同模型）| 待生成 {len(todo)} / {len(items)}")
    sys.stdout.flush()

    results = [None] * len(items)
    for i, it in enumerate(items):
        key = (it.get("question") or "").strip()
        if key in done:
            results[i] = done[key]

    t0 = time.time()
    ok = fail = 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(gen_one, it, i): i for i, it in todo}
            for n, f in enumerate(as_completed(futs), 1):
                idx, ans, used = f.result()
                it = items[idx]
                rec = {
                    "question": it.get("question", ""),
                    "answer": ans or "",
                    "contexts": it.get("contexts", []),
                    "ground_truth": it.get("ground_truth", ""),
                    "_gen_model": used,
                    "_grounding": True,
                }
                if ans:
                    ok += 1
                else:
                    fail += 1
                    rec["_err"] = str(used)
                results[idx] = rec

                if n % 10 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"[{n}/{len(todo)}] ok={ok} fail={fail} "
                          f"({n/el*60:.1f}/min, {el:.0f}s)", flush=True)

                # 每 20 条落盘一次，防止中断丢失
                if n % 20 == 0:
                    json.dump(results, open(OUT, "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)

    json.dump(results, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    empty = sum(1 for r in results if r and not (r.get("answer") or "").strip())
    print("=" * 50)
    print(f"完成 {len(results)} 条 | 成功 {len(results)-empty} | 空/失败 {empty} "
          f"| 用时 {time.time()-t0:.0f}s")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()
