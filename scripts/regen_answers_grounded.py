"""
用「严格 grounding prompt」重新生成全部答案（保留原检索上下文）。
- 输入：eval_answers_210.json（含 question/answer/contexts/ground_truth）
- 输出：eval_answers_210_grounded.json（answer 换成严格 grounding 版本，其余字段原样保留）
- 生成模型在 --models 间 round-robin，自动跳过已完成的题目（断点续跑）
- 某模型 402 耗尽时自动换下一个可用模型
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data/test_query" / "eval_answers_210.json"

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


def gen_one(gen_models, cfg, q, ctxs):
    """依次尝试 gen_models，返回首个成功生成的答案。"""
    from src.online_service.llm_generator import LLMGenerator
    last_err = None
    for m in gen_models:
        try:
            cfg.LLM_MODEL_NAME = m
            g = LLMGenerator(cfg)
            if not g.client:
                continue
            ans = g.generate(STRICT_USR.format(context="\n\n".join(ctxs), query=q),
                             system_prompt=STRICT_SYS, temperature=1)
            if ans:
                return ans, m
        except Exception as e:
            last_err = e
    if last_err:
        print(f"  [warn] 全部生成模型失败: {str(last_err)[:60]}")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/test_query" / "eval_answers_210_grounded.json"))
    ap.add_argument("--models", nargs="*",
                    default=["deepseek-v4-pro-202606", "glm-5"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="跳过已生成的题目")
    args = ap.parse_args()

    from src.config.settings import Config
    cfg = Config()
    cfg.LLM_TEMPERATURE = 1  # 兼容只接受 temperature=1 的模型（glm-5.3-flash/kimi-k3 等）

    src = json.loads(SRC.read_text(encoding="utf-8"))
    print(f"源答案数: {len(src)}")

    out_path = Path(args.out)
    done = {}
    if args.resume and out_path.exists():
        for a in json.loads(out_path.read_text(encoding="utf-8")):
            done[a["question"]] = a
        print(f"续跑：已存在 {len(done)} 条")

    results = list(done.values())
    todo = [a for a in src if a["question"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"待生成: {len(todo)} | 生成模型: {args.models}")

    n_ok = 0
    t0 = time.time()
    for i, a in enumerate(todo, 1):
        q = a["question"]
        ctxs = a.get("contexts", [])
        new_ans, used = gen_one(args.models, cfg, q, ctxs)
        if new_ans:
            rec = dict(a)
            rec["answer"] = new_ans
            rec["_gen_model"] = used
            results.append(rec)
            n_ok += 1
        else:
            # 保留原答案，标记失败
            rec = dict(a)
            rec["_gen_model"] = None
            rec["_gen_failed"] = True
            results.append(rec)
        # 每 10 条落盘一次，防中断丢数据
        if i % 10 == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            print(f"  [{i}/{len(todo)}] ok={n_ok} 耗时 {time.time()-t0:.0f}s", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n完成：成功 {n_ok}/{len(todo)}，结果 -> {out_path}")


if __name__ == "__main__":
    main()
