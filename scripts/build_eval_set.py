# -*- coding: utf-8 -*-
"""
构造 Ragas 评估用的测试集（question + ground_truth）。

设计要点（决定评估结果是否可信，别改坏）：

1. **ground_truth 必须由源文档生成，而不是由检索到的 context 生成。**
   如果用检索结果构造 ground_truth，context_recall / context_precision 会自证为 1.0
   ——衡量的是"检索结果和它自己像不像"，毫无意义。本脚本从 data/clean_md 抽样原始
   文档片段交给 LLM 出题，ground_truth 独立于任何检索行为，因此 context_recall
   才真正衡量"有没有把含答案的那一段召回来"。

2. **问题必须像真实用户提问**（口语化、带场景），不能用文档标题当问题，
   否则测的是字面匹配能力，不是语义检索能力。

3. **答案要具体**（含数字/专名/剂量），抽象的答案无法判定召回是否正确。

4. 固定随机种子 → 可复现；断点续跑 → 中断不丢已完成部分。

用法：
  python scripts/build_eval_set.py --limit 6           # 小样本试跑，验证 prompt 效果
  python scripts/build_eval_set.py                      # 生成 200 条（默认）
  python scripts/build_eval_set.py --n 300 --workers 8
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import Config
from src.online_service.llm_generator import LLMGenerator

CLEAN_MD_DIR = ROOT / "data" / "clean_md"
DEFAULT_OUT = ROOT / "data" / "test_query" / "test_qa_200.json"

# 出题 prompt：约束全部指向"可评判性"
GEN_PROMPT = """你是医学考试命题专家。下面是一段来自《默沙东诊疗手册（大众版）》的科普内容。

【文档片段】
{text}

请基于**且仅基于**上面这段内容，出 {n} 道问答题，要求：

1. 问题要像真实患者或家属的提问：口语化、自然、可带场景（如"我妈70岁了，血压160要不要吃药？"），
   **不要直接抄文档标题**，不要用"本文介绍了什么"这类问法。
2. 答案必须完全来自上面这段文本，**不得添加你自己的医学知识**；文本里没有的信息就不要写。
3. 答案要**具体**：优先包含数字、病名、药名、时间、剂量等可核验的关键信息，
   避免"建议咨询医生"这类无法评判的套话。
4. 答案控制在 60-150 字，一段话，不要分点。
5. {n} 道题之间不要重复。

严格按以下 JSON 数组格式输出，不要有任何额外文字、不要加 markdown 代码块标记：
[
  {{"question": "问题1", "reference_answer": "答案1"}},
  {{"question": "问题2", "reference_answer": "答案2"}}
]"""


def sample_doc_fragment(path: Path, rng: random.Random, max_chars: int = 1600) -> str:
    """从一篇文档里取一段有实质内容的连续文本（非标题、非清单）。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    # 去掉 markdown 标题与过短行，保留正文段落
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if len(s) < 30:  # 过滤短语行（多为目录/标注）
            continue
        lines.append(s)
    body = "\n".join(lines)
    if len(body) <= max_chars:
        return body
    # 随机起点，避免每篇都取开头导致问题同质化
    start = rng.randint(0, max(0, len(body) - max_chars))
    frag = body[start:start + max_chars]
    # 尽量从句号后开始，避免半截句子
    dot = frag.find("。")
    if 0 < dot < 120:
        frag = frag[dot + 1:]
    return frag.strip()


def parse_qa_pairs(raw: str):
    """容忍 LLM 返回 markdown 代码块 / 前后废话 / 尾随逗号。"""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end == -1:
        return []
    s = s[start:end + 1]
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        # 兜底：去掉尾随逗号再试
        s2 = re.sub(r",\s*([\]}])", r"\1", s)
        try:
            data = json.loads(s2)
        except json.JSONDecodeError:
            return []
    out = []
    for it in data:
        if isinstance(it, dict):
            q = str(it.get("question", "")).strip()
            a = str(it.get("reference_answer", it.get("answer", ""))).strip()
            if q and a:
                out.append({"question": q, "reference_answer": a})
    return out


def gen_one(llm: LLMGenerator, cfg: Config, frag: str, n: int, retries: int = 2):
    """对单个文档片段调 LLM 出题，失败重试。"""
    prompt = GEN_PROMPT.format(text=frag, n=n)
    for attempt in range(retries + 1):
        try:
            resp = llm.client.chat.completions.create(
                model=cfg.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,   # 出题需要多样性
                max_tokens=1200,
            )
            pairs = parse_qa_pairs(resp.choices[0].message.content or "")
            if pairs:
                return pairs
        except Exception as e:
            if attempt == retries:
                print(f"    [WARN] 生成失败（{type(e).__name__}）: {str(e)[:80]}")
            else:
                time.sleep(1.5)
    return []


def main():
    ap = argparse.ArgumentParser(description="构造 Ragas 评估测试集（ground_truth 独立于检索）")
    ap.add_argument("--n", type=int, default=200, help="目标样本数（默认 200）")
    ap.add_argument("--per-doc", type=int, default=2, help="每篇文档出几道题（默认 2）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇文档（试跑用）")
    ap.add_argument("--workers", type=int, default=6, help="并发线程数")
    ap.add_argument("--seed", type=int, default=20260829, help="随机种子（保证可复现）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出路径")
    args = ap.parse_args()

    if not CLEAN_MD_DIR.exists():
        print(f"[FAIL] 知识库目录不存在: {CLEAN_MD_DIR}")
        return 1

    docs = sorted(CLEAN_MD_DIR.glob("*.md"))
    if not docs:
        print(f"[FAIL] 未找到任何 .md 文档: {CLEAN_MD_DIR}")
        return 1

    rng = random.Random(args.seed)
    need_docs = -(-args.n // args.per_doc)  # 向上取整
    picked = rng.sample(docs, min(need_docs, len(docs)))
    if args.limit:
        picked = picked[:args.limit]

    print(f"知识库文档数: {len(docs)}")
    print(f"抽样文档数  : {len(picked)}   每篇出题: {args.per_doc}   目标: {args.n} 条")
    print(f"随机种子    : {args.seed}（可复现）   并发: {args.workers}\n")

    cfg = Config()
    llm = LLMGenerator(cfg)

    # 断点续跑：已存在则跳过已完成的文档
    out_path = Path(args.out).resolve()   # 必须绝对化，否则末尾 relative_to(ROOT) 会抛 ValueError
    existing = []
    done_titles = set()
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            done_titles = {e.get("_source", "") for e in existing}
            print(f"[续跑] 已有 {len(existing)} 条，跳过 {len(done_titles)} 篇已完成文档")
        except Exception:
            existing = []

    results = list(existing)
    lock_path = out_path.with_suffix(".partial.json")

    def work(doc: Path):
        frag = sample_doc_fragment(doc, rng)
        if len(frag) < 200:
            return doc.stem, []
        pairs = gen_one(llm, cfg, frag, args.per_doc)
        for p in pairs:
            p["_source"] = doc.stem   # 溯源：便于人工抽检答案是否真的来自该文档
        return doc.stem, pairs

    todo = [d for d in picked if d.stem not in done_titles]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, d): d for d in todo}
        for fu in as_completed(futs):
            doc = futs[fu]
            done += 1
            try:
                title, pairs = fu.result()
            except Exception as e:
                title, pairs = doc.stem, []
                print(f"  [ERR] {doc.stem}: {e}")
            results.extend(pairs)
            print(f"  [{done}/{len(todo)}] {title[:28]:<30} +{len(pairs)} 条  累计 {len(results)}")
            # 边跑边存，中断不丢
            if done % 5 == 0:
                lock_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保留 _source 溯源字段：① 支持断点续跑跳过已完成文档
    # ② 面试时可以证明"标准答案来自知识库第 X 篇文档"，而不是拍脑袋编的
    final = [{"question": r["question"],
              "reference_answer": r["reference_answer"],
              "_source": r.get("_source", "")}
             for r in results if r.get("question") and r.get("reference_answer")]

    # 去重（同题不同文档可能撞车）
    seen, uniq = set(), []
    for it in final:
        key = it["question"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    uniq = uniq[:args.n] if len(uniq) > args.n else uniq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(uniq, ensure_ascii=False, indent=2), encoding="utf-8")
    if lock_path.exists():
        lock_path.unlink()

    print(f"\n[完成] 共 {len(uniq)} 条，耗时 {time.time() - t0:.0f}s")
    print(f"输出: {out_path}")
    print("\n样例预览（前 3 条，请人工确认答案确实来自知识库）:")
    for it in uniq[:3]:
        print(f"  Q: {it['question'][:60]}")
        print(f"  A: {it['reference_answer'][:80]}...\n")
    print("下一步: python scripts/evaluate_rag.py --test-set "
          f"{out_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
