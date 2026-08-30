"""候选 judge 模型筛选脚本。

用途：在正式跑大规模 Ragas 评估前，先验证候选 LLM 能否稳定输出 Ragas
要求的结构化结果（Pydantic / JSON 解析）。踩过的坑：如果模型不支持结构化
输出，Ragas 会把解析失败兜底成 0 分，几百次调用全白烧、且结果看起来
"像真的低分"，极难察觉。

设计要点：
- 单进程内换 config.LLM_MODEL_NAME 复用同一个 BGEEmbeddingProvider，
  避免每个模型都重新加载 BGE-M3（每次约 20s）。
- 每个模型只用 1 条样本（4 次 judge 调用）验证，token 开销约 1 万/模型。
- 判定标准：engine 必须是 ragas，且四项指标不能"全为 0"（全 0 = 解析失败
  或鉴权/额度失败，不是真实的低分）。

用法：
    python scripts/test_judge_models.py
    python scripts/test_judge_models.py --models deepseek-v4-pro glm-5.3-flash
    python scripts/test_judge_models.py --n 2     # 每个模型用 2 条样本（更稳但更费）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import Config  # noqa: E402
from src.offline_pipeline.embedding_provider import BGEEmbeddingProvider  # noqa: E402
from src.online_service.rag_evaluator import RAGEvaluator  # noqa: E402

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# 默认候选：TokenHub 上可用的强文本 LLM（多厂商，便于后续做多裁判交叉验证）
DEFAULT_CANDIDATES = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5.3-flash",
    "glm-5.2",
    "kimi-k3",
    "qwen3.5-plus",
    "minimax-m3",
    "hy3",
]


def build_items(n: int):
    """从已缓存的答案中取前 n 条，拼上 ground_truth 组成 Ragas 输入。

    复用答案缓存是关键：生成阶段已跑完（210/210），筛选模型只需验证
    "打分"链路，不必重跑检索与生成。
    """
    ans_path = ROOT / "data/test_query/eval_answers_210.json"
    set_path = ROOT / "data/test_query/eval_set_210.json"
    answers = json.load(open(ans_path, encoding="utf-8"))
    test_set = json.load(open(set_path, encoding="utf-8"))
    gt_map = {x["question"]: x.get("reference_answer", "") for x in test_set}

    items = []
    for a in answers[:n]:
        items.append({
            "question": a["question"],
            "answer": a.get("answer", ""),
            "contexts": a.get("contexts", []),
            "ground_truth": gt_map.get(a["question"], ""),
        })
    return items


def main():
    ap = argparse.ArgumentParser(description="筛选可用于 Ragas judge 的模型")
    ap.add_argument("--models", nargs="*", default=None,
                    help=f"候选模型列表，默认: {DEFAULT_CANDIDATES}")
    ap.add_argument("--n", type=int, default=1, help="每个模型验证用样本数（默认 1）")
    ap.add_argument("--out", default=str(ROOT / "data/test_query/_judge_probe.json"))
    args = ap.parse_args()

    models = args.models or DEFAULT_CANDIDATES
    items = build_items(args.n)

    config = Config()
    provider = BGEEmbeddingProvider(config)
    evaluator = RAGEvaluator(config, embedding_provider=provider)

    print(f"验证样本数: {len(items)} 条/模型；候选模型: {len(models)} 个\n")
    print(f"{'模型':<24} {'engine':<22} {'判定':<8} 四项分数")
    print("-" * 92)

    results = []
    for model in models:
        # 直接改 config 上的模型名即可：_build_llm 每次评估时重建 LLM
        config.LLM_MODEL_NAME = model
        t0 = time.time()
        # 先按默认 temperature=0.0 试；若该模型只接受 temperature=1
        # （TokenHub 上部分 deepseek 系列会报 400 "only 1 is allowed"），
        # 自动用 1 重试一次，避免把可用模型误判为不可用。
        last_err = None
        for temp in ("0.0", "1"):
            os.environ["LLM_JUDGE_TEMPERATURE"] = temp
            try:
                res = evaluator.evaluate_dataset(items, show_progress=False)
                engine = res.get("engine", "?")
                avg = res.get("average", {})
                scores = [round(avg.get(m, 0), 3) for m in METRICS]
                # 全 0 = 解析失败/鉴权失败，不是真实低分
                ok = engine == "ragas" and any(v > 0 for v in scores)
                if ok or temp == "1":
                    verdict = "可用" if ok else "不可用"
                    print(f"{model:<24} {engine:<22} {verdict:<8} {scores}"
                          f"  temp={temp}  ({time.time()-t0:.0f}s)")
                    results.append({"model": model, "engine": engine, "ok": ok,
                                    "temperature": temp,
                                    "scores": dict(zip(METRICS, scores)),
                                    "seconds": round(time.time() - t0, 1)})
                    break
            except Exception as e:
                last_err = str(e).replace("\n", " ")[:60]
        else:
            print(f"{model:<24} {'EXCEPTION':<22} {'不可用':<8} {last_err}")
            results.append({"model": model, "engine": "exception", "ok": False,
                            "error": last_err})
    os.environ.pop("LLM_JUDGE_TEMPERATURE", None)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    usable = [r["model"] for r in results if r.get("ok")]
    print("\n" + "=" * 92)
    print(f"可用 judge 模型（{len(usable)}/{len(models)}）: {usable}")
    print(f"明细已存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
