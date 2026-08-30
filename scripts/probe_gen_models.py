"""探测当前哪些生成模型仍有可用额度。

直接复用 Config 的 API key / base_url，对每个候选模型发一条极小调用，
捕获真实异常（402 额度耗尽 / 429 限流 / 404 模型不存在 / 其他），
打印成表，便于挑选还能用于 grounding 重生成的模型。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import Config  # noqa: E402
from openai import OpenAI  # noqa: E402

cfg = Config()
client = OpenAI(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_BASE_URL)

# 候选生成模型（含之前用过、以及 TokenHub 上可能还有的）
CANDIDATES = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v4-pro-202606",
    "glm-5.1",
    "glm-5.2",
    "glm-5.3-flash",
    "glm-5",
    "hy3",
    "minimax-m3",
    "kimi-k3",
    "qwen3.5-plus",
    "qwen-turbo",
    "qwen-plus",
    "qwen-max",
]

print(f"{'model':<24} {'status':<8} detail")
print("-" * 72)
for m in CANDIDATES:
    try:
        r = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "请只回复两个字：OK"}],
            max_tokens=8,
        )
        ans = (r.choices[0].message.content or "").strip()
        print(f"{m:<24} {'OK':<8} {ans!r}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:70]
        print(f"{m:<24} {'FAIL':<8} {msg}")
