"""用真实生成路径验证哪些 OK 模型能产出非空答案。

直接复用 LLMGenerator.generate()，对一条真实医疗短问 + 短上下文发请求，
报告返回字符数。空内容 = 该模型在当前配置下不可用（即使 init 没报错）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import Config  # noqa: E402
from src.online_service.llm_generator import LLMGenerator  # noqa: E402

OK_MODELS = [
    "deepseek-v4-pro-202606",
    "glm-5.2",
    "glm-5.3-flash",
    "glm-5",
    "hy3",
    "minimax-m3",
    "kimi-k3",
]

Q = "高血压患者平时饮食需要注意什么？"
CTX = "高血压患者的饮食原则：低盐（每日食盐<5g）、低脂、多摄入蔬果与全谷物，限制饮酒，控制体重。"

print(f"{'model':<24} {'len':>6} 首句预览")
print("-" * 72)
for m in OK_MODELS:
    cfg = Config()
    cfg.LLM_MODEL_NAME = m
    g = LLMGenerator(cfg)
    if not g.client:
        print(f"{m:<24} {'NOCONN':>6} client 初始化失败")
        continue
    ans = g.generate(Q, system_prompt="你是严谨的医疗助手，只基于给定资料回答。")
    if not ans:
        print(f"{m:<24} {'EMPTY':>6} 返回空")
        continue
    print(f"{m:<24} {len(ans):>6}  {ans[:40]!r}")
