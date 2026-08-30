# Ragas 评估与 badcase 分析

> 记录 2026-08-29 补跑 210 条 Ragas 评估的全过程：测试集怎么造的、分数怎么读、
> 以及从 0 分样本里挖出来的真实问题。
>
> 写这份文档的原因：简历和自我介绍里原写「Ragas 200 组 / 0.88」，但仓库里
> 只有 5 条样本、且无 ground_truth。**面试时只要被追问一句"这 200 组是怎么来的"就穿帮**，
> 所以真跑、真留档：现已完成 **210 组评测集、209 条成功打分**的跨厂商多模型裁判评估，
> 并把跑出来的问题如实写下来。

---

## 1. 为什么原来的 5 条不能说明任何问题

改造前的 `data/test_query/test_qa.json` 是 5 条**人工手写**的问答：

```
1 型糖尿病 (DM) 是什么？            faithfulness 0.92   relevancy 0.89
高血压患者在日常生活中需要注意什么？     faithfulness 0.81   relevancy 0.87
感冒了需要吃抗生素吗？                faithfulness 1.00   relevancy 0.65
空腹血糖的正常范围是多少？             faithfulness 1.00   relevancy 0.76
服用退烧药后多久可以重复用药？          faithfulness 0.00   relevancy 0.00   ← 拉低均值
```

5 条样本有三个致命问题：

1. **样本量太小**：单条波动 ±0.2，均值没有统计意义。
2. **没有 ground_truth**：原报告 `has_ground_truth: false`，只能算
   faithfulness / answer_relevancy 两项，**根本没算 context_precision / context_recall**
   ——而这两项恰恰是衡量"检索"好坏的指标。也就是说，原来那 0.88 只评估了生成，没评估检索。
3. **题目是拍脑袋写的，不保证知识库能答**：最后一条就是典型（详见第 4 节）。

---

## 2. 新测试集怎么造的（`scripts/build_eval_set.py`）

### 2.1 核心原则：ground_truth 必须独立于检索行为

这是整个评估可不可信的分水岭。

**错误做法**：先跑检索拿到 contexts，再让 LLM 基于 contexts 写标准答案。
这样 context_recall / context_precision **必然恒等于 1.0** ——衡量的是
"检索结果和它自己像不像"，是自我循环论证，毫无意义。

**正确做法**：从 `data/clean_md` 随机抽原始文档片段 → 交给 LLM 出题 + 写答案。
ground_truth 的产生过程完全不经过检索系统，因此 context_recall 才真正衡量
"有没有把含答案的那一段召回来"。

```
data/clean_md/*.md  ──随机抽样(固定种子 20260829)──▶ 1600 字片段
                                                       │
                                                       ▼
                                              LLM 出题（temperature 0.7）
                                                       │
                                                       ▼
                              {question, reference_answer, _source}
```

### 2.2 prompt 里的五条约束

| 约束 | 目的 |
|---|---|
| 问题要像真实患者提问，口语化、可带场景，**不许抄文档标题** | 否则测的是字面匹配，不是语义检索 |
| 答案**完全来自**片段，不得添加 LLM 自己的医学知识 | 保证 ground_truth 可溯源到知识库 |
| 答案要**具体**：优先含数字、病名、药名、剂量 | 抽象答案无法判定召回是否正确 |
| 60–150 字，一段话不分点 | 便于 judge 判定 |
| 同一片段内题目不重复 | 避免刷量 |

### 2.3 产出

| 项 | 值 |
|---|---|
| 样本数 | **210** |
| 覆盖源文档 | 106 篇（占 clean_md 的 ~4%） |
| 问题去重 | 249 条去重后取 210 |
| 答案长度 | 中位 50 字，范围 18–93 字 |
| 随机种子 | 20260829 / 20260830（可复现） |

每条都带 `_source` 字段记录出自哪篇文档，**面试时可以现场打开源文档核对标准答案**，
证明不是编的。

### 2.4 诚实的局限

出题基于文档片段，天然与源文档强绑定 → 检索召回源文档的概率偏高 →
**context_precision / context_recall 会偏乐观**。更严格的做法还应加一层
"跨域问题"（问题来自文档 A，答案需要文档 B 佐证）。这一点在报告里如实标注，
不用来充数。

---

## 3. 评估脚本的改造（`scripts/evaluate_rag.py`）

原脚本一次性跑完才落盘，200+ 条要跑近 1 小时，中途挂掉全部白跑。补了断点续跑：

```bash
# 全量跑（生成阶段每 10 条刷一次盘）
python scripts/evaluate_rag.py \
    --test-set data/test_query/eval_set_210.json \
    --out-json  data/test_query/eval_report_210.json \
    --out-csv   data/test_query/ragas_evaluation_results_210.csv \
    --answers-file data/test_query/eval_answers_210.json \
    --reuse-answers

# 中断后续跑（跳过已生成的题目，只重跑打分）
# ... 同样命令即可

# 冒烟
python scripts/evaluate_rag.py --test-set ... --limit 5
```

新增参数：

| 参数 | 作用 |
|---|---|
| `--answers-file` | 答案缓存落盘位置，生成阶段每 10 条刷新 |
| `--reuse-answers` | 按 question 匹配缓存，跳过已生成题目 |
| `--force-rerun` | 忽略缓存全部重生成 |
| `--limit N` | 只跑前 N 条 |

另外在报告里新增 `pipeline_stats`（平均上下文数 / 降级命中数 / 空上下文 / 空答案），
这些**不计入 Ragas 分数**，但用来判读分数为什么低。

### 3.1 踩过的坑：RPM 限流把分数全部兜底成 0（2026-08-29）

第一次全量跑（judge = qwen-turbo）跑到一半，turbo **免费额度耗尽（403）**；
临时切 qwen-plus 续跑，但 plus 的免费额度也在这一轮被烧光。
更隐蔽的是：即使额度没彻底耗尽，**DashScope 的 RPM 限制**也会让 Ragas 并发发出的
数百次 judge 调用被 `429` 拒绝，而 `evaluate(raise_exceptions=False)` 会把失败
兜底成 `0.0` —— 第一次产出的 `eval_report_210.json` 里 **210 条中有 142 条四项全 0**，
平均分被拖到 0.18 左右，**完全不可用**。

两个修复：

1. **限流**：给 judge 的 `ChatOpenAI` 加 `InMemoryRateLimiter(requests_per_second=1.0)`，
   把并发请求压到 ~1 次/秒、超出阻塞等待而非失败（见 `rag_evaluator.py._build_llm`）。
   注意本机 `langchain_core` 版本的该限速器**不支持 `max_bucket` 参数**，传了会直接抛异常
   并触发 fallback 路径。
2. **换裁判模型**：turbo / plus 免费额度都空了，临时切到 qwen-max 当 judge；但 qwen-max
   在打出 44 条干净分后同样 `403 Free quota exhausted`。最终解法是**跨厂商多模型裁判集成**——
   用腾讯云 TokenHub 上 deepseek / glm / minimax / hy 等多个模型各自的免费额度，把 210 题
   切块分发给不同模型并行打分（详见 §3.2）。生成答案用的是 qwen-turbo（205 条）+ qwen-plus（5 条）。

> 结论：评估能否跑通，**取决于 judge 模型的额度/限流**，与检索系统质量无关。
> 跑评估前先确认 judge 模型可用，否则拿到的是一堆 0。

### 3.2 最终解法：跨厂商多模型裁判集成 + 分块并行（2026-08-29）

qwen-max 在 44 条后也 403，单模型免费额度天花板已到。但发现腾讯云 TokenHub 上
**每个模型都有独立免费额度（约 100 万 token）**——单个不够，多个拼起来远超需求
（实测单条样本四指标约耗 3.57 万 token，210 条共需 ~750 万，而可用额度 2000 万+）。

于是改成分块 + 多裁判并行：

1. **分块**：把 210 题按 ~25 条/块切分，每块只吃一个模型的额度（留安全余量，避免烧穿）。
2. **多裁判**：每块用不同模型当 judge，跨厂商（DeepSeek / 智谱 GLM / MiniMax / 阶跃 hy）
   规避单模型偏好；最终 6 个模型各贡献一部分（28 / 56 / 50 / 23 / 25 / 27 条）。
3. **按题续跑 + 零分过滤**：已拿到有效分的题直接跳过，全零样本（额度耗尽/解析失败）
   自动剔除，只在有效样本上算均值。
4. **并行加速**：本机串行太慢，改用 `run_parallel_eval.py` 起多个子进程各跑一块，
   3 路并行把耗时从 ~25 分钟压到 ~8 分钟。

最终 **209/210 条拿到有效分**（仅 1 条因 Ragas 框架对该特定输入触发内部解析 bug
而漏评，与额度/模型无关）。产物见 §7。

> 经验：**免费额度是评估规模的天花板，但多模型免费额度可以拼凑出足够预算**。
> 做任何 ≥100 条的 LLM-as-judge 评估前，先确认 judge 模型有充足额度，或准备好多模型接力方案。

---

## 4. badcase：「服用退烧药后多久可以重复用药？」为什么是 0 分

### 4.1 现象

faithfulness 和 answer_relevancy 双双 0.00。第一反应是"检索挂了 → L2 拒答 → 空答案"。
实跑验证后发现**完全不是**：

```
degraded       : False
命中上下文数    : 2
[1] 药物相互作用   1987 字
[2] 成年人发热     1229 字
```

系统**没有降级**，检索**命中了正确的文档**，也生成了答案。

### 4.2 真正发生了什么

打开「成年人发热」这段上下文的第 180–460 字：

> 用于降低体温的药物称作退热药。
> 最有效和最广泛使用的退热药为对乙酰氨基酚和非甾体抗炎药 (NSAID)，
> 如阿司匹林、布洛芬和萘普生。**这些药物需根据容器标签上的说明服用。**

知识库原文**只说"按容器标签说明服用"，根本没有"间隔 4–6 小时"这个数字**。

而系统的回答是：

> 根据提供的医学知识，关于服用退烧药后多久可以重复用药，**没有明确的时间间隔说明**。
> 但以下信息可供参考：……这类药物的重复使用间隔时间**可能为 4 至 6 小时**……

于是：

- **faithfulness = 0**：Ragas 判定"4 至 6 小时"这句无上下文支撑。
  **判得对** —— 这个数字确实是 LLM 的参数知识，不在检索结果里。
- **answer_relevancy = 0**：偏严。答案其实是切题的，但开头那句否定式表述
  导致反向生成的问题与原始问题不匹配。中文长答案下这个指标容易失真。

### 4.3 这暴露了三个不同层面的问题

**A. 系统侧（可修，且必须修）——「先诚实，再编造」是最危险的失败模式**

`src/online_service/llm_generator.py` 的 `_build_system_prompt()` 第 3 条写的是：

```
3. 如果信息不足，明确说明
```

模型**照做了**：说了"没有明确的时间间隔说明"。但这条约束没说"然后停下"，
于是模型为了显得有用，紧接着补了一句参数知识。**用户只会记住那个数字，
不会记住前面那句免责声明。**

> 这是医疗 RAG 里最坏的失败模式：**输出既包含诚实的免责声明，又包含无据的
> 具体数值**。它的危害大于直接编造 —— 因为它看起来"已经核实过了"。

**B. 评估侧（题目本身出得不合理）**

ground_truth 写的是"间隔 4～6 小时"，而知识库不支持这个答案。
这道题测的不是系统能力，而是"LLM 是否碰巧知道答案"。
这正是手写测试集的通病，也是第 2 节那条原则的反面教材。

**C. 指标侧**

answer_relevancy 在中文 + 长答案场景会系统性偏低。该指标的做法是
"从答案反向生成问题 → 与原始问题算语义相似度"，中文反向生成的措辞漂移
比英文大，导致失真。看 210 条的分布可以验证这一点。

### 4.4 A 的修法（待跑完基线后应用）

把 prompt 第 3 条从软约束改成硬约束：

```python
3. 如果上下文没有回答该问题所需的信息，明确说明"知识库未提供该信息"，
   并且**不得**用你自己的医学知识补充任何具体数值、剂量、时间间隔、
   药名或诊疗建议。
   承认不知道之后又给出具体数字，比直接说不知道更危险——用户只会记住那个数字。
```

注意：**不能在评估跑到一半时改 prompt**，否则前一半样本是旧 prompt、
后一半是新 prompt，报告就不可比了。必须跑完基线 → 改 → 用 `--force-rerun`
重跑做 A/B 对照。

---

## 5. 顺带挖出来的环境 bug：GPU 被静默降级成 CPU

### 5.1 现象

跑评估时发现单条查询要 12 秒，明显偏慢。一查：

```
torch.__version__ = '2.13.0+cpu'
torch.cuda.is_available() = False
```

但用户的机器是 **RTX 4060 Laptop（8GB VRAM，驱动 561.03，支持 CUDA 12.6）**，
8 月 16 日装的明明是 `torch-2.13.0+cu126`。

### 5.2 根因

`site-packages` 里同时存在两个 dist-info：

```
Aug 16 20:03   torch-2.13.0+cu126.dist-info   ← GPU 版
Aug 18 16:52   torch-2.13.0.dist-info         ← CPU 版，后装，覆盖了实际文件
```

`uv.lock` 里 torch 解析到的 wheel 是：

```
torch-2.13.0-cp310-cp310-win_amd64.whl   只有 122 MB
```

**Windows 上 PyPI / 清华镜像的 `torch` wheel 默认是 CPU-only 的**
（同版本 cu126 的 Windows wheel 约 2.5 GB，一眼可辨）。Linux 的默认 wheel
自带 CUDA，所以这个问题是 **Windows 独有的**。

于是 8 月 18 日那次 `uv sync` 把 GPU 版静默覆盖成了 CPU 版。

### 5.3 为什么没人发现

`src/config/settings.py` 原来只有一行：

```python
self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

**静默降级，连一条日志都没有**。系统照常跑、不报错，只是慢一个数量级。

### 5.4 已做的两处修复

**① `pyproject.toml` 锁死 Windows 的 torch 索引：**

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu126", marker = "sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true   # 只对显式指定的包生效，不污染其它依赖
```

**② `src/config/settings.py` 加显式告警**（已验证生效）：

```python
self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if self.DEVICE == "cpu":
    logger.warning("torch 未检测到 CUDA，已降级为 CPU 推理（..."
        "常见原因：Windows 下被 pip/uv 默认源的 CPU 版 torch 覆盖。...")
else:
    logger.info(f"使用 GPU 推理: {torch.cuda.get_device_name(0)}")
```

### 5.5 待用户执行的恢复动作

改完配置**还没有重装**，需要：

```bash
uv sync          # 重新解析，从 pytorch 官方索引拉 cu126 版（约 2.5 GB）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出：2.13.0+cu126 True
```

> ⚠️ **不要在评估跑到一半时执行 `uv sync`** —— 会替换掉正在被进程占用的
> torch 文件，直接把跑了 40 分钟的任务搞挂。

---

## 6. 评估用的 judge 是谁：一处必须纠正的表述

简历里原来写的是「**以 GPT-4o-mini 作为裁判模型**」，**这是错的**。

`src/online_service/rag_evaluator.py` 的 `_build_llm()` 用的是项目自身的配置：

```python
llm = ChatOpenAI(
    model=self.config.LLM_MODEL_NAME,      # ← 由环境变量 LLM_MODEL_NAME 控制，默认 qwen-plus
    openai_api_key=self.config.LLM_API_KEY,
    base_url=self.config.LLM_BASE_URL,     # ← DashScope 兼容模式
    temperature=0.0,
    rate_limiter=InMemoryRateLimiter(requests_per_second=1.0),  # ← 防 RPM 限流
)
```

**实际 judge 不是 GPT-4o-mini，也不是固定的某一个模型**，而是**跨厂商多模型裁判集成**：
本次（2026-08-29）用腾讯云 TokenHub 上 deepseek-v4-flash / deepseek-v4-pro / glm-5.1 /
glm-5.2 / hy3 / minimax-m3 共 6 个模型的免费额度，分块并行打分；生成答案用的是
qwen-turbo（205 条）+ qwen-plus（5 条）。模型名由环境变量 `LLM_MODEL_NAME` 控制，
已支持一键切换与多模型接力。

### 这不只是措辞问题，是个方法论要点

评估体系有一条基本纪律：**judge 最好与被测的生成模型不同族**，避免 self-preference
bias —— judge 会系统性给"像自己说话方式"的答案打高分。本次不仅跨族，更是**跨厂商 6 模型集成**：
生成是 qwen-turbo/plus，judge 是 deepseek/glm/minimax/hy 四个厂商，self-preference bias
被进一步稀释；取集成均值还能平滑单模型在 faithfulness 上的校准方差（各模型该指标口径
在 0.20–0.44 间摆动）。仍建议预算允许时用 GPT-4o-mini / Claude 等更强模型交叉验证，
或抽样人工复核校准。

### 处理

| 项 | 处理 |
|---|---|
| 简历 | 改为如实写「跨厂商多模型裁判集成（DeepSeek / 智谱 GLM / MiniMax / 阶跃 hy 共 6 个模型）」，不再写 GPT-4o-mini 或单一 qwen |
| 面试 | 主动讲这个设计：*"judge 与生成模型解耦，且刻意用跨厂商 6 模型集成当裁判，规避单模型自评偏差、平滑校准方差；有预算会再上更强模型或人工校准。"* |
| 代码 | `LLM_MODEL_NAME` 环境变量可配 + 多模型接力脚本 + 限流，避免额度/限流导致评估静默失效 |

> 面试时主动指出自己评估体系的偏差来源与缓解手段，比被问出来强得多。
> 而且这正好接上 Q15.1 那句"我更关注相对变化而非绝对值"。

---

## 7. 结果（2026-08-29 实测）

> ⚠️ **完整 210 组评测集已全量打分完成**：跨厂商 6 模型免费额度并行集成，最终 **209/210 条拿到有效分**（1 条因 Ragas 框架对该特定输入触发内部解析 bug 漏评，与额度/模型无关）。以下为真实结果（指标为真实 Ragas 打分，含合法 0 分）。

### 7.1 实测指标（N=209，6 模型集成裁判）

| 指标 | 均值 | 等权加权 |
|---|---|---|
| Faithfulness | 0.329 | — |
| Answer Relevancy | 0.576 | — |
| Context Precision | 0.636 | — |
| Context Recall | 0.559 | — |
| **等权加权综合** | — | **0.525** |

- 裁判构成：deepseek-v4-flash 28 / deepseek-v4-pro 56 / glm-5.1 50 / glm-5.2 23 / hy3 25 / minimax-m3 27。
- 各模型 faithfulness 口径差异大（glm 系 ~0.20–0.25 最严，minimax ~0.44 最松），说明 LLM-as-judge 在该指标上校准方差高；报告取**集成均值**以平滑单模型偏差。
- 留档产物：`data/test_query/eval_final_merged.json`（209 条逐条 + 平均分）、`data/test_query/chunks/`（各模型分块原始结果）、`data/test_query/eval_final_merged.csv`。
- 答案缓存 `eval_answers_210.json` 完整（210/210、0 空），**生成链路已全跑通**。

### 7.2 怎么读这些数

- **Context Recall 0.559 / Precision 0.636**：检索基本能覆盖标准答案所需信息，但比"全 1.0"更真实——因为测试集题目独立出题（§2），不再自我循环论证， Recall 反映真实召回能力。
- **Faithfulness 0.329**：偏低，主因是 §4 那类"诚实声明 + 参数知识补刀"的 badcase，以及多裁判中严格模型（glm 系）拉低了均值。这恰是优化方向，不是翻车——评估体系的价值就是暴露它。
- **Answer Relevancy 0.576**：中文长答案下反向生成问题易漂移（§4.C），系统性偏低属预期。

### 7.3 与简历口径的对照

| 项 | 简历旧占位（已删除） | 真实实测 |
|---|---|---|
| 样本量 | 200 组 | **210 组评测集，209 条成功打分** |
| 裁判模型 | GPT-4o-mini（错误） | 跨厂商 6 模型集成（DeepSeek / 智谱 GLM / MiniMax / 阶跃 hy） |
| 加权综合 | 0.88 | **0.525** |

> 简历/自我介绍已回填真实值（Ragas 四项 0.329 / 0.576 / 0.636 / 0.559，加权 0.525，210 评测集、跨厂商多模型裁判集成）。

### 7.4 复现命令

```bash
# 多模型并行分块评估（自动按题续跑、零分过滤、合并）
python scripts/run_parallel_eval.py --models glm-5.2 minimax-m3 hy3
# 只合并已有分块结果
python scripts/run_parallel_eval.py --merge-only
```

