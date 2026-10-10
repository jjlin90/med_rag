"""
降级策略验证脚本（使用检索 mock，不连接 Milvus、不加载模型；Config 仍校验 CUDA GPU）

覆盖 Small-to-Big 的五条路径，验证核心原则：
    「降级路径必须比主路径更安全，而不是更粗糙」

    L0 正常          → 粒度 2000 字父块，degraded=False
    L0 数据异常      → 剔除 orphan，仍走 L0，绝不退化成返回父块原文
    L1 同粒度降级    → 粒度仍为 400 字子块（切分参数与入库一致），degraded=True, level=1
    L2 无召回        → 空结果，level=2
    L2 基础设施故障  → 空结果 + error 字段，level=2，与「无召回」严格区分

运行：
    python scripts/test_degrade_policy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Config
from src.online_service.retrieval import (
    Retrieval, RetrievalResult, DEGRADE_OK, DEGRADE_L1, DEGRADE_L2
)

PASS, FAIL = "[PASS]", "[FAIL]"
_failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = PASS if cond else FAIL
    if not cond:
        _failures.append(name)
    print(f"  {mark} {name}" + (f"  -- {detail}" if detail else ""))


def make_retrieval(search_impl) -> Retrieval:
    """构造 Retrieval 但跳过 __init__（避免连接 Milvus / 加载 BGE-M3）。"""
    r = Retrieval.__new__(Retrieval)
    r.config = Config()
    r.top_k_retrieve = r.config.TOP_K_RETRIEVE
    r.top_k_children = r.config.TOP_K_CHILDREN
    r.sparse_weight = r.config.SPARSE_WEIGHT
    r.dense_weight = r.config.DENSE_WEIGHT
    r.milvus_store = None
    r.embedding_provider = None
    r.search = search_impl
    return r


def child(text="子块内容", pid="p1", score=0.9, cid="c1"):
    return {
        'id': cid, 'content': text, 'metadata': {'chunk_type': 'child'},
        'score': score, 'distance': score, 'source': '内分泌科',
        'parent_id': pid, 'parent_content': '父块完整原文' * 200,
    }


def parent(text=None, score=0.8, pid="p9"):
    return {
        'id': pid, 'content': text or ("糖尿病是一种代谢性疾病。" * 200),
        'metadata': {'chunk_type': 'parent'}, 'score': score, 'distance': score,
        'source': '内分泌科', 'parent_id': '', 'parent_content': '',
    }


# ---------------------------------------------------------------- L0 正常
def test_l0_normal():
    print("\n[L0] 正常路径：子块召回 → 回溯父块")
    calls = []

    def search(query, source_filter=None, use_hybrid=True, only_children=False, status=None):
        calls.append(only_children)
        return [child(f"子块{i}", pid="p1", cid=f"c{i}") for i in range(3)]

    r = make_retrieval(search)
    res = r.search_child_to_parent("一型糖尿病")

    check("只调用一次检索（未触发降级）", len(calls) == 1, f"calls={calls}")
    check("过滤已下推 only_children=True", calls[0] is True)
    check("未标记降级", res.degraded is False and res.degrade_level == DEGRADE_OK)
    check("3 子块去重为 1 父块", len(res) == 1, f"实际 {len(res)}")
    check("返回的是父块内容（非子块）", len(res[0]['content']) > 1000,
          f"长度 {len(res[0]['content'])}")


# ---------------------------------------------------------------- L0 数据异常
def test_l0_orphan():
    print("\n[L0] 数据异常：子块缺 parent_id")
    calls = []

    def search(query, source_filter=None, use_hybrid=True, only_children=False, status=None):
        calls.append(only_children)
        # 2 条正常子块 + 2 条缺 parent_id（模拟入库数据问题）
        return [child(cid="c1"), child(cid="c2"),
                {**child(cid="c3"), 'parent_id': ''},
                {**child(cid="c4"), 'parent_id': ''}]

    r = make_retrieval(search)
    res = r.search_child_to_parent("头痛")

    check("orphan 被剔除且计数", res.orphan_count == 2, f"orphan_count={res.orphan_count}")
    check("仍走 L0，未降级", res.degrade_level == DEGRADE_OK)
    check("未触发二次检索（不静默降级）", len(calls) == 1, f"calls={calls}")
    check("剩余正常子块仍产出结果", len(res) == 1)


# ---------------------------------------------------------------- L1 同粒度降级
def test_l1_fallback():
    print("\n[L1] 子块召回为空 → 放开过滤命中父块 → 内存切片")

    def search(query, source_filter=None, use_hybrid=True, only_children=False, status=None):
        # 严格模式（只子块）召回为空；放开过滤后命中父块
        return [] if only_children else [parent(), parent(pid="p10")]

    r = make_retrieval(search)
    res = r.search_child_to_parent("罕见病 xyz")

    check("触发 L1", res.degraded is True and res.degrade_level == DEGRADE_L1,
          f"level={res.degrade_level} reason={res.degrade_reason}")

    if res:
        size = r.config.CHILD_CHUNK_SIZE
        lengths = [len(d['content']) for d in res]
        avg = sum(lengths) / len(lengths)
        check("输出粒度仍是子块（未退化到 2000 字）", avg <= size * 1.5,
              f"平均 {avg:.0f} 字符，子块上限 {size}，共 {len(res)} 条")
        check("切片携带 parent_id（可溯源）",
              all(d.get('parent_id') for d in res))
        check("切片标记为 l1_split（便于离线分析）",
              all(d.get('metadata', {}).get('l1_split') for d in res))
        check("每条切片保留 parent_content",
              all(d.get('parent_content') for d in res))
        # 候选规模受控：降级路径不能比主路径更贵（reranker 开销恒定）
        check("L1 候选总量 ≤ top_k_retrieve",
              len(res) <= r.top_k_retrieve,
              f"{len(res)} 条 ≤ {r.top_k_retrieve}")
        # 实际进 LLM 的只有 reranker 输出的 Top-2，约 800 字，远小于 L0 的 4000 字
        into_llm = sum(sorted(lengths, reverse=True)[:2])
        check("进 LLM 的上下文量（Top-2）低于 L0 父块",
              into_llm < r.config.PARENT_CHUNK_SIZE,
              f"Top-2 合计 {into_llm} 字符 < 单个父块 {r.config.PARENT_CHUNK_SIZE}")


# ---------------------------------------------------------------- L2 无召回
def test_l2_no_recall():
    print("\n[L2] 严格模式与放开过滤后均无召回")
    calls = []

    def search(query, source_filter=None, use_hybrid=True, only_children=False, status=None):
        calls.append(only_children)
        return []

    r = make_retrieval(search)
    res = r.search_child_to_parent("完全无关的查询")

    check("尝试过 L1 后进入 L2", len(calls) == 2, f"calls={calls}")
    check("降级水位 L2", res.degrade_level == DEGRADE_L2)
    check("无文档", len(res) == 0)
    check("无 error 字段（不是故障）", res.error is None, f"error={res.error}")
    check("reason=no_recall", res.degrade_reason == 'no_recall')
    check("is_usable=False（上层应拒答）", res.is_usable is False)


# ---------------------------------------------------------------- L2 故障
def test_l2_error():
    print("\n[L2] 基础设施故障（Milvus 不可达）")
    calls = []

    def search(query, source_filter=None, use_hybrid=True, only_children=False, status=None):
        calls.append(only_children)
        if status is not None:
            status['error'] = "MilvusException: connection refused"
        return []

    r = make_retrieval(search)
    res = r.search_child_to_parent("糖尿病")

    check("故障立即进 L2，不再试探 L1", len(calls) == 1, f"calls={calls}")
    check("降级水位 L2", res.degrade_level == DEGRADE_L2)
    check("reason=retrieval_error", res.degrade_reason == 'retrieval_error')
    check("error 字段已填充（与无召回区分）", res.error is not None, f"error={res.error}")
    check("is_usable=False", res.is_usable is False)


# ---------------------------------------------------------------- L1 开关
def test_l1_disabled():
    print("\n[配置] 关闭 L1 降级开关")

    def search(query, source_filter=None, use_hybrid=True, only_children=False, status=None):
        return [] if only_children else [parent()]

    r = make_retrieval(search)
    r.config.ENABLE_CHILD_FILTER_FALLBACK = False
    res = r.search_child_to_parent("罕见病 xyz")

    check("直接进入 L2，不做父块切片", res.degrade_level == DEGRADE_L2)
    check("无文档", len(res) == 0)
    check("reason=child_recall_empty", res.degrade_reason == 'child_recall_empty')


# ---------------------------------------------------------------- 兼容性
def test_result_compat():
    print("\n[兼容] RetrievalResult 序列代理行为")
    res = RetrievalResult(documents=[child(cid="a"), child(cid="b")])

    check("len()", len(res) == 2)
    check("bool() 非空为真", bool(res) is True)
    check("迭代", [d['id'] for d in res] == ['a', 'b'])
    check("索引", res[0]['id'] == 'a')
    check("切片返回 list（非 RetrievalResult）",
          isinstance(res[:1], list), f"type={type(res[:1]).__name__}")
    check("空结果 bool 为假", bool(RetrievalResult()) is False)
    check("list() 转换", len(list(res)) == 2)


# ---------------------------------------------------------------- 打点
def test_metrics():
    print("\n[打点] 降级指标")
    Retrieval.reset_degrade_metrics()
    Retrieval._bump('l0_ok')
    Retrieval._bump('l1_fallback')
    Retrieval._bump('l2_error')
    m = Retrieval.get_degrade_metrics()

    # 指标四舍五入到 4 位，容差取 1e-3
    check("total 统计", m['total'] == 3, f"total={m['total']}")
    check("l1_rate", abs(m['l1_rate'] - 1 / 3) < 1e-3, f"l1_rate={m['l1_rate']}")
    check("l2_rate（含故障）", abs(m['l2_rate'] - 1 / 3) < 1e-3, f"l2_rate={m['l2_rate']}")
    check("l2_error 独立计数", m['l2_error'] == 1)


# ------------------------------------------------- RAGSystem 层：L2 不得调 LLM
def test_rag_system_l2_blocks_llm():
    """上层编排：L2 无召回时必须拒答，不能把空上下文丢给 LLM 自由发挥。"""
    print("\n[RAGSystem] L2 拦截 —— 不得调用 LLM 编造")

    from src.online_service.rag_system import (
        RAGSystem, NO_CONTEXT_ANSWER, SERVICE_UNAVAILABLE_ANSWER
    )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def generate_with_context(self, query, context, history=None):
            self.calls += 1
            return "模型自由发挥的答案"

    class FakeReranker:
        def rerank(self, query, documents, top_k=2):
            return list(documents)[:top_k]

    class FakeIntent:
        def predict(self, q):
            return {'intent': 'medical', 'confidence': 0.99}

    class FakeSelector:
        def select_strategy(self, q):
            return 'direct'

    def make_system(result):
        rs = RAGSystem.__new__(RAGSystem)
        rs.config = Config()
        rs.llm_generator = FakeLLM()
        rs.reranker = FakeReranker()
        rs.intent_classifier = FakeIntent()
        rs.strategy_selector = FakeSelector()
        rs.query_augmenter = None

        class FakeRetrieval:
            def search_child_to_parent(self, q, sf=None):
                return result

        rs.retrieval = FakeRetrieval()
        return rs

    # 场景 A：无召回
    rs = make_system(RetrievalResult(degraded=True, degrade_level=DEGRADE_L2,
                                     degrade_reason='no_recall'))
    out = rs.generate("罕见病 xyz")
    check("无召回时不调用 LLM", rs.llm_generator.calls == 0,
          f"LLM 被调用 {rs.llm_generator.calls} 次")
    check("返回安全拒答话术", out['answer'] == NO_CONTEXT_ANSWER)
    check("响应标记 degraded", out['degraded'] is True and out['degrade_level'] == DEGRADE_L2)
    check("sources 为空", out['sources'] == [])

    # 场景 B：基础设施故障 —— 话术必须不同（区分「没资料」与「服务挂了」）
    rs = make_system(RetrievalResult(degraded=True, degrade_level=DEGRADE_L2,
                                     degrade_reason='retrieval_error',
                                     error='MilvusException: refused'))
    out = rs.generate("糖尿病")
    check("故障时也不调用 LLM", rs.llm_generator.calls == 0)
    check("话术区分故障与无召回", out['answer'] == SERVICE_UNAVAILABLE_ANSWER)
    check("degrade_reason=retrieval_error", out['degrade_reason'] == 'retrieval_error')


def test_rag_system_l1_allows_llm():
    """反向验证：L1（同粒度降级）有真实依据，应当正常生成。"""
    print("\n[RAGSystem] L1 放行 —— 有依据就正常生成")

    from src.online_service.rag_system import RAGSystem

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def generate_with_context(self, query, context, history=None):
            self.calls += 1
            return "基于检索到的切片生成的答案"

    class FakeReranker:
        def rerank(self, query, documents, top_k=2):
            return list(documents)[:top_k]

    class FakeIntent:
        def predict(self, q):
            return {'intent': 'medical', 'confidence': 0.99}

    class FakeSelector:
        def select_strategy(self, q):
            return 'direct'

    rs = RAGSystem.__new__(RAGSystem)
    rs.config = Config()
    rs.llm_generator = FakeLLM()
    rs.reranker = FakeReranker()
    rs.intent_classifier = FakeIntent()
    rs.strategy_selector = FakeSelector()
    rs.query_augmenter = None

    docs = [{'id': f'p1_{i}_l1child', 'content': '切片内容' * 50,
             'metadata': {}, 'score': 0.7, 'source': '内分泌科',
             'parent_id': 'p1', 'parent_content': '原文'} for i in range(3)]

    class FakeRetrieval:
        def search_child_to_parent(self, q, sf=None):
            return RetrievalResult(documents=docs, degraded=True,
                                   degrade_level=DEGRADE_L1,
                                   degrade_reason='child_recall_empty_relaxed')

    rs.retrieval = FakeRetrieval()
    out = rs.generate("罕见病 xyz")

    check("L1 正常调用 LLM", rs.llm_generator.calls == 1,
          f"LLM 被调用 {rs.llm_generator.calls} 次")
    check("标记 degraded 但 level=1", out['degraded'] is True and out['degrade_level'] == DEGRADE_L1)
    check("sources 正常返回", len(out['sources']) > 0)


def main():
    print("=" * 68)
    print("降级策略验证：L0 严格 / L1 同粒度降级 / L2 安全拒答")
    print("=" * 68)

    Retrieval.reset_degrade_metrics()
    for fn in (test_l0_normal, test_l0_orphan, test_l1_fallback,
               test_l2_no_recall, test_l2_error, test_l1_disabled,
               test_result_compat, test_metrics,
               test_rag_system_l2_blocks_llm, test_rag_system_l1_allows_llm):
        fn()

    print("\n" + "=" * 68)
    if _failures:
        print(f"{FAIL} {len(_failures)} 项未通过：")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print(f"{PASS} 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
