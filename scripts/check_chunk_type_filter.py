"""
Chunk Type 过滤体检脚本
========================

用途：验证 Small-to-Big 链路的「只召回子块」过滤是否真正生效。

背景（本次修复的问题）：
    Milvus schema 原先没有顶层 chunk_type 字段，检索侧只能在应用层用
    `parent_id 非空` 做后置过滤。这导致 2000 字父块先占用 Top-K 槽位再被丢弃，
    极端情况下 Top-K 全为父块，Small-to-Big 静默失效、退化成直接返回大块。

本脚本体检三件事：
    1. 集合 schema 是否含顶层 chunk_type 字段（新库 vs 旧库）
    2. 库里父块 / 子块的实际数量与占比
    3. 下推过滤是否真的只返回子块（可选，需加载 BGE-M3）

用法：
    # 快速体检（不加载模型，几秒钟）
    python scripts/check_chunk_type_filter.py

    # 完整体检（含真实混合检索验证，需加载 BGE-M3，较慢）
    python scripts/check_chunk_type_filter.py --with-search

    # 指定验证用的查询文本
    python scripts/check_chunk_type_filter.py --with-search --query "头痛怎么办"

退出码：0 = 通过；1 = 发现问题需要处理。
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Config  # noqa: E402
from src.offline_pipeline.milvus_store import MilvusStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("check_chunk_type")

OK = "  [OK]  "
BAD = "  [!!]  "
WARN = "  [~]   "


def hr(title: str = ""):
    print()
    print("=" * 68)
    if title:
        print(title)
        print("=" * 68)


def count_by_expr(collection, expr: str) -> int:
    """按过滤表达式统计实体数（优先 count(*)，失败则退化为拉取 id 计数）。"""
    try:
        res = collection.query(expr=expr, output_fields=["count(*)"])
        if res and isinstance(res[0], dict):
            # pymilvus 返回的 key 可能是 'count(*)' 或 'count'
            for k, v in res[0].items():
                if "count" in k.lower():
                    return int(v)
        return len(res)
    except Exception as e:
        logger.debug("count(*) 不可用(%s)，退化为 id 计数", e)
        try:
            res = collection.query(expr=expr, output_fields=["id"], limit=16384)
            return len(res)
        except Exception as e2:
            logger.warning("统计失败 expr=%s : %s", expr, e2)
            return -1


def main():
    parser = argparse.ArgumentParser(description="Small-to-Big 子块过滤体检")
    parser.add_argument("--with-search", action="store_true",
                        help="执行真实混合检索验证（需加载 BGE-M3，较慢）")
    parser.add_argument("--query", default="头痛怎么办",
                        help="验证用的查询文本")
    args = parser.parse_args()

    config = Config()
    problems = []

    hr("1. 连接与 Schema 检查")
    try:
        store = MilvusStore(config)
    except Exception as e:
        print(f"{BAD}无法连接 Milvus：{e}")
        print("      请先启动 Milvus（本项目为 Docker 部署，需先启动 Docker Desktop）")
        return 1

    from pymilvus import Collection, utility  # noqa: E402

    if not utility.has_collection(store.collection_name, using=store.alias):
        print(f"{BAD}集合 {store.collection_name} 不存在，请先执行离线入库")
        return 1

    collection = Collection(store.collection_name, using=store.alias)
    collection.load()
    total = collection.num_entities
    field_names = [f.name for f in collection.schema.fields]
    print(f"  集合: {store.collection_name}")
    print(f"  实体总数: {total}")
    print(f"  字段: {', '.join(field_names)}")

    has_ct = store.has_chunk_type_field()
    if has_ct:
        print(f"{OK}schema 含顶层 chunk_type 字段 → 过滤表达式 chunk_type == 'child'")
        child_expr = "chunk_type == 'child'"
        parent_expr = "chunk_type == 'parent'"
    else:
        print(f"{WARN}schema 无顶层 chunk_type 字段（旧库）")
        print("       → 过滤退化为 parent_id != ''（语义等价，父块 parent_id 恒为空串）")
        print("       → 若要获得显式字段，需重建集合后重新入库")
        child_expr = 'parent_id != ""'
        parent_expr = 'parent_id == ""'

    hr("2. 父块 / 子块数量统计")
    n_child = count_by_expr(collection, child_expr)
    n_parent = count_by_expr(collection, parent_expr)
    print(f"  子块(child): {n_child}")
    print(f"  父块(parent): {n_parent}")

    if n_child <= 0:
        print(f"{BAD}库中没有子块！Small-to-Big 完全无法工作。")
        print("       检查点：入库时 parent_id / chunk_type 是否真的写入了。")
        problems.append("子块数量为 0")
    elif n_parent == 0:
        print(f"{WARN}库中没有父块（只有子块）。若这是刻意为之，请确认")
        print("       parent_content 字段有值，否则回溯后拿不到完整上下文。")
    else:
        ratio = n_child / max(n_parent, 1)
        print(f"{OK}父/子结构正常，平均每个父块切出 {ratio:.2f} 个子块")

    # 抽样看一条子块，确认 parent_id / parent_content 真的有值
    try:
        samples = collection.query(
            expr=child_expr,
            output_fields=["id", "parent_id", "text"],
            limit=1)
        if samples:
            s = samples[0]
            pid = s.get("parent_id") or ""
            text = s.get("text") or ""
            print(f"\n  子块样例:")
            print(f"    id        = {s.get('id')}")
            print(f"    parent_id = {pid!r}")
            print(f"    text 长度 = {len(text)} 字符")
            if not pid:
                print(f"{BAD}子块 parent_id 为空，无法回溯父块")
                problems.append("子块 parent_id 为空")
            else:
                print(f"{OK}子块 parent_id 非空，可正常回溯")
    except Exception as e:
        logger.warning("抽样失败: %s", e)

    hr("3. 下推过滤表达式")
    print(f"  实际生效表达式: {store.child_filter_expr()}")
    combined = MilvusStore.combine_expr("source == '心血管'", store.child_filter_expr())
    print(f"  与来源过滤组合: {combined}")
    if " and " in combined and combined.count("(") >= 2:
        print(f"{OK}多条件组合正确（已加括号保证优先级）")
    else:
        print(f"{BAD}多条件组合异常，请检查 combine_expr")
        problems.append("表达式组合异常")

    if not args.with_search:
        hr("体检结束（快速模式）")
        print("  未执行真实检索验证。加 --with-search 可做端到端验证（需加载 BGE-M3）。")
        if problems:
            print(f"\n发现 {len(problems)} 个问题：")
            for p in problems:
                print(f"  - {p}")
            return 1
        print("\n结论：静态检查全部通过。")
        return 0

    hr("4. 端到端检索验证")
    from src.online_service.retrieval import Retrieval  # noqa: E402

    retrieval = Retrieval(config)
    q = args.query
    print(f"  查询: {q!r}")

    print("\n  --- 未加子块过滤（修复前的行为）---")
    raw = retrieval.search(q, use_hybrid=True, only_children=False)
    raw_parents = [r for r in raw if not r.get("parent_id")]
    print(f"  召回 {len(raw)} 条，其中父块 {len(raw_parents)} 条、子块 "
          f"{len(raw) - len(raw_parents)} 条")
    if raw_parents:
        wasted = len(raw_parents)
        print(f"{WARN}有 {wasted} 个 Top-K 槽位被 2000 字父块占用后丢弃 "
              f"（槽位浪费率 {wasted / max(len(raw), 1):.0%}）")

    print("\n  --- 下推子块过滤（修复后的行为）---")
    kids = retrieval.search(q, use_hybrid=True, only_children=True)
    bad = [r for r in kids if not r.get("parent_id")]
    print(f"  召回 {len(kids)} 条，其中无 parent_id 的异常项 {len(bad)} 条")
    if bad:
        print(f"{BAD}过滤下推未生效！仍混入父块。")
        problems.append("子块过滤下推失效")
    elif kids:
        print(f"{OK}全部 {len(kids)} 条均为子块，过滤下推生效")
    else:
        print(f"{WARN}子块召回为空，请确认知识库内容与过滤表达式")

    print("\n  --- Small-to-Big 完整链路 ---")
    rr = retrieval.search_child_to_parent(q)
    parents = rr.documents
    print(f"  {len(kids)} 子块 → {len(parents)} 去重父块")
    if rr.degraded:
        print(f"  {WARN}降级水位 L{rr.degrade_level} ({rr.degrade_reason})"
              + (f"，错误：{rr.error}" if rr.error else ""))

    for i, p in enumerate(parents[:3], 1):
        print(f"    [{i}] id={p.get('id')}  块长度={len(p.get('content') or '')} 字符  "
              f"score={p.get('score', 0):.4f}")

    if parents:
        avg_len = sum(len(p.get("content") or "") for p in parents) / len(parents)
        if rr.degrade_level >= 1:
            # L1 降级返回的是内存切片的子块（~400 字），长度偏小是预期行为，
            # 不能沿用「父块长度 < 500 即异常」的判据，否则会误报。
            print(f"{OK}L1 降级产出 {len(parents)} 条子块，平均长度 {avg_len:.0f} 字符"
                  f"（粒度未退化，符合预期）")
        else:
            print(f"{OK}回溯成功，父块平均长度 {avg_len:.0f} 字符")
            if avg_len < 500:
                print(f"{WARN}父块平均长度偏小，疑似 parent_content 未写入，"
                      f"实际返回的是子块原文")
                problems.append("parent_content 疑似未写入")
    elif rr.degrade_level >= 2:
        print(f"{BAD}Small-to-Big 无输出（降级水位 L{rr.degrade_level}）")
        problems.append(f"Small-to-Big 无输出: {rr.degrade_reason}")
    else:
        print(f"{BAD}Small-to-Big 未产出任何父块")
        problems.append("Small-to-Big 无输出")

    hr("体检结束（完整模式）")
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("结论：全部通过，Small-to-Big 链路工作正常。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
