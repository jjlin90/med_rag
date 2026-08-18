"""
FAQ 数据清洗脚本
================
针对 data/msd_faq.csv（默沙东诊疗手册大众版导出的问答对）做去噪处理。
原始数据混入大量网页 UI / 导航 / 机构附录噪声，本脚本按规则剥离：

  R1  开头元数据块   `作者： … 已审核/已修订 修改的 浏览专业版 `（99.6% 命中）
  R2  版权声明行     `Co., Inc., Rahway, NJ, USA … 保留所有权利。` / `版权所有`
  R3  导航栏目串     空格分隔的栏目词 / UI 词（如 `症状 诊断 治疗`、`浏览专业版 多媒体`、
                     `阅读更多 由`、`关于 免责声明 版权所有` 等）
  R4  末尾机构附录   最后一个句号之后、由"机构：描述"清单构成的尾部噪声
  R5  交叉引用       `（请参阅 … 。）` / `(请参阅 …)` 形式的站内跳转

清洗后写入 data/msd_faq_clean.csv（原始文件不动），并打印逐规则命中统计与前后对比样例。

用法：
    python scripts/clean_faq.py                 # 清洗并写 data/msd_faq_clean.csv
    python scripts/clean_faq.py --csv x.csv --out y.csv
    python scripts/clean_faq.py --dry-run       # 只打印报告，不写文件
"""

import argparse
import csv
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
DEFAULT_CSV = project_root / "data" / "msd_faq.csv"
DEFAULT_OUT = project_root / "data" / "msd_faq_clean.csv"

# R3 导航判定词汇
# UI 垃圾词：出现即判为导航/页脚噪声，整串删除
UI_JUNK = {
    "浏览专业版", "多媒体", "阅读更多", "关于", "免责声明", "版权所有", "修改的",
}
# 栏目词：若整串仅由这些词构成（空格分隔），判为目录导航，整串删除
SECTION = {
    "病因", "筛查", "预防", "症状", "诊断", "治疗", "概述", "概要", "临床表现",
    "评估", "管理", "治疗监测", "药物治疗", "病因学", "病理生理学", "预后",
    "并发症", "关键点", "疫苗类型", "剂量和建议", "副作用", "筛查和预防",
}
# R4 机构附录判定词
ORG_KW = ["协会", "基金会", "研究所", "中心", "网络", "组织", "机构", "学院", "学会"]


def clean_answer(text: str) -> str:
    """对单条 answer 应用全部清洗规则，返回清洗后文本。"""
    if not text:
        return text
    t = text

    # R1 开头/文中元数据块：从 `作者：` 一直删到最近的 `浏览专业版` 及其后空白。
    # 不锚定行首——很多条目以标题开头、随后才是 `作者：…浏览专业版`。
    # 仍要求必须以 `浏览专业版` 收尾，正文里偶发的"作者"二字不会误命中。
    t = re.sub(r"\s*作者[：:].*?浏览专业版\s*", " ", t, flags=re.S)

    # R2 版权声明行
    t = re.sub(
        r"Co\.,?\s*Inc\.?,?\s*Rahway,?\s*NJ,?\s*USA[^。]*?保留所有权利。?", "", t
    )
    t = re.sub(r"保留所有权利。?", "", t)
    t = re.sub(r"版权所有。?", "", t)

    # R3 导航栏目串（两步，避免误删紧跟在 UI 词后的真实词）：
    #   3a. 先逐个删除 UI 垃圾词（浏览专业版/多媒体/阅读更多/关于/免责声明/版权所有/修改的）
    #   3b. 再删除"纯栏目词"组成的空格串（长度>=2）
    for w in UI_JUNK:
        t = re.sub(r"\s*" + re.escape(w) + r"\s*", " ", t)
    # 按长度降序排列，避免短词（如"筛查"）抢先匹配长词（如"筛查和预防"）的前缀
    section_alt = "|".join(
        re.escape(w) for w in sorted(SECTION, key=len, reverse=True)
    )
    # 前面是 空白/句号（开头补空格使行首也能匹配），连续 >=2 个栏目词
    t = " " + t
    t = re.sub(
        r"(?<=[\s。])(?:" + section_alt + r")(?:\s+(?:" + section_alt + r")){1,}\s*",
        " ",
        t,
    )
    t = t.strip()

    # R4 末尾机构附录：最后一个句号之后的尾部若为机构清单则删除。
    # 判定：尾部出现 >=2 个冒号（多为 "机构：主题" 重复排列），
    # 或机构关键词 >=2 且尾部较长 -> 判为页脚附录并删除。
    idx = t.rfind("。")
    if idx != -1:
        tail = t[idx + 1:].strip()
        if tail:
            colons = tail.count("：") + tail.count(":")
            org_hits = sum(1 for k in ORG_KW if k in tail)
            # 尾部前 60 字内若出现 "机构词+冒号"（如 `基金会：…`、`协会：`），
            # 即为单条机构附录，同样删除
            org_colon = bool(
                re.search(r"(?:协会|基金会|研究所|中心|网络|组织|机构|学院|学会)\s*[：:]", tail[:60])
            )
            if (colons >= 2 and len(tail) > 20) or (org_hits >= 2 and len(tail) > 60) or org_colon:
                t = t[:idx + 1].rstrip()

    # R4b 结尾机构引用块（自带句号，故 R4 的"末句号之后"判定失效）：
    # 若"机构词/缩写 + 冒号"出现在末尾 150 字内，判为页脚，从它前面
    # 最后一个句号处截断（保留全部正文、只丢页脚）；不在末尾的（夹在正文
    # 中间的）一律不动，避免误删正文。
    ORG_TERMS = ORG_KW + ["CDC", "AAP", "AAFP", "ECDC", "NIH", "WHO", "FDA",
                          "美国国家", "美国"]
    last_pos = -1
    for term in ORG_TERMS:
        for mm in re.finditer(re.escape(term) + r"\s*[：:]", t):
            if mm.start() > len(t) - 150:
                last_pos = max(last_pos, mm.start())
    if last_pos != -1:
        pre = t[:last_pos].rfind("。")
        if pre != -1:
            t = t[:pre + 1].rstrip()
        else:
            t = t[:last_pos].rstrip(" 。，")

    # R5 交叉引用（站内跳转）
    t = re.sub(r"（请参阅[^）]*?。）", "", t)
    t = re.sub(r"\(请参阅[^)]*?\)", "", t)

    # 收尾：压缩多余空白、去除首尾空格
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_question(text: str) -> str:
    """question 仅做轻量清洗：去首尾空白、压缩内部空白。"""
    if not text:
        return text
    return re.sub(r"\s+", " ", text).strip()


def main():
    parser = argparse.ArgumentParser(description="清洗 FAQ CSV 噪声")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="原始 FAQ CSV")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="清洗后输出 CSV")
    parser.add_argument("--dry-run", action="store_true", help="只打印报告，不写文件")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[错误] 输入文件不存在：{csv_path}")
        return 1

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [fn.strip().lstrip("\ufeff") for fn in reader.fieldnames]
        raw_rows = list(reader)

    total = len(raw_rows)
    out_rows = []
    stats = {
        "empty_after_clean": 0,
        "len_before_total": 0,
        "len_after_total": 0,
        "samples": [],
    }
    # 各规则命中计数（用规则前/后差异近似：若文本变化则计入"有改动"）
    changed = 0

    for raw in raw_rows:
        q = clean_question(raw.get("question") or "")
        a_raw = raw.get("answer") or ""
        a = clean_answer(a_raw)
        if not q or not a:
            stats["empty_after_clean"] += 1
            continue
        out_rows.append({
            "question": q,
            "answer": a,
            "source": (raw.get("source") or "").strip(),
            "doc_name": (raw.get("doc_name") or "").strip(),
        })
        stats["len_before_total"] += len(a_raw)
        stats["len_after_total"] += len(a)
        if a != a_raw:
            changed += 1
            if len(stats["samples"]) < 6:
                stats["samples"].append((a_raw, a))

    # 报告
    print("=" * 70)
    print("FAQ 清洗报告")
    print("=" * 70)
    print(f"原始条数           : {total}")
    print(f"清洗后有效条数     : {len(out_rows)}")
    print(f"清洗后为空被丢弃   : {stats['empty_after_clean']}")
    print(f"发生过改动的条数   : {changed} ({changed/total*100:.1f}%)")
    if total:
        print(f"answer 平均长度    : {stats['len_before_total']/total:.0f} -> "
              f"{stats['len_after_total']/len(out_rows):.0f} 字符")
    print("-" * 70)
    print("前后对比样例：")
    for i, (b, a) in enumerate(stats["samples"], 1):
        print(f"\n[样例 {i}] 前({len(b)}字): {b[:160]}")
        print(f"        后({len(a)}字): {a[:160]}")
    print("=" * 70)

    if args.dry_run:
        print("[dry-run] 未写入文件。")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "answer", "source", "doc_name"]
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"已写出清洗结果：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
