"""
构建意图分类器训练数据（medical=1 / general=0）。

数据来源：
- 医疗类：data/clean_md 下 2570 个真实词条（疾病/药物名，源自默沙东诊疗手册大众版）
  + 常见症状问句模板（覆盖「头痛/发烧/感冒」等此前误判样本）+ test_qa 医疗问句
- 通用类：通用概念 + 动词模板批量生成（AI/编程/美食/运动/科普/出行等，刻意与医疗无关）

输出：
- data/intent_train/train.json  (90%)
- data/intent_train/val.json    (10%)
格式：[{"text": "...", "label": 0/1}, ...]
"""
import json
import os
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CLEAN_MD_DIR = BASE_DIR / "data/clean_md"
TEST_QA = BASE_DIR / "data/test_query/test_qa.json"
OUT_DIR = BASE_DIR / "data/intent_train"

random.seed(42)

# ---------- 医疗类：从真实词条构造 ----------
ENTITY_TEMPLATES = [
    "{name} 是什么？",
    "{name} 有哪些症状？",
    "{name} 是由什么原因引起的？",
    "{name} 应该怎么治疗和护理？",
    "{name} 严重吗？能治好吗？",
    "得了 {name} 该怎么办？",
    "{name} 需要吃什么药？",
    "{name} 要做哪些检查？",
    "{name} 平时饮食要注意什么？",
    "{name} 会传染吗？",
]

# 常见症状（覆盖此前误判为 general 的样本）
SYMPTOMS = [
    "头痛", "发烧", "咳嗽", "肚子疼", "头晕", "恶心", "失眠", "皮疹",
    "喉咙痛", "流鼻涕", "腹泻", "乏力", "胸闷", "气短", "心慌", "关节痛",
    "腰痛", "牙痛", "感冒", "高血压", "糖尿病", "哮喘", "过敏",
    "贫血", "低血糖", "胃疼", "便秘", "耳鸣", "眼睛干涩", "心慌",
]
SYMPTOM_TEMPLATES = [
    "{s}应该怎么办？",
    "{s}需要吃药吗？",
    "{s}可能是什么原因引起的？",
    "{s}吃什么药比较好？",
    "{s}平时要注意什么？",
    "一直 {s}是不是得了什么病？",
]

# 医疗动作/用药类
MED_ACTION = [
    "抗生素可以随便吃吗？",
    "退烧药多久能重复吃一次？",
    "孕妇能打疫苗吗？",
    "血压计怎么看？",
    "血糖高了一日三餐怎么吃？",
    "吃药后多久能喝酒？",
    "输液比吃药效果更好吗？",
    "中药和西药能一起吃吗？",
]

# ---------- 通用类：与医疗无关的多样问句 ----------
# 通用概念（覆盖多领域，刻意不含医学实体）
GENERAL_CONCEPTS = [
    # 科技/编程
    "人工智能", "机器学习", "区块链", "云计算", "大数据", "物联网", "量子计算",
    "Python", "Java", "前端开发", "后端开发", "数据库", "数据结构", "操作系统",
    "手机", "笔记本电脑", "路由器", "固态硬盘", "显示器", "显卡", "主板",
    "深度学习", "神经网络", "开源", "API", "算法", "代码", "软件", "硬件",
    "浏览器", "操作系统", "服务器", "编程语言", "网络安全", "芯片", "内存", "CPU",
    # 美食/生活
    "红烧肉", "蛋炒饭", "西红柿鸡蛋面", "蛋糕", "面包", "饺子", "可乐", "咖啡",
    "洗衣机", "冰箱", "空调", "吸尘器", "电饭煲", "微波炉", "扫地机器人",
    "火锅", "烧烤", "寿司", "披萨", "奶茶", "啤酒", "红酒", "茶", "面条", "粥",
    "拖把", "洗洁精", "沐浴露", "洗发水", "毛巾", "枕头", "被子", "窗帘",
    # 运动/健康生活（非疾病）
    "篮球", "足球", "羽毛球", "跑步", "游泳", "健身", "瑜伽", "骑行",
    "登山", "滑雪", "乒乓球", "网球", "马拉松", "跳绳", "举重", "徒步",
    # 科普/自然
    "太阳系", "黑洞", "光速", "引力", "火山", "地震", "彩虹", "闪电",
    "进化", "基因", "细胞", "原子", "行星", "恒星", "海洋", "森林", "恐龙", "月亮",
    "风", "雨", "雪", "季节", "潮汐", "沙漠", "冰川", "闪电", "磁铁", "电",
    # 出行/娱乐
    "北京", "上海", "成都", "西安", "云南", "西藏", "电影", "小说", "音乐", "绘画",
    "广州", "杭州", "重庆", "厦门", "三亚", "故宫", "长城", "丽江", "青岛", "武汉",
    "吉他", "钢琴", "摄影", "旅行", "露营", "桌游", "手游", "动漫", "演唱会", "博物馆",
    # 工作/理财/学习
    "简历", "面试", "基金", "股票", "复利", "预算", "英语", "数学", "摄影", "写作",
    "副业", "创业", "PPT", "Excel", "时间管理", "沟通", "领导力", "跳槽", "公积金", "社保",
    # 日常闲聊/其他（刻意非医疗）
    "今天天气怎么样", "周末去哪玩", "怎么追剧", "养猫要注意什么", "怎么挑西瓜",
    "快递怎么寄", "怎么买火车票", "朋友圈怎么发", "怎么存钱", "星座准不准",
]
GENERAL_TEMPLATES = [
    "什么是{concept}？",
    "怎么学习{concept}？",
    "{concept}有什么用？",
    "{concept}难学吗？",
    "推荐一本关于{concept}的书",
    "{concept}和类似的有什么区别？",
    "新手怎么入门{concept}？",
    "{concept}的发展历史是怎样的？",
    "网上关于{concept}的说法靠谱吗？",
    "{concept}相关的职业有哪些？",
    "如何判断{concept}的好坏？",
    "{concept}适合什么人群？",
    "关于{concept}有哪些常见误区？",
    "怎么用{concept}解决实际问题？",
]
# 动词类通用问句
GENERAL_VERBS = [
    "煮米饭", "洗衣服", "去油渍", "拍视频", "做PPT", "写毛笔字", "养绿植",
    "学吉他", "练瑜伽", "减肚子", "背单词", "规划旅行", "挑选电脑", "保存照片",
]
GENERAL_VERB_TEMPLATES = [
    "怎么{verb}？",
    "为什么{verb}总做不好？",
    "{verb}有什么技巧？",
    "新手怎么{verb}？",
]


def clean_entity(name: str) -> str:
    """清理文件名得到实体名（去 .md 与括号内英文/剂量）。"""
    name = name.replace(".md", "")
    name = re.sub(r"\s*\(.*?\)\s*", "", name)
    return name.strip()


def build_medical(target: int = 1500) -> list:
    samples = []

    # 1) 真实词条
    names = []
    for f in CLEAN_MD_DIR.glob("*.md"):
        n = clean_entity(f.name)
        if n:
            names.append(n)
    random.shuffle(names)
    chosen = names[:target // len(ENTITY_TEMPLATES) + 5]  # 足够覆盖 target
    for n in chosen:
        for t in ENTITY_TEMPLATES:
            samples.append({"text": t.format(name=n), "label": 1})
            if len(samples) >= target:
                break
        if len(samples) >= target:
            break

    # 2) 症状问句
    for s in SYMPTOMS:
        for t in SYMPTOM_TEMPLATES:
            samples.append({"text": t.format(s=s), "label": 1})

    # 3) 医疗动作/用药
    for t in MED_ACTION:
        samples.append({"text": t, "label": 1})

    # 4) test_qa 中的医疗问句
    if TEST_QA.exists():
        for q in json.loads(TEST_QA.read_text(encoding="utf-8")):
            samples.append({"text": q["question"], "label": 1})

    return samples


def build_general(target: int = 1500) -> list:
    samples = []
    # 概念模板
    for c in GENERAL_CONCEPTS:
        for t in GENERAL_TEMPLATES:
            samples.append({"text": t.format(concept=c), "label": 0})
    # 动词模板
    for v in GENERAL_VERBS:
        for t in GENERAL_VERB_TEMPLATES:
            samples.append({"text": t.format(verb=v), "label": 0})
    # 打乱后截取目标数量
    random.shuffle(samples)
    return samples[:target]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    medical = build_medical(2500)
    general = build_general(2500)

    # 平衡到较小者
    n = min(len(medical), len(general))
    random.shuffle(medical)
    random.shuffle(general)
    medical = medical[:n]
    general = general[:n]

    all_data = medical + general
    random.shuffle(all_data)

    split = int(len(all_data) * 0.9)
    train = all_data[:split]
    val = all_data[split:]

    def dist(lst):
        c = {0: 0, 1: 0}
        for x in lst:
            c[x["label"]] += 1
        return c

    (OUT_DIR / "train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"医疗样本数: {len(medical)}  通用样本数: {len(general)}")
    print(f"训练集: {len(train)}  -> 标签分布 {dist(train)}")
    print(f"验证集: {len(val)}  -> 标签分布 {dist(val)}")
    print(f"已保存: {OUT_DIR/'train.json'}  {OUT_DIR/'val.json'}")


if __name__ == "__main__":
    main()
