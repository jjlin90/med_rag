"""
FAQ 批量入库脚本
将 data/msd_faq_clean.csv（清洗后的 question,answer,source,doc_name）批量写入
MySQL 的 faq 表。

用法：
    # 默认：读 data/msd_faq_clean.csv，连接参数取自 src/config/settings.py
    #（MYSQL_HOST/PORT/USER/PASSWORD/DATABASE，可用同名环境变量覆盖），自动建库后入库
    python scripts/ingest_faq.py

    # 指定 CSV 路径
    python scripts/ingest_faq.py --csv path/to/faq.csv

    # 自定义 MySQL 连接（覆盖 settings 中的配置）
    python scripts/ingest_faq.py --host 127.0.0.1 --port 3306 --user root --password 123456 --database medical_rag

说明：
    - faq 表由 FAQSearch 在初始化时自动 CREATE TABLE IF NOT EXISTS，无需手动建表。
    - 入库前请先跑 scripts/clean_faq.py 生成清洗后的 data/msd_faq_clean.csv，
      否则默认会读未清洗的原始数据（含大量网页噪声）。
    - 列映射：question->question, answer->answer, source->category, doc_name->keywords
      （category/keywords 用于分类与来源追溯，均为 faq 表已有字段）。
    - 若 MySQL 未启动或库不存在，脚本会给出明确的搭建步骤后退出。
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

# 必须在导入任何 CUDA 相关模块之前设置，避免无谓的 CUDA 初始化
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pymysql  # 直接用它做"建库"步骤；若未安装会给出提示

from src.config.settings import Config
from src.online_service.cache_manager import CacheManager
from src.online_service.faq_search import FAQSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ingest_faq")

DEFAULT_CSV = project_root / "data" / "msd_faq_clean.csv"


def ensure_database(host, port, user, password, database):
    """确保目标数据库存在：先连到 MySQL 服务器（不指定库），建库后断开。"""
    try:
        conn = pymysql.connect(
            host=host, port=port, user=user, password=password,
            charset="utf8mb4", connect_timeout=10,
        )
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.close()
        logger.info(f"数据库 `{database}` 已就绪（不存在则已自动创建）")
        return True
    except pymysql.Error as e:
        logger.error(f"无法连接 MySQL 或创建数据库：{e}")
        return False


def read_faq_rows(csv_path: Path):
    """读取 CSV 并映射为 faq 字典列表，跳过 question/answer 为空的行。"""
    rows = []
    skipped = 0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # 规范化表头（去掉 BOM / 空白）
        reader.fieldnames = [fn.strip().lstrip("\ufeff") for fn in reader.fieldnames]
        for raw in reader:
            question = (raw.get("question") or "").strip()
            answer = (raw.get("answer") or "").strip()
            if not question or not answer:
                skipped += 1
                continue
            category = (raw.get("source") or "").strip() or None
            keywords = (raw.get("doc_name") or "").strip() or None
            rows.append({
                "question": question,
                "answer": answer,
                "category": category,
                "keywords": keywords,
            })
    logger.info(f"从 {csv_path} 解析出 {len(rows)} 条有效 FAQ（跳过 {skipped} 条空行）")
    return rows


def main():
    parser = argparse.ArgumentParser(description="将 FAQ CSV 批量写入 MySQL")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="FAQ CSV 路径")
    parser.add_argument("--host", default=None, help="MySQL host（默认用 faq_search 写死配置）")
    parser.add_argument("--port", type=int, default=None, help="MySQL port")
    parser.add_argument("--user", default=None, help="MySQL user")
    parser.add_argument("--password", default=None, help="MySQL password")
    parser.add_argument("--database", default=None, help="MySQL database 名")
    parser.add_argument("--batch-size", type=int, default=500, help="每批插入条数")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV 文件不存在：{csv_path}")
        return 1

    # 实例化配置，作为连接参数的默认值（用户没传 CLI 参数时，沿用 settings 的配置）
    config = Config()
    cfg_override = {
        "host": args.host or config.MYSQL_HOST,
        "port": args.port or config.MYSQL_PORT,
        "user": args.user or config.MYSQL_USER,
        "password": args.password if args.password is not None else config.MYSQL_PASSWORD,
        "database": args.database or config.MYSQL_DATABASE,
    }

    # 1. 确保数据库存在
    if not ensure_database(**cfg_override):
        logger.error(
            "请先准备 MySQL：\n"
            "  1) 启动 MySQL 服务（默认 127.0.0.1:3306）\n"
            "  2) 确认用户/密码（取自 src/config/settings.py 的 MYSQL_USER/MYSQL_PASSWORD；\n"
            "     若不同请用 --user/--password 指定，或用同名环境变量覆盖）\n"
            "  3) 本脚本会自动 CREATE DATABASE medical_rag，无需手动建库\n"
            "  本步骤失败通常是 MySQL 未启动或凭据不对。"
        )
        return 1

    # 2. 初始化组件（CacheManager 在 Redis 不可用时自动降级，不影响入库）
    # config 已在上方实例化，FAQSearch 会自动用其中的 MySQL 配置连接
    cache = CacheManager(config)
    faq_search = FAQSearch(config, cache)

    # 3. 若用户传了自定义连接参数，覆盖 faq_search 的内部配置并重新连库建表
    if any([args.host, args.port, args.user, args.password is not None, args.database]):
        try:
            if faq_search.connection:
                faq_search.close()
        except Exception:
            pass
        faq_search.db_config = {
            "host": cfg_override["host"],
            "port": cfg_override["port"],
            "user": cfg_override["user"],
            "password": cfg_override["password"],
            "database": cfg_override["database"],
            "charset": "utf8mb4",
        }
        faq_search.connection = None
        faq_search.cursor = None
        faq_search._init_db()  # 重新连接 + 建 faq 表

    if not faq_search.connection:
        logger.error(
            "FAQSearch 未能连接到 MySQL（faq_search.connection 为 None）。\n"
            "请检查 faq_search.py 中的 db_config 或改用 --host/--user/--password 指定。"
        )
        return 1

    # 4. 读取并入库
    rows = read_faq_rows(csv_path)
    if not rows:
        logger.warning("没有可入库的 FAQ 数据，结束。")
        return 0

    total = len(rows)
    inserted = 0
    for i in range(0, total, args.batch_size):
        batch = rows[i:i + args.batch_size]
        n = faq_search.batch_add_faq(batch)
        inserted += n
        logger.info(f"已写入 {min(i + args.batch_size, total)}/{total}（本批 {n} 条）")

    # 5. 汇总
    stats = faq_search.get_faq_stats()
    logger.info(f"入库完成：成功写入 {inserted}/{total} 条。")
    logger.info(f"当前 faq 表统计：{stats}")

    try:
        faq_search.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
