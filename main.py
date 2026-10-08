#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
med_rag 命令行交互入口（对齐 EduRAG 主流程）。

程序入口 → 初始化 → 模式选择（数据处理 / 查询）→ 执行对应流程

使用方式：
    # 交互式查询模式（默认）
    python main.py

    # 数据处理模式（离线入库）
    python main.py --data-processing --data-dir ./data/clean_md
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到 Python 路径，使 src 包可被导入
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 必须在任何 CUDA 初始化之前设置：减少显存碎片导致的分配器空转
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.config.settings import Config
from src.online_service.rag_system import RAGSystem
from src.online_service.conversation_store import ConversationStore


def initialize_rag_system(config: Config) -> RAGSystem:
    """初始化 RAG 系统（加载向量模型、重排模型、意图分类器、LLM 客户端）。"""
    print("正在初始化 RAG 系统，首次启动会加载 BGE-M3 / BGE-Reranker / BERT，请稍候...")
    rag = RAGSystem(config)
    print("RAG 系统初始化完成！")
    return rag


def process_data_mode(config: Config, data_dir: Path) -> None:
    """数据处理模式：清洗 → 分块 → 向量化 → 写入 Milvus。"""
    # 复用现有离线入库脚本，避免重复实现
    from scripts.run_offline_ingest import (
        process_documents,
        setup_directories,
        validate_config,
    )

    setup_directories(config)
    if not validate_config(config, data_dir=data_dir):
        print("配置验证失败，退出")
        raise SystemExit(1)

    process_documents(config, data_dir=data_dir)


def print_help() -> None:
    """打印帮助信息。"""
    print("\n med_rag 帮助")
    print("-" * 40)
    print("- 直接输入问题即可开始问答")
    print("- 来源过滤：可按疾病类别/来源过滤检索范围，不填则全库检索")
    print("- help: 显示本帮助")
    print("- exit / quit / q: 退出系统")
    print("-" * 40)


def query_mode(config: Config, session_id: str = None) -> None:
    """交互式查询模式（对齐 EduRAG 的 rag_qa/main.py）。"""
    rag = initialize_rag_system(config)

    # 会话管理（MySQL 不可用时会自动降级）
    conversation_store = ConversationStore(config)
    if session_id is None:
        session_id = ConversationStore.new_session_id()

    print("\n" + "=" * 60)
    print("欢迎使用 med_rag 医疗知识问答系统！")
    print("=" * 60)
    print(f"会话 ID: {session_id}")
    print("输入 'help' 查看帮助，输入 'exit' / 'quit' / 'q' 退出。")

    while True:
        try:
            print("\n" + "-" * 60)
            question = input("请输入问题: ").strip()

            if question.lower() in ("exit", "quit", "q"):
                print("\n感谢使用 med_rag，再见！")
                break
            if question.lower() == "help":
                print_help()
                continue
            if not question:
                continue

            source_filter = input("请输入来源过滤（直接回车表示不过滤）: ").strip() or None

            print("\n正在生成答案...")
            start = time.time()
            answer = rag.generate_answer(
                question, source_filter=source_filter
            )
            elapsed = time.time() - start

            print("\n" + "=" * 60)
            print(f"【问题】{question}")
            if source_filter:
                print(f"【过滤】{source_filter}")
            print(f"【回答】{answer}")
            print(f"【耗时】{elapsed:.2f} 秒")
            print("=" * 60)

            # 持久化会话历史（对齐 EduRag update_session_history）
            conversation_store.update_session_history(session_id, question, answer)

        except KeyboardInterrupt:
            print("\n\n检测到中断，正在退出...")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请重试或输入 'help' 查看帮助。")


def main() -> None:
    """主函数：解析参数并进入对应模式。"""
    parser = argparse.ArgumentParser(description="med_rag 医疗知识问答系统")
    parser.add_argument(
        "--data-processing",
        action="store_true",
        help="进入数据处理模式（离线清洗、分块、向量化并写入 Milvus）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="待处理的本地文档目录（默认使用 config.CLEAN_MD_DIR）",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="指定会话 ID（用于保存历史记录）",
    )
    args = parser.parse_args()

    config = Config()

    if args.data_processing:
        process_data_mode(config, args.data_dir)
    else:
        query_mode(config, args.session_id)


if __name__ == "__main__":
    main()
