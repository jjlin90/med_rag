import os

# 修复 Windows 上 Intel OpenMP 运行时（libiomp5md.dll）被 torch / FlagEmbedding 与
# numpy 的 MKL 重复加载，导致进程直接崩溃（0xC0000005 访问冲突，无 Python 堆栈）。
# 必须在任何 torch / numpy / FlagEmbedding 导入之前设置本变量，故放在本文件最顶部。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import torch
from dotenv import load_dotenv


class Config:

    def __init__(self):
        # ===================== 项目基础路径配置 =====================
        # 当前文件：src/config/settings.py
        self.BASE_DIR = Path(__file__).parent.parent.parent

        # 原始MSD解压资源目录
        self.RAW_MSD_DIR = self.BASE_DIR / "data/raw/MSDZHConsumerMedicalTopics"
        # 清洗后Markdown输出目录
        self.CLEAN_MD_DIR = self.BASE_DIR / "data/clean_md"
        # 分块结构化JSON保存路径
        self.CHUNK_SAVE_PATH = self.BASE_DIR / "data/split_docs/docs.json"
        # Milvus本地持久化缓存
        self.MILVUS_CACHE_DIR = self.BASE_DIR / "cache/milvus_local"
        # Embedding模型缓存目录
        self.EMBED_CACHE_DIR = self.BASE_DIR / "cache/embedding_cache"
        # 是否把生成的向量额外存一份 JSON 缓存（embeddings.json）。
        # 注意：该缓存全项目无任何读取方（load_embeddings 无人调用），
        # 且 20816×1024 维写 JSON 约 380MB、又慢又占内存，默认关闭。
        # 如需调试向量可临时改为 True。
        self.SAVE_EMBEDDING_CACHE = False
        # RAG评估测试集路径
        self.TEST_QUERY_PATH = self.BASE_DIR / "data/test_query/test_qa.json"

        # ===================== 父子分块超参数 =====================
        self.PARENT_CHUNK_SIZE = 2000  # 父块完整上下文长度
        self.CHILD_CHUNK_SIZE = 400  # 检索细分子块长度
        self.CHUNK_OVERLAP = 60  # 文本重叠字符

        # ===================== 检索、重排权重参数 =====================
        self.TOP_K_RETRIEVE = 8
        self.TOP_K_RERANK = 4
        self.BM25_WEIGHT = 0.4
        self.DENSE_WEIGHT = 0.6

        # ===================== 设备全局配置 =====================
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # ===================== Embedding模型配置 =====================
        self.EMBED_MODEL_NAME = "BGE-M3"
        self.EMBED_DEVICE = self.DEVICE
        self.EMBED_DIM = 1024

        # ===================== 重排模型配置 =====================
        self.RERANK_MODEL_NAME = "BGE-reranker-large"
        self.RERANK_DEVICE = self.DEVICE

        # ===================== Milvus向量库配置 =====================
        self.MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
        self.MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
        self.MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "MED")
        self.MILVUS_CONN_ALIAS = os.getenv("MILVUS_CONN_ALIAS", "MED")
        self.MILVUS_COLLECTION_NAME = "med_msd_consumer_chunk"
        self.MILVUS_INDEX_TYPE = "IVF_FLAT"
        # IVF_FLAT 建索引参数
        self.MILVUS_NLIST = 256
        # 查询阶段参数（检索nprobe个聚类簇）
        self.MILVUS_NPROBE = 16

        # ===================== LLM生成参数 =====================
        # DashScope 兼容模式可用模型：qwen-turbo / qwen-plus / qwen-max 等。
        # "qwen2" 已不可识别，会报 404 model_not_found，故默认用 qwen-turbo。
        self.LLM_MODEL_NAME = "qwen-turbo"
        self.LLM_TEMPERATURE = 0.2
        self.LLM_MAX_TOKENS = 1024

        # ===================== LLM密钥&接口地址（从.env加载） =====================
        load_dotenv(self.BASE_DIR / ".env")
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")
        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

        # ===================== MySQL 配置（FAQ 问答库） =====================
        # 本地开发库，凭据随代码提交（与 .env 里的 LLM_API_KEY 不同，MySQL 是本地库，
        # 提交不算泄露）；若部署到别的环境，用同名环境变量覆盖即可。
        # 注意：这里的密码是占位值，请改成你本机 MySQL 实际密码后再提交/运行。
        self.MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
        self.MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
        self.MYSQL_USER = os.getenv("MYSQL_USER", "root")
        self.MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")
        self.MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "medical_rag")

        # ===================== Redis 配置（缓存，可选） =====================
        # Redis 是可选依赖：不可用时 CacheManager 会自动降级为无缓存模式，不影响主流程。
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        self.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
        self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    def init_dirs(self):
        """自动创建全部缺失文件夹，运行前调用一次，避免路径不存在报错"""
        dir_list = [
            self.RAW_MSD_DIR, self.CLEAN_MD_DIR, self.CHUNK_SAVE_PATH.parent,
            self.MILVUS_CACHE_DIR, self.EMBED_CACHE_DIR,
            self.TEST_QUERY_PATH.parent
        ]
        for d in dir_list:
            d.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # 全局单例配置，项目其他文件统一导入使用
    cfg = Config()
    print(cfg.CHUNK_OVERLAP)
