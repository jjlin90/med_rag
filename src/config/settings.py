import os

# 修复 Windows 上 Intel OpenMP 运行时（libiomp5md.dll）被 torch / FlagEmbedding 与
# numpy 的 MKL 重复加载，导致进程直接崩溃（0xC0000005 访问冲突，无 Python 堆栈）。
# 必须在任何 torch / numpy / FlagEmbedding 导入之前设置本变量，故放在本文件最顶部。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import logging
import torch
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Config:

    def __init__(self):
        # ===================== 项目基础路径配置 =====================
        # 当前文件：src/config/settings.py
        self.BASE_DIR = Path(__file__).parent.parent.parent
        load_dotenv(self.BASE_DIR / ".env")

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

        # ===================== 检索、重排权重参数（对齐 EduRag 双通道架构） =====================
        # 混合检索加权融合权重（WeightedRanker）：sparse 侧重关键词精确命中，dense 侧重语义相似
        self.SPARSE_WEIGHT = 0.7   # 稀疏（词权）权重，EduRag 对齐值
        self.DENSE_WEIGHT = 1.0    # 稠密（语义）权重，EduRag 对齐值

        # 检索阶段参数
        self.TOP_K_RETRIEVE = 16       # 混合检索召回量（粗排，取多一点给精排留余地）
        self.TOP_K_CHILDREN = 5        # Small-to-Big：子块召回数（400字粒度，精细定位）
        self.TOP_K_RERANK = 2          # CrossEncoder 精排最终输出数（EduRag: Top-2 父块）

        # FAQ 快通道参数（对齐 EduRag：softmax 归一化后阈值）
        self.FAQ_NORMALIZED_THRESHOLD = 0.85   # BM25 softmax 归一化后的命中阈值 [0,1]
        self.FAQ_CACHE_TTL = 3600              # FAQ Redis 缓存秒数（1小时）

        # ===================== 降级策略（Degradation Policy） =====================
        # 原则：降级路径必须比主路径「更安全」，而不是「更粗糙」。
        # 粒度的退化（400 字子块 → 2000 字父块）属于质量下降，在医疗场景下
        # 会稀释 LLM 注意力、翻倍 token，且掩盖数据问题，因此不提供开关，永久禁止。

        # L1：子块召回为空时，放开子块过滤重查；命中父块后在内存中切成子块再返回。
        #     输出粒度仍是 400 字子块，只是检索入口从「子块层」换成「父块层」，
        #     输出保持子块粒度，质量与耗时单独验证，默认开启。
        self.ENABLE_CHILD_FILTER_FALLBACK = True

        # L2：L1 仍为空时，是否允许交给 LLM 自由作答。
        #     医疗场景默认 False —— 检索不到依据时让 LLM 用参数知识硬答，
        #     是整条链路上风险最高的行为（会编造且语气权威）。改为返回安全拒答话术。
        self.ALLOW_LLM_WHEN_NO_CONTEXT = False

        # 单调递增的降级水位告警阈值：单次查询降级到该级别即打 WARNING，便于接入告警。
        self.DEGRADE_ALERT_LEVEL = 1          # 0=不打点 1=L1及以上告警 2=仅L2告警

        # ===================== 设备全局配置 =====================
        # 按当前 PyTorch 与硬件运行时检测；无可用 CUDA 时使用 CPU。
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if self.DEVICE == "cpu":
            logger.warning(
                "torch 未检测到可用 CUDA，使用 CPU 推理（torch=%s，构建CUDA=%s）。"
                "GPU 环境请核对硬件、驱动和 PyTorch 构建，耗时在当前设备实测。",
                torch.__version__, torch.version.cuda,
            )
        else:
            logger.info(f"使用 GPU 推理: {torch.cuda.get_device_name(0)}")

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
        # 模型名优先读取 LLM_MODEL_NAME；缺省 qwen-plus 与 .env.example 一致。
        # 模型名称必须与 LLM_BASE_URL 所指供应商匹配。
        # 注意：模型在进程初始化时加载，切换后必须重启进程才生效。
        self.LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-plus")
        self.LLM_TEMPERATURE = 0.2
        self.LLM_MAX_TOKENS = 1024
        # grounding 硬约束（默认开启）：生成阶段强制"只基于检索上下文作答、不足即拒答"，
        # 抑制 LLM 参数化补刀导致的 faithfulness 下降。两轮全量配对复评实测：
        #   离线轮 glm-4.7 裁判 / 210 题：F 0.259→0.464（ΔF=+0.21），AR 0.538→0.470（ΔAR=-0.07）
        #   线上轮 glm-4.5-air 裁判 / 207 题有效（真实服务链路重生成）：F 0.713→0.964（ΔF=+0.25），AR 0.841→0.698（ΔAR=-0.14）
        # 注：早期 30 题抽样曾估 ΔF=+0.42，系小样本方差大所致，已被全量修正。
        # 置 LLM_GROUNDING=false 可切回宽松模式，用于复现"无 grounding vs 有 grounding"的 A/B 对比。
        self.LLM_GROUNDING = os.getenv("LLM_GROUNDING", "true").lower() in ("1", "true", "yes", "on")

        # ===================== LLM密钥&接口地址（从.env加载） =====================
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")
        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

        # ===================== MySQL 配置（FAQ 问答库） =====================
        # 凭据从环境变量或本地 .env 读取，不把实际密码写入源码。
        self.MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
        self.MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
        self.MYSQL_USER = os.getenv("MYSQL_USER", "root")
        self.MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
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
