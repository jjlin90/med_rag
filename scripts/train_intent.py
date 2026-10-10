"""
微调意图分类 BERT 模型（对齐 EduRAG_V7.5 讲义 4.5 配方）。

- 底座：本地 bert-base-chinese（避免联网，且不使用损坏的 _bak 权重）
- 数据：data/intent_train/train.json + val.json（medical=1 / general=0，各约 2500）
- 配方：num_train_epochs=3, per_device_batch_size=8, warmup_steps=500,
        weight_decay=0.01, evaluation_strategy=epoch, load_best_model_at_end=True
- 输出：src/models/bert_query_classifier（正式目录，覆盖 _bak 回退逻辑）

运行：python scripts/train_intent.py
"""
import json
import sys
import time
import logging
from pathlib import Path

# 先执行项目的 Windows 原生依赖初始化与 OpenMP 环境配置，再导入计算库。
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from src.config.settings import Config

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.online_service.intent_classifier import IntentDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_intent")

TRAIN_FILE = BASE_DIR / "data/intent_train/train.json"
VAL_FILE = BASE_DIR / "data/intent_train/val.json"
BASE_MODEL = BASE_DIR / "src/models/bert-base-chinese"      # 预训练底座
OUT_DIR = BASE_DIR / "src/models/bert_query_classifier"     # 正式保存目录


def load_data(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    texts = [d["text"] for d in data]
    labels = [int(d["label"]) for d in data]
    return texts, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def main():
    if not BASE_MODEL.exists():
        logger.error(f"底座模型不存在: {BASE_MODEL}")
        sys.exit(1)
    if not TRAIN_FILE.exists():
        logger.error(f"训练集不存在: {TRAIN_FILE}，请先运行 scripts/build_intent_data.py")
        sys.exit(1)

    cfg = Config()
    device = torch.device(cfg.DEVICE)
    logger.info(f"使用设备: {device}")

    # 1. 加载分词器与底座（num_labels=2，随机初始化分类头）
    tokenizer = BertTokenizer.from_pretrained(str(BASE_MODEL))
    model = BertForSequenceClassification.from_pretrained(
        str(BASE_MODEL), num_labels=2
    )
    model.to(device)

    # 2. 加载数据（IntentDataset 内部负责 tokenize 原始文本）
    train_texts, train_labels = load_data(TRAIN_FILE)
    val_texts, val_labels = load_data(VAL_FILE) if VAL_FILE.exists() else ([], [])

    train_ds = IntentDataset(
        [{"text": t, "label": l} for t, l in zip(train_texts, train_labels)], tokenizer
    )
    val_ds = IntentDataset(
        [{"text": t, "label": l} for t, l in zip(val_texts, val_labels)], tokenizer
    ) if val_texts else None

    # 3. 训练参数（对齐讲义）
    training_args = TrainingArguments(
        output_dir=str(OUT_DIR / "tmp"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(OUT_DIR / "logs"),
        logging_steps=50,
        evaluation_strategy="epoch" if val_ds else "no",
        save_strategy="epoch" if val_ds else "no",
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss" if val_ds else None,
        greater_is_better=False,
        save_total_limit=1,
        fp16=True,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if val_ds else None,
    )

    # 4. 训练
    logger.info("开始训练意图分类 BERT 模型...")
    t0 = time.time()
    trainer.train()
    logger.info(f"训练完成，耗时 {time.time() - t0:.1f}s")

    # 5. 保存最佳模型到正式目录
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    logger.info(f"模型已保存到: {OUT_DIR}")

    # 6. 在验证集评估并打印报告
    if val_ds:
        preds = trainer.predict(val_ds)
        y_pred = np.argmax(preds.predictions, axis=-1)
        target_names = ["general(通用)", "medical(医疗)"]
        logger.info("验证集分类报告:\n" + classification_report(
            val_labels, y_pred, target_names=target_names, digits=4
        ))
        logger.info("混淆矩阵:\n" + str(confusion_matrix(val_labels, y_pred)))

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
