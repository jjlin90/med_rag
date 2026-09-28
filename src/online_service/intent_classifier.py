"""
Intent Classifier Module
BERT意图识别（通用/医疗咨询）
"""

import logging
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

from ..config.settings import Config

logger = logging.getLogger(__name__)

class IntentClassifier:
    """BERT意图分类器"""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE

        # 分类配置
        self.labels = {0: 'general', 1: 'medical'}  # 通用知识，医疗咨询
        self.label_names = ['general', 'medical']

        # 优先加载正式训练目录，其次备份；缺少分类权重时明确失败，不使用随机分类头。
        self.model_dir = config.BASE_DIR / "src/models/bert_query_classifier"
        # 若目标目录不存在但存在 _bak 备份目录，则使用备份目录
        if not self.model_dir.exists():
            backup_dir = config.BASE_DIR / "src/models/bert_query_classifier_bak"
            if backup_dir.exists():
                self.model_dir = backup_dir

        # 初始化模型和tokenizer
        self.model = None
        self.tokenizer = None

        # 初始化
        self._init_model()

    def _init_model(self):
        """初始化BERT模型"""
        try:
            # 加载tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(str(self.model_dir))

            # 检查是否有训练好的模型（同时兼容 .bin 与 .safetensors 两种权重格式）
            has_trained_weights = (
                (self.model_dir / "pytorch_model.bin").exists()
                or (self.model_dir / "model.safetensors").exists()
            )
            if has_trained_weights:
                logger.info("Loading pre-trained BERT model...")
                self.model = BertForSequenceClassification.from_pretrained(str(self.model_dir))
            else:
                raise RuntimeError(
                    'Trained intent classifier weights are missing; '
                    'run scripts/train_intent.py before serving requests. '
                    'An untrained classification head must not route medical queries.'
                )

            # 移动到设备
            self.model.to(self.device)

            # 设置为评估模式
            self.model.eval()

            logger.info("BERT model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {str(e)}")
            raise

    def predict(self, query: str) -> Dict[str, Any]:
        """
        预测查询意图

        Args:
            query: 用户查询

        Returns:
            包含预测结果的字典
        """
        if not self.model or not self.tokenizer:
            return {'error': 'Model not initialized'}

        try:
            # 预处理
            inputs = self._preprocess_text(query)

            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 获取预测结果
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

            result = {
                'intent': self.labels[pred_idx],
                'confidence': confidence,
                'probabilities': {
                    self.labels[i]: probs[0][i].item()
                    for i in range(len(self.labels))
                },
                'raw_query': query
            }

            logger.info(f"Intent prediction: {result['intent']} (confidence: {confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Intent prediction failed: {str(e)}")
            return {'error': str(e)}

    def predict_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        批量预测查询意图

        Args:
            queries: 查询列表

        Returns:
            预测结果列表
        """
        results = []

        for query in queries:
            result = self.predict(query)
            results.append(result)

        return results

    def _preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """预处理文本"""
        # 编码文本
        encoding = self.tokenizer(
            text,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # 移动到设备
        for key, value in encoding.items():
            encoding[key] = value.to(self.device)

        return encoding

    def train(self, training_data: List[Dict], eval_data: Optional[List[Dict]] = None,
              output_dir: Optional[Path] = None):
        """
        训练BERT模型

        Args:
            training_data: 训练数据，格式为 [{'text': str, 'label': int}, ...]
            eval_data: 验证数据，格式同 training_data（可选；提供后启用 epoch 级评估与早停）
            output_dir: 模型保存路径（默认保存到模型目录）
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not initialized")
            return False

        try:
            # 准备数据集
            train_dataset = IntentDataset(training_data, self.tokenizer)
            eval_dataset = IntentDataset(eval_data, self.tokenizer) if eval_data else None
            has_eval = eval_dataset is not None

            out_dir = output_dir or self.model_dir

            # 准备训练参数
            training_args = TrainingArguments(
                output_dir=str(out_dir),
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=str(out_dir / "logs"),
                logging_steps=50,
                evaluation_strategy="epoch" if has_eval else "no",
                save_strategy="epoch" if has_eval else "no",
                load_best_model_at_end=has_eval,
                metric_for_best_model="eval_loss" if has_eval else None,
                greater_is_better=False,
                save_total_limit=1,
                seed=42
            )

            # 创建Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics if has_eval else None,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if has_eval else None
            )

            # 开始训练
            logger.info("Starting BERT training...")
            trainer.train()

            # 保存模型
            trainer.save_model(str(out_dir))
            self.tokenizer.save_pretrained(str(out_dir))

            logger.info("BERT training completed")
            return True

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False

    @staticmethod
    def _compute_metrics(eval_pred):
        """计算验证指标（准确率）"""
        import numpy as np
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    def evaluate(self, test_data: List[Dict]) -> Dict[str, Any]:
        """评估模型性能"""
        if not self.model or not self.tokenizer:
            return {'error': 'Model not initialized'}

        try:
            # 准备数据集
            test_dataset = IntentDataset(test_data, self.tokenizer)

            # 创建Trainer用于评估
            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer
            )

            # 评估
            results = trainer.evaluate(test_dataset)

            # 计算准确率等指标
            predictions = trainer.predict(test_dataset)
            pred_labels = np.argmax(predictions.predictions, axis=1)
            true_labels = test_dataset.labels

            # 计算分类报告
            from sklearn.metrics import classification_report
            report = classification_report(true_labels, pred_labels, target_names=self.label_names)

            evaluation_result = {
                'loss': results['eval_loss'],
                'accuracy': np.mean(pred_labels == true_labels),
                'classification_report': report,
                'predictions': predictions.predictions.tolist(),
                'label_ids': true_labels.tolist()
            }

            logger.info("Model evaluation completed")
            return evaluation_result

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {'error': str(e)}

    def save_training_data(self, data: List[Dict], output_path: Path):
        """保存训练数据"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Training data saved to {output_path}")

    def load_training_data(self, input_path: Path) -> List[Dict]:
        """加载训练数据"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Training data loaded from {input_path}")
        return data

class IntentDataset(torch.utils.data.Dataset):
    """意图分类数据集"""

    def __init__(self, data: List[Dict], tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        # 提取标签
        self.labels = [item['label'] for item in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 编码文本
        encoding = self.tokenizer(
            item['text'],
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # 移除batch维度
        for key, value in encoding.items():
            encoding[key] = value.squeeze()

        # 添加标签
        encoding['labels'] = torch.tensor(item['label'])

        return encoding

class IntentClassifierDataGenerator:
    """意图分类器数据生成器"""

    @staticmethod
    def generate_sample_data(num_samples: int = 1000) -> List[Dict]:
        """生成示例训练数据"""
        data = []

        # 通用知识示例
        general_patterns = [
            "什么是人工智能？",
            "怎么用Python写程序？",
            "什么是机器学习？",
            "如何学习英语？",
            "什么是云计算？",
            "什么是区块链技术？",
            "如何保护个人信息安全？",
            "什么是物联网？",
            "怎么做好时间管理？",
            "什么是虚拟现实？"
        ]

        # 医疗咨询示例
        medical_patterns = [
            "头痛应该怎么办？",
            "发烧38度需要吃药吗？",
            "血压140/90正常吗？",
            "感冒了吃什么药？",
            "胃痛是什么原因？",
            "糖尿病有什么症状？",
            "如何预防心脏病？",
            "失眠应该怎么治？",
            "过敏反应怎么处理？",
            "血压偏高需要注意什么？"
        ]

        # 生成通用知识数据
        for i in range(num_samples // 2):
            data.append({
                'text': np.random.choice(general_patterns),
                'label': 0
            })

        # 生成医疗咨询数据
        for i in range(num_samples // 2):
            data.append({
                'text': np.random.choice(medical_patterns),
                'label': 1
            })

        # 打乱数据
        np.random.shuffle(data)

        return data

    @staticmethod
    def augment_data(data: List[Dict], augment_factor: int = 2) -> List[Dict]:
        """数据增强"""
        augmented = []

        # 同义词替换
        synonyms = {
            "头痛": ["头疼", "头部疼痛", "头部不适"],
            "发烧": ["发热", "体温升高", "高烧"],
            "胃痛": ["胃部疼痛", "胃疼", "腹部疼痛"],
            "感冒": ["伤风", "上呼吸道感染", "流感"],
            "失眠": ["睡眠不足", "睡不着", "睡眠障碍"]
        }

        for item in data:
            augmented.append(item)  # 原始数据

            # 生成增强数据
            for _ in range(augment_factor - 1):
                text = item['text']
                for original, synonym_list in synonyms.items():
                    if original in text:
                        text = text.replace(original, np.random.choice(synonym_list))
                        break

                augmented.append({
                    'text': text,
                    'label': item['label']
                })

        return augmented
