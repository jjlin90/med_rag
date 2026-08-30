"""
LLM Generator Module
Prompt组装、调用LLM生成回答
"""

import logging
import time
from typing import List, Dict, Any, Optional

from ..config.settings import Config

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

class LLMGenerator:
    """LLM生成器"""

    # 宽松模式系统提示（原始版：软约束，易被 LLM 参数化补刀绕过）
    _BASE_SYS = """你是一个专业的医疗智能助手，基于提供的医学知识回答用户问题。

回答要求：
1. 基于提供的上下文内容回答，不要编造信息
2. 回答要准确、专业、易于理解
3. 如果信息不足，明确说明
4. 对于医疗问题，强调仅供参考，不能替代专业医疗建议
5. 保持礼貌和专业的态度"""

    # 严格 grounding 模式系统提示（两轮全量配对复评验证：
    #   离线轮 glm-4.7 裁判 / 210 题：F 0.259→0.464（ΔF=+0.21）
    #   线上轮 glm-4.5-air 裁判 / 207 题有效：F 0.713→0.964（ΔF=+0.25））
    _GROUNDING_SYS = """你是一个严谨的医疗知识问答助手。你必须【严格只】基于下面【相关知识】中给出的内容来回答用户问题，遵守以下规则：

1. 只能使用【相关知识】中明确包含的信息。绝对不允许引入【相关知识】之外的任何医学知识、常识或个人推断。
2. 如果【相关知识】的内容不足以回答用户问题，或完全不相关，必须明确说明"根据提供的资料，无法回答该问题"，不要尝试用外部知识补充。
3. 回答中的每一条事实陈述都必须能在【相关知识】中找到对应依据。
4. 保持专业、简洁；涉及医疗建议时提醒"仅供参考，不能替代专业医疗建议"。"""

    def __init__(self, config: Config):
        self.config = config
        self.model_name = config.LLM_MODEL_NAME
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS

        # 初始化LLM客户端
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化LLM客户端"""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI package not installed")
            return

        try:
            # 初始化客户端
            self.client = OpenAI(
                api_key=self.config.LLM_API_KEY,
                base_url=self.config.LLM_BASE_URL
            )

            # 测试连接
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )

            logger.info(f"LLM client initialized successfully for model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self.client = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """
        生成回答

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大生成长度

        Returns:
            生成的回答
        """
        if not self.client:
            logger.error("LLM client not available")
            return None

        try:
            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # 生成回答
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM generated response ({len(answer)} characters)")
            return answer

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return None

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                        temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """
        流式生成回答

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大生成长度

        Yields:
            流式生成的文本块
        """
        if not self.client:
            logger.error("LLM client not available")
            yield None
            return

        try:
            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # 流式生成
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM streaming generation failed: {str(e)}")
            yield None

    def generate_with_context(self, query: str, context: str, history: Optional[List[Dict]] = None,
                             system_prompt: Optional[str] = None) -> Optional[str]:
        """
        基于上下文生成回答

        Args:
            query: 用户查询
            context: 检索上下文
            history: 对话历史
            system_prompt: 系统提示

        Returns:
            生成的回答
        """
        # 构建完整的提示
        if system_prompt is None:
            system_prompt = self._build_system_prompt()

        # 构建用户提示
        user_prompt = self._build_user_prompt(query, context, history)

        # 生成回答
        answer = self.generate(user_prompt, system_prompt)
        return answer

    def _build_system_prompt(self) -> str:
        """构建系统提示（grounding 开关切换严格/宽松）"""
        if getattr(self.config, "LLM_GROUNDING", True):
            return self._GROUNDING_SYS
        return self._BASE_SYS

    def _build_user_prompt(self, query: str, context: str, history: Optional[List[Dict]] = None) -> str:
        """构建用户提示（grounding 开启时强制"仅基于相关知识"）"""
        if getattr(self.config, "LLM_GROUNDING", True):
            prompt = f"""请仅基于以下【相关知识】回答用户问题，不要使用任何额外知识：

【相关知识】
{context}

【用户问题】
{query}
"""
            if history:
                prompt += "\n【对话历史】\n"
                for msg in history[-3:]:  # 只使用最近3轮历史
                    role = "用户" if msg['role'] == 'user' else "助手"
                    prompt += f"{role}: {msg['content']}\n"
            prompt += "\n请严格依据上述【相关知识】作答："
            return prompt

        # 宽松模式（原逻辑）
        prompt = f"""基于以下医学知识回答问题：

【相关知识】
{context}

【用户问题】
{query}
"""
        if history:
            prompt += "\n【对话历史】\n"
            for msg in history[-3:]:
                role = "用户" if msg['role'] == 'user' else "助手"
                prompt += f"{role}: {msg['content']}\n"
        prompt += "\n请根据提供的知识回答用户的问题："
        return prompt

    def generate_fallback_answer(self, query: str) -> str:
        """生成兜底回答"""
        fallback_prompt = f"""用户问了一个我不知道的问题，请礼貌地说明你无法回答，并建议咨询专业人士。

用户问题：{query}

回答："""

        answer = self.generate(fallback_prompt)
        return answer if answer else "抱歉，我无法回答这个问题，请咨询专业医生。"

    def validate_llm_connection(self) -> bool:
        """验证LLM连接"""
        if not self.client:
            return False

        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            end_time = time.time()

            latency = end_time - start_time
            logger.info(f"LLM connection validated, latency: {latency:.2f}s")
            return True

        except Exception as e:
            logger.error(f"LLM connection validation failed: {str(e)}")
            return False

    def get_llm_stats(self) -> Dict[str, Any]:
        """获取LLM统计信息"""
        stats = {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'client_available': self.client is not None,
            'model_type': 'chat'
        }

        if self.client:
            stats.update({
                'base_url': self.config.LLM_BASE_URL,
                'connection_status': self.validate_llm_connection()
            })

        return stats

    def estimate_token_count(self, text: str) -> int:
        """估算token数量"""
        # 简单估算：1个token约等于1.3个汉字或0.75个英文单词
        if text.isascii():
            return int(len(text) * 0.75)
        else:
            return int(len(text) / 1.3)