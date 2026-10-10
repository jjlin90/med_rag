"""
LLM Generator Module
Prompt组装、调用LLM生成回答
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional

from ..config.settings import Config

logger = logging.getLogger(__name__)


def format_knowledge_document(document: Dict[str, Any]) -> str:
    """The same evidence text is used by generation and live evaluation."""
    title = document.get('source')
    heading = f'【文档标题：{title}】\n' if title and title != 'unknown' else ''
    return heading + (document.get('content') or '')


def evidence_paragraphs(context: str) -> List[str]:
    return [part.strip() for part in context.split('\n\n') if part.strip()]


def indexed_knowledge(context: str) -> str:
    return '\n\n'.join(f'【证据{i}】\n{part}' for i, part in enumerate(evidence_paragraphs(context),1))


def validated_review_answer(raw: str, context: str) -> Optional[str]:
    """Reject malformed reviews and references outside the supplied evidence."""
    try:
        payload = json.loads(raw)
        answer = payload['final_answer']
        checks = payload['checks']
        if not isinstance(answer, str) or not answer.strip() or not isinstance(checks, list) or not checks:
            return None
        normalized_context = ''.join(context.split())
        paragraphs = evidence_paragraphs(context)
        for check in checks:
            if not isinstance(check, dict) or type(check.get('supported')) is not bool:
                return None
            claim = check.get('claim')
            if not isinstance(claim, str) or not claim.strip():
                return None
            if 'evidence_ids' in check:
                identifiers = check['evidence_ids']
                if not isinstance(identifiers,list) or any(type(i) is not int or i < 1 or i > len(paragraphs) for i in identifiers):
                    return None
                if check['supported'] and not identifiers:
                    return None
                continue
            evidence = check.get('evidence')
            if not isinstance(evidence, str):
                return None
            quote = ''.join(evidence.split())
            if check['supported'] and (not quote or quote not in normalized_context):
                return None
        return answer.strip()
    except (ValueError, KeyError, TypeError):
        return None

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
2. 优先直接回答问题。首句必须直接给出结论或核心信息，禁止以“根据提供的资料”“资料显示”“关于这个问题”等元话语开头，也不要复述用户问题。
3. 在拒答前，必须先逐项检查【相关知识】中是否存在任何能回应问题的事实。能回答一部分时，先直接给出该部分：诊断类问题可说明已知症状或关联但不得确诊；预后类问题可说明已知病程或风险；治疗类问题可列出资料明确给出的治疗。不得因为最终结论缺少依据而丢掉这些有用信息。
4. 只有【相关知识】完全没有任何能回应问题的事实时才简短拒答。拒答必须点明缺失的具体主题，例如“相关知识未说明抽动对上学的影响，因此无法判断。”禁止只写“现有资料不足以回答这个问题”这类没有具体对象的模板句，并立即停止，不得追加外部知识。
5. 回答中的每一条事实陈述都必须能在【相关知识】中找到对应依据，尤其不得自行补充数值、剂量、时间间隔、药名、病因或诊疗建议。
6. 使用自然、简洁的中文。是非问题先回答“是/不是/可能/不一定”；方法类问题先列出方法；时间类问题先给时间。通常控制在 1～3 个短段落，仅在确有多个要点时使用列表。
7. 不要添加“仅供参考”“建议咨询医生”等通用免责声明，除非【相关知识】明确包含该建议，或问题涉及需要立即处理的危险信号。"""

    _GROUNDING_REVIEW_SYS = """你是医疗知识回答的事实核查编辑，只输出核查后的答案正文。
逐项核对初稿中的每一条医学陈述与【相关知识】，保留可支持且与用户问题相关的内容，删除无依据内容并修正错误。不得补充外部医学知识。
特别检查：疾病和亚型、人群及部位是否相同；一般规律是否被写成个体诊断；公斤与体重百分比是否混淆；时间指起效、持续、重复治疗还是病程；剂量、比例、单位和适用条件是否忠于原文。不能假定用户未给出的基准体重、检验值或病情，也不能从疾病慢性推断用药必须长期。
没有某种病因的证据不能写成排除该病因；发现另一种病的相关症状不能写成此人患有该病或原怀疑不成立。逐句检查肯定和否定是否都有直接依据，禁止先作结论、后说无法判断。不能借助参数知识推导正文未给出的遗传传递规则、治疗疗程或新的医嘱。对于缺少起效时间等细节的问题，仍应保留正文明确支持的治疗作用或适用条件；不要将部分可回答的问题改成整题拒答。
先回答有依据的核心问题，再解释必要条件。删除无关疾病的治疗与过多背景；保留真实的条件和不确定性。证据只能回应一部分时提供该部分，并准确指出剩余缺口。知识完全没有相关事实时才简短说明具体缺失主题。不得把初稿当作事实依据。
输出JSON对象：{"checks":[{"claim":"医学陈述","evidence_ids":[1,2],"supported":true}],"final_answer":"最终答案正文"}。
逐项核查初稿中的关键医学陈述，supported检查整句话是否成立，尤其是“而非”“不是”“会”“必须”等判断词；只支持前半句、不支持后半句时整个claim为false。不能以未提及某个原因来排除该原因。
true项的evidence_ids必须引用实际给出的【证据编号】，所指原文须支持陈述完整含义；不要重写原文，也不要编造编号。没有直接证据的医学陈述标false，evidence_ids可为空，删除或修正。说明资料缺口的句子不列为医学事实true项。final_answer仅保留有依据且回应问题的内容；若个体原因未明确，先说明单凭症状不能确定，再介绍相关一般事实，不排除其他病因。检查至多12个关键陈述，核查记录不写进final_answer。"""

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
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 json_mode: bool = False) -> Optional[str]:
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
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                **({'response_format':{'type':'json_object'}} if json_mode else {})
            )

            if response.choices[0].finish_reason != 'stop':
                logger.error('LLM response incomplete: %s', response.choices[0].finish_reason)
                return None
            answer = (response.choices[0].message.content or '').strip()
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
        use_review = (system_prompt is None and getattr(self.config, 'LLM_GROUNDING', True)
                      and getattr(self.config, 'LLM_GROUNDING_REVIEW', False))
        if system_prompt is None:
            system_prompt = self._build_system_prompt()

        # 构建用户提示
        user_prompt = self._build_user_prompt(query, context, history)

        # 生成回答
        answer = self.generate(user_prompt, system_prompt)
        if answer and use_review:
            answer = self.review_grounded_answer(query, context, answer, history)
        return answer

    def review_grounded_answer(self, query: str, context: str, answer: str,
                               history: Optional[List[Dict]] = None) -> Optional[str]:
        """Review against original evidence. An unavailable review returns None."""
        prompt = self._build_user_prompt(query, indexed_knowledge(context), history)
        prompt += f"\n\n【待核查初稿（不是事实依据）】\n{answer}\n\n请核查并输出最终答案正文："
        raw = self.generate(prompt, self._GROUNDING_REVIEW_SYS, temperature=0.0,
                            max_tokens=3072,json_mode=True)
        answer = validated_review_answer(raw,context) if raw else None
        if not answer:
            logger.warning('Grounding review unavailable or quotations invalid')
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
                prompt += "\n【对话历史（仅用于理解指代，不是医学事实依据）】\n"
                for msg in history[-3:]:  # 只使用最近3轮历史
                    role = "用户" if msg['role'] == 'user' else "助手"
                    prompt += f"{role}: {msg['content']}\n"
            prompt += "\n请直接输出答案正文。能回答时首句给结论，不要提及‘资料’‘上下文’或回答依据；只有完全无相关事实时，才用一句包含具体缺失主题的话拒答："
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
