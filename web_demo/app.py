"""
医疗 RAG 问答系统 - Streamlit 前端

连接后端 FastAPI 服务（默认 http://localhost:8005）。

功能：
1. 智能问答（单一问答 / 多轮对话），检索策略由后端大模型自动判断。
2. 会话持久化：前端生成 session_id（写入 URL 参数，刷新/重开不丢失），
   后端把每一轮问答存入 MySQL conversations 表，刷新后可自动恢复历史。
3. RAG 评估：调用 /evaluate 端点，默认评估忠实度与答案相关性；
   整批均提供标准答案时，再评估上下文精确率与召回率。

用法：
    cd D:/pythonProject/med_rag
    web_demo/.venv/Scripts/python.exe -m streamlit run web_demo/app.py
"""

import time
import uuid
from datetime import datetime

import requests
import streamlit as st

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="医疗 RAG 问答系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- 常量与默认值 ----------------------
DEFAULT_API_URL = "http://localhost:8005"
CHAT_ENDPOINTS = {
    "单一问答": "/query",
    "多轮对话": "/chat",
}


# ---------------------- 会话状态初始化 ----------------------
def get_session_id() -> str:
    """获取/生成会话 ID，并镜像到 URL 参数以便刷新后恢复历史（对齐 EduRag）。"""
    if "session_id" not in st.session_state:
        existing = st.query_params.get("sid")
        if existing:
            sid = existing if isinstance(existing, str) else existing[0]
        else:
            sid = str(uuid.uuid4())
            st.query_params["sid"] = sid
        st.session_state["session_id"] = sid
    return st.session_state["session_id"]


def init_state():
    defaults = {
        "messages": [],          # 多轮对话历史
        "api_url": DEFAULT_API_URL,
        "last_error": None,
        "health": None,
        "stats": None,
        "history_loaded": False,  # 是否已尝试从后端恢复历史
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
SESSION_ID = get_session_id()


# ---------------------- 工具函数 ----------------------
def check_health(api_url: str) -> dict:
    """检查后端健康状态"""
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e), "services": {}}


def query_api(api_url: str, question: str, source_filter: str,
              use_cache: bool, session_id: str):
    """调用 /query 接口（检索策略由后端大模型自动判断，前端不再指定）"""
    payload = {
        "question": question,
        "source_filter": source_filter or None,
        "use_cache": use_cache,
        "session_id": session_id,
    }
    resp = requests.post(f"{api_url}/query", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def chat_api(api_url: str, messages: list, source_filter: str,
             use_cache: bool, session_id: str):
    """调用 /chat 接口"""
    payload = {
        "messages": messages,
        "source_filter": source_filter or None,
        "use_cache": use_cache,
        "session_id": session_id,
    }
    resp = requests.post(f"{api_url}/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def load_history(api_url: str, session_id: str) -> list:
    """从后端恢复最近会话历史（MySQL conversations 表），映射为 messages 格式。"""
    try:
        resp = requests.get(f"{api_url}/conversation/{session_id}", timeout=10)
        resp.raise_for_status()
        history = resp.json().get("history", [])
        messages = []
        for pair in history:
            messages.append({"role": "user", "content": pair.get("question", "")})
            messages.append({"role": "assistant", "content": pair.get("answer", "")})
        return messages
    except Exception:
        return []


def evaluate_api(api_url: str, items: list) -> dict:
    """调用 /evaluate 端点做 RAG 评估"""
    resp = requests.post(f"{api_url}/evaluate", json={"items": items}, timeout=300)
    resp.raise_for_status()
    return resp.json()


def render_sources(sources: list):
    """折叠展示引用来源"""
    if not sources:
        st.info("未使用医学知识库来源（可能来自 FAQ 或通用知识）")
        return

    with st.expander(f"📚 引用来源（{len(sources)} 条）", expanded=False):
        for i, src in enumerate(sources, 1):
            score = src.get("score", 0.0)
            content = src.get("content", "")
            meta = src.get("metadata", {})
            title = src.get("title") or meta.get("title") or meta.get(
                "doc_name") or f"来源 {i}"

            st.markdown(f"**{i}. {title}**  `score={score:.3f}`")
            st.markdown(
                f"> {content[:500]}{'…' if len(content) > 500 else ''}")
            if meta:
                st.caption(f"元数据：{meta}")
            st.divider()


# ---------------------- 侧边栏 ----------------------
with st.sidebar:
    st.title("⚙️ 设置")

    api_url = st.text_input(
        "后端 API 地址",
        value=st.session_state["api_url"],
        help="FastAPI 服务地址，例如 http://localhost:8005",
    )
    st.session_state["api_url"] = api_url.strip().rstrip("/")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 刷新健康", use_container_width=True):
            st.session_state["health"] = check_health(
                st.session_state["api_url"])
    with col2:
        if st.button("📊 获取统计", use_container_width=True):
            try:
                st.session_state["stats"] = requests.get(
                    f"{st.session_state['api_url']}/stats", timeout=10).json()
            except Exception as e:
                st.session_state["stats"] = {"error": str(e)}

    # 会话管理
    st.divider()
    st.caption(f"会话 ID：`{SESSION_ID[:8]}…`（已写入网址，刷新不丢失）")
    if st.button("🆕 开启新对话", use_container_width=True):
        new_sid = str(uuid.uuid4())
        st.query_params["sid"] = new_sid
        st.session_state["session_id"] = new_sid
        st.session_state["messages"] = []
        st.session_state["history_loaded"] = True
        st.rerun()

    # 健康状态面板
    health = st.session_state.get("health")
    if health is None:
        health = check_health(st.session_state["api_url"])
        st.session_state["health"] = health

    st.subheader("🏥 服务健康")
    if health.get("status") == "healthy":
        st.success("后端运行正常")
    elif health.get("status") == "degraded":
        st.warning("后端降级运行（部分服务不可用）")
    else:
        st.error(f"无法连接后端：{health.get('error', '未知错误')}")

    services = health.get("services", {})
    if services:
        cols = st.columns(2)
        for idx, (name, ok) in enumerate(services.items()):
            with cols[idx % 2]:
                st.markdown(
                    f"{'🟢' if ok else '🔴'} **{name}**: {'正常' if ok else '异常'}")

    # 统计信息
    stats = st.session_state.get("stats")
    if stats:
        st.subheader("📈 系统统计")
        if "error" in stats:
            st.error(stats["error"])
        else:
            faq_stats = stats.get("faq_stats", {})
            if isinstance(faq_stats, dict):
                st.metric("FAQ 总数", faq_stats.get("total", "—"))
                st.metric("最近 7 天新增", faq_stats.get("recent_7_days", "—"))
                st.metric("BM25 就绪",
                          "是" if faq_stats.get("bm25_ready") else "否")

    st.divider()
    st.caption("医疗 RAG 前端 v1.0 · "
               f"最后检查：{datetime.now().strftime('%H:%M:%S')}")

# ---------------------- 页面路由 ----------------------
page = st.sidebar.radio("功能页面", ["智能问答", "RAG 评估"], horizontal=True)

# ===================== 智能问答 =====================
if page == "智能问答":
    st.title("🩺 医疗 RAG 问答系统")
    st.caption("基于 FastAPI 后端 · FAQ 优先 · RAG 增强 · 支持多轮对话 · 会话持久化")

    # 后端不可达提示
    if st.session_state["health"].get("status") not in ("healthy", "degraded"):
        st.error(f"无法连接到后端 API：{st.session_state['api_url']}。"
                 "请先运行 `python scripts/run_api.py` 启动后端服务，再刷新本页面。")

    # 首次进入：尝试从后端恢复历史（仅当本地无历史且未手动开新会话）
    if (not st.session_state["history_loaded"]
            and not st.session_state["messages"]):
        restored = load_history(st.session_state["api_url"], SESSION_ID)
        if restored:
            st.session_state["messages"] = restored
            st.toast(f"已恢复 {len(restored)//2} 轮历史对话", icon="💾")
        st.session_state["history_loaded"] = True

    mode = st.radio(
        "选择交互模式",
        options=list(CHAT_ENDPOINTS.keys()),
        horizontal=True,
        help="单一问答只处理当前问题；多轮对话会携带历史上下文调用 /chat。",
    )

    with st.expander("🔧 高级参数", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            source_filter = st.text_input("来源过滤（source_filter）",
                                          value="",
                                          help="留空表示不过滤")
        with col_b:
            use_cache = st.checkbox("使用缓存", value=True)
        st.info("检索策略现在由后端大模型根据问题自动判断"
                 "（direct / hyde / subquery / backtracking），无需手动选择。")

    # 多轮对话模式下展示历史
    if mode == "多轮对话":
        for msg in st.session_state["messages"]:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])

    question = st.chat_input("请输入您的问题，例如：1 型糖尿病 (DM)")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        if mode == "多轮对话":
            st.session_state["messages"].append({
                "role": "user",
                "content": question,
                "timestamp": time.time(),
            })

        with st.chat_message("assistant"):
            with st.spinner("正在思考中…"):
                try:
                    if mode == "单一问答":
                        result = query_api(
                            st.session_state["api_url"],
                            question,
                            source_filter,
                            use_cache,
                            SESSION_ID,
                        )
                    else:
                        result = chat_api(
                            st.session_state["api_url"],
                            st.session_state["messages"],
                            source_filter,
                            use_cache,
                            SESSION_ID,
                        )
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": result.get("answer", ""),
                            "timestamp": time.time(),
                        })

                    answer = result.get("answer", "")
                    st.markdown(answer)

                    if result.get('degraded'):
                        st.warning('本轮处于备用检索或服务降级状态，请结合回答提示和所附资料阅读。')

                    meta_cols = st.columns(4)
                    meta_cols[0].metric("意图", result.get("intent", "—"))
                    meta_cols[1].metric("策略", result.get("strategy", "—"))
                    meta_cols[2].metric("置信度",
                                        f"{result.get('confidence', 0):.3f}")
                    meta_cols[3].metric(
                        "响应时间",
                        f"{result.get('response_time', 0):.3f}s",
                    )

                    cache_badge = "✅ 命中缓存" if result.get(
                        "used_cache") else "❌ 未命中缓存"
                    st.caption(cache_badge)

                    render_sources(result.get("sources", []))

                    # 刷新健康状态以反映缓存命中
                    st.session_state["health"] = check_health(
                        st.session_state["api_url"])

                except requests.exceptions.ConnectionError:
                    st.error("连接后端失败。请确认已运行 `python scripts/run_api.py`，"
                             "且地址/端口设置正确。")
                except requests.exceptions.Timeout:
                    st.error("后端响应超时，请稍后重试。")
                except Exception as e:
                    st.error(f"请求出错：{e}")

# ===================== RAG 评估 =====================
else:
    st.title("📊 RAG 评估")
    st.caption("对一批问题调用真实 RAG 管线生成答案，再用大模型裁判打分："
               "无标准答案时 Ragas 评估忠实度与答案相关性；结果会标明评估引擎与有效条数。")

    st.info("评估会依次对每题调用后端 /query 生成答案与上下文，"
            "再调用 /evaluate 打分。请耐心等待（耗时取决于问题数量与 LLM 速度）。")

    default_questions = "\n".join([
        "1 型糖尿病 (DM) 是什么？",
        "高血压患者在日常生活中需要注意什么？",
        "感冒了需要吃抗生素吗？",
        "空腹血糖的正常范围是多少？",
        "服用退烧药后多久可以重复用药？",
    ])
    questions_text = st.text_area(
        "评测问题（每行一条）",
        value=default_questions,
        height=180,
        help="每行一个问题；评估结果会保存到 data/test_query/eval_report.json",
    )

    if st.button("🚀 开始评估", use_container_width=True, type="primary"):
        questions = [q.strip() for q in questions_text.splitlines() if q.strip()]
        if not questions:
            st.warning("请至少输入一个问题。")
        else:
            items = []
            progress = st.progress(0, text="生成答案中…")
            for i, q in enumerate(questions, 1):
                try:
                    r = query_api(
                        st.session_state["api_url"], q, "", False, SESSION_ID)
                    contexts = [s.get("content", "")
                                for s in (r.get("sources") or [])]
                    items.append({
                        "question": q,
                        "answer": r.get("answer", ""),
                        "contexts": contexts,
                    })
                except Exception as e:
                    st.error(f"问题生成失败：{q} -> {e}")
                    items.append({"question": q, "answer": "", "contexts": []})
                progress.progress(i / len(questions),
                                  text=f"生成答案中… ({i}/{len(questions)})")

            progress.progress(1.0, text="调用评估中…")
            try:
                result = evaluate_api(st.session_state["api_url"], items)
                avg = result.get("average", {})

                st.subheader("📈 平均得分（0~1）")
                metrics = result.get('metrics') or list(avg)
                labels = {'faithfulness': '忠实度', 'context_precision': '上下文精确度',
                          'answer_relevancy': '答案相关性', 'context_recall': '上下文召回率'}
                columns = st.columns(max(1, len(metrics)))
                for column, metric in zip(columns, metrics):
                    value = avg.get(metric)
                    column.metric(labels.get(metric, metric),
                                  f'{value:.3f}' if isinstance(value, (int, float)) else '未评出')
                st.caption(f"引擎：{result.get('engine', 'unknown')}；完整评估 "
                           f"{result.get('complete_count', 0)}/{result.get('total', 0)} 条；"
                           f"各指标有效数：{result.get('valid_counts', {})}")

                with st.expander("🔍 逐条明细", expanded=True):
                    for s in result.get("scores", []):
                        st.markdown(f"**Q：{s.get('question', '')}**")
                        for metric in metrics:
                            value = s.get(metric)
                            display_value = f'{value:.2f}' if isinstance(value, (int, float)) else '未评出'
                            st.markdown(f"- {labels.get(metric, metric)}：`{display_value}`")
                        if s.get("rationale"):
                            st.caption(f"理由：{s['rationale']}")
                        st.divider()

                import json
                from pathlib import Path
                out_path = Path(__file__).parent.parent / "data/test_query/eval_report.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
                st.success(f"完整报告已保存：{out_path}")
            except Exception as e:
                st.error(f"评估失败：{e}")
            finally:
                progress.empty()
