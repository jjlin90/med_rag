# 医疗 RAG Streamlit 前端

连接后端的 Web 问答界面。

## 快速启动

1. 先启动后端 API（在项目根目录）：

```bash
python scripts/run_api.py
# Windows / VS Code 启动崩溃(0xC0000005) 时改用净化启动器：
.\run_api_safe.ps1
```

> 后端需连通 Redis 缓存：本机若用 Docker 容器 `milvus-redis`（启动带 `--requirepass`），须在 `.env` 设 `REDIS_PASSWORD=1234` 才能连通；未配则健康页 Redis 显示红、自动降级为无缓存模式，不影响问答主流程。

2. 再启动前端（另开终端，在项目根目录）：

```bash
web_demo/.venv/Scripts/python.exe -m streamlit run web_demo/app.py
```

默认打开：`http://localhost:8501`

## 功能

- 单一问答 `/query`
- 多轮对话 `/chat`（按 `session_id` 持久化，刷新不丢历史）
- 实时健康检查
- 系统统计面板
- 引用来源折叠展示
- 检索策略、来源过滤、缓存开关
- **RAG 评估** Tab（对接 `POST /evaluate`，Ragas 四项指标）

## 配置

在左侧边栏修改后端 API 地址，默认 `http://localhost:8005`。
