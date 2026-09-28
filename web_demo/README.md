# 医疗 RAG Streamlit 前端

连接后端的 Web 问答界面。

## 快速启动

1. 先启动后端 API（在项目根目录）：

```bash
python scripts/run_api.py
# Windows / VS Code 启动崩溃(0xC0000005) 时改用净化启动器：
.\run_api_safe.ps1
```

> Redis 是可选缓存；`REDIS_PASSWORD` 必须与实际服务一致。`.env.example` 中的 1234 只是本地示例。连接失败时健康页显示不可用，问答链路以无缓存模式继续。

2. 再启动前端（另开终端，在项目根目录）：

```bash
uv sync --extra demo
.venv/Scripts/python.exe -m streamlit run web_demo/app.py
```

默认打开：`http://localhost:8501`

## 功能

- 单一问答 `/query`
- 多轮对话 `/chat`：界面按 `session_id` 从 MySQL 接口恢复展示，并把当前界面 messages 提交给后端；后端不会自行把数据库历史注入生成
- 实时健康检查
- 系统统计面板
- 引用来源折叠展示
- 检索策略、来源过滤、缓存开关
- **RAG 评估** Tab（对接 `POST /evaluate`，按返回的 metrics 展示；本页面不提供标准答案，正常 Ragas 路径为 F/AR 两项）

## 配置

在左侧边栏修改后端 API 地址，默认 `http://localhost:8005`。
