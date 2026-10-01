@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 启动 Streamlit 前端...
echo 请确保后端 API 已运行：python scripts/run_api.py
echo.
if not exist ".venv\Scripts\python.exe" (
    echo 未找到项目根目录的 .venv，请先在项目根目录运行 uv sync --extra demo。
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run web_demo/app.py
set "streamlit_exit_code=%errorlevel%"
pause
exit /b %streamlit_exit_code%
