@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 启动 Streamlit 前端...
echo 请确保后端 API 已运行：python scripts/run_api.py
echo.
"web_demo/.venv/Scripts/python.exe" -m streamlit run web_demo/app.py
pause
