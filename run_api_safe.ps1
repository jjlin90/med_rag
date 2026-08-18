<#
.SYNOPSIS
    净化环境变量后启动 med_rag 后端 API，规避 VS Code / 多 Python 环境导致的原生 DLL 冲突（0xC0000005 崩溃）。
.DESCRIPTION
    现象：在 VS Code 集成终端里直接 `python scripts/run_api.py` 会 0xC0000005 崩溃（原生库 segfault，无 Python 堆栈）。
    同一份 venv 在干净环境里却能正常启动，说明是 VS Code 终端把其它 Python/conda 环境的路径塞进了 PATH / 环境变量，
    导致 venv 的 torch / FlagEmbedding 加载到不匹配的原生 DLL。

    本脚本在启动前：
      1) 把 PATH 收窄为「仅 venv 自身目录 + Windows 系统目录」，丢掉一切外部 Python/conda 路径；
      2) 清空 VS Code 可能注入的 PYTHONPATH / PYTHONSTARTUP / PYTHONHOME；
      3) 兜底设置 KMP_DUPLICATE_LIB_OK=TRUE；
      4) 用 -X faulthandler 启动，万一仍崩溃会把 segfault 堆栈打到终端，便于定位。
    然后用项目 venv 的 python.exe 启动 scripts/run_api.py。
.USAGE
    .\run_api_safe.ps1                 # 在项目根目录的 PowerShell / VS Code 终端中执行
    .\run_api_safe.ps1 --port 8005     # 透传参数
#>

$ErrorActionPreference = 'Continue'

# 项目根目录 = 本脚本所在目录（兼容直接运行 / 后台运行）
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $root) { $root = $PSScriptRoot }
if (-not $root) { $root = Get-Location }
$venv = Join-Path $root '.venv'
$venvScripts = Join-Path $venv 'Scripts'

# ---- 1) 构造最小化 PATH：仅 venv 自身 + Windows 系统目录 ----
$keepDirs = New-Object System.Collections.Generic.List[string]
$keepDirs.Add($venvScripts)

# torch 的原生 DLL 目录（cu126 wheel 自带 CUDA 运行时，自包含）
$torchLib = Join-Path $venv 'Lib\site-packages\torch\lib'
if (Test-Path $torchLib) { $keepDirs.Add($torchLib) }
$torchBin = Join-Path $venv 'Lib\site-packages\torch\bin'
if (Test-Path $torchBin) { $keepDirs.Add($torchBin) }
# numpy OpenBLAS 的 .libs
$numpyLibs = Join-Path $venv 'Lib\site-packages\numpy\.libs'
if (Test-Path $numpyLibs) { $keepDirs.Add($numpyLibs) }
# FlagEmbedding / BGE 可能用到的 DLL 目录
$flagLib = Join-Path $venv 'Lib\site-packages\FlagEmbedding'
if (Test-Path $flagLib) { $keepDirs.Add($flagLib) }

# Windows 系统目录（运行 python 必需，且不含其它 Python）
$sysDirs = @('C:\Windows\System32', 'C:\Windows\SysWOW64', 'C:\Windows')
foreach ($d in $sysDirs) {
    if (Test-Path $d) { $keepDirs.Add($d) }
}

$env:PATH = $keepDirs -join ';'

# ---- 2) 清理 VS Code / 其它环境注入的变量 ----
$env:PYTHONPATH = ''
Remove-Item Env:PYTHONSTARTUP -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME   -ErrorAction SilentlyContinue
if (-not $env:KMP_DUPLICATE_LIB_OK) { $env:KMP_DUPLICATE_LIB_OK = 'TRUE' }

Write-Host "[run_api_safe] 已收窄 PATH 为:" -ForegroundColor Cyan
$keepDirs | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
Write-Host "[run_api_safe] 启动后端 API（faulthandler 已开，崩溃会显示堆栈）..." -ForegroundColor Cyan

# ---- 3) 启动（faulthandler + 透传参数）----
& (Join-Path $venvScripts 'python.exe') '-X', 'faulthandler', (Join-Path $root 'scripts\run_api.py') @args
