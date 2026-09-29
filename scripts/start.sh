#!/usr/bin/env bash
#
# 一键启动 —— 把 README「快速开始」里的 5 步手工流程收成一条命令。
#
# 为什么要有这个脚本
# ------------------
# 五步手工流程（建 venv → 装依赖 → 备 .env → 播种业务库 → 起服务）每一步都能
# 悄悄漏掉，而**漏掉之后的表现是"看起来正常"**：
#
#   - 忘了 `cp .env.example .env`：服务照常起来，但走的是 Mock 模型，
#     回答永远是同一段话 —— 不报错；
#   - 忘了播种：服务照常起来，工具 Agent 照常"调用成功"，只是永远查不到人
#     —— 与"这位员工没有记录"完全同形，不报错；
#   - venv 里缺包：向量库静默退回内存库，检索的是空索引 —— 不报错。
#
# 这三条都是**部署期漏步 → 运行期静默降级**。脚本的价值不只是省事，
# 而是把这五步变成一次原子操作，并在每一步之后**显式报告它做了什么**。
#
# 用法
# ----
#   ./scripts/start.sh                  # 全流程，最后前台起服务
#   ./scripts/start.sh --check          # 只体检（建环境 + 备配置 + 播种），不起服务
#   ./scripts/start.sh --port 9000      # 换端口
#   ./scripts/start.sh --host 127.0.0.1 # 只监听本机
#   ./scripts/start.sh --no-install     # 跳过依赖安装（已装好时提速）
#   ./scripts/start.sh --no-seed        # 跳过业务库播种（已有数据时别覆盖）
#
# 退出码：0 正常；1 前置条件不满足（Python 版本、缺文件）；2 依赖安装失败。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOST="0.0.0.0"
PORT="8001"
CHECK_ONLY=0
DO_INSTALL=1
DO_SEED=1
VENV_DIR=".venv"
PY="$VENV_DIR/bin/python"

# ---------------------------------------------------------------- 输出工具
if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi

step()  { printf '%s▸ %s%s\n' "$BOLD" "$1" "$RESET"; }
ok()    { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
warn()  { printf '  %s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
die()   { printf '%s✗ %s%s\n' "$RED" "$1" "$RESET" >&2; exit "${2:-1}"; }
note()  { printf '  %s%s%s\n' "$DIM" "$1" "$RESET"; }

usage() { sed -n '/^# 用法$/,/^# 退出码/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# ---------------------------------------------------------------- 参数解析
while [ $# -gt 0 ]; do
    case "$1" in
        --port)       PORT="${2:?--port 需要一个值}"; shift 2 ;;
        --host)       HOST="${2:?--host 需要一个值}"; shift 2 ;;
        --check)      CHECK_ONLY=1; shift ;;
        --no-install) DO_INSTALL=0; shift ;;
        --no-seed)    DO_SEED=0; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            die "未知参数：$1（用 --help 看用法）" ;;
    esac
done

printf '\n%s企业智能助手 · 一键启动%s\n\n' "$BOLD" "$RESET"

# ---------------------------------------------------------------- 1/5 Python
step "1/5 检查 Python"
PY_BOOT=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PY_BOOT="$(command -v "$cand")"; break
        fi
    fi
done
[ -n "$PY_BOOT" ] || die "找不到 Python 3.10+。装一个再回来（macOS: brew install python@3.13）"
ok "$("$PY_BOOT" -c 'import sys; print("Python %d.%d.%d" % sys.version_info[:3])')  →  $PY_BOOT"

# ---------------------------------------------------------------- 2/5 venv
step "2/5 准备虚拟环境与依赖"
if [ ! -x "$PY" ]; then
    note "未找到 $VENV_DIR，正在创建…"
    "$PY_BOOT" -m venv "$VENV_DIR" || die "创建虚拟环境失败"
    ok "已创建 $VENV_DIR"
    DO_INSTALL=1
else
    ok "$VENV_DIR 已存在"
fi

if [ "$DO_INSTALL" -eq 1 ]; then
    if "$PY" -c 'import fastapi' >/dev/null 2>&1; then
        ok "依赖已就绪（跳过安装）"
    else
        note "正在安装 requirements.txt（首次约 1~3 分钟）…"
        if ! "$PY" -m pip install --quiet --upgrade pip >/dev/null 2>&1; then
            warn "升级 pip 失败，继续尝试安装依赖"
        fi
        if ! "$PY" -m pip install --quiet -r requirements.txt; then
            printf '\n' >&2
            warn "依赖安装失败。本机代理常对 TLS 做中间人，表现为证书错误："
            note 'env -u HTTP_PROXY -u HTTPS_PROXY '"$PY"' -m pip install -r requirements.txt \'
            note '    --index-url https://pypi.mirrors.ustc.edu.cn/simple'
            die "依赖安装失败" 2
        fi
        ok "依赖安装完成"
    fi
else
    ok "按 --no-install 跳过"
fi

# ---------------------------------------------------------------- 3/5 .env
step "3/5 准备配置"
if [ -f .env ]; then
    ok ".env 已存在"
    if ! grep -qE '^[[:space:]]*LLM_API_KEY=[^[:space:]]' .env; then
        warn "LLM_API_KEY 是空的 —— 服务会走本地 Mock 模型（回答固定，不是真模型）"
    fi
else
    [ -f .env.example ] || die "既没有 .env 也没有 .env.example，仓库不完整"
    cp .env.example .env
    ok "已从 .env.example 生成 .env"
    warn "还没填 LLM 配置 —— 现在能跑，但走的是 Mock 模型"
    note "填上这三项即接真模型：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME"
fi

# ---------------------------------------------------------------- 4/5 业务库
step "4/5 准备业务数据库"
DB_PATH="$("$PY" -c 'from app import config; from pathlib import Path; print(Path(config.SQLITE_DB_PATH).resolve())' 2>/dev/null || true)"
if [ "$DO_SEED" -eq 0 ]; then
    ok "按 --no-seed 跳过"
elif [ -n "$DB_PATH" ] && [ -f "$DB_PATH" ]; then
    ok "业务库已存在：$DB_PATH"
else
    note "业务库不存在，正在播种（供 3 个只读工具查询）…"
    if "$PY" scripts/seed_enterprise_db.py >/dev/null; then
        ok "已播种：${DB_PATH:-data/enterprise.db}"
    else
        warn "播种失败 —— 工具 Agent 会「调用成功但查不到人」，与「没有这位员工」同形"
    fi
fi

# ---------------------------------------------------------------- 5/5 自检
step "5/5 服务自检"
"$PY" - <<'PYCODE' || die "自检未通过（上面有具体原因）"
import logging
import sys

# 装配工作流会打 INFO 日志，混进这里的体检输出会盖掉真正要看的三行
logging.disable(logging.INFO)

problems = []
try:
    from app import config
    print(f"  向量库后端      {config.VECTOR_DB_TYPE}")
    print(f"  模型            {config.LLM_MODEL_NAME or '(未配置，将走 Mock)'}")
except Exception as exc:                      # noqa: BLE001
    problems.append(f"配置加载失败：{exc}")
try:
    from app.graph.workflow_graph import (
        NODE_NAMES, conditional_branch_count, enterprise_workflow,
    )
    print(f"  工作流          {len(NODE_NAMES)} 节点 / {conditional_branch_count(enterprise_workflow)} 条件边")
except Exception as exc:                      # noqa: BLE001
    problems.append(f"工作流装配失败：{exc}")
if problems:
    for p in problems:
        print(f"  ✗ {p}")
    sys.exit(1)
PYCODE
ok "自检通过"

if [ "$CHECK_ONLY" -eq 1 ]; then
    printf '\n%s体检完成（--check，未启动服务）%s\n\n' "$BOLD" "$RESET"
    exit 0
fi

# ---------------------------------------------------------------- 起服务
printf '\n%s▸ 启动服务%s\n' "$BOLD" "$RESET"
note "面板  http://localhost:${PORT}"
note "接口文档 http://localhost:${PORT}/docs"
note "按 Ctrl-C 停止"
printf '\n'

exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
