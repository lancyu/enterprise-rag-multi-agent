#!/usr/bin/env bash
# ============================================================
# 无 git 环境下的文件级备份（切分改造专用）
#
# 用法：
#   bash scripts/backup.sh T1-1 app/config.py .env
#   bash scripts/backup.sh T3-1 app/rag/indexer.py
#
# 产物：artifacts/backup/<task-id>/<原路径>
# 恢复：cp -R artifacts/backup/<task-id>/. .
#
# 约定：**每次改动业务代码之前**先执行本脚本。
#
# 目录卫生：artifacts/backup/ 只放**进行中**任务的快照。任务完结后把该目录
# 移出仓库（mv，不要 rm）——旧版源码副本留在仓库里会让静态扫描与全文检索
# 反复命中过期实现，是审查时最容易踩的坑。归档位置与搬运方式见
# artifacts/backup/README.md。
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ $# -lt 2 ]; then
  echo "用法: bash scripts/backup.sh <task-id> <file1> [file2 ...]" >&2
  exit 1
fi

TASK_ID="$1"; shift
BACKUP_DIR="artifacts/backup/${TASK_ID}"

if [ -d "$BACKUP_DIR" ]; then
  echo "[backup] 目标已存在，追加覆盖: $BACKUP_DIR"
fi

copied=0
for f in "$@"; do
  if [ ! -e "$f" ]; then
    echo "[backup] 跳过（不存在）: $f" >&2
    continue
  fi
  dest="${BACKUP_DIR}/$(dirname "$f")"
  mkdir -p "$dest"
  cp -R "$f" "${dest}/"
  echo "[backup] $f -> ${BACKUP_DIR}/$f"
  copied=$((copied + 1))
done

if [ "$copied" -eq 0 ]; then
  echo "[backup] 没有文件被备份，请检查路径" >&2
  exit 1
fi

echo "[backup] 完成：${BACKUP_DIR}（${copied} 个）"
echo "[backup] 恢复命令: cp -R ${BACKUP_DIR}/. ."
