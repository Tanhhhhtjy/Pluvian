#!/usr/bin/env bash
# 用法: bash scripts/sync_to_gpu.sh <user@host[:port]> [--code-only|--data-only|--full] [--dry-run] [--delete]
#
# 默认: --code-only
# 密码: 通过环境变量 GPU_SSH_PASS 传入（推荐），或交互式输入
#       export GPU_SSH_PASS='xxx' 然后再跑脚本
#       绝不把密码写进脚本/仓库
#
# 远端目标: /workspace/pluvian
# 需要远端机预先 mkdir -p /workspace/pluvian

set -euo pipefail

# ---- 默认参数 ----
MODE="code-only"
DRY_RUN=""
DELETE_FLAG=""
REMOTE_BASE="/workspace/pluvian"
LOCAL_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_PORT=22

# ---- 解析参数 ----
if [[ $# -lt 1 ]]; then
    echo "[错误] 缺少远端目标。用法: bash scripts/sync_to_gpu.sh <user@host[:port]> [--code-only|--data-only|--full] [--dry-run] [--delete]" >&2
    exit 1
fi

REMOTE="$1"; shift

# 支持 user@host:port 写法
if [[ "$REMOTE" == *":"* ]]; then
    SSH_PORT="${REMOTE##*:}"
    REMOTE="${REMOTE%:*}"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --code-only) MODE="code-only" ;;
        --data-only) MODE="data-only" ;;
        --full)      MODE="full" ;;
        --dry-run)   DRY_RUN="--dry-run" ;;
        --delete)    DELETE_FLAG="--delete-after" ;;
        *) echo "[错误] 未知参数: $1" >&2; exit 1 ;;
    esac
    shift
done

# ---- 密码处理 ----
SSHPASS_BIN="$(command -v sshpass || true)"
SSH_CMD="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts"

if [[ -z "${GPU_SSH_PASS:-}" ]]; then
    echo "[提示] 未设置 GPU_SSH_PASS 环境变量，将使用 ssh 密钥或交互式密码。"
    RSYNC_RSH="$SSH_CMD"
else
    if [[ -z "$SSHPASS_BIN" ]]; then
        echo "[错误] 检测到 GPU_SSH_PASS 但本机没装 sshpass，请: sudo apt install sshpass" >&2
        exit 1
    fi
    RSYNC_RSH="sshpass -e $SSH_CMD"
    export SSHPASS="${GPU_SSH_PASS}"
fi

# ---- 同步条目 ----
# code-only: 代码 + 配置 + 文档（轻量）
CODE_ITEMS=(
    "pipeline/"
    "model/"
    "scripts/"
    "configs/"
    "download_era5.py"
    "README.md"
    "DATA_HANDLING.md"
    "RESEARCH_PROPOSAL_v1.md"
    "GPU_SETUP.md"
    "requirements.txt"
    "requirements-gpu.txt"
    ".gitignore"
)

# data-only: 原始观测 + 派生量（重）
DATA_ITEMS=(
    "radar_nc/"
    "pwv/"
    "stations/"
    "era5/"
    "derived/"
    "2023_05-08_ERA5_PWV.nc"
)

# ---- 准备远端目录 ----
echo "[信息] 远端目标: ${REMOTE}:${REMOTE_BASE} (port ${SSH_PORT})"
echo "[信息] 同步模式: ${MODE}"
[[ -n "$DRY_RUN" ]] && echo "[信息] DRY RUN（预览，不实际传输）"
[[ -n "$DELETE_FLAG" ]] && echo "[警告] 已开启 --delete-after，远端多余文件将被删除！"

if [[ -z "$DRY_RUN" ]]; then
    eval "$RSYNC_RSH ${REMOTE} \"mkdir -p ${REMOTE_BASE}\""
fi

# ---- 执行 rsync ----
RSYNC_OPTS=(
    -avz
    --human-readable
    --partial
    --progress
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='.ipynb_checkpoints/'
    --exclude='.git/'
    --exclude='.claude/'
    --exclude='.DS_Store'
    --exclude='*.log'
    --exclude='ckpt/'
    --exclude='logs/'
    --exclude='runs/'
)

[[ -n "$DRY_RUN" ]]     && RSYNC_OPTS+=("$DRY_RUN")
[[ -n "$DELETE_FLAG" ]] && RSYNC_OPTS+=("$DELETE_FLAG")

sync_items() {
    local label="$1"; shift
    local items=("$@")
    local existing=()
    for it in "${items[@]}"; do
        if [[ -e "${LOCAL_BASE}/${it}" ]]; then
            existing+=("${LOCAL_BASE}/${it}")
        else
            echo "[跳过] 本地不存在: ${it}"
        fi
    done
    if [[ ${#existing[@]} -eq 0 ]]; then
        echo "[警告] ${label} 没有任何条目可同步，跳过。"
        return
    fi
    echo "[执行] ${label} → ${REMOTE}:${REMOTE_BASE}/"
    rsync "${RSYNC_OPTS[@]}" \
        -e "$RSYNC_RSH" \
        "${existing[@]}" \
        "${REMOTE}:${REMOTE_BASE}/"
}

case "$MODE" in
    code-only) sync_items "代码" "${CODE_ITEMS[@]}" ;;
    data-only) sync_items "数据" "${DATA_ITEMS[@]}" ;;
    full)
        sync_items "代码" "${CODE_ITEMS[@]}"
        sync_items "数据" "${DATA_ITEMS[@]}"
        ;;
esac

echo "[完成] 同步结束。"
