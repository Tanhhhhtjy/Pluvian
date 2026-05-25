#!/usr/bin/env bash
# 用法: bash scripts/sync_from_gpu.sh <user@host[:port]> [--dry-run] [--ckpt|--logs|--runs|--all]
#
# 从 GPU 机回传产物到本机。默认 --all。
# 密码: GPU_SSH_PASS 环境变量

set -euo pipefail

MODE="all"
DRY_RUN=""
REMOTE_BASE="/workspace/pluvian"
LOCAL_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_PORT=22

if [[ $# -lt 1 ]]; then
    echo "[错误] 用法: bash scripts/sync_from_gpu.sh <user@host[:port]> [--dry-run] [--ckpt|--logs|--runs|--all]" >&2
    exit 1
fi

REMOTE="$1"; shift
if [[ "$REMOTE" == *":"* ]]; then
    SSH_PORT="${REMOTE##*:}"
    REMOTE="${REMOTE%:*}"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)    MODE="ckpt" ;;
        --logs)    MODE="logs" ;;
        --runs)    MODE="runs" ;;
        --all)     MODE="all" ;;
        --dry-run) DRY_RUN="--dry-run" ;;
        *) echo "[错误] 未知参数: $1" >&2; exit 1 ;;
    esac
    shift
done

SSHPASS_BIN="$(command -v sshpass || true)"
SSH_CMD="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts"

if [[ -z "${GPU_SSH_PASS:-}" ]]; then
    RSYNC_RSH="$SSH_CMD"
else
    if [[ -z "$SSHPASS_BIN" ]]; then
        echo "[错误] 检测到 GPU_SSH_PASS 但本机没装 sshpass" >&2
        exit 1
    fi
    RSYNC_RSH="sshpass -e $SSH_CMD"
    export SSHPASS="${GPU_SSH_PASS}"
fi

RSYNC_OPTS=(-avz --human-readable --partial --progress)
[[ -n "$DRY_RUN" ]] && RSYNC_OPTS+=("$DRY_RUN")

fetch_dir() {
    local sub="$1"
    local local_dst="${LOCAL_BASE}/${sub}/"
    mkdir -p "$local_dst"
    echo "[执行] ${REMOTE}:${REMOTE_BASE}/${sub}/  →  ${local_dst}"
    rsync "${RSYNC_OPTS[@]}" \
        -e "$RSYNC_RSH" \
        "${REMOTE}:${REMOTE_BASE}/${sub}/" \
        "$local_dst" || echo "[警告] ${sub} 拉取失败（可能远端目录还不存在）"
}

case "$MODE" in
    ckpt) fetch_dir "ckpt" ;;
    logs) fetch_dir "logs" ;;
    runs) fetch_dir "runs" ;;
    all)
        fetch_dir "ckpt"
        fetch_dir "logs"
        fetch_dir "runs"
        ;;
esac

echo "[完成] 回传结束。"
