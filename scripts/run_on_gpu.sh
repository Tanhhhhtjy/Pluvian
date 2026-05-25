#!/usr/bin/env bash
# 用法: bash scripts/run_on_gpu.sh <user@host[:port]> <config-yaml-相对路径> [--bg] [--debug] [--venv /opt/venv]
#
# 例:
#   bash scripts/run_on_gpu.sh root@10.0.0.1 configs/ablation_5_budget_loss.yaml --bg
#
# 远端会在 /workspace/pluvian 下执行：
#   source <venv>/bin/activate
#   python scripts/train.py --config <yaml> [--debug]
#
# --bg: nohup 后台运行，日志写到 logs/<config-basename>.log

set -euo pipefail

REMOTE_BASE="/workspace/pluvian"
VENV="/opt/venv"
BG=0
DEBUG=""
SSH_PORT=22

if [[ $# -lt 2 ]]; then
    echo "[错误] 用法: bash scripts/run_on_gpu.sh <user@host[:port]> <config.yaml> [--bg] [--debug] [--venv PATH]" >&2
    exit 1
fi

REMOTE="$1"; shift
CONFIG="$1"; shift
if [[ "$REMOTE" == *":"* ]]; then
    SSH_PORT="${REMOTE##*:}"
    REMOTE="${REMOTE%:*}"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bg)    BG=1 ;;
        --debug) DEBUG="--debug" ;;
        --venv)  VENV="$2"; shift ;;
        *) echo "[错误] 未知参数: $1" >&2; exit 1 ;;
    esac
    shift
done

SSHPASS_BIN="$(command -v sshpass || true)"
SSH_BASE="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts"

if [[ -n "${GPU_SSH_PASS:-}" && -n "$SSHPASS_BIN" ]]; then
    SSH_RUN=(sshpass -e $SSH_BASE)
    export SSHPASS="${GPU_SSH_PASS}"
else
    SSH_RUN=($SSH_BASE)
fi

CONFIG_BASE="$(basename "$CONFIG" .yaml)"
LOG_FILE="logs/${CONFIG_BASE}.log"

if [[ $BG -eq 1 ]]; then
    REMOTE_CMD="cd ${REMOTE_BASE} && mkdir -p logs && source ${VENV}/bin/activate && nohup python scripts/train.py --config ${CONFIG} ${DEBUG} > ${LOG_FILE} 2>&1 & echo \$! > logs/${CONFIG_BASE}.pid && echo '[GPU] 已后台启动, PID:' \$(cat logs/${CONFIG_BASE}.pid) ', 日志:' ${LOG_FILE}"
else
    REMOTE_CMD="cd ${REMOTE_BASE} && source ${VENV}/bin/activate && python scripts/train.py --config ${CONFIG} ${DEBUG}"
fi

echo "[信息] 远端: ${REMOTE} (port ${SSH_PORT})"
echo "[信息] 命令: ${REMOTE_CMD}"

"${SSH_RUN[@]}" "${REMOTE}" "bash -lc '${REMOTE_CMD}'"
