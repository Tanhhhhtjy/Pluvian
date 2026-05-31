#!/bin/bash
# Pluvian training watchdog (with auto-shutdown on completion)
# - Runs on AutoDL remote, independent of any SSH session
# - Checks every 60s if train.py is alive
# - On crash, resumes from latest checkpoint
# - On normal completion: stop AutoDL instance (save $$)

set -u

PLUVIAN=/root/autodl-tmp/pluvian
LOG=$PLUVIAN/logs/watchdog.log
PYTHON=/root/miniconda3/bin/python

mkdir -p $PLUVIAN/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> $LOG
}

is_train_alive() {
    pgrep -f 'scripts/train.py' > /dev/null
}

latest_ckpt() {
    local cfg=$1
    local name=$(basename $cfg .yaml)
    local d=$PLUVIAN/ckpt/$name
    if [ -d "$d" ]; then
        if [ -f "$d/last.pt" ]; then
            echo "$d/last.pt"
            return
        fi
        local newest=$(ls -t $d/*.pt 2>/dev/null | head -1)
        if [ -n "$newest" ]; then echo "$newest"; fi
    fi
}

start_train() {
    local cfg=$1
    local name=$(basename $cfg .yaml)
    local logf=$PLUVIAN/logs/$name.log
    local ckpt=$(latest_ckpt $cfg)
    local resume_args=""
    if [ -n "$ckpt" ]; then
        resume_args="--resume $ckpt"
        log "Resuming $name from $ckpt"
    else
        log "Starting fresh $name"
    fi
    cd $PLUVIAN
    nohup env PYTHONUNBUFFERED=1 $PYTHON -u scripts/train.py \
        --config $cfg $resume_args \
        > $logf 2>&1 < /dev/null &
    disown
    log "Spawned train PID $!"
}

# ----------- shutdown helpers -----------
do_shutdown() {
    log "==== TRAINING COMPLETE — initiating GPU shutdown to save \$\$ ===="
    # Method 1: AutoDL native (preferred if available)
    if command -v shutdown > /dev/null 2>&1; then
        log "Trying: shutdown -h now"
        shutdown -h now 2>>$LOG &
    fi
    sleep 5
    # Method 2: kill init (forces container exit)
    log "Trying: kill -SIGTERM 1"
    kill -SIGTERM 1 2>>$LOG
    sleep 10
    # Method 3: force halt
    log "Trying: halt -f"
    halt -f 2>>$LOG
    # If we're still here after 30s, give up
    sleep 30
    log "Shutdown attempts exhausted. User must manually stop instance."
}

if [ -n "${1:-}" ]; then
    case "$1" in
        /*) CONFIG_NOW="$1" ;;
        *)  CONFIG_NOW="$PLUVIAN/$1" ;;
    esac
else
    CONFIG_NOW=$PLUVIAN/configs/ablation_1_radar_only.yaml
fi
CHECK_INTERVAL=60

log "Watchdog start. monitoring: $CONFIG_NOW, interval=${CHECK_INTERVAL}s"
log "AUTO-SHUTDOWN enabled when [done] appears in training log"

while true; do
    if is_train_alive; then
        sleep $CHECK_INTERVAL
        continue
    fi

    name=$(basename $CONFIG_NOW .yaml)
    cur_log=$PLUVIAN/logs/$name.log

    if [ -f "$cur_log" ]; then
        if grep -q "\[done\]" "$cur_log" 2>/dev/null; then
            log "$name FINISHED normally."
            log "All current ablations done. Triggering GPU shutdown."
            do_shutdown
            exit 0
        fi
        log "$name CRASHED. Will resume in 10s."
        sleep 10
    else
        log "$name not started yet. Starting in 5s."
        sleep 5
    fi

    start_train $CONFIG_NOW
    sleep 30
done
