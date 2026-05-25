#!/bin/bash
# Pluvian training watchdog
# - Runs on AutoDL remote, independent of any SSH session
# - Checks every 60s if train.py is alive
# - On crash, resumes from latest checkpoint
# - Logs to /root/autodl-tmp/pluvian/logs/watchdog.log

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
        # Prefer last.pt, fall back to highest-numbered epoch ckpt
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

CONFIG_NOW=$PLUVIAN/configs/ablation_1_radar_only.yaml
CHECK_INTERVAL=60

log "Watchdog start. monitoring: $CONFIG_NOW, interval=${CHECK_INTERVAL}s"

while true; do
    if is_train_alive; then
        # Healthy. Sleep and continue.
        sleep $CHECK_INTERVAL
        continue
    fi

    # train.py not running. Was it ever started?
    name=$(basename $CONFIG_NOW .yaml)
    cur_log=$PLUVIAN/logs/$name.log

    if [ -f "$cur_log" ]; then
        # Was running before — check if it died with success or crash
        if grep -q "\[done\]" "$cur_log" 2>/dev/null; then
            log "$name has finished. Watchdog moving to next or stopping."
            # TODO: schedule next ablation. For now stop here.
            log "All current ablations done. Watchdog exits."
            exit 0
        fi
        log "$name CRASHED. Will resume in 10s."
        sleep 10
    else
        log "$name not started yet. Starting in 5s."
        sleep 5
    fi

    start_train $CONFIG_NOW
    sleep 30  # give it a moment to allocate GPU before next check
done
