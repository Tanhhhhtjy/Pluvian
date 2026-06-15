#!/usr/bin/env bash
# CDU 训完后一键执行：生成论文需要的全部 CDU 评测产物。
#
# 触发前提：CDU 训练完成，ckpt/ablation_3_cdu_p7d/best.pt 已经落盘。
# 运行方式：
#   bash scripts/run_cdu_post_training.sh
#
# 输出：
#   ckpt/eval/ablation_3_cdu_p7d/{event_test,test_robust}/metric.json
#   ckpt/figures/ab3_cdu_per_lead/per_lead_{event_test,test_robust}.{json,png}
#   ckpt/figures/pooled_csi_cdu/pooled_csi.{json,md}
#   ckpt/figures/paper_case/event_20230730T0900_with_cdu.png
#   ckpt/figures/paper_skill_bar/skill_ladder_v2.{png,pdf} （含 5 模型）
#   ckpt/bootstrap_ci/ab3_cdu/ab3-cdu_{event_test,test_robust}_ci.json
#   docs/paper/tables_1_2.{md,tex}              （含 ab3-cdu 行）
#
# fail-fast，任何一步失败立刻退出。

set -euo pipefail

REPO_ROOT="/data4/WuMingrui/TianJinyu/npj/pluvian"
PY="/home/WuMingrui/miniconda3/envs/npj/bin/python"

CDU_CKPT="ckpt/ablation_3_cdu_p7d/best.pt"
AB1_CKPT="ckpt/ablation_1_p7d/best.pt"
AB2_CKPT="ckpt/ablation_2_p7d_xcoreA/best.pt"
AB3_CKPT="ckpt/ablation_3_era5_p7c/best.pt"
AB3B_CKPT="ckpt/ablation_3b_era5_budget_p7c/best.pt"

SPLITS=("event_test" "test_robust")

cd "$REPO_ROOT"

mkdir -p logs
LOG_TS="$(date +%Y%m%dT%H%MZ)"
LOG_FILE="logs/cdu_post_training_${LOG_TS}.log"

# 全部 stdout/stderr 同时写日志和终端
exec > >(tee -a "$LOG_FILE") 2>&1

export CUDA_VISIBLE_DEVICES=0

echo "=== STEP 0: env summary ==="
echo "repo:    $REPO_ROOT"
echo "python:  $PY"
echo "log:     $LOG_FILE"
echo "started: $(date -Iseconds)"

# ----------------------------------------------------------------------
echo "=== STEP 1: 健康检查 — CDU best.pt 必须存在 ==="
for f in "$CDU_CKPT" "$AB1_CKPT" "$AB2_CKPT" "$AB3_CKPT" "$AB3B_CKPT"; do
    if [[ ! -f "$f" ]]; then
        echo "[ABORT] missing checkpoint: $f"
        exit 2
    fi
    echo "  ok: $f ($(stat -c '%s bytes, mtime=%y' "$f"))"
done

# ----------------------------------------------------------------------
echo "=== STEP 2: 基础评测 (eval.py) on event_test + test_robust ==="
for split in "${SPLITS[@]}"; do
    out_dir="ckpt/eval/ablation_3_cdu_p7d/${split}"
    mkdir -p "$out_dir"
    echo "  >> split=${split} -> ${out_dir}/metric.json"
    "$PY" scripts/eval.py \
        --ckpt   "$CDU_CKPT" \
        --out_dir "$out_dir" \
        --split   "$split"
done

# ----------------------------------------------------------------------
echo "=== STEP 3: per-lead diagnostic (ab3 vs ab3-cdu) ==="
# 注意：eval_per_lead.py 的两个参数叫 --ckpt_ab1 / --ckpt_ab2，是位置语义而非
# 真正的 ab1/ab2，所以这里 ab1=ab3 基线，ab2=ab3-cdu。
for split in "${SPLITS[@]}"; do
    out_dir="ckpt/figures/ab3_cdu_per_lead"
    mkdir -p "$out_dir"
    echo "  >> split=${split}"
    "$PY" scripts/eval_per_lead.py \
        --ckpt_ab1 "$AB3_CKPT" \
        --ckpt_ab2 "$CDU_CKPT" \
        --split    "$split" \
        --out_dir  "$out_dir"
done

# ----------------------------------------------------------------------
echo "=== STEP 4: pooled-CSI (ab3 vs ab3-cdu) ==="
pooled_dir="ckpt/figures/pooled_csi_cdu"
mkdir -p "$pooled_dir"
"$PY" scripts/eval_pooled_csi.py \
    --ckpt "ab3=${AB3_CKPT}" \
    --ckpt "ab3-cdu=${CDU_CKPT}" \
    --split event_test \
    --split test_robust \
    --out_dir "$pooled_dir"

# ----------------------------------------------------------------------
echo "=== STEP 5: case study 重画 (含 ab3-cdu 行 + 两条 diff) ==="
case_out="ckpt/figures/paper_case/event_20230730T0900_with_cdu.png"
mkdir -p "$(dirname "$case_out")"
"$PY" scripts/plot_case_grid.py \
    --model "ab1=${AB1_CKPT}" \
    --model "ab2=${AB2_CKPT}" \
    --model "ab3=${AB3_CKPT}" \
    --model "ab3b=${AB3B_CKPT}" \
    --model "ab3-cdu=${CDU_CKPT}" \
    --diff  "ab3-cdu vs ab3b" \
    --diff  "ab3-cdu vs ab3" \
    --start "2023-07-30T09:00:00" \
    --lead 18 --lead 36 --lead 72 --lead 108 \
    --diff_range 15 \
    --out "$case_out"

# ----------------------------------------------------------------------
echo "=== STEP 6: skill ladder 重绘 (5 模型，含 ab3-cdu) ==="
ladder_out="ckpt/figures/paper_skill_bar/skill_ladder_v2.png"
mkdir -p "$(dirname "$ladder_out")"
"$PY" scripts/plot_skill_ladder.py \
    --include-cdu \
    --out "$ladder_out"

# ----------------------------------------------------------------------
echo "=== STEP 7: bootstrap CI for CDU (两个 split) ==="
boot_dir="ckpt/bootstrap_ci/ab3_cdu"
mkdir -p "$boot_dir"
"$PY" scripts/eval_bootstrap_ci.py \
    --ckpt "ab3-cdu=${CDU_CKPT}" \
    --split event_test \
    --split test_robust \
    --out_dir "$boot_dir"

# ----------------------------------------------------------------------
echo "=== STEP 8: 重写 Tables 1/2 (含 ab3-cdu 行) ==="
"$PY" scripts/make_paper_tables_final.py

# ----------------------------------------------------------------------
echo "=== STEP 9: 一句话总结 — CDU vs ab3 在 CSI@30 上的变化 ==="
"$PY" - <<'PYEOF'
import json
from pathlib import Path

repo = Path("/data4/WuMingrui/TianJinyu/npj/pluvian")
splits = ["event_test", "test_robust"]
print()
print("=========== CDU vs ab3 (baseline) — CSI@30 mm/h ===========")
for sp in splits:
    ab3 = json.loads((repo / f"ckpt/eval/ablation_3_era5_p7c/{sp}/metric.json").read_text())
    cdu = json.loads((repo / f"ckpt/eval/ablation_3_cdu_p7d/{sp}/metric.json").read_text())
    a = float(ab3["csi_30mm"])
    c = float(cdu["csi_30mm"])
    abs_d = c - a
    rel = (c - a) / a * 100.0 if a else float("nan")
    sign = "+" if abs_d >= 0 else ""
    verdict = "提升" if abs_d > 0 else ("回归" if abs_d < 0 else "持平")
    print(f"  {sp:<12} ab3={a:.4f}  ab3-cdu={c:.4f}  delta={sign}{abs_d:+.4f} ({sign}{rel:+.2f}%)  -> {verdict}")
print()
PYEOF

echo "=== DONE: 所有产物已生成 (log: ${LOG_FILE}) ==="
