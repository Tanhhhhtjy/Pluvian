"""Dataset-wide normalize stats scanner.

从 manifest.csv 抽样若干 day × hour 时刻, 调用 build.py 里已有的 modality
loader 提取 7 个 dense modality, 每帧 reservoir-sample N 个有限 pixel,
最终合并算 p2/p50/p98/min/max, 输出 build.py 兼容的 normalize_stats.json.

Schema (与 build.py._normalize_to_uint8 期望的 stats 字典 100% 一致):
{
  "radar":              {"p2": x, "p98": y, "log1p": true,
                         "p50": z, "min": a, "max": b, "n_samples": k},
  "pwv_dense":          {"p2": ..., "p98": ..., "log1p": false, ...},
  "tem_station_dense":  {...},
  "rhu_station_dense":  {...},
  "prs_station_dense":  {...},
  "era5_t_925":         {...},
  "era5_q_925":         {...}
}

(build.py 只读 p2/p98/log1p; p50/min/max 是 sanity check 用, 不影响兼容性.)

CLI:
    python -m pluvian.team_dataset.scan_stats \\
        --out_path docs/team_dataset/normalize_stats.json \\
        --sample_rate 0.2 \\
        --pixels_per_frame 10000 \\
        --n_workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 直接复用 build.py 的 loader, 保证 stats 算的就是 build 写出的数值
from pluvian.team_dataset.build import (  # noqa: E402
    _radar_at, _pwv_at, _station_at, _era5_at, _idw_to_dense,
    RADAR_H, RADAR_W, VAR_COLS,
)

log = logging.getLogger("team_dataset.scan_stats")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


# ----------------------- 模态定义 --------------------------------------------

MODALITY_LOG1P = {
    "radar": True,  # 雷达 mm/h 长尾, 必须 log1p 后再算 percentile
    "pwv_dense": False,
    "tem_station_dense": False,
    "rhu_station_dense": False,
    "prs_station_dense": False,
    "era5_t_925": False,
    "era5_q_925": False,
}
MODALITIES = list(MODALITY_LOG1P.keys())

# 这些模态本来就是 dataset-wide 物理量, 0 不是缺失而是真值, 全留
# 雷达 0 比例极高 (无雨 pixel), 单独处理: 只采非零 pixel, 不然 p2/p50 全是 0
NONZERO_ONLY = {"radar"}


# ----------------------- 抽样函数 --------------------------------------------

def _frame_to_samples(
    timestamp: pd.Timestamp,
    pixels_per_frame: int,
    rng: np.random.Generator,
) -> Optional[Dict[str, np.ndarray]]:
    """单帧: 调 build.py 的 loader -> 抽 N 个 finite pixel per modality.

    任何硬要求模态缺 (radar/era5) -> 整帧 skip 返回 None.
    站点 / PWV 全缺 -> 该模态返回空数组 (但其他 modality 仍写入).
    """
    try:
        radar = _radar_at(timestamp)
        if radar is None:
            return None
        era5 = _era5_at(timestamp)
        if era5 is None:
            return None
        era5_t = era5["t"]
        era5_q = era5["q"]
    except Exception as e:
        log.warning("[%s] hard-required modality load failed: %s", timestamp, e)
        return None

    # PWV (soft)
    try:
        pwv_val, pwv_mask, pwv_coords = _pwv_at(timestamp)
        pwv_dense = (_idw_to_dense(pwv_val, pwv_mask, pwv_coords)
                     if pwv_mask.sum() > 0 else None)
    except Exception as e:
        log.warning("[%s] PWV load error: %s", timestamp, e)
        pwv_dense = None

    # 地面站 (soft) — 复用 build.py 的逻辑, 但避免引入 disk 写, 自己内联简化
    tem_dense = rhu_dense = prs_dense = None
    try:
        sta_val, sta_mask, sta_coords3 = _station_at(timestamp)
        sta_coords = sta_coords3[:, :2]
        if sta_mask.sum() > 0:
            def _dense(col_name):
                col = VAR_COLS.index(col_name)
                col_vals = sta_val[:, col]
                col_mask = (sta_mask > 0) & np.isfinite(col_vals) & (col_vals != 0.0)
                if col_mask.sum() == 0:
                    return None
                return _idw_to_dense(col_vals, col_mask.astype(np.float32), sta_coords)
            tem_dense = _dense("TEM")
            rhu_dense = _dense("RHU")
            prs_dense = _dense("PRS")
    except Exception as e:
        log.warning("[%s] station load error: %s", timestamp, e)

    fields = {
        "radar": radar,
        "pwv_dense": pwv_dense,
        "tem_station_dense": tem_dense,
        "rhu_station_dense": rhu_dense,
        "prs_station_dense": prs_dense,
        "era5_t_925": era5_t,
        "era5_q_925": era5_q,
    }

    samples: Dict[str, np.ndarray] = {}
    for name, arr in fields.items():
        if arr is None:
            samples[name] = np.empty(0, dtype=np.float32)
            continue
        a = np.asarray(arr, dtype=np.float32).ravel()
        finite = np.isfinite(a)
        if name in NONZERO_ONLY:
            finite &= (a > 0)
        pool = a[finite]
        if pool.size == 0:
            samples[name] = np.empty(0, dtype=np.float32)
            continue
        if pool.size <= pixels_per_frame:
            samples[name] = pool.astype(np.float32, copy=False)
        else:
            idx = rng.choice(pool.size, size=pixels_per_frame, replace=False)
            samples[name] = pool[idx].astype(np.float32, copy=False)
    return samples


def _worker(args: Tuple[str, int, int]) -> Tuple[str, Optional[Dict[str, np.ndarray]]]:
    """multiprocessing 入口. seed 派生自 timestamp 保证可复现."""
    ts_iso, pixels_per_frame, seed = args
    rng = np.random.default_rng(seed)
    t = pd.Timestamp(ts_iso)
    out = _frame_to_samples(t, pixels_per_frame, rng)
    return ts_iso, out


# ----------------------- 时间戳生成 ------------------------------------------

def _build_timestamps(
    manifest_path: Path,
    sample_rate: float,
    hours_per_day: int,
    seed: int,
) -> List[pd.Timestamp]:
    """每个 split 各抽 sample_rate 比例的 day, 每天取 hours_per_day 个整点.

    drop 日子直接排除. 返回 ISO 排序后的 timestamp 列表.
    """
    df = pd.read_csv(manifest_path)
    df = df[df["split"] != "drop"]
    rng = np.random.default_rng(seed)

    sampled_days: List[str] = []
    for split, g in df.groupby("split"):
        days = g["date"].tolist()
        n = max(1, int(round(len(days) * sample_rate)))
        n = min(n, len(days))
        pick = rng.choice(len(days), size=n, replace=False)
        sampled_days.extend([days[i] for i in pick])

    # 每天 hours_per_day 个整点 (均匀间隔)
    step = max(1, 24 // hours_per_day)
    out: List[pd.Timestamp] = []
    for d in sorted(sampled_days):
        base = pd.Timestamp(d)
        for h in range(0, 24, step):
            out.append(base + pd.Timedelta(hours=int(h)))
    return out


# ----------------------- 主统计 ----------------------------------------------

def _summarize(samples: np.ndarray, log1p: bool) -> Dict[str, float]:
    """从合并后的 sample pool 算 p2/p50/p98/min/max.

    若 log1p=True, 注意: build.py 在 normalize 时会自己 log1p, 所以这里也得在
    log1p 域上算 p2/p98 才对得上 build.py 的 clip 行为.
    """
    if samples.size == 0:
        return {"p2": 0.0, "p50": 0.0, "p98": 1.0, "min": 0.0, "max": 1.0,
                "n_samples": 0}
    x = samples.astype(np.float64)
    if log1p:
        x = np.log1p(np.clip(x, 0.0, None))
    p2, p50, p98 = np.percentile(x, [2.0, 50.0, 98.0])
    return {
        "p2": float(p2),
        "p50": float(p50),
        "p98": float(p98),
        "min": float(x.min()),
        "max": float(x.max()),
        "n_samples": int(samples.size),
    }


def scan(
    manifest_path: Path,
    out_path: Path,
    sample_rate: float = 0.2,
    pixels_per_frame: int = 10000,
    hours_per_day: int = 6,
    n_workers: int = 4,
    seed: int = 20260619,
) -> dict:
    timestamps = _build_timestamps(manifest_path, sample_rate, hours_per_day, seed)
    log.info("planned timestamps: %d (sample_rate=%.2f, hours/day=%d)",
             len(timestamps), sample_rate, hours_per_day)

    # reservoir: list of arrays per modality, 最后 concat
    buckets: Dict[str, List[np.ndarray]] = {m: [] for m in MODALITIES}

    job_args = [(t.isoformat(), pixels_per_frame, seed + i)
                for i, t in enumerate(timestamps)]

    done = 0
    failed = 0
    if n_workers <= 1:
        for a in job_args:
            ts_iso, out = _worker(a)
            done += 1
            if out is None:
                failed += 1
                continue
            for m, arr in out.items():
                if arr.size:
                    buckets[m].append(arr)
            if done % 50 == 0:
                log.info("progress %d/%d (failed=%d)", done, len(job_args), failed)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_worker, a): a for a in job_args}
            for fut in as_completed(futs):
                ts_iso, out = fut.result()
                done += 1
                if out is None:
                    failed += 1
                else:
                    for m, arr in out.items():
                        if arr.size:
                            buckets[m].append(arr)
                if done % 50 == 0:
                    log.info("progress %d/%d (failed=%d)",
                             done, len(job_args), failed)

    log.info("collected: total=%d, failed=%d", done, failed)

    stats: Dict[str, dict] = {}
    for m in MODALITIES:
        if buckets[m]:
            pool = np.concatenate(buckets[m])
        else:
            pool = np.empty(0, dtype=np.float32)
        s = _summarize(pool, log1p=MODALITY_LOG1P[m])
        s["log1p"] = MODALITY_LOG1P[m]
        stats[m] = s

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 打印 sanity check 表
    log.info("=" * 78)
    log.info("normalize_stats summary (values shown in the domain build.py clips on):")
    log.info("  modality                  p2          p50         p98         "
             "min         max         n")
    for m in MODALITIES:
        s = stats[m]
        log.info("  %-24s  %-10.4g  %-10.4g  %-10.4g  %-10.4g  %-10.4g  %d",
                 m, s["p2"], s["p50"], s["p98"], s["min"], s["max"], s["n_samples"])
    log.info("=" * 78)
    log.info("wrote %s", out_path)
    return stats


# ----------------------- CLI ------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Scan dataset-wide normalize stats")
    ap.add_argument("--manifest", default=str(_REPO_ROOT / "pipeline" / "manifest.csv"))
    ap.add_argument("--out_path", required=True,
                    help="output normalize_stats.json path")
    ap.add_argument("--sample_rate", type=float, default=0.2,
                    help="fraction of days per split to sample")
    ap.add_argument("--pixels_per_frame", type=int, default=10000,
                    help="reservoir sample size per modality per frame")
    ap.add_argument("--hours_per_day", type=int, default=6,
                    help="number of evenly-spaced hourly timestamps per sampled day")
    ap.add_argument("--n_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260619)
    args = ap.parse_args(argv)

    scan(
        manifest_path=Path(args.manifest),
        out_path=Path(args.out_path),
        sample_rate=args.sample_rate,
        pixels_per_frame=args.pixels_per_frame,
        hours_per_day=args.hours_per_day,
        n_workers=args.n_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
