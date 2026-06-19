"""Team dataset per-frame dump.

输入: 单个 UTC 时间戳, 输出: 8 个 dense modality 的 PNG + 1 个 NPZ.

模态:
    radar (mm/h)        — 雷达 6 min, 661x701
    pwv_dense           — PWV 站点 IDW 到 661x701
    tem_station_dense   — 地面站 TEM IDW 到 661x701
    rhu_station_dense   — 地面站 RHU IDW 到 661x701
    prs_station_dense   — 地面站 PRS IDW 到 661x701
    era5_t_925          — ERA5 t@925, 0.25° -> 0.01° bilinear
    era5_q_925          — ERA5 q@925, 同上

只做单帧, 多进程批跑由 Team B wrapper 负责.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

# 让 pluvian/ 子包能找到 pipeline/ 顶层模块
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.radar_io import load_radar_sequence  # noqa: E402
from pipeline.pwv_io import load_pwv_window  # noqa: E402
from pipeline.station_io import load_station_window, VAR_COLS  # noqa: E402
from pipeline.era5_io import load_era5_window_with_grid  # noqa: E402
from pipeline.utils import (  # noqa: E402
    RADAR_LAT, RADAR_LON, RADAR_H, RADAR_W,
    bilinear_interp_to_radar, dbz_to_rainrate,
)

log = logging.getLogger("team_dataset.build")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


# -------------------- sparse -> dense (IDW) ----------------------------------

@lru_cache(maxsize=4)
def _idw_weights(coords_tuple: tuple, k: int = 8, power: float = 2.0):
    """缓存 IDW 邻居索引和权重 (一组站点坐标只算一次).

    coords_tuple: ((lon, lat), ...) 用 tuple 当 key.
    返回: idx (H*W, k) int64, w (H*W, k) float32.
    """
    coords = np.asarray(coords_tuple, dtype=np.float32)
    n = coords.shape[0]
    if n == 0:
        return None, None
    k = min(k, n)

    grid_lon, grid_lat = np.meshgrid(
        RADAR_LON.astype(np.float32), RADAR_LAT.astype(np.float32), indexing="xy")
    grid = np.stack([grid_lon.ravel(), grid_lat.ravel()], axis=1)  # (P, 2)

    # 距离平方 (P, N) — 接受 ~390 站 * 463k pix ~ 180M float, 700MB. 太大.
    # 分块算
    P = grid.shape[0]
    idx = np.empty((P, k), dtype=np.int64)
    w = np.empty((P, k), dtype=np.float32)
    chunk = 20000
    for s in range(0, P, chunk):
        e = min(P, s + chunk)
        d2 = ((grid[s:e, None, 0] - coords[None, :, 0]) ** 2
              + (grid[s:e, None, 1] - coords[None, :, 1]) ** 2)
        part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        d2_k = np.take_along_axis(d2, part, axis=1)
        # IDW 权重 = 1/d^p, 距离 0 时给极大权重
        d_k = np.sqrt(d2_k) + 1e-6
        wi = 1.0 / np.power(d_k, power)
        wi = wi / wi.sum(axis=1, keepdims=True)
        idx[s:e] = part
        w[s:e] = wi.astype(np.float32)
    return idx, w


def _idw_to_dense(values: np.ndarray, mask: np.ndarray, coords: np.ndarray,
                  k: int = 8) -> np.ndarray:
    """sparse 站点值 -> dense 661x701.

    values: (N,) float32
    mask:   (N,) 0/1 表示该站是否有观测
    coords: (N, 2) lon/lat
    """
    valid = mask > 0
    if valid.sum() == 0:
        return np.zeros((RADAR_H, RADAR_W), dtype=np.float32)
    vals = values[valid].astype(np.float32)
    crd = coords[valid]
    idx, w = _idw_weights(tuple(map(tuple, crd.tolist())), k=k)
    if idx is None:
        return np.zeros((RADAR_H, RADAR_W), dtype=np.float32)
    picked = vals[idx]  # (P, k)
    out = (picked * w).sum(axis=1)  # (P,)
    return out.reshape(RADAR_H, RADAR_W).astype(np.float32)


# -------------------- 时间对齐辅助 -------------------------------------------

def _interp_time_pair(t_lo: pd.Timestamp, v_lo: np.ndarray,
                      t_hi: pd.Timestamp, v_hi: np.ndarray,
                      t_target: pd.Timestamp) -> np.ndarray:
    """两个时刻之间线性插值. v 形状任意, 沿 axis=0 不需要."""
    span = (t_hi - t_lo).total_seconds()
    if span <= 0:
        return v_lo.astype(np.float32)
    w = (t_target - t_lo).total_seconds() / span
    w = float(max(0.0, min(1.0, w)))
    return (v_lo * (1 - w) + v_hi * w).astype(np.float32)


# -------------------- 各模态 loader 包到 target time -------------------------

def _radar_at(t: pd.Timestamp) -> Optional[np.ndarray]:
    """返回 (H, W) mm/h, 缺帧返回 None."""
    arr, fm, valid, _ = load_radar_sequence(t, n_frames=1)
    if fm[0] < 0.5:
        return None
    rr = dbz_to_rainrate(arr[0])
    rr[~np.isfinite(rr)] = 0.0
    return rr.astype(np.float32)


def _pwv_at(t: pd.Timestamp) -> tuple:
    """PWV 30 min cadence -> 取包含 t 的两端 30 min 帧线性插值.

    返回 (value (N,), mask (N,), coords (N, 2)).
    """
    floor_30 = t.floor("30min")
    if floor_30 == t:
        vals, mask, coords = load_pwv_window(floor_30, n_steps=1)
        return vals[0], mask[0], coords
    nxt = floor_30 + pd.Timedelta(minutes=30)
    vals, mask, coords = load_pwv_window(floor_30, n_steps=2)
    # 两端都得有观测才算 valid
    both = (mask[0] > 0) & (mask[1] > 0)
    out_val = np.zeros(vals.shape[1], dtype=np.float32)
    span = 30 * 60.0
    w = (t - floor_30).total_seconds() / span
    interp = vals[0] * (1 - w) + vals[1] * w
    out_val[both] = interp[both].astype(np.float32)
    # 单端有也算 (用 ffill — 跟 pipeline 现有行为一致)
    only_lo = (mask[0] > 0) & (mask[1] <= 0)
    out_val[only_lo] = vals[0, only_lo]
    only_hi = (mask[0] <= 0) & (mask[1] > 0)
    out_val[only_hi] = vals[1, only_hi]
    out_mask = ((mask[0] > 0) | (mask[1] > 0)).astype(np.float32)
    return out_val, out_mask, coords


def _station_at(t: pd.Timestamp) -> tuple:
    """地面站 1 h cadence, 包含 t 的两端 1 h 帧.

    返回 (values (N, 6), mask (N,), coords (N, 3)).
    """
    floor_h = t.floor("h")
    if floor_h == t:
        vals, mask, coords = load_station_window(floor_h, n_steps=1)
        return vals[0], mask[0], coords
    vals, mask, coords = load_station_window(floor_h, n_steps=2)
    both = (mask[0] > 0) & (mask[1] > 0)
    span = 3600.0
    w = (t - floor_h).total_seconds() / span
    interp = vals[0] * (1 - w) + vals[1] * w
    out_val = np.zeros_like(vals[0])
    out_val[both] = interp[both].astype(np.float32)
    only_lo = (mask[0] > 0) & (mask[1] <= 0)
    out_val[only_lo] = vals[0, only_lo]
    only_hi = (mask[0] <= 0) & (mask[1] > 0)
    out_val[only_hi] = vals[1, only_hi]
    out_mask = ((mask[0] > 0) | (mask[1] > 0)).astype(np.float32)
    return out_val, out_mask, coords


def _era5_at(t: pd.Timestamp) -> Optional[Dict[str, np.ndarray]]:
    """返回 dict[var] -> (H_radar, W_radar) at level 925, var=t,q.

    load_era5_window_with_grid 给的是 native 0.25° 单时刻 (T=1).
    """
    try:
        cubes, src_lat, src_lon = load_era5_window_with_grid(t, n_frames=1)
    except Exception as e:
        log.warning("ERA5 load failed at %s: %s", t, e)
        return None
    out = {}
    # LEVELS=(925,...); 925 -> index 0
    for var in ("t", "q"):
        if var not in cubes:
            return None
        cube = cubes[var]  # (T=1, L=8, H, W)
        if cube.shape[0] < 1 or cube.shape[1] < 1:
            return None
        field = cube[0, 0]  # 925 hPa
        if not np.isfinite(field).all() or field.sum() == 0:
            # 全零可能是 fallback 零立方 (文件缺) — 视作 missing
            if field.sum() == 0:
                return None
        dense = bilinear_interp_to_radar(field, src_lat, src_lon,
                                         RADAR_LAT, RADAR_LON)
        out[var] = dense.astype(np.float32)
    return out


# -------------------- PNG 归一化 ---------------------------------------------

def _normalize_to_uint8(arr: np.ndarray, mode: str,
                        stats: dict) -> np.ndarray:
    """按 dataset-wide stats 把 float 归一化到 uint8.

    stats[mode] = {"p2": x, "p98": y, "log1p": bool}
    缺 stats 时回退到 per-image min/max (只用在单测).
    """
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    s = stats.get(mode) if stats else None
    if s and s.get("log1p", False):
        a = np.log1p(np.clip(a, 0, None))
    if s and "p2" in s and "p98" in s:
        lo, hi = float(s["p2"]), float(s["p98"])
    else:
        lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype=np.uint8)
    u = np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return u.astype(np.uint8)


def _write_png(path: Path, arr: np.ndarray, mode: str, stats: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = _normalize_to_uint8(arr, mode, stats)
    # 翻转 lat 让北向上 (RADAR_LAT 是 ascending, 显示时上=北)
    Image.fromarray(u8[::-1], mode="L").save(path, optimize=True)


# -------------------- 主函数 -------------------------------------------------

MODALITIES = [
    "radar", "pwv_dense",
    "tem_station_dense", "rhu_station_dense", "prs_station_dense",
    "era5_t_925", "era5_q_925",
]


def _ts_key(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H-%M-%S")


def dump_single_frame(timestamp: pd.Timestamp, out_root: Path,
                      normalize_stats: dict) -> bool:
    """读 5 源 -> 时间对齐 -> 空间对齐 -> 写 PNG + NPZ.

    成功返回 True; 任何不可恢复缺失 (雷达缺帧 / ERA5 缺失 / IO 错) 返回 False.
    站点 / PWV 全 NaN 用 0 填 + warning, 不算失败.
    """
    timestamp = pd.Timestamp(timestamp)
    out_root = Path(out_root)
    key = _ts_key(timestamp)

    try:
        # === 1. 雷达 (硬要求) ===
        radar = _radar_at(timestamp)
        if radar is None:
            log.warning("[%s] radar frame missing -> skip", key)
            return False

        # === 2. ERA5 (硬要求) ===
        era5 = _era5_at(timestamp)
        if era5 is None:
            log.warning("[%s] ERA5 missing/empty -> skip", key)
            return False
        era5_t = era5["t"]
        era5_q = era5["q"]

        # === 3. PWV (软要求) ===
        try:
            pwv_val, pwv_mask, pwv_coords = _pwv_at(timestamp)
        except Exception as e:
            log.warning("[%s] PWV load error: %s", key, e)
            pwv_val = np.zeros(0, dtype=np.float32)
            pwv_mask = np.zeros(0, dtype=np.float32)
            pwv_coords = np.zeros((0, 2), dtype=np.float32)
        if pwv_mask.sum() == 0:
            log.warning("[%s] PWV all-NaN, dense fallback to zero", key)
            pwv_dense = np.zeros((RADAR_H, RADAR_W), dtype=np.float32)
        else:
            pwv_dense = _idw_to_dense(pwv_val, pwv_mask, pwv_coords)

        # === 4. 地面站 (软要求) ===
        try:
            sta_val, sta_mask, sta_coords3 = _station_at(timestamp)
        except Exception as e:
            log.warning("[%s] station load error: %s", key, e)
            sta_val = np.zeros((0, len(VAR_COLS)), dtype=np.float32)
            sta_mask = np.zeros(0, dtype=np.float32)
            sta_coords3 = np.zeros((0, 3), dtype=np.float32)
        sta_coords = sta_coords3[:, :2]
        # 列索引: PRS=0, TEM=3, RHU=4
        idx_prs = VAR_COLS.index("PRS")
        idx_tem = VAR_COLS.index("TEM")
        idx_rhu = VAR_COLS.index("RHU")

        def _sta_dense(col):
            if sta_mask.sum() == 0:
                log.warning("[%s] station %s all-NaN, fallback zero", key, VAR_COLS[col])
                return np.zeros((RADAR_H, RADAR_W), dtype=np.float32), np.zeros(sta_val.shape[0], dtype=np.float32)
            col_vals = sta_val[:, col]
            # per-variable mask: 该站该值有限且 station mask 为 1
            col_mask = (sta_mask > 0) & np.isfinite(col_vals) & (col_vals != 0.0)
            if col_mask.sum() == 0:
                log.warning("[%s] station %s no valid values, fallback zero", key, VAR_COLS[col])
                return np.zeros((RADAR_H, RADAR_W), dtype=np.float32), col_mask.astype(np.float32)
            dense = _idw_to_dense(col_vals, col_mask.astype(np.float32), sta_coords)
            return dense, col_mask.astype(np.float32)

        tem_dense, tem_col_mask = _sta_dense(idx_tem)
        rhu_dense, rhu_col_mask = _sta_dense(idx_rhu)
        prs_dense, prs_col_mask = _sta_dense(idx_prs)

        # === 5. 写 PNG ===
        for mode, arr in [
            ("radar", radar),
            ("pwv_dense", pwv_dense),
            ("tem_station_dense", tem_dense),
            ("rhu_station_dense", rhu_dense),
            ("prs_station_dense", prs_dense),
            ("era5_t_925", era5_t),
            ("era5_q_925", era5_q),
        ]:
            _write_png(out_root / "png" / mode / f"{key}.png",
                       arr, mode, normalize_stats)

        # === 6. 写 NPZ (dense + sparse token) ===
        npz_path = out_root / "npz" / f"{key}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            timestamp=np.array(timestamp.isoformat()),
            radar_mmh=radar,
            pwv_dense=pwv_dense,
            tem_dense=tem_dense,
            rhu_dense=rhu_dense,
            prs_dense=prs_dense,
            era5_t_925=era5_t,
            era5_q_925=era5_q,
            # sparse tokens
            pwv_sparse_value=pwv_val.astype(np.float32),
            pwv_sparse_mask=pwv_mask.astype(np.float32),
            pwv_sparse_lonlat=pwv_coords.astype(np.float32),
            tem_sparse_value=sta_val[:, idx_tem].astype(np.float32),
            tem_sparse_mask=tem_col_mask.astype(np.float32),
            rhu_sparse_value=sta_val[:, idx_rhu].astype(np.float32),
            rhu_sparse_mask=rhu_col_mask.astype(np.float32),
            prs_sparse_value=sta_val[:, idx_prs].astype(np.float32),
            prs_sparse_mask=prs_col_mask.astype(np.float32),
            station_sparse_lonlat=sta_coords.astype(np.float32),
        )
        return True
    except Exception as e:
        log.error("[%s] dump_single_frame failed: %s\n%s",
                  key, e, traceback.format_exc())
        return False


# -------------------- CLI ---------------------------------------------------

def _load_stats(path: Optional[str]) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        log.warning("normalize_stats not found at %s, will use per-image min/max", p)
        return {}
    with open(p, "r") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Team dataset: dump one frame")
    ap.add_argument("--start", required=True, help="ISO timestamp, e.g. 2023-07-30T09:00:00")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--normalize_stats", default=None)
    args = ap.parse_args(argv)

    stats = _load_stats(args.normalize_stats)
    ok = dump_single_frame(pd.Timestamp(args.start), Path(args.out_root), stats)
    print(json.dumps({"timestamp": args.start, "ok": bool(ok)}))
    sys.exit(0 if ok else 1)


# -------------------- 单元测试 -----------------------------------------------

if __name__ == "__main__" and len(sys.argv) == 1:
    # 自测模式: 跑真实时刻
    import tempfile
    test_ts = pd.Timestamp("2023-07-30T09:00:00")
    with tempfile.TemporaryDirectory(prefix="team_dump_smoke_") as td:
        out = Path(td)
        print(f"[smoke] out_root = {out}")
        print(f"[smoke] timestamp = {test_ts}")
        ok = dump_single_frame(test_ts, out, normalize_stats={})
        print(f"[smoke] dump_single_frame returned: {ok}")
        if not ok:
            sys.exit(2)

        # 加载 NPZ 检查
        key = _ts_key(test_ts)
        npz_path = out / "npz" / f"{key}.npz"
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        print(f"[smoke] npz: {npz_path}  size={size_mb:.2f} MB")
        d = np.load(npz_path)
        for mode in MODALITIES:
            if mode == "radar":
                arr = d["radar_mmh"]
            elif mode == "pwv_dense":
                arr = d["pwv_dense"]
            elif mode == "tem_station_dense":
                arr = d["tem_dense"]
            elif mode == "rhu_station_dense":
                arr = d["rhu_dense"]
            elif mode == "prs_station_dense":
                arr = d["prs_dense"]
            elif mode == "era5_t_925":
                arr = d["era5_t_925"]
            elif mode == "era5_q_925":
                arr = d["era5_q_925"]
            else:
                continue
            nan_count = int(np.isnan(arr).sum())
            print(f"[smoke] {mode:24s} shape={arr.shape} "
                  f"min={float(arr.min()):.4g} max={float(arr.max()):.4g} "
                  f"nan={nan_count}")

        # 列 PNG
        import os
        png_root = out / "png"
        print("[smoke] PNG files:")
        for sub in sorted(os.listdir(png_root)):
            files = sorted(os.listdir(png_root / sub))
            for f in files:
                fp = png_root / sub / f
                print(f"  {fp}  {fp.stat().st_size} B")

        if size_mb >= 50:
            print(f"[smoke] WARNING: npz size {size_mb:.2f} MB >= 50 MB limit")
            sys.exit(3)
        print("[smoke] OK")
    sys.exit(0)
elif __name__ == "__main__":
    main()
