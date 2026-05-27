"""Pluvian training entry-point.

Usage:
    python scripts/train.py --config configs/ablation_5_budget_loss.yaml
    python scripts/train.py --config configs/ablation_3_pwv_channel.yaml --debug
    python scripts/train.py --config configs/ablation_6_full_pluvian.yaml --resume ckpt/.../last.pt

Configurable everything via YAML; no hard-coded hyperparams.
Hand-rolled trainer (no Lightning) so the loop stays grep-able.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import Pluvian
from model.losses import (
    WeightedMSE, BMAE, FSSProxy, WaterBudgetLoss,
    FocalRainLoss, TweedieDevianceLoss,
)
from pipeline.data_loader import NPJDataset
from pipeline.utils import RADAR_LAT, RADAR_LON


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load a YAML config with optional `inherits: <other.yaml>` for defaults."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("inherits", None)
    if parent is not None:
        base = load_config(path.parent / parent)
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------

def _starts_for_splits(manifest_path: Path, splits: list[str]) -> list[str]:
    mf = pd.read_csv(manifest_path)
    mf = mf[mf["split"].isin(splits)]
    starts = []
    for _, row in mf.iterrows():
        day = pd.Timestamp(row["date"])
        # Phase 7a: honor per-row `starts_per_day` from the manifest. Legacy
        # manifests without the column fall back to the historical 8/day spacing.
        spd = int(row["starts_per_day"]) if "starts_per_day" in row and not pd.isna(row["starts_per_day"]) else 8
        if spd <= 0:
            continue
        step_hours = 24.0 / spd
        for k in range(spd):
            offset = pd.Timedelta(hours=step_hours * k)
            starts.append((day + offset).isoformat())
    return starts


def build_datasets(cfg: dict) -> tuple[NPJDataset, NPJDataset]:
    manifest = REPO_ROOT / cfg["data"]["manifest"]
    train_starts = _starts_for_splits(manifest, cfg["data"]["splits_train"])
    val_starts = _starts_for_splits(manifest, cfg["data"]["splits_val"])
    win = cfg["data"]["window_minutes"]
    load_mfd = bool(cfg["model"].get("mfd_channel_enabled", False))
    # Skip ERA5 loading when model doesn't use it (huge DataLoader speedup)
    load_era5 = bool(cfg["model"].get("era5_enabled", True)) or \
                bool(cfg["loss"].get("budget", {}).get("enabled", False))
    train_ds = NPJDataset(
        manifest=manifest, window_minutes=win,
        starts=train_starts, drop_pwv=cfg["data"]["drop_pwv_train"],
        load_mfd=load_mfd, load_era5=load_era5,
    )
    val_ds = NPJDataset(
        manifest=manifest, window_minutes=win,
        starts=val_starts, drop_pwv=cfg["data"]["drop_pwv_val"],
        load_mfd=load_mfd, load_era5=load_era5,
    )
    return train_ds, val_ds


def _collate(batch: list[dict]) -> dict:
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        elif isinstance(v0, dict):
            sub = {}
            for kk in v0.keys():
                vv0 = v0[kk]
                if isinstance(vv0, torch.Tensor):
                    sub[kk] = torch.stack([b[k][kk] for b in batch], dim=0)
                else:
                    sub[kk] = [b[k][kk] for b in batch]
            out[k] = sub
        else:
            out[k] = [b[k] for b in batch]
    return out


# ---------------------------------------------------------------------------
# batch slicing — split radar/ERA5 along time into (input, target)
# ---------------------------------------------------------------------------

def _slice_time(t: torch.Tensor, idx: slice) -> torch.Tensor:
    # tensors of shape (B, T, ...) — time is dim 1
    return t[:, idx]


def split_batch(batch: dict, T_in: int, T_out: int, mfd_channels: int):
    """Cleave the dataset window into:
        model_in   — batch dict slimmed to first T_in frames (radar + ERA5)
        rain_tgt   — radar[:, T_in : T_in+T_out]
        era5_future— u/v/q at future window (for budget loss)
    """
    radar = batch["radar"].float()
    radar_in = radar[:, :T_in]
    rain_tgt = radar[:, T_in:T_in + T_out]
    era5 = batch["era5"]
    era5_in = {k: era5[k][:, :T_in].float() for k in era5}
    era5_fut = {k: era5[k][:, T_in:T_in + T_out].float() for k in era5}

    model_in = {
        "radar": radar_in,
        "radar_mask": batch["radar_mask"][:, :T_in],
        "pwv_grid": batch["pwv_grid"],
        "pwv_mask": batch["pwv_mask"],
        "pwv_coords": batch["pwv_coords"],
        "station_grid": batch["station_grid"],
        "station_mask": batch["station_mask"],
        "station_coords": batch["station_coords"],
        "era5": era5_in,
    }
    if mfd_channels > 0:
        if "era5_mfd" in batch:
            mfd = batch["era5_mfd"].float()
            # NPJDataset emits (B, T, L_mfd, H, W) on the radar grid; slice
            # the input-time window. If the channel count drifts, pad/trim.
            mfd_in = mfd[:, :T_in]
            if mfd_in.shape[2] < mfd_channels:
                pad = torch.zeros(
                    mfd_in.shape[0], mfd_in.shape[1],
                    mfd_channels - mfd_in.shape[2], *mfd_in.shape[-2:],
                    dtype=mfd_in.dtype, device=mfd_in.device,
                )
                mfd_in = torch.cat([mfd_in, pad], dim=2)
            elif mfd_in.shape[2] > mfd_channels:
                mfd_in = mfd_in[:, :, :mfd_channels]
            model_in["era5_mfd"] = mfd_in
        else:
            # Fallback only if dataset isn't emitting MFD (e.g. legacy ckpt).
            B = radar.shape[0]
            H, W = radar.shape[-2:]
            model_in["era5_mfd"] = torch.zeros(
                B, T_in, mfd_channels, H, W,
                dtype=radar.dtype, device=radar.device,
            )
    return model_in, rain_tgt, era5_fut


# ---------------------------------------------------------------------------
# losses + metrics
# ---------------------------------------------------------------------------

def build_losses(cfg: dict) -> dict:
    ltype = cfg["loss"]["data"]["type"]
    intensity_on = bool(cfg.get("model", {}).get("intensity_stratified", False))
    # Phase 7b audit M3: when the model emits band logits the configured data
    # loss is silently overridden by _intensity_loss at train time, but the
    # *val* loop still evaluates losses["data"]. That made train/val data-loss
    # plot different quantities. Fail loud: caller must pick exactly one.
    if intensity_on and ltype != "intensity":
        raise ValueError(
            f"intensity_stratified=True conflicts with loss.data.type={ltype!r}; "
            "set loss.data.type='intensity' to use the band-stratified loss"
        )
    if not intensity_on and ltype == "intensity":
        raise ValueError(
            "loss.data.type='intensity' requires model.intensity_stratified=True"
        )
    if ltype == "weighted_mse":
        data_loss = WeightedMSE()
    elif ltype == "bmae":
        data_loss = BMAE()
    elif ltype == "focal_rain":
        fr_cfg = cfg["loss"]["data"].get("focal_rain", {})
        data_loss = FocalRainLoss(
            rain_threshold=fr_cfg.get("threshold", 10.0),
            sharpness=fr_cfg.get("sharpness", 5.0),
            alpha=fr_cfg.get("alpha", 0.75),
            gamma=fr_cfg.get("gamma", 2.0),
        )
    elif ltype == "tweedie":
        tw_cfg = cfg["loss"]["data"].get("tweedie", {})
        data_loss = TweedieDevianceLoss(
            p=tw_cfg.get("p", 1.5), eps=tw_cfg.get("eps", 1e-6),
        )
    elif ltype == "intensity":
        # Sentinel value: train_one_epoch + validate call _intensity_loss
        # directly. We still return a callable so val's data-loss accumulator
        # has something well-defined to log (the same band CE+reg combo).
        data_loss = None
    else:
        raise ValueError(f"unknown data loss type: {ltype}")
    # FSS: support legacy single ``threshold`` or new list of ``thresholds``.
    fss_list: list[tuple[FSSProxy, float]] = []
    if cfg["loss"]["fss"]["enabled"]:
        fss_cfg = cfg["loss"]["fss"]
        thresholds = fss_cfg.get("thresholds")
        if thresholds is None:
            thresholds = [fss_cfg.get("threshold", 1.0)]
        thr_weights = fss_cfg.get("threshold_weights")
        if thr_weights is None:
            thr_weights = [1.0 if t < 10.0 else 0.1 for t in thresholds]
        if len(thr_weights) != len(thresholds):
            raise ValueError("fss.threshold_weights length mismatch")
        win = fss_cfg["window"]
        for thr, w in zip(thresholds, thr_weights):
            fss_list.append((FSSProxy(threshold=thr, window=win), float(w)))
    budget = None
    if cfg["loss"]["budget"]["enabled"]:
        budget = WaterBudgetLoss(
            delta_p_pa=cfg["loss"]["budget"]["delta_p_pa"],
            smooth_sigma=cfg["loss"]["budget"]["smooth_sigma"],
            huber_delta=cfg["loss"]["budget"]["huber_delta"],
        )
    return {"data": data_loss, "fss": fss_list, "budget": budget}


def _intensity_loss(rain_logits: torch.Tensor, rain_pred: torch.Tensor,
                    rain_tgt: torch.Tensor, cfg: dict
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MTLDM-style stratified loss.

    Returns ``(total, ce, weighted_reg)`` where each is a scalar tensor.
    ``rain_logits``: (B, T, K, H, W); ``rain_pred``/``rain_tgt``: (B, T, H, W).
    """
    icfg = cfg["loss"].get("intensity", {})
    edges = icfg.get("band_edges", [0.1, 1.0, 8.0, 30.0])
    sample_w = icfg.get("band_sample_weights", [0.1, 1.0, 2.0, 4.0, 8.0])
    ce_w = float(icfg.get("ce_weight", 1.0))
    reg_w = float(icfg.get("reg_weight", 1.0))

    K = rain_logits.shape[2]
    if len(sample_w) != K:
        raise ValueError(f"band_sample_weights length {len(sample_w)} != K={K}")

    edges_t = torch.tensor(edges, dtype=rain_tgt.dtype, device=rain_tgt.device)
    band_idx = torch.bucketize(rain_tgt.clamp(min=0.0), edges_t).long()

    logits_flat = rain_logits.permute(0, 1, 3, 4, 2).reshape(-1, K)
    target_flat = band_idx.reshape(-1)
    ce = F.cross_entropy(logits_flat, target_flat)

    w_t = torch.tensor(sample_w, dtype=rain_pred.dtype, device=rain_pred.device)
    per_pix_w = w_t[band_idx]
    sq = (rain_pred - rain_tgt) ** 2
    reg = (sq * per_pix_w).mean()

    total = ce_w * ce + reg_w * reg
    return total, ce.detach(), reg.detach()


def _csi(pred: torch.Tensor, target: torch.Tensor, thr: float) -> float:
    p = (pred >= thr)
    t = (target >= thr)
    hits = (p & t).sum().item()
    fa = (p & ~t).sum().item()
    miss = (~p & t).sum().item()
    denom = hits + fa + miss
    return hits / denom if denom > 0 else float("nan")


def _csi_counts(pred: torch.Tensor, target: torch.Tensor, thr: float) -> tuple[int, int, int]:
    """Return (hits, false_alarms, misses) for one batch at threshold `thr`.

    Used by the Phase 7a full-set accumulator: callers sum the three integers
    across the whole val epoch and compute CSI once at the end. Batch-averaged
    Used by the Phase 7a full-set accumulator: callers sum the three integers
    across the whole val epoch and compute CSI once at the end. Batch-averaged
    CSI (the old behaviour) over-weighted small-rain batches and pulled the
    val estimate around by ~0.02-0.05, which was indistinguishable from the
    ab2-vs-ab3 deltas we were trying to compare.
    """
    p = (pred >= thr)
    t = (target >= thr)
    hits = int((p & t).sum().item())
    fa = int((p & ~t).sum().item())
    miss = int((~p & t).sum().item())
    return hits, fa, miss


def _fss_binary(pred: torch.Tensor, target: torch.Tensor, thr: float,
                window: int) -> float:
    # binary FSS over (H,W) per (B*T) slice
    p = (pred >= thr).float()
    t = (target >= thr).float()
    B, T, H, W = p.shape
    p = p.reshape(B * T, 1, H, W)
    t = t.reshape(B * T, 1, H, W)
    pad = window // 2
    fp = F.avg_pool2d(p, window, stride=1, padding=pad, count_include_pad=False)
    ft = F.avg_pool2d(t, window, stride=1, padding=pad, count_include_pad=False)
    num = ((fp - ft) ** 2).mean().item()
    den = ((fp ** 2).mean() + (ft ** 2).mean()).clamp(min=1e-12).item()
    return 1.0 - num / den


def _fss_components(pred: torch.Tensor, target: torch.Tensor, thr: float,
                    window: int) -> tuple[float, float]:
    """Return (sum_sq_diff, sum_ref) summed across pixels for the batch.

    Full-set FSS = 1 - (sum of (fp-ft)^2) / (sum of fp^2 + sum of ft^2).
    Accumulating numerator/denominator sums gives the same answer as
    evaluating on the concatenated tensor, unlike the batch-mean form which
    drifted by ~0.01 per uneven batch (drop_last=False).
    """
    p = (pred >= thr).float()
    t = (target >= thr).float()
    B, T, H, W = p.shape
    p = p.reshape(B * T, 1, H, W)
    t = t.reshape(B * T, 1, H, W)
    pad = window // 2
    fp = F.avg_pool2d(p, window, stride=1, padding=pad, count_include_pad=False)
    ft = F.avg_pool2d(t, window, stride=1, padding=pad, count_include_pad=False)
    num = float(((fp - ft) ** 2).sum().item())
    den = float((fp ** 2).sum().item() + (ft ** 2).sum().item())
    return num, den


def _crps_ensemble(samples: torch.Tensor, target: torch.Tensor) -> float:
    """Proper CRPS for a K-member ensemble via the empirical formula::

        CRPS = mean_i |x_i - y| - 1/(2 K^2) sum_{i,j} |x_i - x_j|

    Parameters
    ----------
    samples : (K, ...) tensor of forecast members.
    target  : (...) tensor — broadcast over members.
    """
    if samples.dim() == target.dim():
        # Treat single-member as a (1, ...) ensemble — degenerates to MAE.
        samples = samples.unsqueeze(0)
    K = samples.shape[0]
    y = target.unsqueeze(0)
    mae_term = (samples - y).abs().mean()
    # Pairwise |x_i - x_j| averaged over K^2 (including i=j -> 0).
    # Reshape so we can broadcast: (K, 1, N) - (1, K, N) -> (K, K, N)
    flat = samples.reshape(K, -1)
    pw = (flat.unsqueeze(0) - flat.unsqueeze(1)).abs().mean()
    return (mae_term - 0.5 * pw).item()



def _crps_marginal(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Backwards-compatible alias — degenerate single-member CRPS == MAE."""
    return _crps_ensemble(pred.unsqueeze(0), target)


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------

def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / (total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

class CkptManager:
    def __init__(self, save_dir: Path, monitor: str, mode: str, top_k: int):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k
        self.records: list[tuple[float, Path]] = []  # (score, path)

    def _better(self, a: float, b: float) -> bool:
        return a > b if self.mode == "max" else a < b

    def update(self, state: dict, score: float, epoch: int):
        path = self.save_dir / f"epoch{epoch:03d}.pt"
        torch.save(state, path)
        torch.save(state, self.save_dir / "last.pt")
        self.records.append((score, path))
        # sort best first
        self.records.sort(key=lambda r: -r[0] if self.mode == "max" else r[0])
        # Copy the historically-best epoch file into best.pt. Audit item #1
        # (2026-05-27): previous version dumped the CURRENT epoch state into
        # best.pt regardless of score, silently invalidating every "best" eval.
        best = self.records[0][1]
        best_link = self.save_dir / "best.pt"
        shutil.copyfile(best, best_link)
        # prune
        keep = self.records[: self.top_k]
        drop = self.records[self.top_k:]
        for _, p in drop:
            if p.exists() and p != best:
                p.unlink()
        self.records = keep


# ---------------------------------------------------------------------------
# OOM-resilient forward
# ---------------------------------------------------------------------------

def _oom_retry(fn, *args, retries: int = 2, **kw):
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == retries:
                raise
            print(f"[OOM] retrying ({attempt+1}/{retries}) after cache clear")


# ---------------------------------------------------------------------------
# train + val loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, opt, losses, cfg, device, dtype, epoch,
                    global_step, total_steps, writer, debug: bool):
    model.train()
    grad_accum = cfg["hardware"]["grad_accum"]
    grad_clip = cfg["optim"]["grad_clip"]
    lr_base = cfg["optim"]["lr"]
    warmup_steps = cfg["optim"]["warmup_epochs"] * max(1, len(loader)) // grad_accum
    T_in = cfg["model"]["input_frames"]
    T_out = cfg["model"]["forecast_frames"]
    mfd_ch = cfg["model"]["mfd_channels"] if cfg["model"]["mfd_channel_enabled"] else 0
    log_every = cfg["log"]["every_n_steps"]
    budget_cfg = cfg["loss"]["budget"]
    intensity_on = bool(cfg["model"].get("intensity_stratified", False))

    lat_t = torch.from_numpy(np.asarray(RADAR_LAT, dtype=np.float32)).to(device)
    lon_t = torch.from_numpy(np.asarray(RADAR_LON, dtype=np.float32)).to(device)

    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    running = {"data": 0.0, "fss": 0.0, "budget": 0.0, "total": 0.0, "n": 0}

    for it, batch in enumerate(loader):
        batch = _to_device(batch, device)
        model_in, rain_tgt, era5_fut = split_batch(batch, T_in, T_out, mfd_ch)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            out = model(model_in)
            rain_pred = out["rain_pred"]
            pwv_pred = out["pwv_pred"]

            # Model pads HW up to multiple of 8 due to stem 8x downsample.
            # Crop back to target shape so loss can compare.
            if rain_pred.shape[-2:] != rain_tgt.shape[-2:]:
                Ht, Wt = rain_tgt.shape[-2:]
                rain_pred = rain_pred[..., :Ht, :Wt].contiguous()
                pwv_pred = pwv_pred[..., :Ht, :Wt].contiguous()

            if intensity_on and "rain_logits" in out:
                rain_logits = out["rain_logits"]
                if rain_logits.shape[-2:] != rain_tgt.shape[-2:]:
                    Ht, Wt = rain_tgt.shape[-2:]
                    rain_logits = rain_logits[..., :Ht, :Wt].contiguous()
                loss_data, _ce_v, _reg_v = _intensity_loss(
                    rain_logits, rain_pred, rain_tgt, cfg
                )
                loss_data = loss_data * cfg["loss"]["data"]["weight"]
            else:
                loss_data = losses["data"](rain_pred, rain_tgt) * cfg["loss"]["data"]["weight"]
            loss_total = loss_data
            loss_fss_v = torch.zeros((), device=device)
            if losses["fss"]:
                fss_w = cfg["loss"]["fss"]["weight"]
                for fss_mod, thr_w in losses["fss"]:
                    loss_fss_v = loss_fss_v + fss_mod(rain_pred, rain_tgt) * thr_w
                loss_fss_v = loss_fss_v * fss_w
                loss_total = loss_total + loss_fss_v
            loss_budget_v = torch.zeros((), device=device)
            if losses["budget"] is not None:
                lvl = budget_cfg["level_idx"]
                u = era5_fut["u"][:, :, lvl]
                v = era5_fut["v"][:, :, lvl]
                q = era5_fut["q"][:, :, lvl]
                out_budget = losses["budget"](
                    pwv_pred=pwv_pred, rain_pred=rain_pred,
                    u_era5=u, v_era5=v, q_era5=q,
                    lat=lat_t, lon=lon_t,
                )
                loss_budget_v = out_budget["loss"] * budget_cfg["weight"]
                loss_total = loss_total + loss_budget_v

        loss_scaled = loss_total / grad_accum
        loss_scaled.backward()

        if (it + 1) % grad_accum == 0:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # LR schedule
            sched_step = global_step
            if cfg["optim"]["scheduler"] == "cosine":
                lr_mult = cosine_with_warmup(sched_step, total_steps, warmup_steps)
            else:
                lr_mult = 1.0
            for pg in opt.param_groups:
                pg["lr"] = lr_base * lr_mult
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

        bs = rain_pred.shape[0]
        running["data"] += loss_data.item() * bs
        running["fss"] += loss_fss_v.item() * bs
        running["budget"] += loss_budget_v.item() * bs
        running["total"] += loss_total.item() * bs
        running["n"] += bs

        if (it + 1) % log_every == 0:
            cur_lr = opt.param_groups[0]["lr"]
            msg = (f"[ep{epoch:03d}|it{it+1:04d}/{len(loader)}] "
                   f"data={loss_data.item():.4f} "
                   f"fss={loss_fss_v.item():.4f} "
                   f"bud={loss_budget_v.item():.4f} "
                   f"tot={loss_total.item():.4f} "
                   f"lr={cur_lr:.2e} "
                   f"step={global_step}")
            print(msg, flush=True)
            if writer is not None:
                writer.add_scalar("train/data", loss_data.item(), global_step)
                writer.add_scalar("train/fss", loss_fss_v.item(), global_step)
                writer.add_scalar("train/budget", loss_budget_v.item(), global_step)
                writer.add_scalar("train/total", loss_total.item(), global_step)
                writer.add_scalar("train/lr", cur_lr, global_step)

        if debug and it + 1 >= 5:
            print("[debug] stopping after 5 batches")
            break

    n = max(1, running["n"])
    print(f"[ep{epoch:03d}] avg data={running['data']/n:.4f} "
          f"fss={running['fss']/n:.4f} budget={running['budget']/n:.4f} "
          f"total={running['total']/n:.4f}  ({time.time()-t0:.1f}s)")
    return global_step


@torch.no_grad()
def validate(model, loader, losses, cfg, device, dtype, epoch, writer, debug: bool):
    model.eval()
    T_in = cfg["model"]["input_frames"]
    T_out = cfg["model"]["forecast_frames"]
    mfd_ch = cfg["model"]["mfd_channels"] if cfg["model"]["mfd_channel_enabled"] else 0
    thresholds = (1.0, 5.0, 10.0, 30.0)
    fss_nbrs = (3, 11)
    # Phase 7a: accumulate raw counts across the whole val epoch and compute
    # one CSI/FSS at the end. The old per-batch average over-weighted small-
    # rain batches by ~0.02-0.05 — same order as the ab2/ab3 deltas we were
    # trying to compare.
    csi_totals = {int(t): {"hits": 0, "fa": 0, "miss": 0} for t in thresholds}
    fss_thresholds = (1.0, 10.0, 30.0)
    fss_totals = {(nbr, int(thr)): {"num": 0.0, "den": 0.0}
                  for nbr in fss_nbrs for thr in fss_thresholds}
    agg = {}
    agg["data"] = []
    agg["budget"] = []
    agg["crps"] = []
    agg["spread"] = []

    n_crps = int(cfg["loss"].get("crps_samples", 4))
    intensity_on = bool(cfg["model"].get("intensity_stratified", False))

    lat_t = torch.from_numpy(np.asarray(RADAR_LAT, dtype=np.float32)).to(device)
    lon_t = torch.from_numpy(np.asarray(RADAR_LON, dtype=np.float32)).to(device)
    budget_cfg = cfg["loss"]["budget"]

    for it, batch in enumerate(loader):
        batch = _to_device(batch, device)
        model_in, rain_tgt, era5_fut = split_batch(batch, T_in, T_out, mfd_ch)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            # Phase 7b audit S1: encoder/fusion is deterministic; run it once
            # and feed (fused, skips) into both the deterministic decoder and
            # the K-member MC-dropout sampler. Previously the encoder ran 2x
            # per val batch (once for `out = model(model_in)`, once inside
            # mc_dropout_predict), wasting ~30-50% of val time.
            fused, skips = model._encode_and_fuse(model_in, return_skips=True)
            decoder_out = model.decoder(fused, encoder_skips=skips)
            if len(decoder_out) == 3:
                rain_pred_t, pwv_pred_t, rain_logits_t = decoder_out
            else:
                rain_pred_t, pwv_pred_t = decoder_out
                rain_logits_t = None
            out = {"rain_pred": rain_pred_t, "pwv_pred": pwv_pred_t,
                   "radar_feat": fused}
            if rain_logits_t is not None:
                out["rain_logits"] = rain_logits_t
            if n_crps > 1 and hasattr(model, "mc_dropout_predict_from_fused"):
                mc_samples = model.mc_dropout_predict_from_fused(
                    fused, skips, n_samples=n_crps,
                )
            else:
                mc_samples = None
        rain_pred = out["rain_pred"].float()
        pwv_pred = out["pwv_pred"].float()

        # Crop model output back to target shape (stem 8x padding)
        if rain_pred.shape[-2:] != rain_tgt.shape[-2:]:
            Ht, Wt = rain_tgt.shape[-2:]
            rain_pred = rain_pred[..., :Ht, :Wt].contiguous()
            pwv_pred = pwv_pred[..., :Ht, :Wt].contiguous()

        rain_tgt_f = rain_tgt.float()

        # Phase 7b audit M3: when band logits drive training, val must use the
        # same intensity loss — not the (now-None or mismatched) weighted_mse.
        if intensity_on and "rain_logits" in out:
            rain_logits = out["rain_logits"].float()
            if rain_logits.shape[-2:] != rain_tgt_f.shape[-2:]:
                Ht, Wt = rain_tgt_f.shape[-2:]
                rain_logits = rain_logits[..., :Ht, :Wt].contiguous()
            data_v, _, _ = _intensity_loss(rain_logits, rain_pred, rain_tgt_f, cfg)
            agg["data"].append(data_v.item())
        else:
            agg["data"].append(losses["data"](rain_pred, rain_tgt_f).item())
        for thr in thresholds:
            h, fa, m = _csi_counts(rain_pred, rain_tgt_f, thr)
            ct = csi_totals[int(thr)]
            ct["hits"] += h; ct["fa"] += fa; ct["miss"] += m
        for nbr in fss_nbrs:
            for thr in fss_thresholds:
                num, den = _fss_components(rain_pred, rain_tgt_f, thr, nbr)
                key = (nbr, int(thr))
                fss_totals[key]["num"] += num
                fss_totals[key]["den"] += den
        # ---- real CRPS via K-member MC-dropout sampling ----
        if mc_samples is not None:
            samples = mc_samples.float()
            if samples.shape[-2:] != rain_tgt_f.shape[-2:]:
                Ht, Wt = rain_tgt_f.shape[-2:]
                samples = samples[..., :Ht, :Wt].contiguous()
            agg["crps"].append(_crps_ensemble(samples, rain_tgt_f))
            agg["spread"].append(samples.std(dim=0).mean().item())
        else:
            # Fallback: degenerate single-member CRPS == MAE.
            agg["crps"].append(_crps_marginal(rain_pred, rain_tgt_f))
            agg["spread"].append(0.0)

        if losses["budget"] is not None:
            lvl = budget_cfg["level_idx"]
            u = era5_fut["u"][:, :, lvl]
            v = era5_fut["v"][:, :, lvl]
            q = era5_fut["q"][:, :, lvl]
            ob = losses["budget"](
                pwv_pred=pwv_pred, rain_pred=rain_pred,
                u_era5=u, v_era5=v, q_era5=q, lat=lat_t, lon=lon_t,
            )
            agg["budget"].append(ob["loss"].item())

        if debug and it + 1 >= 5:
            break

    metrics = {}
    for k, vs in agg.items():
        vs = [v for v in vs if v == v]  # drop NaN
        metrics[f"val/{k}"] = float(np.mean(vs)) if vs else float("nan")
    # Phase 7a: full-set CSI/FSS using accumulated counts.
    for thr_int, ct in csi_totals.items():
        denom = ct["hits"] + ct["fa"] + ct["miss"]
        metrics[f"val/csi_{thr_int}mm"] = ct["hits"] / denom if denom > 0 else float("nan")
    for (nbr, thr_int), ft in fss_totals.items():
        metrics[f"val/fss_{nbr}px_{thr_int}mm"] = (
            1.0 - ft["num"] / ft["den"] if ft["den"] > 0 else float("nan"))
    # Phase 7b audit S2: tag the CRPS provenance. In intensity mode, MC-dropout
    # samples ride on a fixed band lattice and per-pixel std (val/spread) is
    # structurally smaller than for the continuous head — comparing CRPS across
    # head topologies without this tag is misleading.
    metrics["val/crps_method"] = "intensity" if intensity_on else "continuous"
    msg = "[val] " + " ".join(
        f"{k.split('/')[1]}={v:.4f}" if isinstance(v, float) else f"{k.split('/')[1]}={v}"
        for k, v in metrics.items()
    )
    print(msg)
    if writer is not None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(k, v, epoch)
    return metrics


def _to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--debug", action="store_true",
                    help="run 5 batches/epoch and exit after 1 epoch")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override config max_epochs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[config] {args.config}")
    print(json.dumps(cfg, indent=2, default=str))

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device(cfg["hardware"]["device"]
                          if torch.cuda.is_available() else "cpu")
    prec = cfg["hardware"]["precision"]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[prec]

    # ---- data ----
    train_ds, val_ds = build_datasets(cfg)
    print(f"[data] train windows: {len(train_ds)}, val windows: {len(val_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], collate_fn=_collate,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
        collate_fn=_collate, pin_memory=(device.type == "cuda"),
    )

    # ---- model ----
    m = cfg["model"]
    model = Pluvian(
        radar_channels=m["radar_channels"],
        pwv_enabled=m["pwv_enabled"],
        pwv_concat_only=m["pwv_concat_only"],
        era5_enabled=m["era5_enabled"],
        mfd_channel_enabled=m["mfd_channel_enabled"],
        forecast_frames=m["forecast_frames"],
        input_frames=m["input_frames"],
        hidden_dim=m["hidden_dim"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        n_era5_vars=m["n_era5_vars"],
        n_era5_levels=m["n_era5_levels"],
        mfd_channels=m["mfd_channels"],
        intensity_stratified=bool(m.get("intensity_stratified", False)),
        band_centers=tuple(m.get("band_centers", (0.0, 0.5, 4.5, 19.0, 50.0))),
        gated_fusion=bool(m.get("gated_fusion", False)),
    ).to(device)
    print(f"[model] params: {model.num_parameters()/1e6:.2f}M")

    # ---- optim / losses ----
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"], betas=(0.9, 0.95),
    )
    losses = build_losses(cfg)
    for k, v in losses.items():
        if v is None:
            continue
        if k == "fss":
            for fss_mod, _w in v:
                fss_mod.to(device)
        else:
            v.to(device)

    # ---- resume ----
    start_epoch = 0
    global_step = 0
    if args.resume is not None and args.resume.exists():
        state = torch.load(args.resume, map_location=device)
        # Phase 7b audit M2: legacy ab1/2/3 ckpts predate U-Net skip + lead-time
        # cross-attn + gated fusion params. strict=True crashes the resume.
        # Use strict=False and warn loudly when the delta is large so silent
        # architecture mismatches don't slip through.
        result = model.load_state_dict(state["model"], strict=False)
        n_missing = len(result.missing_keys)
        n_unexpected = len(result.unexpected_keys)
        if n_missing or n_unexpected:
            print(f"[resume] non-strict load: {n_missing} missing, "
                  f"{n_unexpected} unexpected keys")
            if n_missing > 10 or n_unexpected > 10:
                print(f"[resume][WARNING] >10 key delta — this looks like a "
                      f"legacy/cross-architecture ckpt. First missing: "
                      f"{result.missing_keys[:5]}; first unexpected: "
                      f"{result.unexpected_keys[:5]}")
        opt.load_state_dict(state["opt"])
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("global_step", 0)
        print(f"[resume] loaded {args.resume} -> epoch {start_epoch}")

    # ---- logging ----
    save_dir = Path(cfg["ckpt"]["save_dir"]) / cfg["name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if cfg["log"]["tensorboard"]:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(save_dir / "tb")
        except ImportError:
            print("[log] tensorboard not installed, skipping")

    ckpt = CkptManager(
        save_dir=save_dir, monitor=cfg["ckpt"]["monitor"],
        mode=cfg["ckpt"]["mode"], top_k=cfg["ckpt"]["save_top_k"],
    )

    max_epochs = args.max_epochs or cfg["optim"]["max_epochs"]
    if args.debug:
        max_epochs = min(max_epochs, 1)
    total_steps = max_epochs * max(1, len(train_loader)) // cfg["hardware"]["grad_accum"]
    metric_log: list[dict] = []

    for epoch in range(start_epoch, max_epochs):
        global_step = train_one_epoch(
            model, train_loader, opt, losses, cfg, device, dtype,
            epoch, global_step, total_steps, writer, args.debug,
        )
        metrics = validate(
            model, val_loader, losses, cfg, device, dtype,
            epoch, writer, args.debug,
        )
        score_key = cfg["ckpt"]["monitor"]
        score = metrics.get(score_key, float("nan"))
        if not (isinstance(score, float) and math.isnan(score)):
            ckpt.update(
                state={
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": cfg,
                    "metrics": metrics,
                },
                score=score, epoch=epoch,
            )
        metric_log.append({"epoch": epoch, **metrics})
        with open(save_dir / "metric_log.json", "w") as f:
            json.dump(metric_log, f, indent=2)
        if args.debug:
            break

    if writer is not None:
        writer.close()
    print(f"[done] checkpoints in {save_dir}")


if __name__ == "__main__":
    main()
