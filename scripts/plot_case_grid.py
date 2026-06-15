"""Render a GT + multi-model precipitation nowcasting case grid.

Example:
    python scripts/plot_case_grid.py \
      --model ab1=ckpt/ablation_1_p7d/best.pt \
      --model ab2=ckpt/ablation_2_p7d/best.pt \
      --model ab3=ckpt/ablation_3_era5_p7c/best.pt \
      --diff ab2-ab1 --diff ab3-ab2 --diff ab3-ab1 \
      --start 2023-07-31T06:00:00 \
      --out ckpt/figures/case_grid/ab3_2023-07-31T06.png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiffSpec:
    left: str
    right: str
    label: str


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("model arguments must use NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("model name cannot be empty")
    return name, Path(path)


def parse_diff_arg(value: str, model_names: set[str]) -> DiffSpec:
    # Prefer ' vs ' or '__minus__' as explicit separator (needed when model
    # names themselves contain '-', e.g. ab3-cdu). Fall back to first '-' for
    # backward compatibility.
    if " vs " in value:
        left, right = [p.strip() for p in value.split(" vs ", 1)]
    elif "__minus__" in value:
        left, right = [p.strip() for p in value.split("__minus__", 1)]
    elif "-" in value:
        # Greedy match: try each split position to find a valid (left, right)
        # pair where both halves are known model names. Prefers the rightmost
        # valid split so 'ab3-cdu-ab3' is parsed as ('ab3-cdu', 'ab3').
        positions = [i for i, ch in enumerate(value) if ch == "-"]
        candidates = [(value[:i].strip(), value[i+1:].strip()) for i in positions]
        valid = [(l, r) for (l, r) in candidates
                 if l in model_names and r in model_names]
        if not valid:
            raise ValueError(
                f"diff '{value}': could not match a (LEFT, RIGHT) pair to model names "
                f"{sorted(model_names)}; use ' vs ' or '__minus__' as separator"
            )
        # Prefer the split that uses the longest LEFT (matches model names with '-')
        left, right = max(valid, key=lambda lr: len(lr[0]))
    else:
        raise ValueError("diff arguments must use LEFT-RIGHT, e.g. ab3-ab2")
    missing = [name for name in (left, right) if name not in model_names]
    if missing:
        raise ValueError(f"diff references unknown model(s): {', '.join(missing)}")
    return DiffSpec(left=left, right=right, label=f"{left} - {right}")


def parse_leads(leads_min: list[int], forecast_frames: int) -> list[int]:
    out: list[int] = []
    for lead in leads_min:
        if lead <= 0 or lead % 6 != 0:
            raise ValueError(f"lead {lead} must be a positive multiple of 6 minutes")
        idx = lead // 6 - 1
        if idx < 0 or idx >= forecast_frames:
            raise ValueError(f"lead {lead} outside forecast horizon {forecast_frames * 6} minutes")
        out.append(idx)
    return out


def _load_model(path: Path, device):
    import torch

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from model.pluvian import Pluvian

    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("config") or state.get("cfg")
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
        gated_fusion=bool(m.get("gated_fusion", False)),
        intensity_stratified=bool(m.get("intensity_stratified", False)),
        band_centers=tuple(m.get("band_centers", (0.0, 0.5, 4.5, 19.0, 50.0))),
    )
    model.load_state_dict(state["model"], strict=False)
    return model.to(device).eval(), cfg


def _to_device(x, device):
    import torch

    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _make_batch(sample: dict, cfg: dict, device):
    import torch

    m = cfg["model"]
    T_in = int(m["input_frames"])
    batch = {
        "radar": sample["radar"][:T_in].unsqueeze(0).float(),
        "radar_mask": sample["radar_mask"][:T_in].unsqueeze(0).float(),
        "pwv_grid": sample["pwv_grid"].unsqueeze(0).float(),
        "pwv_mask": sample["pwv_mask"].unsqueeze(0).float(),
        "pwv_coords": sample["pwv_coords"].unsqueeze(0).float(),
        "station_grid": sample["station_grid"].unsqueeze(0).float(),
        "station_mask": sample["station_mask"].unsqueeze(0).float(),
        "station_coords": sample["station_coords"].unsqueeze(0).float(),
        "era5": {k: v[:T_in].unsqueeze(0).float() for k, v in sample["era5"].items()},
    }
    if bool(m.get("mfd_channel_enabled", False)):
        mfd_channels = int(m.get("mfd_channels", 0))
        mfd = sample["era5_mfd"][:T_in].unsqueeze(0).float()
        if mfd.shape[2] < mfd_channels:
            pad = torch.zeros(
                mfd.shape[0], mfd.shape[1], mfd_channels - mfd.shape[2],
                *mfd.shape[-2:], dtype=mfd.dtype,
            )
            mfd = torch.cat([mfd, pad], dim=2)
        elif mfd.shape[2] > mfd_channels:
            mfd = mfd[:, :, :mfd_channels]
        batch["era5_mfd"] = mfd
    return _to_device(batch, device)


def _assert_compatible(configs: list[dict]) -> tuple[int, int]:
    first = configs[0]["model"]
    T_in = int(first["input_frames"])
    T_out = int(first["forecast_frames"])
    for cfg in configs[1:]:
        m = cfg["model"]
        if int(m["input_frames"]) != T_in or int(m["forecast_frames"]) != T_out:
            raise ValueError("all models must share input_frames and forecast_frames")
    return T_in, T_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="NAME=checkpoint path; can be repeated")
    ap.add_argument("--diff", action="append", default=[],
                    help="LEFT-RIGHT diff row; names must match --model names")
    ap.add_argument("--start", required=True,
                    help="case start timestamp, e.g. 2023-07-31T06:00:00")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lead", action="append", type=int, default=None,
                    help="lead time in minutes; repeatable; default 18/36/72/108")
    ap.add_argument("--manifest", type=Path, default=Path("pipeline/manifest.csv"))
    ap.add_argument("--window_minutes", type=int, default=180)
    ap.add_argument("--diff_range", type=float, default=30.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from pipeline.data_loader import NPJDataset
    from scripts.plot_prediction import DBZ_CMAP, DBZ_NORM

    model_args = [parse_model_arg(value) for value in args.model]
    model_names = {name for name, _ in model_args}
    if len(model_names) != len(model_args):
        raise ValueError("model names must be unique")
    diffs = [parse_diff_arg(value, model_names) for value in args.diff]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = []
    configs = []
    for name, path in model_args:
        model, cfg = _load_model(path, device)
        loaded.append((name, model, cfg))
        configs.append(cfg)
    T_in, T_out = _assert_compatible(configs)
    lead_indices = parse_leads(args.lead or [18, 36, 72, 108], T_out)
    load_mfd = any(bool(cfg["model"].get("mfd_channel_enabled", False)) for cfg in configs)

    ds = NPJDataset(
        manifest=repo_root / args.manifest,
        starts=[args.start],
        load_mfd=load_mfd,
        load_era5=True,
        window_minutes=args.window_minutes,
    )
    sample = ds[0]
    gt = sample["radar"][T_in:T_in + T_out].detach().cpu().numpy()
    Ht, Wt = gt.shape[-2:]

    preds: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, model, cfg in loaded:
            out = model(_make_batch(sample, cfg, device))["rain_pred"][0]
            preds[name] = out.detach().float().cpu().numpy()[:, :Ht, :Wt]
            print(f"[case] {name}: max={preds[name].max():.2f}", flush=True)

    rows: list[tuple[str, str, np.ndarray]] = [("Ground truth", "dbz", gt)]
    rows.extend((name, "dbz", preds[name]) for name, _ in model_args)
    for diff in diffs:
        rows.append((diff.label, "diff", preds[diff.left] - preds[diff.right]))

    n_rows = len(rows)
    n_cols = len(lead_indices)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols, 2.45 * n_rows + 0.5),
        squeeze=False,
    )
    extent = [113.0, 120.0, 36.0, 42.6]
    diff_im = None
    for r, (label, kind, cube) in enumerate(rows):
        for c, lead_idx in enumerate(lead_indices):
            ax = axes[r, c]
            if kind == "diff":
                diff_im = ax.imshow(
                    cube[lead_idx], origin="lower", extent=extent,
                    cmap="RdBu_r", vmin=-args.diff_range, vmax=args.diff_range,
                    aspect="auto",
                )
            else:
                ax.imshow(
                    cube[lead_idx], origin="lower", extent=extent,
                    cmap=DBZ_CMAP, norm=DBZ_NORM, aspect="auto",
                )
                ax.contour(
                    cube[lead_idx], levels=[30.0], colors="magenta",
                    linewidths=0.8, origin="lower", extent=extent,
                )
            if r == 0:
                ax.set_title(f"+{(lead_idx + 1) * 6} min", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    title = args.title or f"Pluvian case grid start={args.start}"
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.subplots_adjust(right=0.86, top=0.965, hspace=0.18, wspace=0.10)
    dbz_sm = plt.cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_CMAP)
    dbz_ax = fig.add_axes([0.885, 0.53, 0.014, 0.32])
    fig.colorbar(dbz_sm, cax=dbz_ax, label="dBZ")
    if diff_im is not None:
        diff_ax = fig.add_axes([0.885, 0.18, 0.014, 0.24])
        fig.colorbar(diff_im, cax=diff_ax, label="difference")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[case] saved {args.out}")


if __name__ == "__main__":
    main()
