"""Phase 7d — per-lead-time skill audit: where across the 108-min horizon does
+PWV (ab2) help or hurt, and where does skill collapse?

eval.py reports CSI/FSS averaged over ALL 18 lead frames, which hides *where*
the gain/loss lives. This breaks the same CSI(thr) / FSS(nbr) down per forecast
frame and overlays ab1 (radar-only) vs ab2 (+PWV).

    python scripts/eval_per_lead.py --ckpt_ab1 ckpt/ablation_1_p7d/best.pt \
        --ckpt_ab2 ckpt/ablation_2_p7d/best.pt --split event_test \
        --out_dir ckpt/figures/per_lead
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from model import Pluvian
from pipeline.data_loader import NPJDataset
from scripts.train import (_starts_for_splits, _collate, split_batch, _to_device,
                           _csi_counts, _fss_components)

THRESHOLDS = (1.0, 5.0, 10.0, 30.0)
FSS_NBRS = (3, 11)


def load_ckpt(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = state["config"]; m = cfg["model"]
    model = Pluvian(
        radar_channels=m["radar_channels"], pwv_enabled=m["pwv_enabled"],
        pwv_concat_only=m["pwv_concat_only"], era5_enabled=m["era5_enabled"],
        mfd_channel_enabled=m["mfd_channel_enabled"],
        forecast_frames=m["forecast_frames"], input_frames=m["input_frames"],
        hidden_dim=m["hidden_dim"], n_layers=m["n_layers"], n_heads=m["n_heads"],
        n_era5_vars=m["n_era5_vars"], n_era5_levels=m["n_era5_levels"],
        mfd_channels=m["mfd_channels"],
        intensity_stratified=bool(m.get("intensity_stratified", False)),
        band_centers=tuple(m.get("band_centers", (0.0, 0.5, 4.5, 19.0, 50.0))),
        gated_fusion=bool(m.get("gated_fusion", False)),
        cdu_decoder=bool(m.get("cdu_decoder", False)),
    ).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, cfg


@torch.no_grad()
def eval_per_lead(model, cfg, manifest, splits, device):
    T_in = cfg["model"]["input_frames"]; T_out = cfg["model"]["forecast_frames"]
    mfd_ch = cfg["model"]["mfd_channels"] if cfg["model"]["mfd_channel_enabled"] else 0
    starts = _starts_for_splits(manifest, splits)
    load_mfd = bool(cfg["model"].get("mfd_channel_enabled", False))
    load_era5 = bool(cfg["model"].get("era5_enabled", True)) or load_mfd
    ds = NPJDataset(manifest=manifest, window_minutes=cfg["data"]["window_minutes"],
                    starts=starts, drop_pwv=cfg["data"]["drop_pwv_val"],
                    load_mfd=load_mfd, load_era5=load_era5)
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], collate_fn=_collate)
    csi = {int(t): [{"h": 0, "fa": 0, "m": 0} for _ in range(T_out)] for t in THRESHOLDS}
    fss = {n: [{"num": 0.0, "den": 0.0} for _ in range(T_out)] for n in FSS_NBRS}
    nwin = 0
    for batch in loader:
        batch = _to_device(batch, device)
        model_in, rain_tgt, _ = split_batch(batch, T_in, T_out, mfd_ch)
        rain_pred = model(model_in)["rain_pred"].float()
        if rain_pred.shape[-2:] != rain_tgt.shape[-2:]:
            Ht, Wt = rain_tgt.shape[-2:]
            rain_pred = rain_pred[..., :Ht, :Wt].contiguous()
        rain_tgt = rain_tgt.float()
        nwin += rain_pred.shape[0]
        for t in range(T_out):
            p_t, g_t = rain_pred[:, t:t + 1], rain_tgt[:, t:t + 1]
            for thr in THRESHOLDS:
                h, fa, mi = _csi_counts(p_t, g_t, thr)
                c = csi[int(thr)][t]; c["h"] += h; c["fa"] += fa; c["m"] += mi
            for n in FSS_NBRS:
                num, den = _fss_components(p_t, g_t, 1.0, n)
                fss[n][t]["num"] += num; fss[n][t]["den"] += den
    out = {"n_windows": nwin, "csi": {}, "fss": {}}
    for thr in THRESHOLDS:
        out["csi"][str(int(thr))] = [
            (c["h"] / (c["h"] + c["fa"] + c["m"])) if (c["h"] + c["fa"] + c["m"]) else float("nan")
            for c in csi[int(thr)]]
    for n in FSS_NBRS:
        out["fss"][str(n)] = [(1.0 - f["num"] / f["den"]) if f["den"] else float("nan")
                              for f in fss[n]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_ab1", type=Path, required=True)
    ap.add_argument("--ckpt_ab2", type=Path, required=True)
    ap.add_argument("--split", type=str, default="event_test")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    m1, cfg1 = load_ckpt(args.ckpt_ab1, device)
    m2, cfg2 = load_ckpt(args.ckpt_ab2, device)
    manifest = REPO_ROOT / cfg2["data"]["manifest"]
    splits = [args.split]
    print(f"[per-lead] split={splits}")
    r1 = eval_per_lead(m1, cfg1, manifest, splits, device)
    r2 = eval_per_lead(m2, cfg2, manifest, splits, device)
    T = len(r1["csi"]["1"]); lead = [(t + 1) * 6 for t in range(T)]
    res = {"split": args.split, "lead_min": lead, "ab1": r1, "ab2": r2}
    (args.out_dir / f"per_lead_{args.split}.json").write_text(json.dumps(res, indent=2))

    # figure: CSI(thr) vs lead, ab1 dashed vs ab2 solid; + FSS panel
    fig, axes = plt.subplots(1, 5, figsize=(26, 4.6))
    for ax, thr in zip(axes[:4], THRESHOLDS):
        k = str(int(thr))
        ax.plot(lead, r1["csi"][k], "o--", color="C0", label="ab1 radar-only", alpha=.85)
        ax.plot(lead, r2["csi"][k], "o-", color="C3", label="ab2 +PWV", alpha=.85)
        ax.set_title(f"CSI @ {int(thr)}mm"); ax.set_xlabel("lead time (min)")
        ax.set_ylabel("CSI"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    axF = axes[4]
    for n, ls in zip(FSS_NBRS, ["-", "--"]):
        axF.plot(lead, r1["fss"][str(n)], ls, color="C0", alpha=.8, label=f"ab1 fss{n}px")
        axF.plot(lead, r2["fss"][str(n)], ls, color="C3", alpha=.8, label=f"ab2 fss{n}px")
    axF.set_title("FSS @1mm (3px,11px)"); axF.set_xlabel("lead time (min)")
    axF.set_ylabel("FSS"); axF.grid(alpha=.3); axF.legend(fontsize=8)
    fig.suptitle(f"Per-lead-time skill — {args.split}  (n={r2['n_windows']} windows)  "
                 "ab1 radar-only (dashed) vs ab2 +PWV (solid)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.out_dir / f"per_lead_{args.split}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("  saved:", out)

    # console: where does +PWV win/lose per threshold (averaged)
    for thr in THRESHOLDS:
        k = str(int(thr))
        d = np.array(r2["csi"][k]) - np.array(r1["csi"][k])
        print(f"  CSI@{int(thr):2d}mm  ab2-ab1 by lead: " +
              " ".join(f"{x:+.3f}" for x in d) + f"  | mean {np.nanmean(d):+.3f}")


if __name__ == "__main__":
    main()
