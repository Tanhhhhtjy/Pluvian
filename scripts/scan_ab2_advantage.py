"""Scan all starts in event_test + test_robust + val and rank by how much
ab2 (+PWV) beats ab1 (radar-only), to pick representative timepoints whose
prediction figures visibly show the PWV advantage.

Per sample we score:
  * csi advantage  : CSI(ab2)-CSI(ab1) at dBZ thresholds 10/20/30/40 (mean)
  * err_red_storm  : mean over lead frames of (|ab1-GT|-|ab2-GT|), restricted
                     to storm pixels (GT >= 20 dBZ).  + = ab2 closer to truth.
  * err_red_all    : same but over the whole grid.

Outputs eval/scan_ab2_advantage.csv sorted by err_red_storm desc.

Usage:
    python scripts/scan_ab2_advantage.py CKPT_AB1 CKPT_AB2 [OUT_CSV]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.data_loader import NPJDataset
from scripts.train import _starts_for_splits
from scripts.plot_pwv_mechanism_demo import build_model, make_batch, T_IN, T_OUT

THRS = [10.0, 20.0, 30.0, 40.0]
STORM_THR = 20.0
SPLITS = ["event_test", "test_robust", "val"]


def csi(pred, tgt, thr):
    p = pred >= thr
    t = tgt >= thr
    h = np.logical_and(p, t).sum()
    fa = np.logical_and(p, ~t).sum()
    mi = np.logical_and(~p, t).sum()
    d = h + fa + mi
    return float(h / d) if d > 0 else np.nan


def main(ckpt_ab1, ckpt_ab2, out_csv="eval/scan_ab2_advantage.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = ROOT / "pipeline" / "manifest.csv"
    starts = _starts_for_splits(manifest, SPLITS)
    print(f"scanning {len(starts)} starts over {SPLITS}")

    mf = pd.read_csv(manifest)
    day2split = dict(zip(mf["date"].astype(str).str[:10], mf["split"]))

    ds = NPJDataset(manifest=manifest, starts=starts, load_mfd=False, window_minutes=180)
    ab1 = build_model(ckpt_ab1, device)
    ab2 = build_model(ckpt_ab2, device)

    rows = []
    for i, st in enumerate(starts):
        try:
            sample = ds[i]
        except Exception as e:
            print(f"  skip {st}: {e}")
            continue
        gt = sample["radar"][T_IN:T_IN + T_OUT].numpy()
        Ht, Wt = gt.shape[-2:]
        with torch.no_grad():
            p1 = ab1(make_batch(sample, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]
            p2 = ab2(make_batch(sample, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]

        csi_adv = np.nanmean([csi(p2, gt, t) - csi(p1, gt, t) for t in THRS])
        ae1 = np.abs(p1 - gt)
        ae2 = np.abs(p2 - gt)
        err_red_all = float((ae1 - ae2).mean())
        storm = gt >= STORM_THR
        err_red_storm = float((ae1[storm] - ae2[storm]).mean()) if storm.any() else np.nan
        rows.append(dict(
            start=st, day=st[:10], split=day2split.get(st[:10], "?"),
            gt_max=float(gt.max()), storm_frac=float(storm.mean()),
            csi_adv=csi_adv, err_red_all=err_red_all, err_red_storm=err_red_storm,
        ))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(starts)}")

    df = pd.DataFrame(rows).sort_values("err_red_storm", ascending=False)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("saved:", out_csv)
    print("\n=== TOP 12 by err_red_storm (ab2 beats ab1 in storm region) ===")
    print(df.head(12).to_string(index=False))
    print("\n=== TOP 8 by csi_adv ===")
    print(df.sort_values("csi_adv", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3] if len(a) > 3 else "eval/scan_ab2_advantage.csv")
