"""Build pipeline/manifest.csv.

Phase 7c (data-defect fix 2026-05-30): the split is now constrained to the
radar∩PWV intersection so the +PWV ablation is valid end-to-end.

Two unrecoverable data defects forced this:
  1. GNSS-PWV files only cover 2023-02~08 (no September). The previous split
     placed val (Sept 1–10) and test_robust (Sept 11–25) entirely in
     September → both had PWV=0. Model selection (best.pt) ran on a no-PWV
     val, blind to PWV gain; val/test ab2-vs-ab1 deltas were architecture
     confounds, not PWV effects.
  2. Radar is genuinely missing on some heavy-rain days (README claims only
     dry days were omitted, but stations PRE_1h shows ~10 May–Aug days with
     widespread heavy rain and no radar). Those stay dropped.

New splits — ALL within the May–Aug radar∩PWV window; September is dropped
entirely (it has radar but no PWV, so it cannot serve a PWV ablation):
  * event_test  (23·7 case 07-29~08-01, Figure 1 only)            24 starts/day
  * val         (8 stratified May–Aug radar days, model select)   24 starts/day
  * test_robust (8 stratified May–Aug radar days, held out)       24 starts/day
  * train       (remaining May–Aug radar days)                     8 starts/day
  * train       (zero-day no-radar weak-rain days)                 8 starts/day
  * drop        (no-radar heavy-rain days — unusable, no radar)    0

val/test are stratified across months and rain intensity (each spans
May–Aug and holds ≥1 day with 50–84 mm/h peak rain plus moderate/light
days) so the held-out sets are not all-dry. Caveat for the paper: this is
day-level holdout; a few held-out days share a multi-day synoptic process
with train days (weak temporal adjacency leakage, standard in nowcasting).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from pipeline import radar_io, station_io

OUT = Path(__file__).resolve().parent / "manifest.csv"

# 23·7 main event (held out for figure-1 case study, never used for model
# selection). DATA_HANDLING.md §8.2.
EVENT_TEST_DAYS = pd.to_datetime(["2023-07-29", "2023-07-30", "2023-07-31",
                                  "2023-08-01"])

# Model-selection set: 8 radar-present May–Aug days, stratified across months
# and rain intensity (heavy / moderate / light). All have PWV.
VAL_DAYS = pd.to_datetime([
    "2023-05-12", "2023-05-19", "2023-06-19", "2023-06-28",
    "2023-07-12", "2023-07-24", "2023-08-11", "2023-08-23",
])
# Held-out generalization set: 8 radar-present May–Aug days, same
# stratification, includes 50–84 mm/h heavy-rain days. All have PWV.
TEST_ROBUST_DAYS = pd.to_datetime([
    "2023-05-17", "2023-05-31", "2023-06-11", "2023-06-26",
    "2023-07-03", "2023-07-18", "2023-08-08", "2023-08-20",
])


def build(train_starts: int = 8, val_starts: int = 24,
          test_starts: int = 24, event_starts: int = 24):
    rows = []
    rng_may_aug = pd.date_range("2023-05-01", "2023-08-31", freq="D")
    for d in rng_may_aug:
        has_radar = radar_io.has_radar_day(d)
        if has_radar:
            if d in EVENT_TEST_DAYS:
                split, spd = "event_test", event_starts
            elif d in VAL_DAYS:
                split, spd = "val", val_starts
            elif d in TEST_ROBUST_DAYS:
                split, spd = "test_robust", test_starts
            else:
                split, spd = "train", train_starts
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "split": split, "zero_day": False,
                         "pre_total_mm": None, "n_wet": None,
                         "starts_per_day": spd})
        else:
            total, n_wet = station_io.daily_precip_diagnosis(d)
            if total < 50 and n_wet < 50:
                rows.append({"date": d.strftime("%Y-%m-%d"),
                             "split": "train", "zero_day": True,
                             "pre_total_mm": round(total, 2),
                             "n_wet": int(n_wet),
                             "starts_per_day": train_starts})
            else:
                rows.append({"date": d.strftime("%Y-%m-%d"),
                             "split": "drop", "zero_day": False,
                             "pre_total_mm": round(total, 2),
                             "n_wet": int(n_wet),
                             "starts_per_day": 0})

    # September is dropped entirely: it has radar but no PWV (PWV ends in
    # August), so it cannot serve the +PWV ablation or model selection.

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")
    print(df["split"].value_counts())
    print(f"zero_days: {int(df['zero_day'].sum())}")
    for split in ("train", "val", "test_robust", "event_test"):
        sub = df[df["split"] == split]
        n_samples = int(sub["starts_per_day"].sum())
        print(f"expected {split} samples: {n_samples} "
              f"({len(sub)} days)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-starts", type=int, default=8,
                    help="windows/day for train (default 8 = every 3h)")
    ap.add_argument("--val-starts", type=int, default=24,
                    help="windows/day for val (default 24 = every hour)")
    ap.add_argument("--test-starts", type=int, default=24,
                    help="windows/day for test_robust (default 24)")
    ap.add_argument("--event-starts", type=int, default=24,
                    help="windows/day for event_test 23·7 case (default 24)")
    args = ap.parse_args()
    build(train_starts=args.train_starts,
          val_starts=args.val_starts,
          test_starts=args.test_starts,
          event_starts=args.event_starts)


if __name__ == "__main__":
    main()
