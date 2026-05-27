"""Build pipeline/manifest.csv.

For 2023-05-01 ... 2023-08-31:
  - radar present -> split='train' (held-out 23·7 main event -> split='val')
  - radar missing -> classify by station-network daily PRE_1h:
        total<50mm AND >0.1mm stations<50  -> zero_day, split='train'
        else -> split='drop'
For 2023-09-01 ... 2023-09-30 with radar -> split='test_robust'.

Phase 7a: adds a per-row `starts_per_day` column controlling how many sample
windows are drawn from each day. Audit (2026-05-27) revealed val=32 samples
(4 days * 8 starts) carried CI ~= +/-0.05, so the new default is 24 starts/day
(every hour). With 4 val days that yields 96 val samples.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from pipeline import radar_io, station_io

OUT = Path(__file__).resolve().parent / "manifest.csv"

# 23·7 main event (held out) per DATA_HANDLING.md §8.2
VAL_DAYS = pd.to_datetime(["2023-07-29", "2023-07-30", "2023-07-31",
                           "2023-08-01"])


def build(starts_per_day: int = 24, extra_val_days: list[str] | None = None):
    extra_val = pd.to_datetime(extra_val_days) if extra_val_days else pd.DatetimeIndex([])
    val_set = pd.DatetimeIndex(VAL_DAYS).union(extra_val)

    rows = []
    rng_may_aug = pd.date_range("2023-05-01", "2023-08-31", freq="D")
    for d in rng_may_aug:
        has_radar = radar_io.has_radar_day(d)
        if has_radar:
            if d in val_set:
                split = "val"
            else:
                split = "train"
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "split": split, "zero_day": False,
                         "pre_total_mm": None, "n_wet": None,
                         "starts_per_day": starts_per_day})
        else:
            total, n_wet = station_io.daily_precip_diagnosis(d)
            if total < 50 and n_wet < 50:
                rows.append({"date": d.strftime("%Y-%m-%d"),
                             "split": "train", "zero_day": True,
                             "pre_total_mm": round(total, 2),
                             "n_wet": int(n_wet),
                             "starts_per_day": starts_per_day})
            else:
                rows.append({"date": d.strftime("%Y-%m-%d"),
                             "split": "drop", "zero_day": False,
                             "pre_total_mm": round(total, 2),
                             "n_wet": int(n_wet),
                             "starts_per_day": starts_per_day})

    for d in pd.date_range("2023-09-01", "2023-09-30", freq="D"):
        if radar_io.has_radar_day(d):
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "split": "test_robust", "zero_day": False,
                         "pre_total_mm": None, "n_wet": None,
                         "starts_per_day": starts_per_day})

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows (starts_per_day={starts_per_day})")
    print(df["split"].value_counts())
    print(f"zero_days: {int(df['zero_day'].sum())}")
    n_val_samples = int((df["split"] == "val").sum()) * starts_per_day
    n_train_samples = int((df["split"] == "train").sum()) * starts_per_day
    print(f"expected val samples: {n_val_samples}; "
          f"train samples: {n_train_samples}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts-per-day", type=int, default=24,
                    help="windows sampled per day (default 24 = every hour)")
    ap.add_argument("--extra-val-day", action="append", default=None,
                    help="extra val day(s) (YYYY-MM-DD); repeat to add multiple")
    args = ap.parse_args()
    build(starts_per_day=args.starts_per_day,
          extra_val_days=args.extra_val_day)


if __name__ == "__main__":
    main()
