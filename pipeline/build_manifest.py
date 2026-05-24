"""Build pipeline/manifest.csv.

For 2023-05-01 ... 2023-08-31:
  - radar present -> split='train' (held-out 23·7 main event -> split='val')
  - radar missing -> classify by station-network daily PRE_1h:
        total<50mm AND >0.1mm stations<50  -> zero_day, split='train'
        else -> split='drop'
For 2023-09-01 ... 2023-09-30 with radar -> split='test_robust'.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from pipeline import radar_io, station_io

OUT = Path(__file__).resolve().parent / "manifest.csv"

# 23·7 main event (held out) per DATA_HANDLING.md §8.2
VAL_DAYS = pd.to_datetime(["2023-07-29", "2023-07-30", "2023-07-31",
                           "2023-08-01"])


def main():
    rows = []
    rng_may_aug = pd.date_range("2023-05-01", "2023-08-31", freq="D")
    for d in rng_may_aug:
        has_radar = radar_io.has_radar_day(d)
        if has_radar:
            if d in VAL_DAYS:
                split = "val"
            else:
                split = "train"
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "split": split, "zero_day": False,
                         "pre_total_mm": None, "n_wet": None})
        else:
            total, n_wet = station_io.daily_precip_diagnosis(d)
            if total < 50 and n_wet < 50:
                rows.append({"date": d.strftime("%Y-%m-%d"),
                             "split": "train", "zero_day": True,
                             "pre_total_mm": round(total, 2),
                             "n_wet": int(n_wet)})
            else:
                rows.append({"date": d.strftime("%Y-%m-%d"),
                             "split": "drop", "zero_day": False,
                             "pre_total_mm": round(total, 2),
                             "n_wet": int(n_wet)})

    for d in pd.date_range("2023-09-01", "2023-09-30", freq="D"):
        if radar_io.has_radar_day(d):
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "split": "test_robust", "zero_day": False,
                         "pre_total_mm": None, "n_wet": None})

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")
    print(df["split"].value_counts())
    print(f"zero_days: {int(df['zero_day'].sum())}")


if __name__ == "__main__":
    main()
