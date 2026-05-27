"""Build pipeline/manifest.csv.

Phase 7a (audit 2026-05-27): the split policy is now leakage-aware.
The old val set was the 23·7 京津冀 暴雨 case (2023-07-29 ~ 08-01) — the
same 4 days that the case study figures visualize. Using it for both
model selection AND the marquee figure is data leakage.

New splits:
  * train       (May-Aug days with radar, minus 23·7 case)         8 starts/day
  * val         (first half of September: 7 robust-test days)     24 starts/day
  * test_robust (second half of September: 7 robust-test days)    24 starts/day
  * event_test  (the 23·7 case study — Figure 1 only)             24 starts/day
  * train       (zero-day no-radar weak-rain days)                 8 starts/day

This separates the model-selection signal from the figure-1 case and
keeps train sample count at the historical 816 (102 × 8) while giving
val a larger 168-sample (7 × 24) base for tighter CI.
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

# September robust-test pool. Split deterministically by date: the earlier
# half becomes val (model selection); the later half is held-out test.
_SEPT_VAL_DAYS = pd.to_datetime([
    "2023-09-01", "2023-09-02", "2023-09-03",
    "2023-09-07", "2023-09-08", "2023-09-09", "2023-09-10",
])
_SEPT_TEST_DAYS = pd.to_datetime([
    "2023-09-11", "2023-09-15", "2023-09-16", "2023-09-17",
    "2023-09-23", "2023-09-24", "2023-09-25",
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

    for d in pd.date_range("2023-09-01", "2023-09-30", freq="D"):
        if not radar_io.has_radar_day(d):
            continue
        if d in _SEPT_VAL_DAYS:
            split, spd = "val", val_starts
        elif d in _SEPT_TEST_DAYS:
            split, spd = "test_robust", test_starts
        else:
            # Unanticipated September radar day — keep as test_robust by default
            split, spd = "test_robust", test_starts
        rows.append({"date": d.strftime("%Y-%m-%d"),
                     "split": split, "zero_day": False,
                     "pre_total_mm": None, "n_wet": None,
                     "starts_per_day": spd})

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
