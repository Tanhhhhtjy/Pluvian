"""Phase 7d Track-A data fix: (1) remove false-negative zero-days, (2) raise
train window density.

zero_day=True means radar files are MISSING for that day. The loader zero-fills
radar and marks it valid "known no-rain". But several such train days have real
gauge rain (pre_total_mm up to 43 mm, dozens of wet stations) -> the model is
supervised with radar=0 on days it actually rained = false-negative bias that
pulls forecasts dry. Genuinely-dry zero-days (0 mm, 0 wet) are legitimate
no-rain negatives and are KEPT (at low density to avoid over-weighting zeros).

Also raise non-zero train days from 8 -> 24 starts/day (val/test already 24);
~3x more windows from the same data.

    python scripts/fix_manifest_p7d.py [--apply]
Without --apply it prints the plan only.
"""
import argparse
from pathlib import Path
import pandas as pd

MAN = Path(__file__).resolve().parent.parent / "pipeline" / "manifest.csv"
RAIN_MM = 1.0   # a zero-day with >= this gauge rain (or any wet station) is a false negative
DENSITY = 24    # non-zero train days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    m = pd.read_csv(MAN)
    nwet = m["n_wet"].fillna(0.0)
    pmm = m["pre_total_mm"].fillna(0.0)

    is_train = m["split"] == "train"
    is_zero = m["zero_day"] == True  # noqa: E712
    rainy = is_zero & ((nwet > 0) | (pmm >= RAIN_MM))

    drop_rows = m[is_train & rainy]
    keep_dry = m[is_train & is_zero & ~rainy]
    print(f"[zero-day] train zero-days: {int((is_train & is_zero).sum())}  "
          f"-> drop(false-neg rain) {len(drop_rows)} | keep(dry) {len(keep_dry)}")
    print("  dropping:", ", ".join(f"{r.date}({r.pre_total_mm or 0:.0f}mm/{int(r.n_wet or 0)}w)"
                                   for r in drop_rows.itertuples()))

    new = m.copy()
    new.loc[is_train & rainy, "split"] = "drop"
    new.loc[is_train & rainy, "starts_per_day"] = 0
    # density: non-zero train days -> 24; dry zero-days stay 8
    new.loc[(new["split"] == "train") & (new["zero_day"] != True), "starts_per_day"] = DENSITY  # noqa: E712

    def windows(df):
        t = df[df["split"] == "train"]
        return int(t["starts_per_day"].fillna(0).sum())
    print(f"[windows] train windows: {windows(m)} -> {windows(new)}")
    print(f"[days]    train days: {int((m['split']=='train').sum())} -> "
          f"{int((new['split']=='train').sum())}")

    if args.apply:
        new.to_csv(MAN, index=False)
        print(f"[apply] wrote {MAN}")
    else:
        print("[dry-run] pass --apply to write")


if __name__ == "__main__":
    main()
