from __future__ import annotations

from scripts.eval_bootstrap_ci import aggregate_rows, bootstrap_ci, format_markdown


def _row(date: str, hits: int, fa: int, miss: int, fss_num: float,
         fss_den: float, mae: float) -> dict:
    return {
        "date": date,
        "mae": mae,
        "crps": mae,
        "weighted_mse": mae * 10.0,
        "csi_1mm_hits": hits,
        "csi_1mm_fa": fa,
        "csi_1mm_miss": miss,
        "csi_5mm_hits": hits,
        "csi_5mm_fa": fa,
        "csi_5mm_miss": miss,
        "csi_10mm_hits": hits,
        "csi_10mm_fa": fa,
        "csi_10mm_miss": miss,
        "csi_30mm_hits": hits,
        "csi_30mm_fa": fa,
        "csi_30mm_miss": miss,
        "fss_3px_num": fss_num,
        "fss_3px_den": fss_den,
        "fss_11px_num": fss_num,
        "fss_11px_den": fss_den,
    }


def test_aggregate_rows_uses_fullset_counts():
    rows = [
        _row("2023-07-29", hits=9, fa=1, miss=0, fss_num=2.0, fss_den=10.0, mae=1.0),
        _row("2023-07-30", hits=1, fa=9, miss=10, fss_num=4.0, fss_den=20.0, mae=3.0),
    ]

    agg = aggregate_rows(rows)

    assert agg["n_windows"] == 2
    assert agg["mae"] == 2.0
    assert agg["csi_30mm"] == 10 / 30
    assert agg["fss_3px"] == 1.0 - 6.0 / 30.0


def test_bootstrap_single_day_ci_equals_estimate():
    rows = [
        _row("2023-07-29", hits=2, fa=1, miss=1, fss_num=1.0, fss_den=4.0, mae=2.0),
        _row("2023-07-29", hits=2, fa=0, miss=0, fss_num=1.0, fss_den=4.0, mae=4.0),
    ]

    report = bootstrap_ci(rows, n_boot=20, seed=123)

    assert report["estimate"]["mae"] == 3.0
    assert report["ci95"]["mae"] == [3.0, 3.0]
    assert report["ci95"]["csi_30mm"] == [0.666667, 0.666667]


def test_format_markdown_contains_model_split_and_ci():
    report = {
        "model": "ab2",
        "split": "event_test",
        "estimate": {"mae": 1.0, "csi_30mm": 0.25, "fss_3px": 0.5, "n_windows": 2, "n_days": 1},
        "ci95": {"mae": [1.0, 1.0], "csi_30mm": [0.25, 0.25], "fss_3px": [0.5, 0.5]},
    }

    text = format_markdown([report])

    assert "ab2" in text
    assert "event_test" in text
    assert "0.2500 [0.2500, 0.2500]" in text
