from __future__ import annotations

from scripts.bootstrap_delta_ci import DEFAULT_COMPARISONS, paired_delta_ci, format_delta_markdown


def _row(date: str, mae: float, hits: int, fa: int, miss: int) -> dict:
    return {
        "date": date,
        "mae": mae,
        "crps": mae,
        "weighted_mse": mae * 2,
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
        "fss_3px_num": 1.0,
        "fss_3px_den": 4.0,
        "fss_11px_num": 1.0,
        "fss_11px_den": 4.0,
    }


def test_paired_delta_ci_single_day_is_exact_delta():
    baseline = [_row("2023-07-29", mae=5.0, hits=1, fa=1, miss=2)]
    candidate = [_row("2023-07-29", mae=3.0, hits=2, fa=1, miss=1)]

    report = paired_delta_ci(
        baseline, candidate, baseline_name="ab1", candidate_name="ab2",
        split="event_test", n_boot=20, seed=0,
    )

    assert report["comparison"] == "ab2 - ab1"
    assert report["estimate"]["mae"] == -2.0
    assert report["ci95"]["mae"] == [-2.0, -2.0]
    assert report["estimate"]["csi_30mm"] == 0.25
    assert report["ci95"]["csi_30mm"] == [0.25, 0.25]


def test_paired_delta_ci_requires_same_dates():
    baseline = [_row("2023-07-29", mae=5.0, hits=1, fa=1, miss=2)]
    candidate = [_row("2023-07-30", mae=3.0, hits=2, fa=1, miss=1)]

    try:
        paired_delta_ci(baseline, candidate, "ab1", "ab2", "event_test")
    except ValueError as exc:
        assert "same date set" in str(exc)
    else:
        raise AssertionError("expected date-set mismatch failure")


def test_format_delta_markdown_includes_signed_ci():
    report = {
        "comparison": "ab2 - ab1",
        "split": "event_test",
        "n_days": 1,
        "n_windows_baseline": 1,
        "n_windows_candidate": 1,
        "estimate": {
            "csi_1mm": 0.01, "csi_10mm": 0.02, "csi_30mm": -0.025,
            "fss_3px": 0.035, "fss_11px": 0.04, "crps": -0.36,
        },
        "ci95": {
            "csi_1mm": [0.0, 0.02], "csi_10mm": [0.01, 0.03],
            "csi_30mm": [-0.05, -0.01], "fss_3px": [0.01, 0.05],
            "fss_11px": [0.02, 0.06], "crps": [-0.6, -0.1],
        },
    }

    text = format_delta_markdown([report])

    assert "ab2 - ab1" in text
    assert "-0.0250 [-0.0500, -0.0100]" in text


def test_default_comparisons_include_ab3_route_gates():
    assert ("ab3_era5", "ab1_p7d") in DEFAULT_COMPARISONS
    assert ("ab3_era5", "ab2_p7d") in DEFAULT_COMPARISONS
