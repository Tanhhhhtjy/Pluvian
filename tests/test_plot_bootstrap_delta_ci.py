from __future__ import annotations

from scripts.plot_bootstrap_delta_ci import plot_rows_for_split


def test_plot_rows_for_split_filters_metric_and_order():
    reports = [
        {
            "comparison": "xcoreB - ab2_p7d",
            "split": "event_test",
            "estimate": {"csi_30mm": 0.0201},
            "ci95": {"csi_30mm": [-0.0088, 0.0321]},
        },
        {
            "comparison": "ab3_era5 - ab2_p7d",
            "split": "event_test",
            "estimate": {"csi_30mm": 0.0100},
            "ci95": {"csi_30mm": [-0.0050, 0.0200]},
        },
        {
            "comparison": "ab2_p7d - ab1_p7d",
            "split": "event_test",
            "estimate": {"csi_30mm": -0.0252},
            "ci95": {"csi_30mm": [-0.0432, -0.0126]},
        },
        {
            "comparison": "ab2_p7d - ab1_p7d",
            "split": "test_robust",
            "estimate": {"csi_30mm": 0.0027},
            "ci95": {"csi_30mm": [-0.0151, 0.0292]},
        },
    ]

    rows = plot_rows_for_split(reports, split="event_test", metric="csi_30mm")

    assert [r.comparison for r in rows] == ["ab2_p7d - ab1_p7d", "ab3_era5 - ab2_p7d", "xcoreB - ab2_p7d"]
    assert rows[0].estimate == -0.0252
    assert rows[0].lo == -0.0432
    assert rows[0].hi == -0.0126


def test_plot_rows_for_split_skips_missing_metric():
    reports = [{
        "comparison": "ab2_p7d - ab1_p7d",
        "split": "event_test",
        "estimate": {"crps": -0.36},
        "ci95": {"crps": [-0.63, -0.09]},
    }]

    assert plot_rows_for_split(reports, split="event_test", metric="csi_30mm") == []
