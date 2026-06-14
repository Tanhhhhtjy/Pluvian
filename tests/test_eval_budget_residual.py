from __future__ import annotations

import json

from scripts.eval_budget_residual import (
    BIN_LABELS,
    format_markdown,
    make_plot,
    summarize_rows,
)


def test_summarize_rows_aggregates_residual_and_intensity_bins():
    rows = [
        {
            "model": "ab2_p7d",
            "split": "event_test",
            "time": "2023-07-29T00:00:00",
            "abs_residual_mean": 0.010,
            "residual_rmse": 0.020,
            "abs_residual_p95": 0.050,
            "loss_mean": 0.001,
            "rain_rate_mean": 2.0,
            "dpwv_abs_mean": 0.003,
            "div_abs_mean": 0.004,
            "precip_abs_mean": 0.005,
            "bin_counts": {"0-1": 10, "1-5": 20, "5-10": 0, "10-30": 0, ">=30": 0},
            "bin_abs_sums": {"0-1": 0.20, "1-5": 1.00, "5-10": 0.0, "10-30": 0.0, ">=30": 0.0},
        },
        {
            "model": "ab2_p7d",
            "split": "event_test",
            "time": "2023-07-29T06:00:00",
            "abs_residual_mean": 0.030,
            "residual_rmse": 0.040,
            "abs_residual_p95": 0.090,
            "loss_mean": 0.003,
            "rain_rate_mean": 4.0,
            "dpwv_abs_mean": 0.009,
            "div_abs_mean": 0.008,
            "precip_abs_mean": 0.006,
            "bin_counts": {"0-1": 0, "1-5": 10, "5-10": 10, "10-30": 0, ">=30": 0},
            "bin_abs_sums": {"0-1": 0.0, "1-5": 0.20, "5-10": 0.70, "10-30": 0.0, ">=30": 0.0},
        },
    ]

    summary = summarize_rows(rows)

    assert len(summary) == 1
    row = summary[0]
    assert row["model"] == "ab2_p7d"
    assert row["split"] == "event_test"
    assert row["n_windows"] == 2
    assert row["n_days"] == 1
    assert row["abs_residual_mean"] == 0.02
    assert row["residual_rmse"] == 0.03
    assert row["bin_abs_mean"]["0-1"] == 0.02
    assert row["bin_abs_mean"]["1-5"] == 0.04
    assert row["bin_abs_mean"]["5-10"] == 0.07
    assert row["bin_abs_mean"]["10-30"] is None
    assert BIN_LABELS[-1] == ">=30"


def test_format_markdown_flags_physical_diagnostic_scope():
    summary = [{
        "model": "ab2_p7d",
        "split": "event_test",
        "n_windows": 2,
        "n_days": 1,
        "abs_residual_mean": 0.02,
        "residual_rmse": 0.03,
        "abs_residual_p95": 0.07,
        "loss_mean": 0.002,
        "rain_rate_mean": 3.0,
        "dpwv_abs_mean": 0.006,
        "div_abs_mean": 0.006,
        "precip_abs_mean": 0.006,
        "bin_abs_mean": {"0-1": 0.02, "1-5": 0.04, "5-10": None, "10-30": None, ">=30": None},
    }]

    text = format_markdown(summary)

    assert "physical-consistency diagnostic" in text
    assert "not a forecast-skill metric" in text
    assert "ab2_p7d" in text
    assert "0.020000" in text
    assert "0.040000" in text


def test_summary_is_strict_json_safe_and_plot_writes_file(tmp_path):
    summary = [{
        "model": "ab2_p7d",
        "split": "event_test",
        "n_windows": 2,
        "n_days": 1,
        "abs_residual_mean": 0.02,
        "residual_rmse": 0.03,
        "abs_residual_p95": 0.07,
        "loss_mean": 0.002,
        "rain_rate_mean": 3.0,
        "dpwv_abs_mean": 0.006,
        "div_abs_mean": 0.006,
        "precip_abs_mean": 0.006,
        "bin_abs_mean": {label: None for label in BIN_LABELS},
    }]

    json.dumps(summary, allow_nan=False)
    out = tmp_path / "budget_residual.png"
    make_plot(summary, out)
    assert out.exists()
    assert out.stat().st_size > 0
