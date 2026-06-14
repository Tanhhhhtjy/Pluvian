from __future__ import annotations

from scripts.make_phase7d_paper_tables import (
    format_delta_markdown,
    format_delta_booktabs,
    format_metric_booktabs,
    format_metric_markdown,
)


def _report(model: str, split: str, csi1: float, csi10: float,
            csi30: float, fss3: float, crps: float) -> dict:
    return {
        "model": model,
        "split": split,
        "estimate": {
            "csi_1mm": csi1,
            "csi_10mm": csi10,
            "csi_30mm": csi30,
            "fss_3px": fss3,
            "crps": crps,
        },
    }


def test_format_metric_markdown_contains_selected_models_and_metrics():
    reports = [
        _report("ab1_p7d", "event_test", 0.40, 0.56, 0.31, 0.58, 5.18),
        _report("ab2_p7d", "event_test", 0.43, 0.59, 0.28, 0.61, 4.82),
        _report("ab3_era5", "event_test", 0.44, 0.60, 0.30, 0.62, 4.70),
    ]

    text = format_metric_markdown(reports, split="event_test")

    assert "ab1_p7d" in text
    assert "ab2_p7d" in text
    assert "ab3_era5" in text
    assert "0.280" in text
    assert "CRPS" in text


def test_format_metric_markdown_orders_ab3_before_xcore_fallbacks():
    reports = [
        _report("xcoreA", "event_test", 0.42, 0.58, 0.29, 0.60, 4.97),
        _report("ab3_era5", "event_test", 0.44, 0.60, 0.30, 0.62, 4.70),
        _report("ab2_p7d", "event_test", 0.43, 0.59, 0.28, 0.61, 4.82),
        _report("ab1_p7d", "event_test", 0.40, 0.56, 0.31, 0.58, 5.18),
    ]

    text = format_metric_markdown(reports, split="event_test")

    assert text.index("ab1_p7d") < text.index("ab2_p7d") < text.index("ab3_era5") < text.index("xcoreA")


def test_format_metric_booktabs_is_latex_ready():
    reports = [_report("ab2_p7d", "event_test", 0.43, 0.59, 0.28, 0.61, 4.82)]

    text = format_metric_booktabs(reports, split="event_test")

    assert "\\begin{tabular}" in text
    assert "\\toprule" in text
    assert "ab2\\_p7d" in text
    assert "0.280" in text


def test_format_delta_booktabs_keeps_signed_csi30_delta():
    reports = [{
        "comparison": "ab2_p7d - ab1_p7d",
        "split": "event_test",
        "estimate": {"csi_1mm": 0.0347, "csi_10mm": 0.0255,
                     "csi_30mm": -0.0252, "fss_3px": 0.0350,
                     "crps": -0.3601},
        "ci95": {"csi_1mm": [0.0158, 0.0526],
                 "csi_10mm": [0.0021, 0.0410],
                 "csi_30mm": [-0.0432, -0.0126],
                 "fss_3px": [0.0217, 0.0451],
                 "crps": [-0.6333, -0.0869]},
    }]

    text = format_delta_booktabs(reports, split="event_test")

    assert "ab2\\_p7d $-$ ab1\\_p7d" in text
    assert "$-0.025$" in text
    assert "[-0.043, -0.013]" in text


def test_format_delta_markdown_keeps_signed_csi30_delta():
    reports = [{
        "comparison": "ab2_p7d - ab1_p7d",
        "split": "event_test",
        "estimate": {"csi_1mm": 0.0347, "csi_10mm": 0.0255,
                     "csi_30mm": -0.0252, "fss_3px": 0.0350,
                     "crps": -0.3601},
        "ci95": {"csi_1mm": [0.0158, 0.0526],
                 "csi_10mm": [0.0021, 0.0410],
                 "csi_30mm": [-0.0432, -0.0126],
                 "fss_3px": [0.0217, 0.0451],
                 "crps": [-0.6333, -0.0869]},
    }]

    text = format_delta_markdown(reports, split="event_test")

    assert "ab2_p7d - ab1_p7d" in text
    assert "-0.025 [-0.043, -0.013]" in text
