from __future__ import annotations

from scripts.make_phase7d_auxiliary_tables import (
    build_drop_pwv_rows,
    build_sharpness_rows,
    format_drop_pwv_markdown,
    format_sharpness_markdown,
)


def test_build_drop_pwv_rows_keeps_signed_deltas():
    full_reports = [{
        "model": "ab2_p7d",
        "split": "event_test",
        "estimate": {
            "csi_1mm": 0.4316,
            "csi_10mm": 0.5907,
            "csi_30mm": 0.2826,
            "fss_3px": 0.6141,
            "crps": 4.8274,
        },
    }]
    drop_metrics = {"event_test": {
        "csi_1mm": 0.4258,
        "csi_10mm": 0.6004,
        "csi_30mm": 0.1262,
        "fss_3px": 0.6083,
        "crps": 4.9815,
    }}

    rows = build_drop_pwv_rows(full_reports, drop_metrics)

    assert len(rows) == 1
    assert rows[0]["split"] == "event_test"
    assert rows[0]["delta"]["csi_30mm"] == -0.1564
    assert rows[0]["delta"]["crps"] == 0.1541


def test_format_drop_pwv_markdown_flags_diagnostic_scope():
    rows = [{
        "split": "event_test",
        "full": {"csi_1mm": 0.4316, "csi_10mm": 0.5907, "csi_30mm": 0.2826, "fss_3px": 0.6141, "crps": 4.8274},
        "drop": {"csi_1mm": 0.4258, "csi_10mm": 0.6004, "csi_30mm": 0.1262, "fss_3px": 0.6083, "crps": 4.9815},
        "delta": {"csi_1mm": -0.0058, "csi_10mm": 0.0097, "csi_30mm": -0.1564, "fss_3px": -0.0058, "crps": 0.1541},
    }]

    text = format_drop_pwv_markdown(rows)

    assert "not a radar-only baseline" in text
    assert "event_test" in text
    assert "-0.156" in text
    assert "+0.154" in text


def test_build_sharpness_rows_marks_diffusion_invalid():
    sharpness = {"agg": {
        "gt_p999": 46.7,
        "diff_frac_over65": 0.035,
        "baseline": {"hk": 0.071, "p999": 39.6},
        "gan_coreA": {"hk": 0.686, "p999": 43.7},
        "diff": {"hk": 2.711, "p999": 100.8},
    }}

    rows = build_sharpness_rows(sharpness)

    by_method = {row["method"]: row for row in rows}
    assert by_method["GAN-coreA"]["claim"] == "realism/sharpness only"
    assert by_method["diffusion residual"]["claim"] == "invalid without amplitude control"
    assert by_method["diffusion residual"]["frac_over65"] == 0.035


def test_format_sharpness_markdown_separates_skill_from_realism():
    rows = [{
        "lane": "realism",
        "method": "GAN-coreA",
        "high_k": 0.6862,
        "p999": 43.7324,
        "gt_p999": 46.725,
        "frac_over65": None,
        "claim": "realism/sharpness only",
    }]

    text = format_sharpness_markdown(rows)

    assert "CSI/FSS/CRPS skill is not inferred" in text
    assert "GAN-coreA" in text
    assert "0.686" in text
