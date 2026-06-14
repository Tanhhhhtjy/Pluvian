from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_phase7d_results import build_report, summarize_per_lead


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_summarize_per_lead_computes_short_csi30_gap(tmp_path: Path):
    per_lead = {
        "split": "event_test",
        "lead_min": [6, 18, 24, 42, 48],
        "ab1": {"csi": {"30": [0.50, 0.50, 0.50, 0.50, 0.50]}},
        "ab2": {"csi": {"30": [0.40, 0.55, 0.45, 0.35, 0.60]}},
    }
    path = tmp_path / "per_lead_event_test.json"
    _write_json(path, per_lead)

    row = summarize_per_lead(path, model="candidate", threshold="30")

    assert row["model"] == "candidate"
    assert row["split"] == "event_test"
    assert row["positive_leads"] == 2
    assert row["mean_delta"] == -0.03
    assert row["short_delta"] == -0.05
    assert row["min_delta"] == -0.15
    assert row["max_delta"] == 0.10


def test_build_report_marks_missing_ab3_pending(tmp_path: Path):
    _write_json(
        tmp_path / "ckpt/eval/ablation_2_p7d_xcoreA/event_test/metric.json",
        {"csi_1mm": 0.4, "csi_5mm": 0.5, "csi_10mm": 0.6,
         "csi_30mm": 0.3, "fss_3px": 0.7, "fss_11px": 0.8,
         "crps": 4.0},
    )

    report = build_report(tmp_path)

    assert "ablation_2_p7d_xcoreA" in report
    assert "ablation_2_p7d_droppwv" in report
    assert "ablation_3_era5_p7c" in report
    assert "pending" in report
    assert "csi30_short_delta" in report
