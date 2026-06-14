from __future__ import annotations

import json

from scripts.plot_training_monitor import build_summary, parse_training_log


def test_parse_training_log_pairs_validation_with_last_epoch_summary():
    text = """
[ep017] avg data=14.3400 fss=0.0789 budget=0.0000 spec=0.0000 total=14.4189  (1025.0s)
[val] data=61.3791 budget=nan crps=3.6063 spread=0.0566 csi_1mm=0.3571 csi_5mm=0.4534 csi_10mm=0.5230 csi_30mm=0.2811 fss_3px_1mm=0.5375 fss_3px_10mm=0.7070 fss_3px_30mm=0.4907 fss_11px_1mm=0.5536 fss_11px_10mm=0.7362 fss_11px_30mm=0.5588 crps_method=intensity
[ep018|it0020/1664] data=71.8944 fss=0.0548 bud=0.0000 spec=0.0000 tot=71.9491 lr=2.47e-04 step=7493
"""

    parsed = parse_training_log(text)

    assert len(parsed.epoch_summaries) == 1
    assert parsed.epoch_summaries[0].epoch == 17
    assert parsed.epoch_summaries[0].seconds == 1025.0
    assert len(parsed.validation_rows) == 1
    assert parsed.validation_rows[0].epoch == 17
    assert parsed.validation_rows[0].metrics["csi_30mm"] == 0.2811
    assert parsed.validation_rows[0].metrics["crps"] == 3.6063


def test_parse_training_log_tracks_latest_iteration_progress():
    text = """
[ep018|it0020/1664] data=71.8944 fss=0.0548 bud=0.0000 spec=0.0000 tot=71.9491 lr=2.47e-04 step=7493
[ep018|it0040/1664] data=12.9599 fss=0.0682 bud=0.0000 spec=0.0000 tot=13.0281 lr=2.47e-04 step=7498
"""

    parsed = parse_training_log(text)

    assert parsed.latest_progress is not None
    assert parsed.latest_progress.epoch == 18
    assert parsed.latest_progress.iteration == 40
    assert parsed.latest_progress.total_iterations == 1664
    assert parsed.latest_progress.lr == 2.47e-04


def test_build_summary_emits_strict_json_safe_values(tmp_path):
    text = """
[ep000] avg data=1.0000 fss=0.1000 budget=0.0000 spec=0.0000 total=1.1000  (10.0s)
[val] data=2.0 budget=nan crps=3.0 csi_30mm=0.4 spread=0.1 crps_method=intensity
"""

    summary = build_summary(parse_training_log(text), max_epochs=2, source=tmp_path / "train.log")

    json.dumps(summary, allow_nan=False)
    assert summary["latest_validation"]["budget"] is None
