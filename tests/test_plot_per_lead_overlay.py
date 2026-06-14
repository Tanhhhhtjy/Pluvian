from __future__ import annotations

import json

from scripts.plot_per_lead_overlay import load_series_specs, parse_series_spec


def test_parse_series_spec_supports_optional_json_key():
    spec = parse_series_spec("xcoreA=ckpt/figures/xcore/per_lead_event_test.json:ab2")

    assert spec.label == "xcoreA"
    assert str(spec.path) == "ckpt/figures/xcore/per_lead_event_test.json"
    assert spec.key == "ab2"


def test_parse_series_spec_defaults_to_candidate_key():
    spec = parse_series_spec("ab3=ckpt/figures/ab3/per_lead_event_test.json")

    assert spec.label == "ab3"
    assert spec.key == "ab2"


def test_load_series_specs_skips_missing_when_requested(tmp_path):
    data_path = tmp_path / "per_lead_event_test.json"
    data_path.write_text(json.dumps({
        "split": "event_test",
        "lead_min": [6, 12],
        "ab1": {"csi": {"30": [0.6, 0.5]}, "fss": {"3": [0.7, 0.6]}},
        "ab2": {"csi": {"30": [0.5, 0.55]}, "fss": {"3": [0.72, 0.61]}},
    }), encoding="utf-8")

    series = load_series_specs([
        parse_series_spec(f"ab1={data_path}:ab1"),
        parse_series_spec(f"ab2={data_path}:ab2"),
        parse_series_spec(f"ab3={tmp_path / 'missing.json'}:ab2"),
    ], metric="csi", threshold="30", skip_missing=True)

    assert [row.label for row in series] == ["ab1", "ab2"]
    assert series[1].values == [0.5, 0.55]
    assert series[1].lead_min == [6, 12]
