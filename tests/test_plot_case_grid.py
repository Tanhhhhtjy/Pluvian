from __future__ import annotations

from pathlib import Path

from scripts.plot_case_grid import parse_diff_arg, parse_leads, parse_model_arg


def test_parse_model_arg_requires_name_and_path():
    name, path = parse_model_arg("ab3=ckpt/ablation_3_era5_p7c/best.pt")

    assert name == "ab3"
    assert path == Path("ckpt/ablation_3_era5_p7c/best.pt")


def test_parse_diff_arg_selects_named_models():
    diff = parse_diff_arg("ab3-ab2", {"ab1", "ab2", "ab3"})

    assert diff.label == "ab3 - ab2"
    assert diff.left == "ab3"
    assert diff.right == "ab2"


def test_parse_leads_converts_minutes_to_zero_based_indices():
    assert parse_leads([18, 36, 72, 108], forecast_frames=18) == [2, 5, 11, 17]


def test_parse_leads_rejects_non_6min_lead():
    try:
        parse_leads([20], forecast_frames=18)
    except ValueError as exc:
        assert "multiple of 6" in str(exc)
    else:
        raise AssertionError("expected ValueError")
