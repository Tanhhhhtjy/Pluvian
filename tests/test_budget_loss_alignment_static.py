"""Static guards for the Track C water-budget training path.

Local workstations may not have torch, so these tests inspect source text. The
runtime smoke test still runs on the remote GPU machine before launching a real
budget-loss experiment.
"""
from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _source(relpath: str) -> str:
    return (REPO / relpath).read_text()


def _function_source(relpath: str, name: str) -> str:
    src = _source(relpath)
    tree = ast.parse(src, filename=relpath)
    lines = src.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"missing function {name} in {relpath}")


def test_train_budget_path_aligns_era5_to_budget_grid():
    helper = _function_source("scripts/train.py", "_budget_loss_inputs")
    sampler = _function_source("scripts/train.py", "_sample_radar_grid_to_latlon")
    assert "_sample_radar_grid_to_latlon" in helper
    assert "F.grid_sample" in sampler
    assert "era5_lat" in helper
    assert "era5_lon" in helper
    assert "lat_mask" in helper
    assert "lon_mask" in helper
    assert "RADAR_LAT" in helper
    assert "RADAR_LON" in helper


def test_train_budget_path_converts_dbz_to_rainrate_units():
    helper = _function_source("scripts/train.py", "_budget_loss_inputs")
    assert "_dbz_to_rainrate_torch" in helper
    assert "rain_budget = _dbz_to_rainrate_torch" in helper


def test_train_defines_torch_dbz_to_rainrate_helper():
    helper = _function_source("scripts/train.py", "_dbz_to_rainrate_torch")
    assert "10.0 **" in helper
    assert "300.0" in helper
    assert "1.0 / 1.4" in helper


def test_train_budget_calls_use_aligned_inputs():
    src = _source("scripts/train.py")
    assert "budget_inputs = _budget_loss_inputs(" in src
    assert "pwv_pred=pwv_budget" in src
    assert "rain_pred=rain_budget" in src
    assert "u_era5=u_budget" in src
    assert "v_era5=v_budget" in src
    assert "q_era5=q_budget" in src


def test_dataset_exposes_native_era5_grid_for_budget_loss():
    src = _source("pipeline/data_loader.py")
    assert "load_era5_window_with_grid" in src
    assert '"era5_lat"' in src
    assert '"era5_lon"' in src
