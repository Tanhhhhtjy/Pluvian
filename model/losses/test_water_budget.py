"""Self-tests for the differentiable water-budget loss module.

Run with:
    cd /Data/tanh/npj
    python -m model.losses.test_water_budget

Covers:
    1. Magnitude on a synthetic advected PWV field (residual ~ 0).
    2. Differentiability w.r.t. pwv_pred and rain_pred.
    3. Spherical vs Euclidean divergence at 40 N (~23% gap).
    4. Zero-input -> zero loss.
    5. Parity with numpy compute_mfd._spherical_divergence (< 5%).
    6. Gaussian smoother parity with scipy.ndimage.gaussian_filter (if scipy
       available).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/Data/tanh/npj")
# avoid importing the model/ package __init__ (which pulls model.pluvian, not
# yet available during T4.2): load the loss module directly by file path.
import importlib.util as _ilu

_here = Path(__file__).resolve().parent
_spec = _ilu.spec_from_file_location("_pluvian_water_budget",
                                     str(_here / "water_budget.py"))
_wb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_wb)
GaussianSmooth2d = _wb.GaussianSmooth2d
SphericalDivergence = _wb.SphericalDivergence
TimeDerivative = _wb.TimeDerivative
WaterBudgetLoss = _wb.WaterBudgetLoss
R_EARTH = _wb.R_EARTH
G_ACCEL = _wb.G_ACCEL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _print_header(name: str):
    print(f"\n--- {name} ---")


def _ok(msg: str):
    print(f"  [OK] {msg}")


def _fail(msg: str):
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)


def _grid(H: int = 32, W: int = 32, lat0: float = 36.0, lat1: float = 42.0,
          lon0: float = 113.0, lon1: float = 120.0):
    lat = torch.linspace(lat0, lat1, H, dtype=torch.float64)
    lon = torch.linspace(lon0, lon1, W, dtype=torch.float64)
    return lat, lon


# ---------------------------------------------------------------------------
# 1. magnitude test
# ---------------------------------------------------------------------------

def test_advection_residual():
    _print_header("Test 1: advected-PWV residual ~ 0")
    B, T, H, W = 1, 8, 48, 48
    lat, lon = _grid(H, W)
    dt = 360.0  # 6 minutes
    u_ms = 10.0  # m/s eastward
    # build PWV(t, x) = G(x - u t)  with x in metres
    # use 1-D longitudinal coordinate in metres at the mid latitude
    mid_lat = float(lat.mean())
    dlon_m = R_EARTH * math.cos(math.radians(mid_lat)) * math.radians(float(lon[1] - lon[0]))
    x_m = torch.arange(W, dtype=torch.float64) * dlon_m
    y_idx = torch.zeros(H, dtype=torch.float64)  # uniform in latitude
    sigma_x = 6 * dlon_m  # gaussian width

    pwv_list = []
    for t in range(T):
        # peak shifts east with time
        x0 = (W * dlon_m * 0.3) + u_ms * (t * dt)
        prof = torch.exp(-0.5 * ((x_m - x0) / sigma_x) ** 2)  # (W,)
        slab = prof.unsqueeze(0).expand(H, W) + y_idx.unsqueeze(1) * 0.0
        pwv_list.append(slab)
    pwv = torch.stack(pwv_list, dim=0).unsqueeze(0)  # (1,T,H,W)  units: arbitrary "mm"
    pwv = pwv * 50.0  # peak ~50 mm

    # Set scale = delta_p/g so that scale * q = PWV exactly.  Then div(qV) =
    # div(PWV/scale * V) and scale*div(qV) = V . grad(PWV) when V is constant.
    delta_p = 30000.0
    scale = delta_p / G_ACCEL
    q = (pwv / scale).to(torch.float64)
    u = torch.full_like(q, u_ms)
    v = torch.zeros_like(q)
    rain = torch.zeros_like(q)

    loss_mod = WaterBudgetLoss(delta_p_pa=delta_p, g=G_ACCEL,
                               dt_seconds=dt, smooth_sigma=0.0,
                               use_huber=False).double()
    out = loss_mod(pwv.double(), rain.double(), u, v, q, lat, lon)
    res = out["budget_residual"]
    # ignore the spatial/time boundaries (1 grid cell)
    interior = res[:, 1:-1, 2:-2, 2:-2]
    pwv_interior = pwv[:, 1:-1, 2:-2, 2:-2]
    pwv_dt_interior = out["dpwv_dt"][:, 1:-1, 2:-2, 2:-2]
    rel = interior.abs().mean() / (pwv_dt_interior.abs().mean() + 1e-12)
    print(f"  mean|residual| / mean|dPWV/dt| (interior) = {float(rel):.3e}")
    print(f"  mean|dPWV/dt| = {float(pwv_dt_interior.abs().mean()):.3e} mm/s")
    print(f"  mean|div_qV|  = {float(out['div_qV_mm_s'][:,1:-1,2:-2,2:-2].abs().mean()):.3e} mm/s")
    if float(rel) > 0.05:
        _fail(f"interior residual relative magnitude {float(rel):.3e} > 5% — advection identity violated")
    _ok("interior |residual| < 5% of |dPWV/dt|")


# ---------------------------------------------------------------------------
# 2. differentiability
# ---------------------------------------------------------------------------

def test_differentiability():
    _print_header("Test 2: differentiability")
    torch.manual_seed(0)
    B, T, H, W = 1, 4, 16, 16
    lat, lon = _grid(H, W)
    pwv = torch.randn(B, T, H, W, requires_grad=True)
    rain = torch.randn(B, T, H, W).abs().clone().requires_grad_(True)  # mm/h positive, leaf
    u = torch.randn(B, T, H, W) * 5
    v = torch.randn(B, T, H, W) * 5
    q = torch.randn(B, T, H, W).abs() * 1e-3
    loss_mod = WaterBudgetLoss()
    out = loss_mod(pwv, rain, u, v, q, lat, lon)
    out["loss"].backward()
    g_pwv = pwv.grad
    g_rain = rain.grad
    if g_pwv is None or float(g_pwv.abs().sum()) == 0.0:
        _fail("no gradient flowed to pwv_pred")
    if g_rain is None or float(g_rain.abs().sum()) == 0.0:
        _fail("no gradient flowed to rain_pred")
    print(f"  |grad_pwv|.sum()={float(g_pwv.abs().sum()):.3e}  "
          f"|grad_rain|.sum()={float(g_rain.abs().sum()):.3e}")
    _ok("grads flow to both pwv_pred and rain_pred")


# ---------------------------------------------------------------------------
# 3. spherical vs euclidean
# ---------------------------------------------------------------------------

def test_spherical_vs_euclidean():
    _print_header("Test 3: spherical vs Euclidean divergence at 40 N")
    H, W = 32, 32
    # case A: real grid at 40 N (cos~0.766)
    lat_a = torch.linspace(39.0, 41.0, H, dtype=torch.float64)
    lon_a = torch.linspace(113.0, 115.0, W, dtype=torch.float64)
    # case B: equator (cos=1)
    lat_b = torch.linspace(-1.0, 1.0, H, dtype=torch.float64)
    lon_b = torch.linspace(113.0, 115.0, W, dtype=torch.float64)

    # same field magnitude on both
    qu = torch.zeros(1, H, W, dtype=torch.float64)
    # impose a linear gradient in x: qu = c * lon_index
    idx = torch.arange(W, dtype=torch.float64)
    qu[0] = idx.unsqueeze(0).expand(H, W)
    qv = torch.zeros_like(qu)

    div_op = SphericalDivergence().double()
    div_a = div_op(qu, qv, lat_a, lon_a)   # 40 N
    div_b = div_op(qu, qv, lat_b, lon_b)   # equator
    # ratio (spherical 40N) / (spherical equator) should equal 1/cos(40 N)
    ratio = float(div_a[:, H // 2, W // 2] / div_b[:, H // 2, W // 2])
    expected = 1.0 / math.cos(math.radians(40.0))
    print(f"  div(40N)/div(eq) = {ratio:.4f}   1/cos(40)={expected:.4f}")
    rel = abs(ratio - expected) / expected
    if rel > 0.02:
        _fail(f"spherical/equator ratio off by {rel*100:.2f}% > 2%")
    _ok(f"ratio matches 1/cos(lat) within {rel*100:.3f}%")


# ---------------------------------------------------------------------------
# 4. zero input
# ---------------------------------------------------------------------------

def test_zero_input():
    _print_header("Test 4: zero inputs -> zero loss")
    B, T, H, W = 1, 4, 16, 16
    lat, lon = _grid(H, W)
    z = torch.zeros(B, T, H, W)
    loss_mod = WaterBudgetLoss(use_huber=False)
    out = loss_mod(z, z, z, z, z, lat, lon)
    lv = float(out["loss"])
    print(f"  loss = {lv:.3e}")
    if abs(lv) > 1e-12:
        _fail(f"loss should be 0 on zero input, got {lv}")
    _ok("loss is 0")


# ---------------------------------------------------------------------------
# 5. parity with numpy reference (compute_mfd._spherical_divergence)
# ---------------------------------------------------------------------------

def test_parity_with_numpy_mfd():
    _print_header("Test 5: parity with numpy compute_mfd._spherical_divergence")
    # Inline copy of pipeline.compute_mfd._spherical_divergence to avoid pulling
    # xarray as a hard test dependency.  Mirrors that function verbatim.
    R_E = 6_371_000.0
    def np_div(qu, qv, lat_asc, lon_asc):
        dlon = np.deg2rad(float(np.mean(np.diff(lon_asc))))
        dlat = np.deg2rad(float(np.mean(np.diff(lat_asc))))
        coslat = np.cos(np.deg2rad(lat_asc)).astype(np.float32)
        dx = (R_E * coslat * dlon).astype(np.float32)
        dy = np.float32(R_E * dlat)
        dqu_dx = np.empty_like(qu)
        dqu_dx[:, :, 1:-1] = (qu[:, :, 2:] - qu[:, :, :-2]) / (2.0 * dx[None, :, None])
        dqu_dx[:, :,  0]   = (qu[:, :,  1] - qu[:, :,  0]) /        dx[None, :]
        dqu_dx[:, :, -1]   = (qu[:, :, -1] - qu[:, :, -2]) /        dx[None, :]
        dqv_dy = np.empty_like(qv)
        dqv_dy[:, 1:-1, :] = (qv[:, 2:, :] - qv[:, :-2, :]) / (2.0 * dy)
        dqv_dy[:,  0,  :]  = (qv[:,  1, :] - qv[:,  0, :]) /        dy
        dqv_dy[:, -1,  :]  = (qv[:, -1, :] - qv[:, -2, :]) /        dy
        return dqu_dx + dqv_dy
    torch.manual_seed(1)
    T, H, W = 3, 21, 25
    lat_np = np.linspace(36.0, 42.0, H).astype(np.float32)
    lon_np = np.linspace(113.0, 120.0, W).astype(np.float32)
    qu_np = np.random.randn(T, H, W).astype(np.float32) * 1e-3
    qv_np = np.random.randn(T, H, W).astype(np.float32) * 1e-3

    div_np = np_div(qu_np, qv_np, lat_np, lon_np)  # (T,H,W)

    qu_t = torch.from_numpy(qu_np).double()
    qv_t = torch.from_numpy(qv_np).double()
    lat_t = torch.from_numpy(lat_np).double()
    lon_t = torch.from_numpy(lon_np).double()
    div_op = SphericalDivergence().double()
    div_t = div_op(qu_t, qv_t, lat_t, lon_t).numpy()

    # interior comparison (strip 1-cell boundary)
    diff_int = np.abs(div_np[:, 1:-1, 1:-1] - div_t[:, 1:-1, 1:-1])
    ref_int = np.abs(div_np[:, 1:-1, 1:-1])
    rel = diff_int.max() / (ref_int.max() + 1e-12)
    print(f"  interior max-abs rel error = {rel:.3e}")
    # boundary too
    diff_all = np.abs(div_np - div_t)
    rel_all = diff_all.max() / (np.abs(div_np).max() + 1e-12)
    print(f"  whole-field   max-abs rel error = {rel_all:.3e}")
    if rel > 0.05:
        _fail(f"interior rel-error {rel:.3e} > 5%")
    if rel_all > 0.05:
        _fail(f"full-field rel-error {rel_all:.3e} > 5%")
    _ok("torch divergence matches numpy compute_mfd within < 5%")


# ---------------------------------------------------------------------------
# 6. gaussian smoother parity with scipy (if scipy present)
# ---------------------------------------------------------------------------

def test_gaussian_smoother_vs_scipy():
    _print_header("Test 6: GaussianSmooth2d vs scipy.ndimage.gaussian_filter")
    try:
        from scipy.ndimage import gaussian_filter
    except Exception as e:  # pragma: no cover
        print(f"  [SKIP] scipy not available: {e}")
        return
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((1, 1, 64, 64)).astype(np.float64)
    sigma = 2.0
    sp = gaussian_filter(arr[0, 0], sigma=sigma, mode="nearest", truncate=4.0)
    sm = GaussianSmooth2d(sigma=sigma).double()
    t = torch.from_numpy(arr)
    out = sm(t).numpy()[0, 0]
    diff = np.abs(out - sp).max() / (np.abs(sp).max() + 1e-12)
    print(f"  max-abs rel error = {diff:.3e}")
    if diff > 1e-3:
        _fail(f"gaussian smoother parity poor: {diff:.3e}")
    _ok("parity with scipy gaussian_filter < 1e-3")


# ---------------------------------------------------------------------------
# 7. time derivative shape + correctness
# ---------------------------------------------------------------------------

def test_time_derivative_linear():
    _print_header("Test 7: TimeDerivative on a linear ramp")
    T = 6
    dt = 360.0
    x = torch.arange(T, dtype=torch.float64).view(1, T, 1, 1).expand(2, T, 4, 4).contiguous()
    # x[t] = t  -> dx/dt = 1 step per dt time units
    deriv = TimeDerivative(dt_seconds=dt)(x)
    expected_interior = 1.0 / dt
    int_vals = deriv[:, 1:-1].flatten()
    rel = (int_vals - expected_interior).abs().max() / expected_interior
    print(f"  interior dt-derivative rel-err vs 1/dt = {float(rel):.3e}")
    if float(rel) > 1e-12:
        _fail("time derivative not exact on linear ramp")
    _ok("time derivative exact on linear ramp")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    test_advection_residual()
    test_differentiability()
    test_spherical_vs_euclidean()
    test_zero_input()
    test_parity_with_numpy_mfd()
    test_gaussian_smoother_vs_scipy()
    test_time_derivative_linear()
    print("\nAll water-budget tests passed.")


if __name__ == "__main__":
    main()
