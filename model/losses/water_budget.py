"""Differentiable water-budget operators for the Pluvian I-1 soft physics loss.

The budget equation we constrain (per unit area, vertically integrated):

    d(PWV)/dt + nabla . (q V)_col  =  -P + E

with E folded into the residual.  Quantities are unified to mm/s:

    PWV               [mm]      ( = kg/m^2 )
    d(PWV)/dt         [mm/s]
    nabla . (q V)     [1/s]     when q is kg/kg, V is m/s (single level)
    scale * div(qV)   [mm/s]    using scale = delta_p / g      ( kg/(m^2 s) )
    P (rain rate)     [mm/h]    -> divided by 3600 inside loss

The single-level 850 hPa proxy for column moisture flux divergence multiplies the
divergence by an effective layer thickness `delta_p` (Pa) over gravity `g`
(default delta_p=30000 Pa, g=9.81 m/s^2 -> scale ~ 3060 kg/(m^2)).

Spherical FD mirrors `pipeline/compute_mfd.py::_spherical_divergence`:

    dx = R cos(lat) dlon    (varies with latitude)
    dy = R dlat             (constant)
    R  = 6_371_000 m

All ops are written so that autograd flows to `pwv_pred`, `rain_pred`, and the
ERA5 fields if requested (no in-place writes; F.pad + slicing).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

R_EARTH = 6_371_000.0      # m
G_ACCEL = 9.80665          # m/s^2


# ---------------------------------------------------------------------------
# Gaussian smoothing (PyTorch port of scipy.ndimage.gaussian_filter mode="nearest")
# ---------------------------------------------------------------------------

class GaussianSmooth2d(nn.Module):
    """Separable 2-D Gaussian smoother over the last two dims (H, W).

    Matches `scipy.ndimage.gaussian_filter(..., sigma=sigma, mode="nearest",
    truncate=4.0)` to within float32 conv accuracy.  Pure F.conv2d / F.pad
    (replicate) so the operation is differentiable.

    Input  : (..., H, W) tensor (any number of leading dims).
    Output : same shape, smoothed along (H, W).
    """

    def __init__(self, sigma: float = 2.0, truncate: float = 4.0):
        super().__init__()
        self.sigma = float(sigma)
        self.truncate = float(truncate)
        radius = int(self.truncate * self.sigma + 0.5)
        self.radius = max(radius, 1)
        x = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / self.sigma) ** 2)
        k = k / k.sum()
        # store as buffers so .to(device) works
        self.register_buffer("kernel_h", k.view(1, 1, -1, 1))   # (1,1,K,1)
        self.register_buffer("kernel_w", k.view(1, 1, 1, -1))   # (1,1,1,K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return x
        orig_shape = x.shape
        H, W = orig_shape[-2], orig_shape[-1]
        leading = int(torch.tensor(orig_shape[:-2]).prod().item()) if len(orig_shape) > 2 else 1
        x2 = x.reshape(leading, 1, H, W)
        kh = self.kernel_h.to(dtype=x2.dtype)
        kw = self.kernel_w.to(dtype=x2.dtype)
        r = self.radius
        x2 = F.pad(x2, (0, 0, r, r), mode="replicate")
        x2 = F.conv2d(x2, kh)
        x2 = F.pad(x2, (r, r, 0, 0), mode="replicate")
        x2 = F.conv2d(x2, kw)
        return x2.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Spherical divergence
# ---------------------------------------------------------------------------

class SphericalDivergence(nn.Module):
    """Differentiable spherical-FD divergence on a regular lon/lat grid.

    Computes d(qu)/dx + d(qv)/dy with:
        dx_j = R cos(lat_j) dlon
        dy   = R dlat
    using centred differences in the interior and one-sided differences on the
    boundary.  Latitude must be ascending.

    Inputs are (..., H, W) tensors (any leading dims).  The grid 1-D tensors
    `lat (H,)` and `lon (W,)` may be float32; they are converted to the input
    dtype.  All ops are autograd-friendly: only F.pad + slicing, no in-place
    assignment.
    """

    def __init__(self, smooth_sigma: float = 0.0):
        super().__init__()
        self.smoother = GaussianSmooth2d(sigma=smooth_sigma) if smooth_sigma and smooth_sigma > 0 else None

    @staticmethod
    def _ddx(field: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """d field / d(longitude) with centred + one-sided ends.

        field : (..., H, W)
        dx    : (H,) in metres (latitude-dependent)
        returns same shape as field.
        """
        # centred interior: (f[..., 2:] - f[..., :-2]) / (2 dx)
        interior = (field[..., 2:] - field[..., :-2]) / (2.0 * dx.view(*([1] * (field.dim() - 2)), -1, 1))
        # forward/backward at the two boundary columns
        left = (field[..., 1:2] - field[..., 0:1]) / dx.view(*([1] * (field.dim() - 2)), -1, 1)
        right = (field[..., -1:] - field[..., -2:-1]) / dx.view(*([1] * (field.dim() - 2)), -1, 1)
        return torch.cat([left, interior, right], dim=-1)

    @staticmethod
    def _ddy(field: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """d field / d(latitude) with centred + one-sided ends.

        field : (..., H, W)
        dy    : scalar tensor in metres
        """
        interior = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dy)
        top = (field[..., 1:2, :] - field[..., 0:1, :]) / dy
        bot = (field[..., -1:, :] - field[..., -2:-1, :]) / dy
        return torch.cat([top, interior, bot], dim=-2)

    def forward(self,
                qu: torch.Tensor,
                qv: torch.Tensor,
                lat: torch.Tensor,
                lon: torch.Tensor) -> torch.Tensor:
        if qu.shape != qv.shape:
            raise ValueError(f"qu/qv shape mismatch: {qu.shape} vs {qv.shape}")
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("lat/lon must be 1-D")
        if lat.numel() != qu.shape[-2] or lon.numel() != qu.shape[-1]:
            raise ValueError("lat/lon length must match field H/W")

        dtype = qu.dtype
        device = qu.device
        lat = lat.to(device=device, dtype=dtype)
        lon = lon.to(device=device, dtype=dtype)

        # require ascending latitude (matches era5_io convention)
        if lat[-1].item() <= lat[0].item():
            raise ValueError("lat must be ascending (lat[0] < lat[-1])")

        dlon = (lon[1:] - lon[:-1]).mean() * math.pi / 180.0
        dlat = (lat[1:] - lat[:-1]).mean() * math.pi / 180.0
        coslat = torch.cos(lat * (math.pi / 180.0))
        dx = R_EARTH * coslat * dlon                  # (H,)
        dy = R_EARTH * dlat                            # scalar tensor

        dqu_dx = self._ddx(qu, dx)
        dqv_dy = self._ddy(qv, dy)
        div = dqu_dx + dqv_dy
        if self.smoother is not None:
            div = self.smoother(div)
        return div


# ---------------------------------------------------------------------------
# Time derivative
# ---------------------------------------------------------------------------

class TimeDerivative(nn.Module):
    """d/dt along a specified dim with centred + one-sided edges.

    Default dt = 360 s (6 minutes, radar cadence).  Differentiable.
    """

    def __init__(self, dt_seconds: float = 360.0, dim: int = -3):
        super().__init__()
        self.dt = float(dt_seconds)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # we expect x with the time dim explicitly addressable; default -3
        # (e.g. (B, T, H, W) -> time at dim=-3)
        dt = self.dt
        T = x.shape[self.dim]
        if T < 2:
            raise ValueError("need at least 2 time steps")
        # build slices generically
        def sl(start: Optional[int], stop: Optional[int]) -> tuple:
            return tuple(slice(start, stop) if i == self.dim % x.dim() else slice(None)
                         for i in range(x.dim()))

        # interior centred: (x[..., 2:, ...] - x[..., :-2, ...]) / (2 dt)
        interior = (x[sl(2, None)] - x[sl(None, -2)]) / (2.0 * dt)
        first = (x[sl(1, 2)] - x[sl(0, 1)]) / dt
        last = (x[sl(-1, None)] - x[sl(-2, -1)]) / dt
        return torch.cat([first, interior, last], dim=self.dim)


# ---------------------------------------------------------------------------
# Water budget loss
# ---------------------------------------------------------------------------

class WaterBudgetLoss(nn.Module):
    """Soft physics constraint:  residual = dPWV/dt + scale*div(qV) + P  ~ 0.

    Args
    ----
    delta_p_pa
        Layer-thickness proxy used to convert single-level div(qV) [1/s] into
        column-integrated VIMFD [kg/(m^2 s) = mm/s].  Default 30000 Pa (a
        700-1000 hPa column proxy when wind is taken at 850 hPa).
    g
        Gravitational acceleration (m/s^2).
    dt_seconds
        Time step of the model output (default 360 s = 6 min radar cadence).
    smooth_sigma
        Optional Gaussian smoothing (in grid points) applied to the divergence
        field, mirroring `compute_mfd.py` (sigma=2).  Set to 0 to disable.
    reduction
        'mean' (default) | 'sum' | 'none'.  Returned in the 'loss' field.
    use_huber
        If True, use Huber/Smooth-L1 on the residual; else mean |residual|.
    huber_delta
        Threshold for Huber (in mm/s).  Default 1e-4 mm/s.

    forward(...) returns a dict with at least:
        'loss'             : scalar
        'budget_residual'  : (B, T, H, W)
        'dpwv_dt'          : (B, T, H, W)
        'div_qV_mm_s'      : (B, T, H, W)
        'P_mm_s'           : (B, T, H, W)
    """

    def __init__(self,
                 delta_p_pa: float = 30000.0,
                 g: float = G_ACCEL,
                 dt_seconds: float = 360.0,
                 smooth_sigma: float = 2.0,
                 reduction: str = "mean",
                 use_huber: bool = True,
                 huber_delta: float = 1e-4):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"bad reduction: {reduction}")
        self.delta_p_pa = float(delta_p_pa)
        self.g = float(g)
        self.scale = self.delta_p_pa / self.g          # kg/m^2 per (kg/kg)
        self.divergence = SphericalDivergence(smooth_sigma=smooth_sigma)
        self.time_deriv = TimeDerivative(dt_seconds=dt_seconds, dim=-3)
        self.reduction = reduction
        self.use_huber = bool(use_huber)
        self.huber_delta = float(huber_delta)

    def forward(self,
                pwv_pred: torch.Tensor,       # (B, T, H, W) mm
                rain_pred: torch.Tensor,      # (B, T, H, W) mm/h
                u_era5: torch.Tensor,         # (B, T, H, W) m/s
                v_era5: torch.Tensor,         # (B, T, H, W) m/s
                q_era5: torch.Tensor,         # (B, T, H, W) kg/kg
                lat: torch.Tensor,            # (H,) deg
                lon: torch.Tensor,            # (W,) deg
                mask: Optional[torch.Tensor] = None,
                ) -> dict:
        if not (pwv_pred.shape == rain_pred.shape == u_era5.shape == v_era5.shape == q_era5.shape):
            raise ValueError(
                f"shape mismatch: pwv {pwv_pred.shape} rain {rain_pred.shape} "
                f"u {u_era5.shape} v {v_era5.shape} q {q_era5.shape}"
            )
        if pwv_pred.dim() != 4:
            raise ValueError(f"expected (B,T,H,W), got {pwv_pred.shape}")

        qu = q_era5 * u_era5
        qv = q_era5 * v_era5
        div = self.divergence(qu, qv, lat, lon)            # 1/s
        div_mm_s = self.scale * div                         # mm/s

        dpwv_dt = self.time_deriv(pwv_pred)                 # mm/s

        P_mm_s = rain_pred / 3600.0                         # mm/h -> mm/s

        residual = dpwv_dt + div_mm_s + P_mm_s

        if mask is not None:
            if mask.shape != residual.shape:
                raise ValueError(
                    f"mask shape {mask.shape} != residual {residual.shape}"
                )
            mask_f = mask.to(dtype=residual.dtype)
            r_for_loss = residual * mask_f
            n = mask_f.sum().clamp(min=1.0)
        else:
            r_for_loss = residual
            n = torch.tensor(float(residual.numel()), device=residual.device, dtype=residual.dtype)

        if self.use_huber:
            d = self.huber_delta
            abs_r = r_for_loss.abs()
            quad = 0.5 * (r_for_loss ** 2) / d
            lin = abs_r - 0.5 * d
            elementwise = torch.where(abs_r <= d, quad, lin)
        else:
            elementwise = r_for_loss.abs()

        if self.reduction == "none":
            loss = elementwise
        elif self.reduction == "sum":
            loss = elementwise.sum()
        else:  # mean over valid points
            loss = elementwise.sum() / n

        return {
            "loss": loss,
            "budget_residual": residual,
            "dpwv_dt": dpwv_dt,
            "div_qV_mm_s": div_mm_s,
            "P_mm_s": P_mm_s,
        }
