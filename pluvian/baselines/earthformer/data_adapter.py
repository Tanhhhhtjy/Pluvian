"""NPJDataset → Earthformer (B, T, H, W, C) adapter.

Earthformer's initial_downsample_stack_conv uses downscale [3, 2, 2] (=12),
so spatial dims must be divisible by 12. Our radar grid is 661×701; we
zero-pad bottom-right to 672×708 and record a (H_pad, W_pad) validity mask
so evaluation can drop the pad. lon/lat origin (top-left) is preserved.

Pad units are mm/h (already converted in NPJDataset via dbz_to_rainrate).
Pad value = 0 (no-rain), consistent with how missing radar tiles are
already padded inside the pipeline.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

# Allow direct script invocation (`python .../data_adapter.py`) by
# bootstrapping the repo root onto sys.path so `pipeline.*` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

    class Dataset:  # type: ignore[no-redef]
        pass

from pipeline.data_loader import NPJDataset
from pipeline.utils import RADAR_H, RADAR_W

# Earthformer initial downsample stack [3, 2, 2] requires multiples of 12.
EF_H_PAD = 672  # 661 + 11
EF_W_PAD = 708  # 701 + 7
assert EF_H_PAD % 12 == 0 and EF_W_PAD % 12 == 0


def _make_pad_mask() -> np.ndarray:
    m = np.zeros((EF_H_PAD, EF_W_PAD), dtype=np.float32)
    m[:RADAR_H, :RADAR_W] = 1.0
    return m


_PAD_MASK = _make_pad_mask()


class NPJEarthformerDataset(Dataset):
    """Wraps NPJDataset and emits Earthformer-shaped (T, H, W, 1) tensors.

    A sample dict has:
      - 'in':   (n_input, EF_H_PAD, EF_W_PAD, 1) float32, mm/h, pad=0
      - 'tgt':  (n_forecast, EF_H_PAD, EF_W_PAD, 1) float32, mm/h, pad=0
      - 'mask': (EF_H_PAD, EF_W_PAD) float32, 1 inside radar grid else 0
      - 'time': str start timestamp
      - 'meta': dict (date, split, zero_day, radar_sample_valid)
    """

    def __init__(
        self,
        split: str | None = None,
        n_input: int = 12,
        n_forecast: int = 18,
        manifest: Path | None = None,
        starts=None,
    ):
        self.n_input = int(n_input)
        self.n_forecast = int(n_forecast)
        window_min = (self.n_input + self.n_forecast) * 6  # 6-min frames
        # We disable ERA5 loading: baseline is radar-only.
        kwargs = dict(window_minutes=window_min, load_era5=False)
        if manifest is not None:
            kwargs["manifest"] = Path(manifest)
        if starts is not None:
            kwargs["starts"] = starts
        self._inner = NPJDataset(**kwargs)
        if split is not None and starts is None:
            keep = [i for i, m in enumerate(self._inner._meta)
                    if m.get("split") == split]
            if not keep:
                raise ValueError(f"split={split!r} produced 0 windows")
            self._inner._starts = [self._inner._starts[i] for i in keep]
            self._inner._meta = [self._inner._meta[i] for i in keep]
        self.split = split

    def __len__(self) -> int:
        return len(self._inner)

    def _pad(self, radar: np.ndarray) -> np.ndarray:
        # radar: (T, RADAR_H, RADAR_W) mm/h
        T = radar.shape[0]
        out = np.zeros((T, EF_H_PAD, EF_W_PAD, 1), dtype=np.float32)
        out[:, :RADAR_H, :RADAR_W, 0] = radar.astype(np.float32, copy=False)
        return out

    def __getitem__(self, idx):
        sample = self._inner[idx]
        radar = sample["radar"]
        if _HAS_TORCH and isinstance(radar, torch.Tensor):
            radar = radar.numpy()
        radar = np.asarray(radar, dtype=np.float32)
        if radar.shape[0] != self.n_input + self.n_forecast:
            raise RuntimeError(
                f"expected {self.n_input + self.n_forecast} frames, got "
                f"{radar.shape[0]}")
        if radar.shape[1:] != (RADAR_H, RADAR_W):
            raise RuntimeError(
                f"radar shape {radar.shape[1:]} != ({RADAR_H}, {RADAR_W})")
        full = self._pad(radar)
        in_seq = full[: self.n_input]
        tgt_seq = full[self.n_input:]
        out = {
            "in": in_seq,
            "tgt": tgt_seq,
            "mask": _PAD_MASK,
            "time": sample.get("time"),
            "meta": sample.get("meta", {}),
        }
        if _HAS_TORCH:
            out["in"] = torch.from_numpy(in_seq)
            out["tgt"] = torch.from_numpy(tgt_seq)
            out["mask"] = torch.from_numpy(_PAD_MASK)
        return out


def get_pluvian_dataset(
    split: str | None = None,
    n_input: int = 12,
    n_forecast: int = 18,
    manifest: Path | None = None,
    starts=None,
) -> NPJEarthformerDataset:
    """Factory mirroring the plan's Day 1 contract."""
    return NPJEarthformerDataset(
        split=split,
        n_input=n_input,
        n_forecast=n_forecast,
        manifest=manifest,
        starts=starts,
    )


if __name__ == "__main__":
    # Pick a single valid val window so the smoke test is fast and
    # deterministic (val is small and stable).
    ds = get_pluvian_dataset(split="val", n_input=12, n_forecast=18)
    print(f"[info] split=val len={len(ds)}")
    sample = ds[0]
    in_seq = sample["in"]
    tgt_seq = sample["tgt"]
    mask = sample["mask"]
    if _HAS_TORCH and isinstance(in_seq, torch.Tensor):
        in_np = in_seq.numpy(); tgt_np = tgt_seq.numpy(); mask_np = mask.numpy()
    else:
        in_np, tgt_np, mask_np = in_seq, tgt_seq, mask
    in_valid = in_np[:, :RADAR_H, :RADAR_W, 0]
    tgt_valid = tgt_np[:, :RADAR_H, :RADAR_W, 0]
    rmin = float(np.nanmin([in_valid.min(), tgt_valid.min()]))
    rmax = float(np.nanmax([in_valid.max(), tgt_valid.max()]))
    # Pad must be exactly zero everywhere outside the valid radar grid.
    pad_in = in_np.copy(); pad_in[:, :RADAR_H, :RADAR_W, 0] = 0.0
    pad_tgt = tgt_np.copy(); pad_tgt[:, :RADAR_H, :RADAR_W, 0] = 0.0
    pad_max = float(max(abs(pad_in).max(), abs(pad_tgt).max()))
    print(f"[info] in dtype={in_np.dtype} tgt dtype={tgt_np.dtype} "
          f"mask dtype={mask_np.dtype}")
    print(f"[info] pad region max |val| = {pad_max:.3g} (should be 0)")
    print(f"[info] time={sample['time']} meta={sample['meta']}")
    assert in_np.shape == (12, EF_H_PAD, EF_W_PAD, 1), in_np.shape
    assert tgt_np.shape == (18, EF_H_PAD, EF_W_PAD, 1), tgt_np.shape
    assert mask_np.shape == (EF_H_PAD, EF_W_PAD), mask_np.shape
    assert pad_max == 0.0, "pad region must be zero"
    print(f"[OK] in_shape={tuple(in_np.shape)} "
          f"tgt_shape={tuple(tgt_np.shape)} "
          f"mask_shape={tuple(mask_np.shape)} "
          f"mm/h_range=[{rmin:.2f}, {rmax:.2f}]")
