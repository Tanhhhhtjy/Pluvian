"""Revalidation harness for Audit-T1 must-fixes."""
import sys, time
sys.path.insert(0, "/Data/tanh/npj")
import numpy as np
import pandas as pd

from pipeline import pwv_io
from pipeline.data_loader import NPJDataset

print("[M1] n_pwv_stations =", len(pwv_io.list_pwv_stations()))
assert len(pwv_io.list_pwv_stations()) == 176, "expected 176 PWV stations"

start = pd.Timestamp("2023-07-28 12:00")

# Cold sample
t0 = time.time()
ds = NPJDataset(starts=[start], window_minutes=180)
sample = ds[0]
cold = time.time() - t0
print(f"[M3] cold single-sample wall = {cold:.2f}s")

# Warm sample (caches hot)
t0 = time.time()
sample2 = ds[0]
warm = time.time() - t0
print(f"[M3] warm single-sample wall = {warm:.2f}s")
assert warm < 5.0, f"warm > 5s: {warm}"

# drop_pwv check
ds_drop = NPJDataset(starts=[start], window_minutes=180, drop_pwv=True)
s_drop = ds_drop[0]
pwv_abs = float(np.asarray(s_drop["pwv_grid"]).__abs__().sum())
mask_sum = float(np.asarray(s_drop["pwv_mask"]).sum())
print(f"[M2] drop_pwv: pwv_abs_sum={pwv_abs}, mask_sum={mask_sum}")
assert pwv_abs == 0 and mask_sum == 0

# Batch of 5 sequential 3h windows
starts5 = [pd.Timestamp("2023-07-28 00:00") + pd.Timedelta(hours=3*i) for i in range(5)]
ds5 = NPJDataset(starts=starts5, window_minutes=180)
t0 = time.time()
for i in range(5):
    _ = ds5[i]
batch = time.time() - t0
print(f"[M3] 5 samples wall = {batch:.2f}s  ({batch/5:.2f}s/sample mean)")

print("OK — all must-fix checks pass")
