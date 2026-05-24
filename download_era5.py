#!/usr/bin/env python3
"""
ERA5 压力层数据下载脚本（细粒度版）
按 variable × level × month 拆分请求，避开 CDS cost limit
"""

import cdsapi
import os
import sys
import time

OUT_DIR = '/Data/tanh/npj/era5'
os.makedirs(OUT_DIR, exist_ok=True)

MONTHS = ['05', '06', '07', '08', '09']
LEVELS = ['925', '850', '700', '600', '500', '400', '300', '250']
VARS = [
    ('u_component_of_wind',  'u'),
    ('v_component_of_wind',  'v'),
    ('specific_humidity',    'q'),
    ('temperature',          't'),
]

c = cdsapi.Client()

total = len(MONTHS) * len(LEVELS) * len(VARS)
done = 0
for mo in MONTHS:
    for lvl in LEVELS:
        for var_full, var_short in VARS:
            done += 1
            target = f'{OUT_DIR}/era5_{var_short}_{lvl}_2023{mo}.nc'
            if os.path.exists(target) and os.path.getsize(target) > 100_000:
                print(f'[skip {done}/{total}] {os.path.basename(target)}')
                continue
            req = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'download_format': 'unarchived',
                'variable': var_full,
                'pressure_level': lvl,
                'year': '2023',
                'month': mo,
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': [f'{h:02d}:00' for h in range(24)],
                'area': [45, 110, 33, 122],
            }
            print(f'[req {done}/{total}] {var_short}@{lvl}hPa 2023-{mo}')
            t0 = time.time()
            try:
                c.retrieve('reanalysis-era5-pressure-levels', req, target)
                dt = time.time() - t0
                sz = os.path.getsize(target) / 1e6
                print(f'  → {sz:.1f} MB, {dt/60:.1f} min')
            except Exception as e:
                print(f'  [FAIL] {e}', file=sys.stderr)

print('All done.')
