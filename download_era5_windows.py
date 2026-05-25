"""
Pluvian ERA5 数据下载脚本 (Windows 版)
=================================

在 Windows 上跑这个脚本，把 ERA5 压力层数据全部下载到本地，
之后通过 AutoDL JupyterLab 上传到远程服务器。

------- 一次性准备（你只需做一次） -------

1) 安装 Python 3.10+：
   - 去 https://www.python.org/downloads/windows/ 下载安装
   - 安装时**勾上** "Add Python to PATH"

2) 打开 PowerShell（搜索栏输入 powershell）

3) 安装 cdsapi：
       pip install cdsapi

4) 创建 CDS API 凭据文件
   在 PowerShell 里执行（一次到位）：

       $cdsapirc = @"
       url: https://cds.climate.copernicus.eu/api
       key: 9dbb3976-2eef-4f59-9303-f87ff09ec2b5
       "@
       $cdsapirc | Out-File -Encoding ASCII -FilePath "$env:USERPROFILE\.cdsapirc"

   验证: type $env:USERPROFILE\.cdsapirc 应能看到 url + key

------- 跑下载 -------

    python download_era5_windows.py

预计 15-30 分钟（取决于你的国际带宽，每个文件 ~4MB，共 160 个）
下载产物在 D:/Pluvian_ERA5/ （默认）或脚本里 OUT_DIR 改的位置

------- 下载完后上传到 AutoDL -------

把 D:/Pluvian_ERA5/ 整个文件夹打包成 era5.zip（在文件资源管理器里右键 → 发送到 → 压缩文件夹）
然后用 AutoDL JupyterLab Web 上传到 /root/autodl-tmp/，告诉我即可。
"""

import cdsapi
import os
import sys
import time

# ============================================
# 配置
# ============================================
OUT_DIR = r"D:\Pluvian_ERA5"   # 改成你想存的路径；D:\ 如果没有就改 C:\Pluvian_ERA5
os.makedirs(OUT_DIR, exist_ok=True)

MONTHS = ['05', '06', '07', '08', '09']
LEVELS = ['925', '850', '700', '600', '500', '400', '300', '250']
VARS = [
    ('u_component_of_wind',  'u'),
    ('v_component_of_wind',  'v'),
    ('specific_humidity',    'q'),
    ('temperature',          't'),
]

# ============================================
# 主循环
# ============================================
c = cdsapi.Client()

total = len(MONTHS) * len(LEVELS) * len(VARS)
done = 0
skipped = 0
failed = []

for mo in MONTHS:
    for lvl in LEVELS:
        for var_full, var_short in VARS:
            done += 1
            target = os.path.join(OUT_DIR, f'era5_{var_short}_{lvl}_2023{mo}.nc')

            # 已下完则跳过
            if os.path.exists(target) and os.path.getsize(target) > 100_000:
                skipped += 1
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

            print(f'[{done}/{total}] {var_short}@{lvl}hPa 2023-{mo}', flush=True)
            t0 = time.time()
            try:
                c.retrieve('reanalysis-era5-pressure-levels', req, target)
                sz = os.path.getsize(target) / 1e6
                dt = time.time() - t0
                print(f'   -> {sz:.1f} MB, {dt:.1f}s', flush=True)
            except Exception as e:
                print(f'   [FAIL] {e}', flush=True)
                failed.append((var_short, lvl, mo, str(e)))

print(f'\n========================================')
print(f'完成 {done - skipped - len(failed)} 个, 跳过 {skipped} 个, 失败 {len(failed)} 个')
if failed:
    print(f'失败清单:')
    for f in failed:
        print(f'  {f}')

print(f'\n下载文件在: {OUT_DIR}')
print(f'下一步: 把整个 {OUT_DIR} 文件夹打成 era5.zip 上传到 AutoDL')
