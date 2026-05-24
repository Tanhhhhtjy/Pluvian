"""23·7 京津冀 case study — 6 物理诊断图.

Output: /Data/tanh/npj/figures/case_23p7/figN_*.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.dates import DateFormatter, DayLocator

sys.path.insert(0, "/Data/tanh/npj")
from pipeline.pwv_io import list_pwv_stations, _read_one_station as pwv_read
from pipeline.station_io import list_stations, _read_one_station as sta_read
from pipeline.utils import dbz_to_rainrate

OUT = Path("/Data/tanh/npj/figures/case_23p7")
OUT.mkdir(parents=True, exist_ok=True)

# Core box (Jing-Jin-Ji centre)
BOX_LON = (115.0, 118.0)
BOX_LAT = (39.0, 41.0)
# Wider plot domain
DOM_LON = (113.0, 120.0)
DOM_LAT = (36.0, 42.6)

T_START = pd.Timestamp("2023-07-25 00:00")
T_END   = pd.Timestamp("2023-08-02 00:00")

LOG = []
def log(msg):
    print(msg, flush=True)
    LOG.append(msg)

# ----------------------------------------------------------------------------
# Helpers
def open_monthly(template_dir, var, months=(("2023", "07"), ("2023", "08"))):
    """Concatenate monthly nc files into one xr.Dataset along valid_time."""
    paths = []
    for y, m in months:
        p = Path(template_dir) / f"{var}_{y}{m}.nc"
        if p.exists():
            paths.append(p)
    if not paths:
        raise FileNotFoundError(f"none of {var} months exist in {template_dir}")
    parts = [xr.open_dataset(p) for p in paths]
    ds = xr.concat(parts, dim="valid_time")
    return ds


def box_mean_timeseries(ds, varname, lon_rng=BOX_LON, lat_rng=BOX_LAT):
    """Area mean over box. ERA5 lat is decreasing."""
    lat0, lat1 = lat_rng
    if float(ds.latitude[0]) > float(ds.latitude[-1]):
        lat_sel = slice(lat1, lat0)
    else:
        lat_sel = slice(lat0, lat1)
    sub = ds.sel(latitude=lat_sel,
                 longitude=slice(lon_rng[0], lon_rng[1]))
    return sub[varname].mean(dim=["latitude", "longitude"]).to_pandas()


# ----------------------------------------------------------------------------
# fig1 timeseries
def fig1():
    log("\n=== fig1 timeseries ===")
    times_30min = pd.date_range(T_START, T_END, freq="30min")

    # (a) PWV box-mean per 30min
    pwv_sta = [s for s in list_pwv_stations()
               if BOX_LON[0] <= s["lon"] <= BOX_LON[1]
               and BOX_LAT[0] <= s["lat"] <= BOX_LAT[1]]
    log(f"  PWV stations in box: {len(pwv_sta)}")
    pwv_mat = np.full((len(times_30min), len(pwv_sta)), np.nan, dtype=np.float32)
    for j, s in enumerate(pwv_sta):
        try:
            df = pwv_read(s["path"])
        except Exception as e:
            log(f"    pwv read fail {s['code']}: {e}")
            continue
        m = (df.time >= T_START) & (df.time <= T_END)
        sub = df.loc[m, ["time", "PWV"]].set_index("time")
        sub = sub[~sub.index.duplicated()]
        if sub.empty:
            continue
        re = sub.reindex(times_30min, method="nearest", tolerance=pd.Timedelta("20min"))
        pwv_mat[:, j] = re.PWV.values
    # mask absurd values (PWV physical range 0-90)
    pwv_mat[(pwv_mat < 0) | (pwv_mat > 100)] = np.nan
    pwv_avg = np.nanmean(pwv_mat, axis=1)
    log(f"  PWV avg range: {np.nanmin(pwv_avg):.1f} - {np.nanmax(pwv_avg):.1f} mm")

    # (b) MFD 850
    mfd = open_monthly("/Data/tanh/npj/derived/mfd", "era5_mfd_850")
    mfd_ts = box_mean_timeseries(mfd, "mfd")
    mfd_ts = mfd_ts[(mfd_ts.index >= T_START) & (mfd_ts.index <= T_END)]
    log(f"  MFD850 range: {mfd_ts.min()*1e5:.2f} - {mfd_ts.max()*1e5:.2f} x10^-5")

    # (c) CAPE
    cape = open_monthly("/Data/tanh/npj/derived/instability", "era5_cape")
    cape_ts = box_mean_timeseries(cape, "cape")
    cape_ts = cape_ts[(cape_ts.index >= T_START) & (cape_ts.index <= T_END)]
    log(f"  CAPE range: {cape_ts.min():.0f} - {cape_ts.max():.0f} J/kg")

    # (d) hourly precip station avg
    sta = [s for s in list_stations()
           if BOX_LON[0] <= s["lon"] <= BOX_LON[1]
           and BOX_LAT[0] <= s["lat"] <= BOX_LAT[1]]
    log(f"  surface stations in box: {len(sta)}")
    times_h = pd.date_range(T_START, T_END, freq="h")
    pre = np.full((len(times_h), len(sta)), np.nan, dtype=np.float32)
    for j, s in enumerate(sta):
        try:
            df = sta_read(s["path"])
        except Exception:
            continue
        m = (df.time >= T_START) & (df.time <= T_END)
        sub = df.loc[m, ["time", "PRE_1h"]].set_index("time")
        sub = sub[~sub.index.duplicated()]
        re = sub.reindex(times_h, method="nearest", tolerance=pd.Timedelta("30min"))
        pre[:, j] = re.PRE_1h.values
    pre[pre < 0] = np.nan  # filter sentinels
    pre_avg = np.nanmean(pre, axis=1)
    log(f"  pre_avg max: {np.nanmax(pre_avg):.2f} mm/h")

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    ax = axes[0]
    ax.plot(times_30min, pwv_avg, color="navy", lw=1.2)
    ax.set_ylabel("PWV (mm)")
    ax.set_title("(a) GNSS-PWV box-mean (115-118°E, 39-41°N)", loc="left", fontsize=11)
    ax.axhline(50, color="grey", ls=":", lw=0.7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(mfd_ts.index, mfd_ts.values * 1e5, color="darkgreen", lw=1.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("MFD 850 (×10$^{-5}$ kg/kg/s)")
    ax.set_title("(b) 850 hPa moisture-flux convergence", loc="left", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(cape_ts.index, cape_ts.values, color="darkorange", lw=1.2)
    ax.axhline(1000, color="grey", ls=":", lw=0.7)
    ax.set_ylabel("CAPE (J/kg)")
    ax.set_title("(c) Most-unstable CAPE", loc="left", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.bar(times_h, pre_avg, width=pd.Timedelta("45min"),
           color="steelblue", edgecolor="none")
    ax.set_ylabel("PRE (mm/h)")
    ax.set_title("(d) Station mean hourly precipitation", loc="left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("UTC date")

    # Event start marker
    for a in axes:
        a.axvline(pd.Timestamp("2023-07-29 00:00"), color="red", ls="--", lw=0.6, alpha=0.6)

    fig.suptitle("23·7 Jing-Jin-Ji extreme rainfall — physical time series",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = OUT / "fig1_timeseries.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log(f"  saved {out}")
    return pwv_avg, mfd_ts, cape_ts, pre_avg


# ----------------------------------------------------------------------------
# fig2 PWV jump
def fig2():
    log("\n=== fig2 PWV jump ===")
    # 3 BJT times -> UTC
    bjt_times = [pd.Timestamp("2023-07-28 08:00") - pd.Timedelta(hours=8),
                 pd.Timestamp("2023-07-28 16:00") - pd.Timedelta(hours=8),
                 pd.Timestamp("2023-07-28 23:30") - pd.Timedelta(hours=8)]
    labels = ["2023-07-28 08:00 BJT", "2023-07-28 16:00 BJT", "2023-07-28 23:30 BJT"]

    pwv_sta = [s for s in list_pwv_stations()
               if DOM_LON[0] <= s["lon"] <= DOM_LON[1]
               and DOM_LAT[0] <= s["lat"] <= DOM_LAT[1]]
    coords = np.array([[s["lon"], s["lat"]] for s in pwv_sta])

    # Pre-read all stations' data
    vals_by_t = [np.full(len(pwv_sta), np.nan) for _ in bjt_times]
    for j, s in enumerate(pwv_sta):
        try:
            df = pwv_read(s["path"]).set_index("time")
        except Exception:
            continue
        for ti, t in enumerate(bjt_times):
            if t in df.index:
                vals_by_t[ti][j] = df.loc[t, "PWV"].mean() if hasattr(df.loc[t, "PWV"], "mean") else df.loc[t, "PWV"]
            else:
                # nearest within 20 min
                try:
                    pos = df.index.get_indexer([t], method="nearest", tolerance=pd.Timedelta("20min"))[0]
                    if pos >= 0:
                        vals_by_t[ti][j] = df["PWV"].iloc[pos]
                except Exception:
                    pass
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    vmin, vmax = 0, 80
    for ti, (t, lab) in enumerate(zip(bjt_times, labels)):
        ax = axes[ti]
        v = vals_by_t[ti]
        good = np.isfinite(v) & (v >= 0) & (v <= 100)
        sc = ax.scatter(coords[good, 0], coords[good, 1], c=v[good],
                        cmap="viridis", vmin=vmin, vmax=vmax,
                        s=28, edgecolors="k", linewidths=0.3)
        ax.set_xlim(*DOM_LON)
        ax.set_ylim(*DOM_LAT)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude (°E)")
        if ti == 0:
            ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"({chr(97+ti)}) {lab}\nmean={np.nanmean(v[good]):.1f} mm  N={good.sum()}",
                     fontsize=10)
        # Beijing/Tianjin/Shijiazhuang
        ax.plot([116.4, 117.2, 114.5], [39.9, 39.1, 38.0], "r^", markersize=6)
        for nm, x, y in [("BJ", 116.4, 39.9), ("TJ", 117.2, 39.1), ("SJZ", 114.5, 38.0)]:
            ax.annotate(nm, (x, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color="darkred")
        ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("PWV (mm)")
    fig.suptitle("PWV station snapshots — 7-28 moisture loading", fontsize=12)
    out = OUT / "fig2_pwv_jump.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {out}")


# ----------------------------------------------------------------------------
# fig3 MFD 3-level + wind
def fig3():
    log("\n=== fig3 MFD 3-level ===")
    t_target = pd.Timestamp("2023-07-29 02:00")
    levels = (925, 850, 700)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), constrained_layout=True)
    vmax = 1e-6
    for li, lev in enumerate(levels):
        mfd_ds = xr.open_dataset(
            f"/Data/tanh/npj/derived/mfd/era5_mfd_{lev}_202307.nc")
        # pick nearest valid_time
        mfd = mfd_ds["mfd"].sel(valid_time=t_target, method="nearest")
        lat = mfd_ds.latitude.values
        lon = mfd_ds.longitude.values

        # Wind
        u = xr.open_dataset(f"/Data/tanh/npj/era5/era5_u_{lev}_202307.nc"
                            )["u"].sel(valid_time=t_target, method="nearest").squeeze().values
        v = xr.open_dataset(f"/Data/tanh/npj/era5/era5_v_{lev}_202307.nc"
                            )["v"].sel(valid_time=t_target, method="nearest").squeeze().values
        u_lat = xr.open_dataset(f"/Data/tanh/npj/era5/era5_u_{lev}_202307.nc"
                                ).latitude.values
        u_lon = xr.open_dataset(f"/Data/tanh/npj/era5/era5_u_{lev}_202307.nc"
                                ).longitude.values

        ax = axes[li]
        # MFD pcolormesh (note ERA5 lat decreasing)
        # Plot in [-1, 1] x 1e-6
        LON, LAT = np.meshgrid(lon, lat)
        pc = ax.pcolormesh(LON, LAT, mfd.values, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        # Wind barbs/quiver subsampled
        ULON, ULAT = np.meshgrid(u_lon, u_lat)
        step = 3
        ax.quiver(ULON[::step, ::step], ULAT[::step, ::step],
                  u[::step, ::step], v[::step, ::step],
                  scale=200, width=0.003, color="k", alpha=0.7)

        ax.set_xlim(*DOM_LON)
        ax.set_ylim(*DOM_LAT)
        ax.set_aspect("equal")
        ax.set_title(f"({chr(97+li)}) {lev} hPa MFD\n"
                     f"max={mfd.values.max()*1e6:.2f} ×10$^{{-6}}$",
                     fontsize=10)
        ax.set_xlabel("Longitude (°E)")
        if li == 0:
            ax.set_ylabel("Latitude (°N)")
        ax.plot([116.4, 117.2, 114.5], [39.9, 39.1, 38.0], "k^", markersize=5)
        ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(pc, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("MFD (kg/kg/s) — positive = convergence")
    fig.suptitle(f"3-level moisture convergence  —  {t_target.strftime('%Y-%m-%d %H:%M')} UTC",
                 fontsize=12)
    out = OUT / "fig3_mfd_convergence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {out}")


# ----------------------------------------------------------------------------
# fig4 CAPE + K-index
def fig4():
    log("\n=== fig4 CAPE/K ===")
    t = pd.Timestamp("2023-07-28 06:00")
    cape_ds = xr.open_dataset("/Data/tanh/npj/derived/instability/era5_cape_202307.nc")
    k_ds = xr.open_dataset("/Data/tanh/npj/derived/instability/era5_kindex_202307.nc")
    cape = cape_ds["cape"].sel(valid_time=t, method="nearest")
    k = k_ds["kindex"].sel(valid_time=t, method="nearest")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, fld, ds, cmap, vmin, vmax, lab, thr in [
        (axes[0], cape, cape_ds, "YlOrRd", 0, 2500,
         "CAPE (J/kg)", 1500),
        (axes[1], k, k_ds, "YlOrRd", 20, 45,
         "K-index (°C)", 38)
    ]:
        LON, LAT = np.meshgrid(ds.longitude.values, ds.latitude.values)
        pc = ax.pcolormesh(LON, LAT, fld.values, cmap=cmap, vmin=vmin,
                           vmax=vmax, shading="auto")
        cs = ax.contour(LON, LAT, fld.values, levels=[thr], colors="navy",
                        linewidths=1.0)
        ax.clabel(cs, fmt=f"{thr}")
        ax.set_xlim(*DOM_LON)
        ax.set_ylim(*DOM_LAT)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.plot([116.4, 117.2, 114.5], [39.9, 39.1, 38.0], "k^", markersize=5)
        ax.grid(True, alpha=0.3)
        cb = fig.colorbar(pc, ax=ax, shrink=0.85)
        cb.set_label(lab)
        ax.set_title(lab + f"  max={float(fld.max()):.0f}", fontsize=10)
    axes[0].set_title(f"(a) CAPE  max={float(cape.max()):.0f} J/kg", fontsize=10)
    axes[1].set_title(f"(b) K-index  max={float(k.max()):.1f} °C", fontsize=10)
    fig.suptitle(f"Instability indices — {t.strftime('%Y-%m-%d %H:%M')} UTC  (BJT 14:00, daily CAPE peak)",
                 fontsize=12)
    out = OUT / "fig4_cape_kindex.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {out}")


# ----------------------------------------------------------------------------
# fig5 trigger visualisation: radar dBZ + PWV scatter at 4 frames
def fig5():
    log("\n=== fig5 trigger ===")
    bjt_frames = ["2023-07-28 18:00", "2023-07-28 21:00",
                  "2023-07-29 00:00", "2023-07-29 03:00"]
    utc_frames = [pd.Timestamp(t) - pd.Timedelta(hours=8) for t in bjt_frames]

    from pipeline.radar_io import date_dir
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.0), constrained_layout=True)
    radar_lon = np.round(np.arange(113.0, 120.0 + 1e-6, 0.01), 2)
    radar_lat = np.round(np.arange(36.0, 42.6 + 1e-6, 0.01), 2)

    # Pre-build PWV station df cache once
    pwv_sta = [s for s in list_pwv_stations()
               if DOM_LON[0] <= s["lon"] <= DOM_LON[1]
               and DOM_LAT[0] <= s["lat"] <= DOM_LAT[1]]
    pwv_dfs = {}
    for s in pwv_sta:
        try:
            pwv_dfs[s["code"]] = pwv_read(s["path"]).set_index("time")
        except Exception:
            pass

    for fi, (bjt, utc) in enumerate(zip(bjt_frames, utc_frames)):
        ax = axes[fi]
        # Radar nearest
        fn = date_dir(utc) / f"{utc.strftime('%Y%m%d_%H%M%S')}.nc"
        if not fn.exists():
            # find nearest within 6 minutes
            cand = sorted(date_dir(utc).glob("*.nc"))
            tdiffs = [(abs(pd.Timestamp(p.stem.replace("_", " ")) - utc), p) for p in cand
                      if len(p.stem) == 15]
            if tdiffs:
                tdiffs.sort()
                fn = tdiffs[0][1]
        if fn.exists():
            with xr.open_dataset(fn) as ds:
                arr = np.squeeze(ds["data0"].values).astype(np.float32)
        else:
            arr = np.zeros((len(radar_lat), len(radar_lon)), dtype=np.float32)
            log(f"  missing radar {utc}")
        arr[~np.isfinite(arr)] = 0.0
        # Mask <5 dBZ
        masked = np.ma.masked_less(arr, 5.0)
        pc = ax.pcolormesh(radar_lon, radar_lat, masked, cmap="turbo",
                           vmin=5, vmax=60, shading="auto")
        # PWV scatter
        xs, ys, vs = [], [], []
        for s in pwv_sta:
            df = pwv_dfs.get(s["code"])
            if df is None:
                continue
            try:
                pos = df.index.get_indexer([utc], method="nearest",
                                           tolerance=pd.Timedelta("30min"))[0]
            except Exception:
                pos = -1
            if pos < 0:
                continue
            val = df["PWV"].iloc[pos]
            if not np.isfinite(val) or val < 0 or val > 100:
                continue
            xs.append(s["lon"]); ys.append(s["lat"]); vs.append(val)
        sc = ax.scatter(xs, ys, c=vs, cmap="viridis", vmin=30, vmax=75,
                        s=18, edgecolors="k", linewidths=0.25)
        ax.set_xlim(*DOM_LON)
        ax.set_ylim(*DOM_LAT)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude (°E)")
        if fi == 0:
            ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"({chr(97+fi)}) {bjt} BJT", fontsize=10)
        ax.plot([116.4, 117.2, 114.5], [39.9, 39.1, 38.0], "w^", markersize=5,
                markeredgecolor="k")

    cb1 = fig.colorbar(pc, ax=axes[-1], shrink=0.8, location="right", pad=0.02)
    cb1.set_label("Radar dBZ")
    cb2 = fig.colorbar(sc, ax=axes[0], shrink=0.8, location="left", pad=0.02)
    cb2.set_label("PWV (mm)")
    fig.suptitle("Convection initiation — radar dBZ overlaid with GNSS-PWV stations",
                 fontsize=12)
    out = OUT / "fig5_trigger_visualization.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {out}")


# ----------------------------------------------------------------------------
# fig6 composite mechanism
def fig6():
    log("\n=== fig6 mechanism ===")
    t = pd.Timestamp("2023-07-28 21:00")  # UTC, BJT 05:00 next day-ish... -- task said 21UTC 7-28
    cape_ds = xr.open_dataset("/Data/tanh/npj/derived/instability/era5_cape_202307.nc")
    mfd_ds = xr.open_dataset("/Data/tanh/npj/derived/mfd/era5_mfd_850_202307.nc")
    u_ds = xr.open_dataset("/Data/tanh/npj/era5/era5_u_850_202307.nc")
    v_ds = xr.open_dataset("/Data/tanh/npj/era5/era5_v_850_202307.nc")
    cape = cape_ds["cape"].sel(valid_time=t, method="nearest")
    mfd = mfd_ds["mfd"].sel(valid_time=t, method="nearest")
    u = u_ds["u"].sel(valid_time=t, method="nearest").squeeze()
    v = v_ds["v"].sel(valid_time=t, method="nearest").squeeze()

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    LON, LAT = np.meshgrid(cape_ds.longitude.values, cape_ds.latitude.values)
    pc = ax.pcolormesh(LON, LAT, cape.values, cmap="YlOrRd",
                       vmin=0, vmax=2000, shading="auto")
    cb = fig.colorbar(pc, ax=ax, shrink=0.85)
    cb.set_label("CAPE (J/kg)")

    MLON, MLAT = np.meshgrid(mfd_ds.longitude.values, mfd_ds.latitude.values)
    cs = ax.contour(MLON, MLAT, mfd.values,
                    levels=np.arange(5e-7, 4e-6, 5e-7),
                    colors="white", linestyles="--", linewidths=0.9)
    ax.clabel(cs, fmt=lambda x: f"{x*1e6:.1f}", fontsize=7)

    ULON, ULAT = np.meshgrid(u_ds.longitude.values, u_ds.latitude.values)
    step = 2
    ax.quiver(ULON[::step, ::step], ULAT[::step, ::step],
              u.values[::step, ::step], v.values[::step, ::step],
              scale=250, width=0.003, color="k", alpha=0.7)

    # PWV stations with PWV>65
    pwv_sta = [s for s in list_pwv_stations()
               if DOM_LON[0] <= s["lon"] <= DOM_LON[1]
               and DOM_LAT[0] <= s["lat"] <= DOM_LAT[1]]
    xs, ys = [], []
    for s in pwv_sta:
        try:
            df = pwv_read(s["path"]).set_index("time")
            pos = df.index.get_indexer([t], method="nearest",
                                       tolerance=pd.Timedelta("30min"))[0]
            if pos < 0:
                continue
            val = df["PWV"].iloc[pos]
            if np.isfinite(val) and val > 65 and val < 100:
                xs.append(s["lon"]); ys.append(s["lat"])
        except Exception:
            continue
    ax.scatter(xs, ys, s=80, facecolors="none", edgecolors="red",
               linewidths=1.5, label=f"PWV>65 mm (N={len(xs)})")

    ax.set_xlim(*DOM_LON)
    ax.set_ylim(*DOM_LAT)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="upper right")
    ax.plot([116.4, 117.2, 114.5], [39.9, 39.1, 38.0], "k^", markersize=7)
    for nm, x, y in [("BJ", 116.4, 39.9), ("TJ", 117.2, 39.1), ("SJZ", 114.5, 38.0)]:
        ax.annotate(nm, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color="black")
    ax.set_title(f"Three-factor superposition at {t.strftime('%Y-%m-%d %H:%M')} UTC\n"
                 f"shading: CAPE  |  dashed contours: 850 hPa MFD (×10$^{{-6}}$)  |  "
                 f"vectors: 850 hPa wind  |  red circles: PWV>65 mm",
                 fontsize=11)
    out = OUT / "fig6_mechanism.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {out}")


if __name__ == "__main__":
    try:
        fig1()
    except Exception as e:
        import traceback; log("FIG1 ERROR\n" + traceback.format_exc())
    try:
        fig2()
    except Exception as e:
        import traceback; log("FIG2 ERROR\n" + traceback.format_exc())
    try:
        fig3()
    except Exception as e:
        import traceback; log("FIG3 ERROR\n" + traceback.format_exc())
    try:
        fig4()
    except Exception as e:
        import traceback; log("FIG4 ERROR\n" + traceback.format_exc())
    try:
        fig5()
    except Exception as e:
        import traceback; log("FIG5 ERROR\n" + traceback.format_exc())
    try:
        fig6()
    except Exception as e:
        import traceback; log("FIG6 ERROR\n" + traceback.format_exc())
    (OUT / "_run.log").write_text("\n".join(LOG))
    print("DONE")
