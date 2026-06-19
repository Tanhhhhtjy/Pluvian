"""Team dataset 批量调度器.

读取 manifest, 按 split 过滤, 展开成每天 144 个 6-min 时刻列表,
用 spawn-context multiprocessing Pool 调 build.dump_single_frame.

主进程累加 n_ok / n_skip, 每 100 帧打 progress + ETA,
最后写 out_root/run_log.json.

用法:
    python -m pluvian.team_dataset.run_batch \
        --out_root /data4/.../dataset_for_team \
        --normalize_stats docs/team_dataset/normalize_stats.json \
        --n_workers 8 \
        [--splits event_test test_robust train val] \
        [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Allow `python -m pluvian.team_dataset.run_batch` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -------------------- worker (top-level for spawn pickling) ------------------

def _worker(args: dict) -> dict:
    """单帧 dump worker. args = {ts_iso, out_root, stats}.

    永远不抛, 失败返回 ok=False + err.
    """
    ts_iso = args["ts_iso"]
    out_root = args["out_root"]
    stats = args["stats"]
    try:
        # 延迟 import: 子进程才加载 numpy / pandas / build, 主进程轻一点
        import pandas as _pd
        from pluvian.team_dataset.build import dump_single_frame
        ok = dump_single_frame(_pd.Timestamp(ts_iso), Path(out_root), stats)
        return {"ts": ts_iso, "ok": bool(ok), "err": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {
            "ts": ts_iso,
            "ok": False,
            "err": f"{type(e).__name__}: {e}",
            "tb": traceback.format_exc(limit=3),
        }


# -------------------- 时刻展开 ----------------------------------------------

def _enumerate_timestamps(manifest_path: Path, splits: List[str]) -> List[str]:
    """每天 144 个 6-min 时刻 (00:00, 00:06, ..., 23:54).

    drop / zero_day 都纳入交付集 (除非 split 是 'drop').
    """
    mf = pd.read_csv(manifest_path)
    mf = mf[mf["split"].isin(splits)].copy()
    out: List[str] = []
    for _, row in mf.iterrows():
        day = pd.Timestamp(row["date"]).normalize()
        for k in range(144):
            ts = day + pd.Timedelta(minutes=6 * k)
            out.append(ts.strftime("%Y-%m-%dT%H:%M:%S"))
    return out


# -------------------- main ---------------------------------------------------

def _load_stats(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[warn] normalize_stats not found at {p}, will use per-image min/max")
        return {}
    with open(p, "r") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Team dataset 批量 dump")
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--normalize_stats", default=None)
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--splits", nargs="+",
                    default=["event_test", "test_robust", "train", "val"],
                    help="manifest split labels (drop 不会被默认包含)")
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "pipeline" / "manifest.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 个时刻 (dev), 0 = 全跑")
    ap.add_argument("--progress_every", type=int, default=100)
    args = ap.parse_args(argv)

    args.out_root.mkdir(parents=True, exist_ok=True)
    stats = _load_stats(args.normalize_stats)

    timestamps = _enumerate_timestamps(args.manifest, args.splits)
    if args.limit > 0:
        timestamps = timestamps[: args.limit]
    n_total = len(timestamps)
    print(f"[run_batch] splits={args.splits} n_total={n_total} "
          f"n_workers={args.n_workers}")
    if n_total == 0:
        print("[run_batch] empty timestamp list, exit.")
        return

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    worker_args = [
        {"ts_iso": ts, "out_root": str(args.out_root), "stats": stats}
        for ts in timestamps
    ]

    n_ok = 0
    n_skip = 0
    skip_list: List[Tuple[str, str]] = []

    nw = max(1, min(args.n_workers, n_total))
    pool = None
    interrupted = False
    try:
        if nw == 1:
            results_iter = (_worker(wa) for wa in worker_args)
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(nw)
            results_iter = pool.imap_unordered(_worker, worker_args, chunksize=1)

        n_done = 0
        for res in results_iter:
            n_done += 1
            if res["ok"]:
                n_ok += 1
            else:
                n_skip += 1
                skip_list.append((res["ts"], res.get("err") or "ok=False"))
                if len(skip_list) <= 5:
                    print(f"  [skip] {res['ts']} err={res.get('err')}")

            if n_done % args.progress_every == 0 or n_done == n_total:
                el = time.time() - t0
                rate = n_done / el if el > 0 else 0
                eta = (n_total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{n_total}] ok={n_ok} skip={n_skip} "
                      f"elapsed={el:.1f}s rate={rate:.2f} fr/s ETA={eta:.0f}s")

    except KeyboardInterrupt:
        interrupted = True
        print("\n[run_batch] KeyboardInterrupt -> terminating pool")
        if pool is not None:
            pool.terminate()
            pool.join()
    finally:
        if pool is not None and not interrupted:
            pool.close()
            pool.join()

    finished_at = datetime.now(timezone.utc).isoformat()
    elapsed = time.time() - t0

    log_payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_sec": round(elapsed, 2),
        "splits": args.splits,
        "n_workers": nw,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_skip": n_skip,
        "interrupted": interrupted,
        "skip_list": skip_list,
        "out_root": str(args.out_root),
        "normalize_stats": args.normalize_stats,
    }
    log_path = args.out_root / "run_log.json"
    with open(log_path, "w") as f:
        json.dump(log_payload, f, indent=2)
    print(f"[run_batch] done: ok={n_ok} skip={n_skip} elapsed={elapsed:.1f}s")
    print(f"[run_batch] log -> {log_path}")

    # 失败率 > 50% 时给非零退出码
    if n_ok == 0 and n_total > 0:
        sys.exit(2)
    if n_skip > n_total * 0.5:
        sys.exit(3)


if __name__ == "__main__":
    main()
