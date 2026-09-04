#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For each of the 13 filters (CPU only for the 4 DL filters), run it on seq08
with the standard 40s-start/60s-duration outage window IN ITS OWN
SUBPROCESS (via _seq08_run_one_worker.py — see that file's docstring for
why: running all 13 in one shared process silently corrupted at least one
result), then plot the saved trajectory against the cached FGO-Batch ground
truth for the 3.tex big-subfigure figure.

Mercator coordinates are computed once per algorithm and sanity-checked
against the raw-ENU RMSE (what the official ATE numbers are actually
computed from) before plotting, printed as
  raw-ENU rmse=... mercator-check rmse=...
A gap of order 1.2-1.5x is expected Web Mercator scale distortion at this
latitude for large absolute errors; anything wildly larger is a bug.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import contextily as ctx
import pymap3d as pm
import gc

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_PY_DIR = _REPO_ROOT / "scripts/positioning/python"
sys.path.insert(0, str(_PY_DIR))
sys.path.insert(0, str(_HERE))

import data_loader
from visualize import _latlon_to_mercator, _apply_mercator_tick_labels

OUT_DIR = _REPO_ROOT / "images" / "c3" / "seq08"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_DIR = _HERE / "full_benchmark_results" / "seq08_traj_cache"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

SEQ = "08"
OUTAGE_START, OUTAGE_DURATION = 40.0, 60.0

ALGOS = [
    ("esekfg_vanilla",  "classical", "ES-EKF Groves"),
    ("esekfg_enhanced", "classical", "ES-EKF Groves+"),
    ("esekfs_vanilla",  "classical", "ES-EKF Sola"),
    ("esekfs_enhanced", "classical", "ES-EKF Sola+"),
    ("iekf_vanilla",    "classical", "IEKF"),
    ("iekf_enhanced",   "classical", "IEKF+"),
    ("imu_only",        "classical", "IMU-only"),
    ("deep_iekf",       "dl",        "Deep IEKF (CPU)"),
    ("tlio",            "dl",        "TLIO (CPU)"),
    ("deep_kf",         "dl",        "DKF (CPU)"),
    ("tartan_imu",      "dl",        "Tartan IMU (CPU)"),
    ("isam2",           "smoother",  "iSAM2"),
    ("isam2_fixedlag",  "smoother",  "iSAM2 FL"),
]

nav = data_loader.get_kitti_dataset(SEQ)
lla0 = nav.lla0

gt = np.load(_HERE / "full_benchmark_results" / "gt_cache" / f"{SEQ}.npz")
p_gt = gt["p"]

A = int(OUTAGE_START * nav.sample_rate)
B = int((OUTAGE_START + OUTAGE_DURATION) * nav.sample_rate)

lat_gt, lon_gt, _ = pm.enu2geodetic(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2], lla0[0], lla0[1], lla0[2])
x_gt, y_gt = _latlon_to_mercator(lat_gt, lon_gt)
x_go, y_go = (x_gt[A:B], y_gt[A:B]) if B > A else (np.array([]), np.array([]))


def plot_mercator(x_gt_, y_gt_, x_est, y_est, x_go_, y_go_, has_outage, label, save_path):
    x_all = np.concatenate([x_gt_, x_est])
    y_all = np.concatenate([y_gt_, y_est])
    pad = 0.05 * max(x_all.max() - x_all.min(), y_all.max() - y_all.min(), 50.0)
    xlim = (x_all.min() - pad, x_all.max() + pad)
    ylim = (y_all.min() - pad, y_all.max() + pad)
    x_mid, y_mid = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    half = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2
    xlim = (x_mid - half, x_mid + half)
    ylim = (y_mid - half, y_mid + half)

    fig, ax = plt.subplots(figsize=(7, 7))
    if has_outage:
        ax.plot(x_go_, y_go_, color="red", linewidth=5.0, alpha=0.35,
                solid_capstyle="round", label="Outage window", zorder=2)
    ax.plot(x_gt_, y_gt_, "k", linewidth=1.6, label="GT (FGO-Batch)", zorder=3)
    ax.plot(x_est, y_est, "b", linestyle="--", linewidth=1.4, alpha=0.9, label="Estimate", zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    try:
        ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto", alpha=1.0)
    except Exception as e:
        print(f"  Warning: could not add basemap: {e}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _apply_mercator_tick_labels(ax)
    ax.set_xlabel("Easting", fontsize=9)
    ax.set_ylabel("Northing", fontsize=9)
    ax.set_title(label, fontsize=11)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    del fig, ax
    gc.collect()


for key, kind, label in ALGOS:
    print(f"=== {key} ({label}) ===")
    traj_npz = TRAJ_DIR / f"{key}.npz"
    cmd = [sys.executable, str(_HERE / "_seq08_run_one_worker.py"),
           "--approach", key, "--kind", kind, "--seq", SEQ, "--out", str(traj_npz)]
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  FAILED: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else '?'}")
        continue
    print(f"  {proc.stdout.strip()}")

    p_est = np.load(traj_npz)["p"]
    n = min(len(p_est), len(p_gt))

    diff = p_gt[:n] - p_est[:n]
    error_3D = np.linalg.norm(diff, axis=1)
    rmse = float(np.sqrt(np.mean(error_3D ** 2)))
    # The Mercator cross-check below is a purely HORIZONTAL distance, so it has
    # to be compared against the horizontal (E,N) RMSE. Comparing it against the
    # 3D RMSE — as this check originally did — silently mixes in the vertical
    # error and makes vertically-drifting filters look broken: deep_iekf's
    # error here is 2.86m horizontal but 4.35m vertical, which dragged the ratio
    # to 0.84 and tripped the warning even though its projection was exact.
    rmse_2D = float(np.sqrt(np.mean(np.sum(diff[:, :2] ** 2, axis=1))))

    lat_est, lon_est, _ = pm.enu2geodetic(p_est[:n, 0], p_est[:n, 1], p_est[:n, 2], lla0[0], lla0[1], lla0[2])
    x_est, y_est = _latlon_to_mercator(lat_est, lon_est)
    merc_rmse = float(np.sqrt(np.mean(np.hypot(x_est - x_gt[:n], y_est - y_gt[:n]) ** 2)))
    ratio = merc_rmse / rmse_2D if rmse_2D > 0 else float("nan")
    flag = "  <-- CHECK THIS" if not (1.3 <= ratio <= 1.7) else ""
    print(f"  raw-ENU rmse={rmse:.2f}m (2D {rmse_2D:.2f}m)  mercator-check "
          f"rmse={merc_rmse:.2f}m  ratio={ratio:.2f}{flag}")

    save_path = OUT_DIR / f"{key}.png"
    plot_mercator(x_gt[:n], y_gt[:n], x_est, y_est, x_go, y_go, B > A, label, str(save_path))
    print(f"  saved -> {save_path}")

print("done")
