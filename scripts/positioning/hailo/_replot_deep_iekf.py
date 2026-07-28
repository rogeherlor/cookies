#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained: replot deep_iekf.png from the already-patched trajectory
npz (full_benchmark_results/seq08_traj_cache/deep_iekf.npz). No import of
_seq08_algo_trajectory_figs.py — that file's top-level code reruns the
whole 13-algorithm loop on import."""
import sys
import gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import contextily as ctx
import pymap3d as pm

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_PY_DIR = _REPO_ROOT / "scripts/positioning/python"
sys.path.insert(0, str(_PY_DIR))

import data_loader
from visualize import _latlon_to_mercator, _apply_mercator_tick_labels

SEQ = "08"
OUTAGE_START, OUTAGE_DURATION = 40.0, 60.0

nav = data_loader.get_kitti_dataset(SEQ)
lla0 = nav.lla0

gt = np.load(_HERE / "full_benchmark_results" / "gt_cache" / f"{SEQ}.npz")
p_gt = gt["p"]
p_est = np.load(_HERE / "full_benchmark_results" / "seq08_traj_cache" / "deep_iekf.npz")["p"]
n = min(len(p_est), len(p_gt))

diff = p_gt[:n] - p_est[:n]
rmse = float(np.sqrt(np.mean(np.linalg.norm(diff, axis=1) ** 2)))
print(f"patched deep_iekf raw-ENU rmse: {rmse:.2f}m (expect ~208.86)")

A = int(OUTAGE_START * nav.sample_rate)
B = int((OUTAGE_START + OUTAGE_DURATION) * nav.sample_rate)

lat_gt, lon_gt, _ = pm.enu2geodetic(p_gt[:n, 0], p_gt[:n, 1], p_gt[:n, 2], lla0[0], lla0[1], lla0[2])
lat_est, lon_est, _ = pm.enu2geodetic(p_est[:n, 0], p_est[:n, 1], p_est[:n, 2], lla0[0], lla0[1], lla0[2])
x_gt, y_gt = _latlon_to_mercator(lat_gt, lon_gt)
x_est, y_est = _latlon_to_mercator(lat_est, lon_est)
x_go, y_go = (x_gt[A:B], y_gt[A:B]) if B > A else (np.array([]), np.array([]))

merc_rmse = float(np.sqrt(np.mean(np.hypot(x_est - x_gt, y_est - y_gt) ** 2)))
print(f"mercator-check rmse: {merc_rmse:.2f}m  ratio={merc_rmse/rmse:.2f}")

x_all = np.concatenate([x_gt, x_est])
y_all = np.concatenate([y_gt, y_est])
pad = 0.05 * max(x_all.max() - x_all.min(), y_all.max() - y_all.min(), 50.0)
xlim = (x_all.min() - pad, x_all.max() + pad)
ylim = (y_all.min() - pad, y_all.max() + pad)
x_mid, y_mid = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
half = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2
xlim = (x_mid - half, x_mid + half)
ylim = (y_mid - half, y_mid + half)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(x_go, y_go, color="red", linewidth=5.0, alpha=0.35,
        solid_capstyle="round", label="Outage window", zorder=2)
ax.plot(x_gt, y_gt, "k", linewidth=1.6, label="GT (FGO-Batch)", zorder=3)
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
ax.set_title("Deep IEKF (CPU)", fontsize=11)
ax.tick_params(labelsize=7)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, loc="best")
fig.tight_layout()
save_path = _REPO_ROOT / "images" / "c3" / "seq08" / "deep_iekf.png"
fig.savefig(save_path, dpi=150)
plt.close(fig)
print("saved ->", save_path)
