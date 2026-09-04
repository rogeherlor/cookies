#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-off: produce three separate PNGs for the 3.tex seq08 ground-truth
comparison figure (map overlay, discrepancy-vs-time, zoomed crop around the
peak discrepancy), reusing the cached FGO-Batch trajectory from
full_benchmark_results/gt_cache/08.npz instead of recomputing it (avoids
needing the GTSAM conda env, which lacks matplotlib/contextily).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import pymap3d as pm

_HERE = Path(__file__).resolve().parent
_HAILO_DIR = _HERE.parent
_PY_DIR = _HAILO_DIR.parent / 'python'
sys.path.insert(0, str(_PY_DIR))

from visualize import _latlon_to_mercator, _apply_mercator_tick_labels  # noqa: E402
from data_loader import get_kitti_dataset  # noqa: E402

_REPO_ROOT = _HERE.parent.parent.parent.parent
OUT_DIR = _REPO_ROOT / 'images' / 'c3' / 'seq08'
OUT_DIR.mkdir(exist_ok=True)

drive = '2011_09_30_drive_0028_extract'
nav = get_kitti_dataset('08')

lat_kitti = nav.lla[:, 0]
lon_kitti = nav.lla[:, 1]

gt = np.load(_HAILO_DIR / 'full_benchmark_results' / 'gt_cache' / '08.npz')
p_batch = gt['p']
lat_b, lon_b, _ = pm.enu2geodetic(p_batch[:, 0], p_batch[:, 1], p_batch[:, 2],
                                   nav.lla0[0], nav.lla0[1], nav.lla0[2])

x_kitti, y_kitti = _latlon_to_mercator(lat_kitti, lon_kitti)
x_b, y_b = _latlon_to_mercator(lat_b, lon_b)

n = min(len(lat_kitti), len(lat_b))
d_east = x_kitti[:n] - x_b[:n]
d_north = y_kitti[:n] - y_b[:n]
discrepancy = np.hypot(d_east, d_north)
t = np.arange(n) / nav.sample_rate

i_peak = int(np.argmax(discrepancy))
t_peak = t[i_peak]
print(f"peak discrepancy {discrepancy[i_peak]:.2f} m at t={t_peak:.1f}s (sample {i_peak})")

# ── (a) full map overlay ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_b, y_b, color='black', linewidth=1.6, label='FGO-Batch', zorder=3)
ax.plot(x_kitti, y_kitti, color='blue', linewidth=1.4, linestyle='--', alpha=0.9,
        label='KITTI OXTS (raw fix)', zorder=4)
ax.set_aspect('equal', adjustable='datalim')
try:
    ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik,
                     zoom='auto', alpha=1.0)
except Exception as e:
    print(f"Warning: could not add basemap: {e}")
_apply_mercator_tick_labels(ax)
ax.set_xlabel('Easting (Web Mercator)', fontsize=11)
ax.set_ylabel('Northing (Web Mercator)', fontsize=11)
ax.legend(loc='best', fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / 'seq08_gt_map.png', dpi=150)
plt.close(fig)

# ── (b) discrepancy vs time ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t, discrepancy, color='purple', linewidth=1.5)
ax.axvline(t_peak, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('KITTI vs FGO-Batch discrepancy (m)', fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'seq08_gt_discrepancy.png', dpi=150)
plt.close(fig)

# ── (c) zoomed crop around the peak ─────────────────────────────────────────
half_window_s = 12.0
mask = (t > t_peak - half_window_s) & (t < t_peak + half_window_s)
pad = 15.0  # metres of margin around the zoomed trajectory segment

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_b, y_b, color='black', linewidth=2.0, label='FGO-Batch', zorder=3)
ax.plot(x_kitti, y_kitti, color='blue', linewidth=2.4, linestyle='--', alpha=0.9,
        label='KITTI OXTS (raw fix)', zorder=4)
x_zoom = x_kitti[:n][mask]
y_zoom = y_kitti[:n][mask]
ax.set_xlim(x_zoom.min() - pad, x_zoom.max() + pad)
ax.set_ylim(y_zoom.min() - pad, y_zoom.max() + pad)
ax.set_aspect('equal', adjustable='box')
try:
    ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik,
                     zoom=19, alpha=1.0)
except Exception as e:
    print(f"Warning: could not add basemap: {e}")
_apply_mercator_tick_labels(ax)
ax.set_xlabel('Easting (Web Mercator)', fontsize=11)
ax.set_ylabel('Northing (Web Mercator)', fontsize=11)
ax.legend(loc='best', fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / 'seq08_gt_zoom.png', dpi=150)
plt.close(fig)

print("saved:", OUT_DIR / 'seq08_gt_map.png')
print("saved:", OUT_DIR / 'seq08_gt_discrepancy.png')
print("saved:", OUT_DIR / 'seq08_gt_zoom.png')
