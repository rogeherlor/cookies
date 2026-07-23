# -*- coding: utf-8 -*-
"""
gt_comparison_satellite.py — KITTI OXTS vs FGO-Batch ground truth, overlaid on
a real map background, one image per sequence.

Reuses the exact map-background approach already proven in visualize.py
(_latlon_to_mercator / OpenStreetMap.Mapnik via contextily, EPSG:3857) rather
than Esri World Imagery — the earlier Esri-based attempt rendered blurry,
upscaled tiles for some sequences (e.g. seq06), whereas OSM/Mapnik at
visualize.py's zoom levels renders crisp street-level detail.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import pymap3d as pm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from visualize import _latlon_to_mercator, _apply_mercator_tick_labels  # noqa: E402
from data_loader import get_kitti_dataset  # noqa: E402
from smoothers import fgo_batch_runner  # noqa: E402

SEQ_TO_DRIVE = {
    '01': '2011_10_03_drive_0042_extract',
    '04': '2011_09_30_drive_0016_extract',
    '06': '2011_09_30_drive_0020_extract',
    '07': '2011_09_30_drive_0027_extract',
    '08': '2011_09_30_drive_0028_extract',
    '09': '2011_09_30_drive_0033_extract',
    '10': '2011_09_30_drive_0034_extract',
}

OUT_DIR = _HERE.parent.parent.parent / 'outputs' / 'gt_comparison'


def main():
    seqs = sys.argv[1:] or list(SEQ_TO_DRIVE.keys())

    for seq in seqs:
        drive = SEQ_TO_DRIVE[seq]
        print(f"=== seq{seq} ({drive}) ===")
        nav = get_kitti_dataset(drive)

        lat_kitti = nav.lla[:, 0]
        lon_kitti = nav.lla[:, 1]

        batch_result = fgo_batch_runner.run(nav_data=nav)
        p_batch = batch_result['p']
        lat_b, lon_b, _ = pm.enu2geodetic(p_batch[:, 0], p_batch[:, 1], p_batch[:, 2],
                                           nav.lla0[0], nav.lla0[1], nav.lla0[2])

        x_kitti, y_kitti = _latlon_to_mercator(lat_kitti, lon_kitti)
        x_b, y_b = _latlon_to_mercator(lat_b, lon_b)

        # Discrepancy vs time, using KITTI's own sample times.
        n = min(len(lat_kitti), len(lat_b))
        d_east = x_kitti[:n] - x_b[:n]
        d_north = y_kitti[:n] - y_b[:n]
        discrepancy = np.hypot(d_east, d_north)
        t = np.arange(n) / nav.sample_rate

        fig, (ax_map, ax_err) = plt.subplots(1, 2, figsize=(18, 8))

        ax_map.plot(x_kitti, y_kitti, color='cyan', linewidth=2.2,
                    label='KITTI OXTS (raw GT)', zorder=3)
        ax_map.plot(x_b, y_b, color='red', linewidth=2, linestyle='--',
                    label='FGO-Batch', zorder=4)

        # Equalize aspect by expanding the data limits (not shrinking the axes
        # box) — for long/thin trajectories (e.g. seq06's oval), the default
        # adjustable='box' shrinks the visible box down to a sliver matching
        # the narrower axis, leaving the rest of the figure blank. Must run
        # BEFORE add_basemap so the fetched tile extent matches the final
        # (expanded) view.
        ax_map.set_aspect('equal', adjustable='datalim')

        try:
            ctx.add_basemap(ax_map, crs='EPSG:3857',
                            source=ctx.providers.OpenStreetMap.Mapnik,
                            zoom='auto', alpha=1.0)
        except Exception as e:
            print(f"Warning: could not add basemap for seq{seq}: {e}")
        _apply_mercator_tick_labels(ax_map)
        ax_map.set_xlabel('Easting (Web Mercator)', fontsize=11)
        ax_map.set_ylabel('Northing (Web Mercator)', fontsize=11)
        ax_map.set_title(f'seq{seq} ({drive}) — trajectory over OSM map', fontsize=12)
        ax_map.legend(loc='best', fontsize=10)

        ax_err.plot(t, discrepancy, color='purple', linewidth=1.5)
        ax_err.set_xlabel('Time (s)', fontsize=11)
        ax_err.set_ylabel('KITTI vs FGO-Batch discrepancy (m)', fontsize=11)
        ax_err.set_title(
            f'Discrepancy — mean {discrepancy.mean():.2f} m, max {discrepancy.max():.2f} m',
            fontsize=12)
        ax_err.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = OUT_DIR / f'seq{seq}_{drive}_satellite.png'
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"saved -> {out_path}")


if __name__ == '__main__':
    main()
