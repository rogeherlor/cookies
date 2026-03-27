# -*- coding: utf-8 -*-
"""
visualize_compare.py — Multi-filter comparison plots for ins_compare.py.

Generates 7 publication-quality figures comparing all filter variants:
    1. compare_trajectory_2d.png  — top-down 2D trajectory overlay
    2. compare_position_error.png — E/N/U error time series
    3. compare_error_magnitude.png — 2D horizontal error magnitude vs time
    4. compare_metrics_bar.png    — grouped bar chart of key metrics
    5. compare_outage_zoom.png    — 2D trajectory during outage only
    6. compare_uncertainty.png    — position std bands (3×2 grid, filters only)
    7. compare_metrics_table.png  — color-coded metrics table

Public API:
    generate_comparison_plots(filter_results, p_gt, gnss_outage_info,
                              sample_rate, output_dir) -> list[str]
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend; works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Optional map background support (same as visualize.py)
try:
    import contextily as ctx
    import pymap3d as pm
    from visualize import _latlon_to_mercator, _apply_mercator_tick_labels
    HAS_MAP_SUPPORT = True
except ImportError:
    HAS_MAP_SUPPORT = False

# ── Per-filter colors and markers ─────────────────────────────────────────────
_FILTER_STYLE = {
    'ekf_vanilla':   {'color': '#1f77b4', 'ls': '--',  'marker': 'o', 'label': 'EKF'},
    'ekf_enhanced':  {'color': '#0a3d7a', 'ls': '-',   'marker': 's', 'label': 'EKF+'},
    'eskf_vanilla':  {'color': '#ff7f0e', 'ls': '--',  'marker': '^', 'label': 'ESKF'},
    'eskf_enhanced': {'color': '#a63c00', 'ls': '-',   'marker': 'D', 'label': 'ESKF+'},
    'iekf_vanilla':  {'color': '#2ca02c', 'ls': '--',  'marker': 'v', 'label': 'IEKF'},
    'iekf_enhanced': {'color': '#145214', 'ls': '-',   'marker': 'P', 'label': 'IEKF+'},
    'imu_only':      {'color': '#d62728', 'ls': ':',   'marker': 'x', 'label': 'IMU Only'},
    'rts_smoother':  {'color': '#9467bd', 'ls': '-',   'marker': '*', 'label': 'Ground Truth (RTS)'},
    # Deep learning filters
    'iekf_ai_imu':   {'color': '#c2185b', 'ls': '-',   'marker': 'o', 'label': 'IEKF AI-IMU'},
    'tlio':          {'color': '#00838f', 'ls': '-',   'marker': 's', 'label': 'TLIO'},
    'deep_kf':       {'color': '#6a0dad', 'ls': '-',   'marker': '^', 'label': 'Deep KF'},
    'tartan_imu':    {'color': '#bf360c', 'ls': '-',   'marker': 'D', 'label': 'Tartan IMU'},
    'isam2':         {'color': '#1565c0', 'ls': '-',   'marker': 'P', 'label': 'iSAM2'},
}


# Each entry: (filename_suffix, set_of_keys, plot_title)
_TRAJECTORY_GROUPS = [
    ('ekf',         {'ekf_vanilla', 'ekf_enhanced'},  '2D Trajectory \u2014 EKF'),
    ('eskf',        {'eskf_vanilla', 'eskf_enhanced'}, '2D Trajectory \u2014 ESKF'),
    ('iekf',        {'iekf_vanilla', 'iekf_enhanced'}, '2D Trajectory \u2014 IEKF'),
    ('imu_only',    {'imu_only'},                      '2D Trajectory \u2014 IMU Only'),
    ('iekf_ai_imu', {'iekf_ai_imu'},                   '2D Trajectory \u2014 IEKF AI-IMU'),
    ('tlio',        {'tlio'},                          '2D Trajectory \u2014 TLIO'),
    ('deep_kf',     {'deep_kf'},                       '2D Trajectory \u2014 Deep KF'),
    ('tartan_imu',  {'tartan_imu'},                    '2D Trajectory \u2014 Tartan IMU'),
    ('isam2',       {'isam2'},                         '2D Trajectory \u2014 iSAM2'),
]


def _style(key: str) -> dict:
    return _FILTER_STYLE.get(key, {'color': 'gray', 'ls': '-', 'marker': '.', 'label': key})


def _add_outage_shade(ax, gnss_outage_info, time_array=None, alpha=0.15):
    """Add a light-gray band marking the GNSS outage window."""
    t1 = gnss_outage_info['start']
    t2 = gnss_outage_info['end']
    if time_array is None or (t1 == 0 and t2 == 0):
        return
    ax.axvspan(t1, t2, alpha=alpha, color='gray', label='GNSS outage')


def _save(fig, path: str, dpi: int = 150) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


# ── 1. 2D Trajectory overlay ──────────────────────────────────────────────────

def _plot_trajectory_2d(filter_results, p_gt, gnss_outage_info, output_dir,
                        lla0=None, p_rts=None, gt_label='Ground Truth',
                        out_fname='compare_trajectory_2d.png',
                        title='2D Trajectory Comparison') -> str:
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    use_map = lla0 is not None and HAS_MAP_SUPPORT

    fig, ax = plt.subplots(figsize=(10, 8))

    if use_map:
        lat_gt, lon_gt, _ = pm.enu2geodetic(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                                             lla0[0], lla0[1], lla0[2])
        x_gt, y_gt = _latlon_to_mercator(lat_gt, lon_gt)
        # Per-axis limits: GT + all non-IMU filters, capped at 1× GT span beyond GT boundary
        xs_fil, ys_fil = [], []
        for entry in filter_results:
            if entry['key'] == 'imu_only':
                continue
            lat_e, lon_e, _ = pm.enu2geodetic(entry['p'][:, 0], entry['p'][:, 1],
                                               entry['p'][:, 2], lla0[0], lla0[1], lla0[2])
            xe, ye = _latlon_to_mercator(lat_e, lon_e)
            xs_fil.append(xe); ys_fil.append(ye)
        x_fil = np.concatenate(xs_fil) if xs_fil else x_gt
        y_fil = np.concatenate(ys_fil) if ys_fil else y_gt
        # Build limits: min-GT + expand for filter (capped at 1.5× GT half-extent).
        # Then expand the shorter axis to match the longer one so the map has equal
        # x/y scale without set_aspect ever needing to shrink the other axis.
        _xl = _axis_limits(x_gt, x_fil)
        _yl = _axis_limits(y_gt, y_fil)
        _xr, _yr = _xl[1] - _xl[0], _yl[1] - _yl[0]
        if _xr >= _yr:
            _cy = (_yl[0] + _yl[1]) / 2.0
            map_xlim, map_ylim = _xl, (_cy - _xr / 2, _cy + _xr / 2)
        else:
            _cx = (_xl[0] + _xl[1]) / 2.0
            map_xlim, map_ylim = (_cx - _yr / 2, _cx + _yr / 2), _yl
        # Set before add_basemap so contextily fetches the right tile region.
        ax.set_xlim(*map_xlim)
        ax.set_ylim(*map_ylim)
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik,
                            zoom='auto', alpha=1.0)
        except Exception as e:
            print(f"Warning: Could not add map background: {e}")
        for entry in filter_results:
            st = _style(entry['key'])
            p  = entry['p']
            lat_e, lon_e, _ = pm.enu2geodetic(p[:, 0], p[:, 1], p[:, 2],
                                               lla0[0], lla0[1], lla0[2])
            x_e, y_e = _latlon_to_mercator(lat_e, lon_e)
            ax.plot(x_e, y_e, color=st['color'], ls=st['ls'], lw=2.0,
                    label=st['label'], alpha=0.92, zorder=5)
        ax.plot(x_gt, y_gt, color='white', lw=3.5, zorder=1)   # white halo for readability
        ax.plot(x_gt, y_gt, 'k--', lw=1.5, label=gt_label, zorder=2)
        if A > 0 and B > A:
            ax.plot(x_gt[A:B+1], y_gt[A:B+1], color='red', lw=5.0, alpha=0.45,
                    zorder=3, label='GT outage')  # transparent red highlight
        if p_rts is not None:
            lat_r, lon_r, _ = pm.enu2geodetic(p_rts[:, 0], p_rts[:, 1], p_rts[:, 2],
                                               lla0[0], lla0[1], lla0[2])
            x_r, y_r = _latlon_to_mercator(lat_r, lon_r)
            ax.plot(x_r, y_r, color='#9467bd', lw=2.5, ls='-',
                    label='Ground Truth (RTS)', zorder=4)
            if A > 0 and B > A:
                ax.plot(x_r[A:B+1], y_r[A:B+1], color='#9467bd', lw=5.0,
                        ls='-', zorder=4)   # highlight outage segment, no extra legend entry
        if A > 0 and B > A:
            ax.plot(x_gt[A], y_gt[A], 'gv', ms=10, zorder=6, label='Outage start')
            ax.plot(x_gt[B], y_gt[B], 'r^', ms=10, zorder=6, label='Outage end')
        # Re-apply limits after drawing (artist additions can trigger autoscale).
        # Because map_xlim and map_ylim already have equal ranges, set_aspect is a no-op.
        ax.set_aspect('equal')
        ax.set_xlim(*map_xlim)
        ax.set_ylim(*map_ylim)
        _apply_mercator_tick_labels(ax)
        ax.set_xlabel('Easting (Web Mercator)')
        ax.set_ylabel('Northing (Web Mercator)')
        ax.set_title(f'{title} (Map View)')
    else:
        non_imu = [e for e in filter_results if e['key'] != 'imu_only']
        for entry in filter_results:
            st = _style(entry['key'])
            p  = entry['p']
            ax.plot(p[:, 0], p[:, 1], color=st['color'], ls=st['ls'], lw=2.0,
                    label=st['label'], alpha=0.92, zorder=5)
        ax.plot(p_gt[:, 0], p_gt[:, 1], 'k--', lw=1.5, label=gt_label, zorder=2)
        if A > 0 and B > A:
            ax.plot(p_gt[A:B+1, 0], p_gt[A:B+1, 1], color='red', lw=5.0, alpha=0.45,
                    zorder=3, label='GT outage')  # transparent red highlight
        if p_rts is not None:
            ax.plot(p_rts[:, 0], p_rts[:, 1], color='#9467bd', lw=2.5, ls='-',
                    label='Ground Truth (RTS)', zorder=4)
            if A > 0 and B > A:
                ax.plot(p_rts[A:B+1, 0], p_rts[A:B+1, 1], color='#9467bd', lw=5.0,
                        ls='-', zorder=4)   # highlight outage segment, no extra legend entry
        if A > 0 and B > A:
            ax.plot(p_gt[A, 0], p_gt[A, 1], 'gv', ms=10, zorder=6, label='Outage start')
            ax.plot(p_gt[B, 0], p_gt[B, 1], 'r^', ms=10, zorder=6, label='Outage end')
        # Per-axis limits: GT + non-IMU filters, capped at 1× GT span beyond GT boundary
        x_fil = np.concatenate([e['p'][:, 0] for e in non_imu]) if non_imu else p_gt[:, 0]
        y_fil = np.concatenate([e['p'][:, 1] for e in non_imu]) if non_imu else p_gt[:, 1]
        ax.set_xlim(*_axis_limits(p_gt[:, 0], x_fil))
        ax.set_ylim(*_axis_limits(p_gt[:, 1], y_fil))
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_title(title)

    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, out_fname)
    return _save(fig, path)


# ── 2. Position error time series (E / N / U) ─────────────────────────────────

def _plot_position_error(filter_results, p_gt, gnss_outage_info,
                         sample_rate, output_dir) -> str:
    N    = len(p_gt)
    time = np.arange(N) / sample_rate
    labels = ['East error [m]', 'North error [m]', 'Up error [m]']

    # Pass 1: collect all error arrays
    all_err = {e['key']: e['p'] - p_gt for e in filter_results}

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for j, ax in enumerate(axes):
        # Y-limit from non-IMU filters for this component
        non_imu_comp = np.concatenate(
            [all_err[k][:, j] for k in all_err if k != 'imu_only'])
        ylim_top = float(np.percentile(np.abs(non_imu_comp), 98)) * 1.1
        ylim_top = max(ylim_top, 0.1)

        for entry in filter_results:
            st  = _style(entry['key'])
            comp = all_err[entry['key']][:, j]
            ax.plot(time, comp,
                    color=st['color'], ls=st['ls'], lw=1.0,
                    label=st['label'], alpha=0.85)
            if entry['key'] == 'imu_only':
                peak = float(np.max(np.abs(comp)))
                if peak > ylim_top:
                    ax.text(0.98, 0.95, f'IMU peak: ±{peak:.0f} m',
                            transform=ax.transAxes, ha='right', va='top',
                            fontsize=7, color=st['color'],
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        _add_outage_shade(ax, gnss_outage_info, time)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_ylim(-ylim_top, ylim_top)
        ax.set_ylabel(labels[j])
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc='upper left', fontsize=7, ncol=2)
    axes[2].set_xlabel('Time [s]')
    fig.suptitle('Position Error Time Series', fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'compare_position_error.png')
    return _save(fig, path)


# ── 3. 2D horizontal error magnitude ─────────────────────────────────────────

def _plot_error_magnitude(filter_results, p_gt, gnss_outage_info,
                          sample_rate, output_dir, p_rts=None) -> str:
    N    = len(p_gt)
    time = np.arange(N) / sample_rate

    # Pass 1: compute all magnitudes
    all_mag = {}
    for entry in filter_results:
        err = entry['p'] - p_gt
        all_mag[entry['key']] = np.sqrt(err[:, 0]**2 + err[:, 1]**2)

    # Y-limit from non-IMU filters
    non_imu_mags = np.concatenate([m for k, m in all_mag.items() if k != 'imu_only'])
    ylim_top = float(np.percentile(non_imu_mags, 98)) * 1.1
    ylim_top = max(ylim_top, 0.1)

    fig, ax = plt.subplots(figsize=(12, 5))

    for entry in filter_results:
        st  = _style(entry['key'])
        mag = all_mag[entry['key']]
        ax.plot(time, mag,
                color=st['color'], ls=st['ls'], lw=1.2,
                label=st['label'], alpha=0.85)
        if entry['key'] == 'imu_only':
            peak = float(np.max(mag))
            if peak > ylim_top:
                ax.text(0.98, 0.95, f'IMU peak: {peak:.0f} m',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=8, color=st['color'],
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    if p_rts is not None:
        err_rts = p_rts - p_gt
        mag_rts = np.sqrt(err_rts[:, 0]**2 + err_rts[:, 1]**2)
        ax.plot(time, mag_rts, color='#9467bd', ls='-', lw=2.0,
                label='Ground Truth (RTS)', alpha=0.9, zorder=10)

    _add_outage_shade(ax, gnss_outage_info, time)
    ax.set_ylim(0, ylim_top)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('2D Position Error [m]')
    ax.set_title('Horizontal Position Error Magnitude')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, 'compare_error_magnitude.png')
    return _save(fig, path)


# ── 4. Metrics bar chart ───────────────────────────────────────────────────────

def _plot_metrics_bar(filter_results, output_dir) -> str:
    metric_keys  = ['ATE 2D', 'ATE 3D', 'RTE 1s', 'RTE 5s', 'Pos RMSE 2D']
    n_filters    = len(filter_results)
    n_metrics    = len(metric_keys)
    bar_w        = 0.8 / n_filters
    x            = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(13, 6))

    # Pass 1: collect values for all filters
    all_vals = {}
    for entry in filter_results:
        mets = entry['metrics']
        all_vals[entry['key']] = [
            mets.get('ate',          {}).get('rmse_2D', float('nan')),
            mets.get('ate',          {}).get('rmse_3D', float('nan')),
            mets.get('rte_1s',       {}).get('rmse',    float('nan')),
            mets.get('rte_5s',       {}).get('rmse',    float('nan')),
            mets.get('position_rmse',{}).get('2D',      float('nan')),
        ]

    # Y-limit from non-IMU filters only
    non_imu_finite = [v for k, vals in all_vals.items()
                      if k != 'imu_only' for v in vals if np.isfinite(v)]
    ylim_top = (max(non_imu_finite) * 1.3) if non_imu_finite else 50.0

    # Pass 2: draw bars
    for idx, entry in enumerate(filter_results):
        vals = all_vals[entry['key']]
        st   = _style(entry['key'])
        # Clip bar heights to ylim_top for drawing
        draw_vals = [min(v, ylim_top) if np.isfinite(v) else 0.0 for v in vals]
        bars = ax.bar(
            x + idx * bar_w - 0.4 + bar_w / 2,
            draw_vals, width=bar_w * 0.9,
            color=st['color'], label=st['label'], alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            if not np.isfinite(val):
                continue
            bx = bar.get_x() + bar.get_width() / 2
            if val > ylim_top * 0.98:
                # Annotate clipped bar with arrow + actual value
                ax.annotate(
                    f'{val:.0f}',
                    xy=(bx, ylim_top),
                    xytext=(bx, ylim_top * 0.80),
                    ha='center', fontsize=6,
                    color=st['color'],
                    arrowprops=dict(arrowstyle='->', color=st['color'], lw=0.8),
                )
            else:
                ax.text(bx, bar.get_height() + ylim_top * 0.01,
                        f'{val:.1f}', ha='center', va='bottom',
                        fontsize=6, rotation=90)

    ax.set_ylim(0, ylim_top)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_ylabel('Error [m]')
    ax.set_title('Filter Metric Comparison')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(output_dir, 'compare_metrics_bar.png')
    return _save(fig, path)


# ── 5. Outage period zoom ─────────────────────────────────────────────────────

def _top_filters_outage(filter_results, p_gt, A, B, n=3):
    """Return the top-n filter entries ranked by mean 2D error during [A:B]."""
    ranked = []
    for entry in filter_results:
        if entry['key'] == 'imu_only':
            continue
        err = entry['p'][A:B, :2] - p_gt[A:B, :2]
        ranked.append((float(np.sqrt((err**2).sum(axis=1)).mean()), entry))
    ranked.sort(key=lambda x: x[0])
    return [e for _, e in ranked[:n]]


def _axis_limits(gt_vals, fil_vals, n=1.5):
    """Return (lim_min, lim_max) for one axis.

    Centre on GT midpoint.  Measure the GT half-extent in each direction
    (positive / negative from centre).  Cap the filter extent in each
    direction at n × that GT half-extent.  A 3 % pad is added after capping.
    """
    cx = (float(gt_vals.min()) + float(gt_vals.max())) / 2
    gt_pos = float(gt_vals.max()) - cx          # GT half-extent toward +
    gt_neg = cx - float(gt_vals.min())          # GT half-extent toward -
    pad = max(gt_pos + gt_neg, 1.0) * 0.03
    fil_pos = max(float(fil_vals.max()) - cx, 0.0)
    fil_neg = max(cx - float(fil_vals.min()), 0.0)
    lim_max = cx + min(max(fil_pos, gt_pos), n * gt_pos) + pad
    lim_min = cx - min(max(fil_neg, gt_neg), n * gt_neg) - pad
    return lim_min, lim_max


def _plot_outage_zoom(filter_results, p_gt, gnss_outage_info, output_dir,
                      lla0=None, p_rts=None, gt_label='Ground Truth') -> str:
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']

    if A >= B:
        return None   # no outage configured

    top3 = _top_filters_outage(filter_results, p_gt, A, B, n=3)
    best_label = _style(top3[0]['key'])['label'] if top3 else ''

    use_map = lla0 is not None and HAS_MAP_SUPPORT
    fig, ax = plt.subplots(figsize=(9, 7))

    if use_map:
        lat_gt, lon_gt, _ = pm.enu2geodetic(p_gt[A:B, 0], p_gt[A:B, 1], p_gt[A:B, 2],
                                             lla0[0], lla0[1], lla0[2])
        x_gt, y_gt = _latlon_to_mercator(lat_gt, lon_gt)
        # Build bounding box from GT + all top-3 filters
        xs_fil, ys_fil = [], []
        for e in top3:
            lat_e, lon_e, _ = pm.enu2geodetic(e['p'][A:B, 0], e['p'][A:B, 1], e['p'][A:B, 2],
                                               lla0[0], lla0[1], lla0[2])
            xe, ye = _latlon_to_mercator(lat_e, lon_e)
            xs_fil.append(xe); ys_fil.append(ye)
        x_fil = np.concatenate(xs_fil) if xs_fil else x_gt
        y_fil = np.concatenate(ys_fil) if ys_fil else y_gt
        ax.set_xlim(*_axis_limits(x_gt, x_fil))
        ax.set_ylim(*_axis_limits(y_gt, y_fil))
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik,
                            zoom='auto', alpha=1.0)
        except Exception as e:
            print(f"Warning: Could not add map background: {e}")
        for entry in filter_results:
            if entry['key'] == 'imu_only':
                continue
            st = _style(entry['key'])
            p  = entry['p']
            lat_e, lon_e, _ = pm.enu2geodetic(p[A:B, 0], p[A:B, 1], p[A:B, 2],
                                               lla0[0], lla0[1], lla0[2])
            x_e, y_e = _latlon_to_mercator(lat_e, lon_e)
            ax.plot(x_e, y_e, color=st['color'], ls=st['ls'], lw=2.0,
                    label=st['label'], alpha=0.92, zorder=5)
        if p_rts is not None:
            lat_r, lon_r, _ = pm.enu2geodetic(p_rts[A:B, 0], p_rts[A:B, 1], p_rts[A:B, 2],
                                               lla0[0], lla0[1], lla0[2])
            x_r, y_r = _latlon_to_mercator(lat_r, lon_r)
            ax.plot(x_r, y_r, color='#9467bd', lw=3.5, ls='-',
                    label='Ground Truth (RTS)', zorder=11)
        ax.plot(x_gt, y_gt, color='white', lw=5.0, zorder=9)
        ax.plot(x_gt, y_gt, 'k-', lw=3.0, label=gt_label, zorder=10)
        ax.plot(x_gt[0],  y_gt[0],  'gv', ms=12, zorder=12, label='Outage start')
        ax.plot(x_gt[-1], y_gt[-1], 'r^', ms=12, zorder=12, label='Outage end')
        ax.set_aspect('equal')
        _apply_mercator_tick_labels(ax)
        ax.set_xlabel('Easting (Web Mercator)')
        ax.set_ylabel('Northing (Web Mercator)')
        ax.set_title(f'GNSS Outage Period: {gnss_outage_info["start"]}s – {gnss_outage_info["end"]}s'
                     f'  |  zoom: {best_label}  (Map View)')
    else:
        for entry in filter_results:
            if entry['key'] == 'imu_only':
                continue
            st = _style(entry['key'])
            p  = entry['p']
            ax.plot(p[A:B, 0], p[A:B, 1], color=st['color'], ls=st['ls'], lw=2.0,
                    label=st['label'], alpha=0.92, zorder=5)
        if p_rts is not None:
            ax.plot(p_rts[A:B, 0], p_rts[A:B, 1], color='#9467bd', lw=3.5, ls='-',
                    label='Ground Truth (RTS)', zorder=11)
        ax.plot(p_gt[A:B, 0], p_gt[A:B, 1], 'k-', lw=3.0,
                label=gt_label, zorder=10)
        ax.plot(p_gt[A, 0],   p_gt[A, 1],   'gv', ms=12, zorder=12, label='Outage start')
        ax.plot(p_gt[B-1, 0], p_gt[B-1, 1], 'r^', ms=12, zorder=12, label='Outage end')
        # Per-axis limits: GT outage + top-3 filters, capped at 1× GT span beyond GT boundary
        x_fil = np.concatenate([e['p'][A:B, 0] for e in top3]) if top3 else p_gt[A:B, 0]
        y_fil = np.concatenate([e['p'][A:B, 1] for e in top3]) if top3 else p_gt[A:B, 1]
        ax.set_xlim(*_axis_limits(p_gt[A:B, 0], x_fil))
        ax.set_ylim(*_axis_limits(p_gt[A:B, 1], y_fil))
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_title(f'GNSS Outage Period: {gnss_outage_info["start"]}s – {gnss_outage_info["end"]}s'
                     f'  |  zoom: {best_label}')

    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, 'compare_outage_zoom.png')
    return _save(fig, path)


# ── 6. Uncertainty corridors (3×2 grid, filters only) ────────────────────────

def _plot_uncertainty(filter_results, p_gt, gnss_outage_info,
                      sample_rate, output_dir) -> str:
    # Select only the 6 filter variants (skip imu_only)
    filt_entries = [e for e in filter_results if e['key'] != 'imu_only']
    if not filt_entries:
        return None

    N    = len(p_gt)
    time = np.arange(N) / sample_rate

    # Arrange 3 rows (EKF / ESKF / IEKF) × 2 cols (vanilla / enhanced)
    row_order = ['ekf_vanilla', 'eskf_vanilla', 'iekf_vanilla',
                 'ekf_enhanced', 'eskf_enhanced', 'iekf_enhanced']
    key_to_pos = {
        'ekf_vanilla':   (0, 0), 'ekf_enhanced':  (0, 1),
        'eskf_vanilla':  (1, 0), 'eskf_enhanced': (1, 1),
        'iekf_vanilla':  (2, 0), 'iekf_enhanced': (2, 1),
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    err_mag = np.sqrt((p_gt[:, 0])**2 + (p_gt[:, 1])**2)   # not used directly

    for entry in filt_entries:
        key  = entry['key']
        pos  = key_to_pos.get(key)
        if pos is None:
            continue
        ax   = axes[pos[0], pos[1]]
        st   = _style(key)

        p       = entry['p']
        std_pos = entry['std_pos']
        err     = np.sqrt((p[:, 0] - p_gt[:, 0])**2 + (p[:, 1] - p_gt[:, 1])**2)
        std_2d  = np.sqrt(std_pos[:, 0]**2 + std_pos[:, 1]**2)

        ax.plot(time, err, color=st['color'], lw=1.2, label='2D error')
        ax.fill_between(time, 0, 3 * std_2d,
                        color=st['color'], alpha=0.2, label='3σ bound')
        _add_outage_shade(ax, gnss_outage_info, time)
        ax.set_title(st['label'], fontsize=9)
        ax.set_ylabel('Error / Std [m]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)

    for ax in axes[2, :]:
        ax.set_xlabel('Time [s]')

    fig.suptitle('2D Position Error and 3σ Uncertainty Band', fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'compare_uncertainty.png')
    return _save(fig, path)


# ── 7. Color-coded metrics table ──────────────────────────────────────────────

def _plot_metrics_table(filter_results, output_dir) -> str:
    col_labels = ['ATE 2D [m]', 'ATE 3D [m]', 'RTE 1s [m]',
                  'RTE 5s [m]', 'Pos RMSE 2D [m]', 'Outage max [m]']
    row_labels = [_style(e['key'])['label'] for e in filter_results]

    data = []
    for entry in filter_results:
        mets = entry['metrics']
        data.append([
            mets.get('ate',          {}).get('rmse_2D', float('nan')),
            mets.get('ate',          {}).get('rmse_3D', float('nan')),
            mets.get('rte_1s',       {}).get('rmse',    float('nan')),
            mets.get('rte_5s',       {}).get('rmse',    float('nan')),
            mets.get('position_rmse',{}).get('2D',      float('nan')),
            (mets.get('outage_analysis') or {}).get('max',  float('nan')),
        ])

    data_arr = np.array(data, dtype=float)

    # Per-column normalised rank: 0 (best=green) → 1 (worst=red)
    cell_colors = []
    for i in range(len(row_labels)):
        row_c = []
        for j in range(len(col_labels)):
            col = data_arr[:, j]
            finite = col[np.isfinite(col)]
            if len(finite) < 2:
                row_c.append('#dddddd')
            else:
                rank = (col[i] - finite.min()) / (finite.max() - finite.min() + 1e-12)
                r  = int(255 * rank)
                g  = int(255 * (1 - rank))
                row_c.append(f'#{r:02x}{g:02x}80')
        cell_colors.append(row_c)

    fig, ax = plt.subplots(figsize=(len(col_labels) * 1.8, len(row_labels) * 0.7 + 1))
    ax.axis('off')

    cell_text = [[f'{v:.2f}' if np.isfinite(v) else '—' for v in row] for row in data]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    fig.suptitle('Filter Metrics Summary  (green = best, red = worst per column)',
                 fontsize=11, y=0.98)
    plt.tight_layout()

    path = os.path.join(output_dir, 'compare_metrics_table.png')
    return _save(fig, path)


# ── Public entry point ─────────────────────────────────────────────────────────

def generate_comparison_plots(
    filter_results: list,
    p_gt: np.ndarray,
    gnss_outage_info: dict,
    sample_rate: float,
    output_dir: str,
    lla0=None,
    p_rts=None,
    gt_label: str = 'Ground Truth',
) -> list:
    """
    Generate all comparison plots and save them to output_dir.

    Args:
        filter_results : list of dicts, one per filter:
                         {'name', 'key', 'p', 'v', 'r', 'std_pos', 'metrics'}
        p_gt           : Nx3 ground-truth ENU position [m]
        gnss_outage_info : {'start', 'end', 'duration', 'start_idx', 'end_idx'}
        sample_rate    : IMU sample rate [Hz]
        output_dir     : directory where PNGs are saved
        lla0           : optional [lat, lon, alt] origin; enables map background
        p_rts          : optional Nx3 RTS smoother position [m]; drawn as a
                         purple overlay when p_gt is the KITTI reference
        gt_label       : legend label for the p_gt line (e.g. 'Ground Truth (RTS)'
                         or 'KITTI GPS GT')

    Returns:
        List of file paths of generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    plots = [
        (f'trajectory_2d_{sfx}', lambda fr=keys, t=ttl, s=sfx: _plot_trajectory_2d(
            [e for e in filter_results if e['key'] in fr],
            p_gt, gnss_outage_info, output_dir,
            lla0=lla0, p_rts=p_rts, gt_label=gt_label,
            out_fname=f'compare_trajectory_2d_{s}.png',
            title=t))
        for sfx, keys, ttl in _TRAJECTORY_GROUPS
    ] + [
        ('position_error',   lambda: _plot_position_error(
            filter_results, p_gt, gnss_outage_info, sample_rate, output_dir)),
        ('error_magnitude',  lambda: _plot_error_magnitude(
            filter_results, p_gt, gnss_outage_info, sample_rate, output_dir, p_rts=p_rts)),
        ('metrics_bar',      lambda: _plot_metrics_bar(
            filter_results, output_dir)),
        ('outage_zoom',      lambda: _plot_outage_zoom(
            filter_results, p_gt, gnss_outage_info, output_dir,
            lla0=lla0, p_rts=p_rts, gt_label=gt_label)),
        ('uncertainty',      lambda: _plot_uncertainty(
            filter_results, p_gt, gnss_outage_info, sample_rate, output_dir)),
        ('metrics_table',    lambda: _plot_metrics_table(
            filter_results, output_dir)),
    ]

    for name, fn in plots:
        try:
            path = fn()
            if path:
                generated.append(path)
        except Exception as e:
            print(f"[visualize_compare] WARNING: could not generate '{name}': {e}")

    return generated
