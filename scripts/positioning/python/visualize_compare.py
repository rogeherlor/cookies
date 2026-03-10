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

# ── Per-filter colors and markers ─────────────────────────────────────────────
_FILTER_STYLE = {
    'ekf_vanilla':   {'color': '#1f77b4', 'ls': '--',  'marker': 'o', 'label': 'EKF Vanilla'},
    'ekf_enhanced':  {'color': '#aec7e8', 'ls': '-',   'marker': 's', 'label': 'EKF Enhanced'},
    'eskf_vanilla':  {'color': '#ff7f0e', 'ls': '--',  'marker': '^', 'label': 'ESKF Vanilla'},
    'eskf_enhanced': {'color': '#ffbb78', 'ls': '-',   'marker': 'D', 'label': 'ESKF Enhanced'},
    'iekf_vanilla':  {'color': '#2ca02c', 'ls': '--',  'marker': 'v', 'label': 'IEKF Vanilla'},
    'iekf_enhanced': {'color': '#98df8a', 'ls': '-',   'marker': 'P', 'label': 'IEKF Enhanced'},
    'imu_only':      {'color': '#d62728', 'ls': ':',   'marker': 'x', 'label': 'IMU Only'},
}


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

def _plot_trajectory_2d(filter_results, p_gt, gnss_outage_info, output_dir) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(p_gt[:, 0], p_gt[:, 1], 'k-', lw=2, label='Ground Truth', zorder=10)

    # Mark outage endpoints on GT
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    if A > 0 and B > A:
        ax.plot(p_gt[A, 0], p_gt[A, 1], 'gv', ms=10, zorder=12, label='Outage start')
        ax.plot(p_gt[B, 0], p_gt[B, 1], 'r^', ms=10, zorder=12, label='Outage end')

    for entry in filter_results:
        st = _style(entry['key'])
        p  = entry['p']
        ax.plot(p[:, 0], p[:, 1],
                color=st['color'], ls=st['ls'], lw=1.2,
                label=st['label'], alpha=0.85)

    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title('2D Trajectory Comparison')
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'compare_trajectory_2d.png')
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
                          sample_rate, output_dir) -> str:
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

def _plot_outage_zoom(filter_results, p_gt, gnss_outage_info, output_dir) -> str:
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']

    if A >= B:
        return None   # no outage configured

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(p_gt[A:B, 0], p_gt[A:B, 1], 'k-', lw=2.5,
            label='Ground Truth', zorder=10)
    ax.plot(p_gt[A, 0],   p_gt[A, 1],   'gv', ms=10, zorder=12, label='Outage start')
    ax.plot(p_gt[B-1, 0], p_gt[B-1, 1], 'r^', ms=10, zorder=12, label='Outage end')

    for entry in filter_results:
        if entry['key'] == 'imu_only':
            continue   # usually off-scale; included but clipped below
        st = _style(entry['key'])
        p  = entry['p']
        ax.plot(p[A:B, 0], p[A:B, 1],
                color=st['color'], ls=st['ls'], lw=1.5,
                label=st['label'], alpha=0.9)

    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title(f'GNSS Outage Period: {gnss_outage_info["start"]}s – {gnss_outage_info["end"]}s')
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal')
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
            mets.get('outage_analysis', {}).get('max',  float('nan')),
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

    Returns:
        List of file paths of generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    plots = [
        ('trajectory_2d',    lambda: _plot_trajectory_2d(
            filter_results, p_gt, gnss_outage_info, output_dir)),
        ('position_error',   lambda: _plot_position_error(
            filter_results, p_gt, gnss_outage_info, sample_rate, output_dir)),
        ('error_magnitude',  lambda: _plot_error_magnitude(
            filter_results, p_gt, gnss_outage_info, sample_rate, output_dir)),
        ('metrics_bar',      lambda: _plot_metrics_bar(
            filter_results, output_dir)),
        ('outage_zoom',      lambda: _plot_outage_zoom(
            filter_results, p_gt, gnss_outage_info, output_dir)),
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
