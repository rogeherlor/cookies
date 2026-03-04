"""
Visualization Module for Navigation Systems

This module provides comprehensive visualization functions for trajectory analysis:
- 2D/3D trajectory plots with optional map backgrounds
- Error time series
- Statistical summaries
- GNSS outage analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Optional map background support
try:
    import contextily as ctx
    import pymap3d as pm
    HAS_MAP_SUPPORT = True
except ImportError:
    HAS_MAP_SUPPORT = False
    print("Note: Install contextily for map backgrounds: pip install contextily")


def _latlon_to_mercator(lat, lon):
    """Convert lat/lon (degrees) to Web Mercator (EPSG:3857) x/y in metres."""
    R = 6378137.0  # WGS-84 equatorial radius
    x = np.radians(lon) * R
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R
    return x, y


def _mercator_to_latlon(x, y):
    """Convert Web Mercator (EPSG:3857) x/y in metres back to lat/lon (degrees)."""
    R = 6378137.0
    lon = np.degrees(x / R)
    lat = np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)
    return lat, lon


def _apply_mercator_tick_labels(ax):
    """
    Replace numeric Mercator-metre tick labels with dual labels:
        "<km> km  (<degrees>°)"
    so the axes show both the projected coordinate and the geographic equivalent.
    """
    import matplotlib.ticker as mticker

    def x_fmt(x, pos):
        _, lon = _mercator_to_latlon(x, 0)  # lat irrelevant; lon depends only on x (easting)
        return f'{x/1000:.1f} km\n({lon:.3f}°)'

    def y_fmt(y, pos):
        lat, _ = _mercator_to_latlon(0, y)  # lon irrelevant; lat depends only on y (northing)
        return f'{y/1000:.1f} km\n({lat:.3f}°)'

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_fmt))


def plot_trajectory_2d(p_est, p_gt, gnss_outage_info, dataset_name, save_path=None, lla0=None, gps_available=None):
    """
    Plot 2D top-view trajectory with optional map background.
    
    Args:
        p_est: Estimated trajectory (Nx3) [E, N, U]
        p_gt: Ground truth trajectory (Nx3) [E, N, U]
        gnss_outage_info: Dict with outage information
        dataset_name: Name of dataset
        save_path: Optional path to save figure
        lla0: Optional [lat, lon, alt] origin for map background (enables map if available)
        gps_available: Optional boolean array indicating GPS measurement availability at each timestep
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    # Check if we should add map background
    use_map = lla0 is not None and HAS_MAP_SUPPORT
    
    if use_map:
        # Convert ENU → lat/lon → Web Mercator (EPSG:3857, metres)
        # Using Mercator avoids the cos(lat) distortion: axes are metric so
        # set_aspect('equal') gives the true proportions.
        lat_gt, lon_gt, _ = pm.enu2geodetic(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                                              lla0[0], lla0[1], lla0[2])
        lat_est, lon_est, _ = pm.enu2geodetic(p_est[:, 0], p_est[:, 1], p_est[:, 2],
                                                lla0[0], lla0[1], lla0[2])
        x_gt, y_gt = _latlon_to_mercator(lat_gt, lon_gt)
        x_est, y_est = _latlon_to_mercator(lat_est, lon_est)

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(x_gt, y_gt, 'k', linewidth=2, label='Ground Truth', zorder=3)
        ax.plot(x_est, y_est, 'b', linewidth=2, alpha=0.8, label='EKF Estimate', zorder=3)

        if A != 0 or B != 0:
            lat_est_o, lon_est_o, _ = pm.enu2geodetic(p_est[A:B, 0], p_est[A:B, 1], p_est[A:B, 2],
                                                       lla0[0], lla0[1], lla0[2])
            lat_gt_o, lon_gt_o, _ = pm.enu2geodetic(p_gt[A:B, 0], p_gt[A:B, 1], p_gt[A:B, 2],
                                                     lla0[0], lla0[1], lla0[2])
            x_est_o, y_est_o = _latlon_to_mercator(lat_est_o, lon_est_o)
            x_gt_o, y_gt_o = _latlon_to_mercator(lat_gt_o, lon_gt_o)
            ax.plot(x_est_o, y_est_o, 'r', linewidth=2.5,
                    label='EKF during GNSS outage', zorder=4)
            ax.plot(x_gt_o, y_gt_o, 'g', linewidth=2.5,
                    label='Ground Truth during GNSS outage', zorder=4)

        # Plot GPS measurement points if available
        if gps_available is not None:
            gps_indices = np.where(gps_available)[0]
            if len(gps_indices) > 0:
                x_gps, y_gps = _latlon_to_mercator(lat_gt[gps_indices], lon_gt[gps_indices])
                ax.scatter(x_gps, y_gps, c='orange', s=15, marker='o',
                           label=f'GPS Measurements ({len(gps_indices)})', zorder=3.5, alpha=0.6)

        # Add map background — tiles are already in EPSG:3857
        # zoom=15 gives crisp street-level tiles; lower (13-14) for longer routes
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik,
                            zoom=17, alpha=1.0)
        except Exception as e:
            print(f"Warning: Could not add map background: {e}")

        # Both axes are now in metres → equal aspect is exact
        ax.set_aspect('equal')

        # Show both Mercator metres and the equivalent lat/lon on each tick
        _apply_mercator_tick_labels(ax)

        ax.set_xlabel('Easting (Web Mercator)', fontsize=12)
        ax.set_ylabel('Northing (Web Mercator)', fontsize=12)
        ax.set_title(f'EKF Trajectory Estimation - {dataset_name} (Map View)', fontsize=14)
    else:
        # Standard ENU plot without map
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(p_gt[:, 0], p_gt[:, 1], 'k', linewidth=1.5, label='Ground Truth')
        ax.plot(p_est[:, 0], p_est[:, 1], 'b', linewidth=1.5, alpha=0.7, label='EKF Estimate')

        if A != 0 or B != 0:
            ax.plot(p_est[A:B, 0], p_est[A:B, 1], 'r', linewidth=2, label='EKF during GNSS outage')
            ax.plot(p_gt[A:B, 0], p_gt[A:B, 1], 'g', linewidth=2, label='Ground Truth during GNSS outage')

        # Plot GPS measurement points if available
        if gps_available is not None:
            gps_indices = np.where(gps_available)[0]
            if len(gps_indices) > 0:
                ax.scatter(p_gt[gps_indices, 0], p_gt[gps_indices, 1], c='orange', s=15, marker='o',
                           label=f'GPS Measurements ({len(gps_indices)})', zorder=3.5, alpha=0.6)

        ax.set_xlabel('East (m)', fontsize=12)
        ax.set_ylabel('North (m)', fontsize=12)
        ax.set_title(f'EKF Trajectory Estimation - {dataset_name}', fontsize=14)
        ax.axis('equal')

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, ax


def plot_trajectory_3d(p_est, p_gt, gnss_outage_info, dataset_name, save_path=None):
    """
    Plot 3D trajectory.
    
    Args:
        p_est: Estimated trajectory (Nx3) [E, N, U]
        p_gt: Ground truth trajectory (Nx3) [E, N, U]
        gnss_outage_info: Dict with outage information
        dataset_name: Name of dataset
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2], 'k', linewidth=1.5, label='Ground Truth')
    ax.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2], 'b', linewidth=1.5, alpha=0.7, label='EKF Estimate')
    
    if A != 0 or B != 0:
        ax.plot(p_est[A:B, 0], p_est[A:B, 1], p_est[A:B, 2], 'r', linewidth=2, label='GNSS Outage')
    
    ax.set_xlabel('East (m)', fontsize=11)
    ax.set_ylabel('North (m)', fontsize=11)
    ax.set_zlabel('Up (m)', fontsize=11)
    ax.set_title(f'3D Trajectory - {dataset_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, ax


def plot_position_errors(time, inst_errors, gnss_outage_info, save_path=None):
    """
    Plot position errors over time (E, N, U components).
    
    Args:
        time: Time array in seconds
        inst_errors: Dict with instantaneous errors
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    errors_labels = ['East', 'North', 'Up']
    
    for i, (ax, label) in enumerate(zip(axes, errors_labels)):
        ax.plot(time, inst_errors[['E', 'N', 'U'][i]], linewidth=1)
        if A != 0 or B != 0:
            ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
        ax.set_ylabel(f'{label} Error (m)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    axes[0].set_title('Position Errors Over Time (Absolute Error)', fontsize=14)
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, axes


def plot_2d_horizontal_error(time, error_2d, ate_rmse, gnss_outage_info, save_path=None):
    """
    Plot 2D horizontal position error with ATE reference.
    
    Args:
        time: Time array in seconds
        error_2d: 2D horizontal error array
        ate_rmse: ATE RMSE value for reference line
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time, error_2d, linewidth=1.5, color='blue')
    
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    
    ax.axhline(y=ate_rmse, color='green', linestyle='--', linewidth=2, 
               label=f"ATE RMSE: {ate_rmse:.2f}m")
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('2D Position Error (m)', fontsize=12)
    ax.set_title('2D Horizontal Position Error Over Time (Absolute Error)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, ax


def plot_total_errors(time, inst_errors, gnss_outage_info, save_path=None):
    """
    Plot 2D and 3D total position errors over time.
    
    Args:
        time: Time array in seconds
        inst_errors: Dict with instantaneous errors (must include '2D' and '3D')
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 2D horizontal error
    ax = axes[0]
    ax.plot(time, inst_errors['2D'], linewidth=1.5, color='#9467bd', label='2D Horizontal Error')
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    ax.set_ylabel('2D Error (m)', fontsize=11)
    ax.set_title('Total Position Errors Over Time (Absolute Error)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # 3D total error
    ax = axes[1]
    ax.plot(time, inst_errors['3D'], linewidth=1.5, color='#d62728', label='3D Total Error')
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    ax.set_ylabel('3D Error (m)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, axes


def plot_velocity_errors(time, v_est, v_gt, gnss_outage_info, save_path=None):
    """
    Plot velocity errors over time (E, N, U components).
    
    Args:
        time: Time array in seconds
        v_est: Estimated velocities (Nx3)
        v_gt: Ground truth velocities (Nx3)
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    errors_labels = ['East', 'North', 'Up']
    vel_errors = [np.abs(v_gt[:, i] - v_est[:, i]) for i in range(3)]
    
    for i, (ax, label) in enumerate(zip(axes, errors_labels)):
        ax.plot(time, vel_errors[i], linewidth=1)
        if A != 0 or B != 0:
            ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
        ax.set_ylabel(f'{label} Vel Error (m/s)', fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    
    axes[0].set_title('Velocity Errors Over Time (Absolute Error)', fontsize=14)
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, axes


def plot_velocity_total_errors(time, v_est, v_gt, gnss_outage_info, save_path=None):
    """
    Plot 2D and 3D total velocity errors over time.
    
    Args:
        time: Time array in seconds
        v_est: Estimated velocities (Nx3) [E, N, U]
        v_gt: Ground truth velocities (Nx3) [E, N, U]
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    # Compute 2D and 3D velocity errors
    vel_error_E = np.abs(v_gt[:, 0] - v_est[:, 0])
    vel_error_N = np.abs(v_gt[:, 1] - v_est[:, 1])
    vel_error_U = np.abs(v_gt[:, 2] - v_est[:, 2])
    vel_error_2D = np.sqrt(vel_error_E**2 + vel_error_N**2)
    vel_error_3D = np.sqrt(vel_error_E**2 + vel_error_N**2 + vel_error_U**2)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 2D horizontal velocity error
    ax = axes[0]
    ax.plot(time, vel_error_2D, linewidth=1.5, color='#9467bd', label='2D Horizontal Velocity Error')
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    ax.set_ylabel('2D Velocity Error (m/s)', fontsize=11)
    ax.set_title('Total Velocity Errors Over Time (Absolute Error)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # 3D total velocity error
    ax = axes[1]
    ax.plot(time, vel_error_3D, linewidth=1.5, color='#d62728', label='3D Total Velocity Error')
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    ax.set_ylabel('3D Velocity Error (m/s)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, axes


def plot_orientation_errors(time, r_est, r_gt, gnss_outage_info, save_path=None):
    """
    Plot orientation errors over time (Roll, Pitch, Yaw).
    
    Args:
        time: Time array in seconds
        r_est: Estimated orientations (Nx3) in radians
        r_gt: Ground truth orientations (Nx3) in radians
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    orient_labels = ['Roll', 'Pitch', 'Yaw']
    # Compute orientation errors with proper angle wrapping
    orient_errors = []
    for i in range(3):
        diff = r_gt[:, i] - r_est[:, i]
        # Wrap angles to [-π, π] to handle discontinuities
        diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))
        orient_errors.append(np.abs(diff_wrapped))
    
    for i, (ax, label) in enumerate(zip(axes, orient_labels)):
        ax.plot(time, np.degrees(orient_errors[i]), linewidth=1)
        if A != 0 or B != 0:
            ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
        ax.set_ylabel(f'{label} Error (deg)', fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    
    axes[0].set_title('Orientation Errors Over Time (Absolute Angular Error)', fontsize=14)
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, axes


def plot_orientation_total_errors(time, r_est, r_gt, gnss_outage_info, save_path=None):
    """
    Plot 2D and 3D total orientation errors over time.
    
    Args:
        time: Time array in seconds
        r_est: Estimated orientations (Nx3) in radians [roll, pitch, yaw]
        r_gt: Ground truth orientations (Nx3) in radians [roll, pitch, yaw]
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    # Compute orientation errors with proper angle wrapping
    orient_errors = []
    for i in range(3):
        diff = r_gt[:, i] - r_est[:, i]
        # Wrap angles to [-π, π] to handle discontinuities
        diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))
        orient_errors.append(np.abs(diff_wrapped))
    
    # Compute 2D (roll + pitch) and 3D (roll + pitch + yaw) total errors
    orient_error_2D = np.sqrt(orient_errors[0]**2 + orient_errors[1]**2)
    orient_error_3D = np.sqrt(orient_errors[0]**2 + orient_errors[1]**2 + orient_errors[2]**2)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 2D orientation error (roll + pitch)
    ax = axes[0]
    ax.plot(time, np.degrees(orient_error_2D), linewidth=1.5, color='#9467bd', label='2D Orientation Error (Roll+Pitch)')
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    ax.set_ylabel('2D Orientation Error (deg)', fontsize=11)
    ax.set_title('Total Orientation Errors Over Time (Absolute Angular Error)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # 3D total orientation error (roll + pitch + yaw)
    ax = axes[1]
    ax.plot(time, np.degrees(orient_error_3D), linewidth=1.5, color='#d62728', label='3D Total Orientation Error (Roll+Pitch+Yaw)')
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    ax.set_ylabel('3D Orientation Error (deg)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, axes


def plot_error_statistics(results, dataset_name, save_path=None):
    """
    Plot summary bar charts of all error statistics.
    
    Args:
        results: Dict with evaluation results
        dataset_name: Name of dataset
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position RMSE
    ax = axes[0, 0]
    pos_rmse = [results['position_rmse']['E'], results['position_rmse']['N'], 
                results['position_rmse']['2D'], results['position_rmse']['U'], 
                results['position_rmse']['3D']]
    ax.bar(['East', 'North', '2D', 'Up', '3D'], pos_rmse, 
           color=['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#d62728'])
    ax.set_ylabel('RMSE (m)', fontsize=11)
    ax.set_title('Position RMSE', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Velocity RMSE
    ax = axes[0, 1]
    vel_rmse = [results['velocity_rmse']['E'], results['velocity_rmse']['N'], 
                results['velocity_rmse']['2D'], results['velocity_rmse']['U'], 
                results['velocity_rmse']['3D']]
    ax.bar(['East', 'North', '2D', 'Up', '3D'], vel_rmse, 
           color=['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#d62728'])
    ax.set_ylabel('RMSE (m/s)', fontsize=11)
    ax.set_title('Velocity RMSE', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Orientation RMSE
    ax = axes[1, 0]
    orient_rmse = [results['orientation_rmse']['roll'], results['orientation_rmse']['pitch'], results['orientation_rmse']['yaw']]
    ax.bar(['Roll', 'Pitch', 'Yaw'], np.degrees(orient_rmse), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('RMSE (deg)', fontsize=11)
    ax.set_title('Orientation RMSE', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # RTE comparison
    ax = axes[1, 1]
    rte_rmse = [results['rte_1s']['rmse'], results['rte_5s']['rmse'], results['rte_10s']['rmse']]
    ax.bar(['1s', '5s', '10s'], rte_rmse, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('RTE RMSE (m)', fontsize=11)
    ax.set_title('Relative Trajectory Error (RTE)', fontsize=12)
    ax.set_xlabel('Segment Duration', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Error Statistics Summary - {dataset_name}', fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig, axes


def generate_all_plots(results, p_est, v_est, r_est, p_gt, v_gt, r_gt, 
                      sample_rate, output_dir, run_id, accel_flu=None, gyro_flu=None, lla0=None, gps_available=None):
    """
    Generate all visualization plots and save to output directory.
    
    Args:
        results: Dict with evaluation results
        p_est: Estimated positions (Nx3)
        v_est: Estimated velocities (Nx3)
        r_est: Estimated orientations (Nx3)
        p_gt: Ground truth positions (Nx3)
        v_gt: Ground truth velocities (Nx3)
        r_gt: Ground truth orientations (Nx3)
        sample_rate: Sampling rate in Hz
        output_dir: Output directory path
        run_id: Run identifier for filenames
        accel_flu: Optional accelerometer data (Nx3) for IMU visualization
        gyro_flu: Optional gyroscope data (Nx3) for gyro visualization
        lla0: Optional [lat, lon, alt] origin for map background in trajectory plots
        gps_available: Optional boolean array indicating GPS measurement availability at each timestep
    
    Returns:
        List of generated file paths
    """
    time = np.arange(len(p_est)) / sample_rate
    gnss_outage_info = results['gnss_outage']
    dataset_name = results['dataset']
    
    generated_files = []
    
    # 1. 2D Trajectory (with map background if lla0 provided)
    plot_trajectory_2d(p_est, p_gt, gnss_outage_info, dataset_name,
                      save_path=os.path.join(output_dir, f'{run_id}_trajectory.png'),
                      lla0=lla0, gps_available=gps_available)
    generated_files.append(f'{run_id}_trajectory.png')
    
    # 2. Position Errors
    plot_position_errors(time, results['instantaneous_errors'], gnss_outage_info,
                        save_path=os.path.join(output_dir, f'{run_id}_position_errors.png'))
    generated_files.append(f'{run_id}_position_errors.png')
    
    # 3. 2D Horizontal Error
    plot_2d_horizontal_error(time, results['instantaneous_errors']['2D'], 
                            results['ate']['rmse'], gnss_outage_info,
                            save_path=os.path.join(output_dir, f'{run_id}_2d_error.png'))
    generated_files.append(f'{run_id}_2d_error.png')
    
    # 4. Total Errors (2D and 3D)
    plot_total_errors(time, results['instantaneous_errors'], gnss_outage_info,
                     save_path=os.path.join(output_dir, f'{run_id}_total_errors.png'))
    generated_files.append(f'{run_id}_total_errors.png')
    
    # 5. Velocity Errors
    plot_velocity_errors(time, v_est, v_gt, gnss_outage_info,
                        save_path=os.path.join(output_dir, f'{run_id}_velocity_errors.png'))
    generated_files.append(f'{run_id}_velocity_errors.png')
    
    # 6. Total Velocity Errors (2D and 3D)
    plot_velocity_total_errors(time, v_est, v_gt, gnss_outage_info,
                              save_path=os.path.join(output_dir, f'{run_id}_velocity_total_errors.png'))
    generated_files.append(f'{run_id}_velocity_total_errors.png')
    
    # 7. Orientation Errors
    plot_orientation_errors(time, r_est, r_gt, gnss_outage_info,
                           save_path=os.path.join(output_dir, f'{run_id}_orientation_errors.png'))
    generated_files.append(f'{run_id}_orientation_errors.png')
    
    # 8. Total Orientation Errors (2D and 3D)
    plot_orientation_total_errors(time, r_est, r_gt, gnss_outage_info,
                                 save_path=os.path.join(output_dir, f'{run_id}_orientation_total_errors.png'))
    generated_files.append(f'{run_id}_orientation_total_errors.png')
    
    # 9. Error Statistics
    plot_error_statistics(results, dataset_name,
                         save_path=os.path.join(output_dir, f'{run_id}_error_statistics.png'))
    generated_files.append(f'{run_id}_error_statistics.png')
    
    # 10. 3D Trajectory
    plot_trajectory_3d(p_est, p_gt, gnss_outage_info, dataset_name,
                      save_path=os.path.join(output_dir, f'{run_id}_trajectory_3d.png'))
    generated_files.append(f'{run_id}_trajectory_3d.png')
    
    # 11. IMU Accelerometer Data (if provided)
    if accel_flu is not None:
        plot_imu_data(time, accel_flu, gnss_outage_info,
                     save_path=os.path.join(output_dir, f'{run_id}_imu_accel.png'))
        generated_files.append(f'{run_id}_imu_accel.png')
    
    # 12. Gyroscope Data (if provided)
    if gyro_flu is not None:
        plot_gyro_data(time, gyro_flu, gnss_outage_info,
                      save_path=os.path.join(output_dir, f'{run_id}_gyro.png'))
        generated_files.append(f'{run_id}_gyro.png')
    
    return generated_files


def plot_imu_data(time, accel_flu, gnss_outage_info, save_path=None):
    """
    Plot IMU accelerometer data over time (Forward, Left, Up components).
    
    Args:
        time: Time array in seconds
        accel_flu: Accelerometer data (Nx3) in FLU frame [forward, left, up] (m/s²)
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    accel_labels = ['Forward', 'Left', 'Up']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (ax, label, color) in enumerate(zip(axes, accel_labels, colors)):
        ax.plot(time, accel_flu[:, i], linewidth=1.0, color=color, label=f'{label} Acceleration')
        
        if A != 0 or B != 0:
            ax.axvspan(time[A], time[B-1], alpha=0.2, color='red', label='GNSS Outage')
        
        ax.set_ylabel(f'{label} (m/s²)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
    
    axes[0].set_title('IMU Accelerometer Data Over Time', fontsize=14)
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return


def plot_gyro_data(time, gyro_flu, gnss_outage_info, save_path=None):
    """
    Plot gyroscope data over time (Roll rate, Pitch rate, Yaw rate).
    
    Args:
        time: Time array in seconds
        gyro_flu: Gyroscope data (Nx3) in FLU frame [roll rate, pitch rate, yaw rate] (rad/s)
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    
    Returns:
        Figure and axes objects
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    gyro_labels = ['Roll Rate', 'Pitch Rate', 'Yaw Rate']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (ax, label, color) in enumerate(zip(axes, gyro_labels, colors)):
        # Convert rad/s to deg/s for better readability
        ax.plot(time, np.degrees(gyro_flu[:, i]), linewidth=1.0, color=color, label=label)
        
        if A != 0 or B != 0:
            ax.axvspan(time[A], time[B-1], alpha=0.2, color='red', label='GNSS Outage')
        
        ax.set_ylabel(f'{label} (deg/s)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
    
    axes[0].set_title('Gyroscope Data Over Time', fontsize=14)
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return


def show_interactive_plot(p_est, p_gt, gnss_outage_info, dataset_name, lla0=None, gps_available=None):
    """
    Display interactive trajectory plot for exploration with optional map background.
    
    Args:
        p_est: Estimated positions (Nx3)
        p_gt: Ground truth positions (Nx3)
        gnss_outage_info: Dict with outage information
        dataset_name: Name of dataset
        lla0: Optional [lat, lon, alt] origin for map background
        gps_available: Optional boolean array indicating GPS measurement availability at each timestep
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    # Check if we should add map background
    use_map = lla0 is not None and HAS_MAP_SUPPORT
    
    if use_map:
        # Convert ENU → lat/lon → Web Mercator (EPSG:3857, metres)
        lat_gt, lon_gt, _ = pm.enu2geodetic(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                                              lla0[0], lla0[1], lla0[2])
        lat_est, lon_est, _ = pm.enu2geodetic(p_est[:, 0], p_est[:, 1], p_est[:, 2],
                                                lla0[0], lla0[1], lla0[2])
        x_gt, y_gt = _latlon_to_mercator(lat_gt, lon_gt)
        x_est, y_est = _latlon_to_mercator(lat_est, lon_est)

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(x_gt, y_gt, 'k', linewidth=2, label='Ground Truth', zorder=3)
        ax.plot(x_est, y_est, 'b', linewidth=2, alpha=0.8, label='EKF Estimate', zorder=3)

        if A != 0 or B != 0:
            lat_est_o, lon_est_o, _ = pm.enu2geodetic(p_est[A:B, 0], p_est[A:B, 1], p_est[A:B, 2],
                                                       lla0[0], lla0[1], lla0[2])
            lat_gt_o, lon_gt_o, _ = pm.enu2geodetic(p_gt[A:B, 0], p_gt[A:B, 1], p_gt[A:B, 2],
                                                     lla0[0], lla0[1], lla0[2])
            x_est_o, y_est_o = _latlon_to_mercator(lat_est_o, lon_est_o)
            x_gt_o, y_gt_o = _latlon_to_mercator(lat_gt_o, lon_gt_o)
            ax.plot(x_est_o, y_est_o, 'r', linewidth=2.5,
                    label='EKF during GNSS outage', zorder=4)
            ax.plot(x_gt_o, y_gt_o, 'g', linewidth=2.5,
                    label='Ground Truth during GNSS outage', zorder=4)

        # Plot GPS measurement points if available
        if gps_available is not None:
            gps_indices = np.where(gps_available)[0]
            if len(gps_indices) > 0:
                x_gps, y_gps = _latlon_to_mercator(lat_gt[gps_indices], lon_gt[gps_indices])
                ax.scatter(x_gps, y_gps, c='orange', s=15, marker='o',
                           label=f'GPS Measurements ({len(gps_indices)})', zorder=3.5, alpha=0.6)

        # Add map background — tiles are already in EPSG:3857
        # zoom=15 gives crisp street-level tiles; lower (13-14) for longer routes
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik,
                            zoom=15, alpha=1.0)
        except Exception as e:
            print(f"Warning: Could not add map background: {e}")

        # Both axes are now in metres → equal aspect is exact
        ax.set_aspect('equal')

        # Show both Mercator metres and the equivalent lat/lon on each tick
        _apply_mercator_tick_labels(ax)

        ax.set_xlabel('Easting (Web Mercator)', fontsize=12)
        ax.set_ylabel('Northing (Web Mercator)', fontsize=12)
        ax.set_title(f'EKF Trajectory - {dataset_name}', fontsize=14)
    else:
        # Standard ENU plot without map
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(p_gt[:, 0], p_gt[:, 1], 'k', linewidth=1.5, label='Ground Truth')
        ax.plot(p_est[:, 0], p_est[:, 1], 'b', linewidth=1.5, alpha=0.7, label='EKF Estimate')

        if A != 0 or B != 0:
            ax.plot(p_est[A:B, 0], p_est[A:B, 1], 'r', linewidth=2, label='EKF during GNSS outage')
            ax.plot(p_gt[A:B, 0], p_gt[A:B, 1], 'g', linewidth=2, label='Ground Truth during GNSS outage')

        # Plot GPS measurement points if available
        if gps_available is not None:
            gps_indices = np.where(gps_available)[0]
            if len(gps_indices) > 0:
                ax.scatter(p_gt[gps_indices, 0], p_gt[gps_indices, 1], c='orange', s=15, marker='o',
                           label=f'GPS Measurements ({len(gps_indices)})', zorder=3.5, alpha=0.6)

        ax.set_xlabel('East (m)', fontsize=12)
        ax.set_ylabel('North (m)', fontsize=12)
        ax.set_title(f'EKF Trajectory Estimation - {dataset_name} (Interactive - Scroll to Zoom)', fontsize=14)
        ax.axis('equal')

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
