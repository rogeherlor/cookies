"""
EKF State Visualization Module

Visualize internal EKF state including:
- Estimated IMU biases over time
- Covariance evolution (uncertainty)
- Bias estimation quality
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_bias_estimates(time, bias_acc, bias_gyr, gnss_outage_info, save_path=None):
    """
    Plot estimated IMU biases over time.
    
    Args:
        time: Time array in seconds
        bias_acc: Accelerometer bias estimates (Nx3) [m/s²]
        bias_gyr: Gyroscope bias estimates (Nx3) [rad/s]
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Accelerometer biases
    ax = axes[0]
    ax.plot(time, bias_acc[:, 0], linewidth=1.5, label='Forward bias', color='#1f77b4')
    ax.plot(time, bias_acc[:, 1], linewidth=1.5, label='Left bias', color='#ff7f0e')
    ax.plot(time, bias_acc[:, 2], linewidth=1.5, label='Up bias', color='#2ca02c')
    
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    
    ax.set_ylabel('Accel Bias (m/s²)', fontsize=11)
    ax.set_title('Estimated IMU Biases Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Gyroscope biases
    ax = axes[1]
    ax.plot(time, np.degrees(bias_gyr[:, 0]), linewidth=1.5, label='Roll rate bias', color='#1f77b4')
    ax.plot(time, np.degrees(bias_gyr[:, 1]), linewidth=1.5, label='Pitch rate bias', color='#ff7f0e')
    ax.plot(time, np.degrees(bias_gyr[:, 2]), linewidth=1.5, label='Yaw rate bias', color='#2ca02c')
    
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    
    ax.set_ylabel('Gyro Bias (deg/s)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return


def plot_uncertainty_evolution(time, std_pos, std_vel, std_orient, gnss_outage_info, save_path=None):
    """
    Plot evolution of state uncertainty (standard deviation from covariance).
    
    Args:
        time: Time array in seconds
        std_pos: Position std (Nx3) [m]
        std_vel: Velocity std (Nx3) [m/s]
        std_orient: Orientation std (Nx3) [rad]
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Position uncertainty (2D horizontal)
    ax = axes[0]
    std_2d = np.sqrt(std_pos[:, 0]**2 + std_pos[:, 1]**2)
    ax.plot(time, std_2d, linewidth=1.5, color='#9467bd', label='2D Horizontal Std')
    ax.plot(time, std_pos[:, 2], linewidth=1.5, color='#2ca02c', label='Vertical Std', alpha=0.7)
    
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    
    ax.set_ylabel('Position Std (m)', fontsize=11)
    ax.set_title('State Uncertainty Evolution (1-sigma)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    
    # Velocity uncertainty
    ax = axes[1]
    vel_std_2d = np.sqrt(std_vel[:, 0]**2 + std_vel[:, 1]**2)
    ax.plot(time, vel_std_2d, linewidth=1.5, color='#9467bd', label='2D Horizontal Std')
    ax.plot(time, std_vel[:, 2], linewidth=1.5, color='#2ca02c', label='Vertical Std', alpha=0.7)
    
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    
    ax.set_ylabel('Velocity Std (m/s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    
    # Orientation uncertainty
    ax = axes[2]
    ax.plot(time, np.degrees(std_orient[:, 0]), linewidth=1.5, label='Roll Std', color='#1f77b4')
    ax.plot(time, np.degrees(std_orient[:, 1]), linewidth=1.5, label='Pitch Std', color='#ff7f0e')
    ax.plot(time, np.degrees(std_orient[:, 2]), linewidth=1.5, label='Yaw Std', color='#d62728')
    
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    
    ax.set_ylabel('Orientation Std (deg)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return


def plot_bias_uncertainty(time, bias_acc, bias_gyr, std_bias_acc, std_bias_gyr, gnss_outage_info, save_path=None):
    """
    Plot bias estimates with uncertainty bounds.
    
    Args:
        time: Time array in seconds
        bias_acc: Accelerometer bias estimates (Nx3) [m/s²]
        bias_gyr: Gyroscope bias estimates (Nx3) [rad/s]
        std_bias_acc: Accelerometer bias std (Nx3) [m/s²]
        std_bias_gyr: Gyroscope bias std (Nx3) [rad/s]
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Accelerometer biases with uncertainty
    ax = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['Forward', 'Left', 'Up']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(time, bias_acc[:, i], linewidth=1.5, label=f'{label} bias', color=color)
        ax.fill_between(time, 
                        bias_acc[:, i] - std_bias_acc[:, i], 
                        bias_acc[:, i] + std_bias_acc[:, i],
                        alpha=0.2, color=color, label=f'{label} ±1σ')
    
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    
    ax.set_ylabel('Accel Bias (m/s²)', fontsize=11)
    ax.set_title('Bias Estimates with Uncertainty (±1σ)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    
    # Gyroscope biases with uncertainty
    ax = axes[1]
    for i, (color, label) in enumerate(zip(colors, ['Roll rate', 'Pitch rate', 'Yaw rate'])):
        ax.plot(time, np.degrees(bias_gyr[:, i]), linewidth=1.5, label=f'{label} bias', color=color)
        ax.fill_between(time, 
                        np.degrees(bias_gyr[:, i] - std_bias_gyr[:, i]), 
                        np.degrees(bias_gyr[:, i] + std_bias_gyr[:, i]),
                        alpha=0.2, color=color, label=f'{label} ±1σ')
    
    if A != 0 or B != 0:
        ax.axvspan(time[A], time[B], alpha=0.2, color='red', label='GNSS Outage')
    
    ax.set_ylabel('Gyro Bias (deg/s)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return


def plot_filter_consistency(time, errors_pos, std_pos, gnss_outage_info, save_path=None):
    """
    Plot filter consistency: actual errors vs predicted uncertainty.
    A consistent filter should have errors within ±2σ bounds ~95% of the time.
    
    Args:
        time: Time array in seconds
        errors_pos: Actual position errors (Nx3) [m]
        std_pos: Predicted position std (Nx3) [m]
        gnss_outage_info: Dict with outage information
        save_path: Optional path to save figure
    """
    A = gnss_outage_info['start_idx']
    B = gnss_outage_info['end_idx']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    labels = ['East', 'North', 'Up']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        # Plot actual error
        ax.plot(time, errors_pos[:, idx], linewidth=1.5, label='Actual Error', color=color, alpha=0.8)
        
        # Plot ±2σ bounds (95% confidence)
        ax.plot(time, 2*std_pos[:, idx], '--', linewidth=1.5, label='+2σ bound', color='red', alpha=0.6)
        ax.plot(time, -2*std_pos[:, idx], '--', linewidth=1.5, label='-2σ bound', color='red', alpha=0.6)
        ax.fill_between(time, -2*std_pos[:, idx], 2*std_pos[:, idx], 
                        alpha=0.1, color='red', label='95% confidence')
        
        if A != 0 or B != 0:
            ax.axvspan(time[A], time[B], alpha=0.2, color='gray', label='GNSS Outage')
        
        ax.set_ylabel(f'{label} Error (m)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Calculate percentage of errors within bounds
        within_bounds = np.abs(errors_pos[:, idx]) <= 2*std_pos[:, idx]
        pct_within = 100 * np.sum(within_bounds) / len(errors_pos)
        ax.text(0.02, 0.95, f'{pct_within:.1f}% within ±2σ', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[0].set_title('Filter Consistency Check (Actual Error vs Predicted Uncertainty)', fontsize=14)
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return


def plot_convergence_dashboard(time, p_est, p_gt, v_est, v_gt, r_est, r_gt,
                                std_pos, std_vel, std_orient,
                                bias_acc, bias_gyr, std_bias_acc, std_bias_gyr,
                                gnss_outage_info, gps_available=None,
                                save_path=None):
    """
    Comprehensive convergence dashboard: shows at a glance whether the filter
    converged, diverged, or became overconfident.

    Layout (6 rows x 2 cols):
      Row 0: Orientation estimates vs truth (roll/pitch | yaw)
      Row 1: Orientation error with +/-2sigma  (roll/pitch | yaw)
      Row 2: Position error with +/-2sigma (East/North | Up)
      Row 3: Velocity error with +/-2sigma (East/North | Up)
      Row 4: Accel bias estimates +/-1sigma (fwd/left | up)
      Row 5: Gyro bias estimates +/-1sigma (roll/pitch rate | yaw rate)
    """
    A = gnss_outage_info.get('start_idx', 0)
    B = gnss_outage_info.get('end_idx', 0)

    fig, axes = plt.subplots(6, 2, figsize=(18, 22), sharex=True)

    def shade_outage(ax):
        if A != 0 or B != 0:
            ax.axvspan(time[A], time[min(B, len(time)-1)],
                       alpha=0.15, color='red', zorder=0)

    def shade_gps(ax):
        """Light green ticks at every GPS update."""
        if gps_available is not None:
            for k in range(len(gps_available)):
                if gps_available[k]:
                    ax.axvline(time[k], color='green', alpha=0.06, lw=0.5)

    # ── colours ──────────────────────────────────────────────────────────
    c_rp = ['#1f77b4', '#ff7f0e']         # roll, pitch
    c_y  = '#d62728'                       # yaw
    c_en = ['#1f77b4', '#ff7f0e']         # east, north
    c_u  = '#2ca02c'                       # up
    c_flu = ['#1f77b4', '#ff7f0e', '#2ca02c']  # fwd, left, up

    # ═══════════════════════ ROW 0: Orientation est vs truth ═════════════
    # Roll / Pitch
    ax = axes[0, 0]
    shade_outage(ax); shade_gps(ax)
    for j, (c, lbl) in enumerate(zip(c_rp, ['Roll', 'Pitch'])):
        ax.plot(time, np.degrees(r_est[:, j]), color=c, lw=1.2, label=f'{lbl} est')
        ax.plot(time, np.degrees(r_gt[:, j]), '--', color=c, lw=1.0, alpha=0.6,
                label=f'{lbl} truth')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Roll / Pitch Estimates vs Truth', fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    # Yaw
    ax = axes[0, 1]
    shade_outage(ax); shade_gps(ax)
    ax.plot(time, np.degrees(r_est[:, 2]), color=c_y, lw=1.2, label='Yaw est')
    ax.plot(time, np.degrees(r_gt[:, 2]), '--', color=c_y, lw=1.0, alpha=0.6,
            label='Yaw truth')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Yaw Estimate vs Truth', fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ═══════════════════════ ROW 1: Orientation error +/- 2sigma ═════════
    orient_err = np.degrees(r_est - r_gt)
    orient_std_deg = np.degrees(std_orient)

    ax = axes[1, 0]
    shade_outage(ax); shade_gps(ax)
    for j, (c, lbl) in enumerate(zip(c_rp, ['Roll', 'Pitch'])):
        ax.plot(time, orient_err[:, j], color=c, lw=1.0, label=f'{lbl} err')
        ax.fill_between(time, -2*orient_std_deg[:, j], 2*orient_std_deg[:, j],
                        color=c, alpha=0.12)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel('Error (deg)')
    ax.set_title('Roll / Pitch Error  (shaded = ±2σ)', fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    shade_outage(ax); shade_gps(ax)
    ax.plot(time, orient_err[:, 2], color=c_y, lw=1.0, label='Yaw err')
    ax.fill_between(time, -2*orient_std_deg[:, 2], 2*orient_std_deg[:, 2],
                    color=c_y, alpha=0.12)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel('Error (deg)')
    ax.set_title('Yaw Error  (shaded = ±2σ)', fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ═══════════════════════ ROW 2: Position error +/- 2sigma ════════════
    pos_err = p_est - p_gt

    ax = axes[2, 0]
    shade_outage(ax); shade_gps(ax)
    for j, (c, lbl) in enumerate(zip(c_en, ['East', 'North'])):
        ax.plot(time, pos_err[:, j], color=c, lw=1.0, label=f'{lbl} err')
        ax.fill_between(time, -2*std_pos[:, j], 2*std_pos[:, j],
                        color=c, alpha=0.12)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel('Error (m)')
    ax.set_title('Horizontal Position Error  (shaded = ±2σ)', fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    ax = axes[2, 1]
    shade_outage(ax); shade_gps(ax)
    ax.plot(time, pos_err[:, 2], color=c_u, lw=1.0, label='Up err')
    ax.fill_between(time, -2*std_pos[:, 2], 2*std_pos[:, 2],
                    color=c_u, alpha=0.12)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel('Error (m)')
    ax.set_title('Vertical Position Error  (shaded = ±2σ)', fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ═══════════════════════ ROW 3: Velocity error +/- 2sigma ════════════
    vel_err = v_est - v_gt

    ax = axes[3, 0]
    shade_outage(ax); shade_gps(ax)
    for j, (c, lbl) in enumerate(zip(c_en, ['East', 'North'])):
        ax.plot(time, vel_err[:, j], color=c, lw=1.0, label=f'{lbl} err')
        ax.fill_between(time, -2*std_vel[:, j], 2*std_vel[:, j],
                        color=c, alpha=0.12)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel('Error (m/s)')
    ax.set_title('Horizontal Velocity Error  (shaded = ±2σ)', fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    ax = axes[3, 1]
    shade_outage(ax); shade_gps(ax)
    ax.plot(time, vel_err[:, 2], color=c_u, lw=1.0, label='Up err')
    ax.fill_between(time, -2*std_vel[:, 2], 2*std_vel[:, 2],
                    color=c_u, alpha=0.12)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel('Error (m/s)')
    ax.set_title('Vertical Velocity Error  (shaded = ±2σ)', fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ═══════════════════════ ROW 4: Accel bias +/- 1sigma ════════════════
    ax = axes[4, 0]
    shade_outage(ax); shade_gps(ax)
    for j, (c, lbl) in enumerate(zip(c_flu[:2], ['Fwd', 'Left'])):
        ax.plot(time, bias_acc[:, j], color=c, lw=1.0, label=f'{lbl} bias')
        ax.fill_between(time,
                        bias_acc[:, j] - std_bias_acc[:, j],
                        bias_acc[:, j] + std_bias_acc[:, j],
                        color=c, alpha=0.15)
    ax.set_ylabel('Bias (m/s²)')
    ax.set_title('Accel Bias Fwd/Left  (shaded = ±1σ)', fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    ax = axes[4, 1]
    shade_outage(ax); shade_gps(ax)
    ax.plot(time, bias_acc[:, 2], color=c_flu[2], lw=1.0, label='Up bias')
    ax.fill_between(time,
                    bias_acc[:, 2] - std_bias_acc[:, 2],
                    bias_acc[:, 2] + std_bias_acc[:, 2],
                    color=c_flu[2], alpha=0.15)
    ax.set_ylabel('Bias (m/s²)')
    ax.set_title('Accel Bias Up  (shaded = ±1σ)', fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ═══════════════════════ ROW 5: Gyro bias +/- 1sigma ═════════════════
    ax = axes[5, 0]
    shade_outage(ax); shade_gps(ax)
    for j, (c, lbl) in enumerate(zip(c_rp, ['Roll rate', 'Pitch rate'])):
        ax.plot(time, np.degrees(bias_gyr[:, j]), color=c, lw=1.0,
                label=f'{lbl} bias')
        ax.fill_between(time,
                        np.degrees(bias_gyr[:, j] - std_bias_gyr[:, j]),
                        np.degrees(bias_gyr[:, j] + std_bias_gyr[:, j]),
                        color=c, alpha=0.15)
    ax.set_ylabel('Bias (deg/s)')
    ax.set_title('Gyro Bias Roll/Pitch Rate  (shaded = ±1σ)', fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('Time (s)', fontsize=10)

    ax = axes[5, 1]
    shade_outage(ax); shade_gps(ax)
    ax.plot(time, np.degrees(bias_gyr[:, 2]), color=c_y, lw=1.0,
            label='Yaw rate bias')
    ax.fill_between(time,
                    np.degrees(bias_gyr[:, 2] - std_bias_gyr[:, 2]),
                    np.degrees(bias_gyr[:, 2] + std_bias_gyr[:, 2]),
                    color=c_y, alpha=0.15)
    ax.set_ylabel('Bias (deg/s)')
    ax.set_title('Gyro Bias Yaw Rate  (shaded = ±1σ)', fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('Time (s)', fontsize=10)

    # ── Overall title ────────────────────────────────────────────────────
    fig.suptitle('EKF Convergence Dashboard', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return
