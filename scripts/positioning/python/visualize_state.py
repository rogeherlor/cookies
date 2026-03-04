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
