"""
Trajectory Evaluation Metrics for Navigation Systems

This module provides standard metrics for evaluating trajectory estimation accuracy:
- Absolute Translation Error (ATE) — no alignment, both trajectories in the same ENU frame
- Relative Trajectory Error (RTE) for drift analysis
- KITTI odometry metrics (t_rel, r_rel) with SE(3) relative pose error
- Traditional RMSE metrics
- Peak error analysis

Based on:
- Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012
- Geiger et al., "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite", CVPR 2012
"""

import numpy as np
import logging
from math import sqrt
from sklearn.metrics import mean_squared_error


def compute_ate(p_est, p_gt):
    """
    Compute Absolute Translation Error (ATE) — RMSE of position errors.

    Both trajectories must already be in the same coordinate frame (ENU with
    the same origin). No alignment is applied: for navigation systems positions
    are metric and share a common reference, so alignment would hide real errors.

    Args:
        p_est: Estimated trajectory (Nx3 numpy array) [E, N, U]
        p_gt: Ground truth trajectory (Nx3 numpy array) [E, N, U]

    Returns:
        dict: Statistics including rmse (total, E, N, 2D, U, 3D), mean, median, std, min, max, and raw errors
    """
    # Component-wise errors
    diff = p_gt - p_est
    error_E = np.abs(diff[:, 0])
    error_N = np.abs(diff[:, 1])
    error_U = np.abs(diff[:, 2])
    
    # Aggregate errors
    error_2D = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
    error_3D = np.linalg.norm(diff, axis=1)
    
    return {
        'rmse': np.sqrt(np.mean(error_3D**2)),
        'rmse_E': np.sqrt(np.mean(error_E**2)),
        'rmse_N': np.sqrt(np.mean(error_N**2)),
        'rmse_2D': np.sqrt(np.mean(error_2D**2)),
        'rmse_U': np.sqrt(np.mean(error_U**2)),
        'rmse_3D': np.sqrt(np.mean(error_3D**2)),
        'mean': np.mean(error_3D),
        'median': np.median(error_3D),
        'std': np.std(error_3D),
        'min': np.min(error_3D),
        'max': np.max(error_3D),
        'errors': error_3D
    }


def compute_rte(p_est, p_gt, delta=10):
    """
    Compute Relative Trajectory Error (RTE) over segments of length delta.
    
    RTE measures drift over short time windows - better for assessing local 
    consistency without being affected by global drift. Evaluates how well
    relative motion is preserved.
    
    Args:
        p_est: Estimated trajectory (Nx3 numpy array)
        p_gt: Ground truth trajectory (Nx3 numpy array)
        delta: Segment length in samples (e.g., delta=10 means 1 second at 10Hz)
    
    Returns:
        dict: Statistics including rmse, mean, median, std, min, max, and raw errors
    """
    n = len(p_est)
    trans_errors = []
    
    for i in range(n - delta):
        # Compute relative transformation in ground truth
        gt_rel = p_gt[i + delta] - p_gt[i]
        
        # Compute relative transformation in estimate
        est_rel = p_est[i + delta] - p_est[i]
        
        # Translation error
        trans_error = np.linalg.norm(gt_rel - est_rel)
        trans_errors.append(trans_error)
    
    trans_errors = np.array(trans_errors)
    
    return {
        'rmse': np.sqrt(np.mean(trans_errors**2)),
        'mean': np.mean(trans_errors),
        'median': np.median(trans_errors),
        'std': np.std(trans_errors),
        'min': np.min(trans_errors),
        'max': np.max(trans_errors),
        'errors': trans_errors
    }


def compute_traditional_rmse(est, gt):
    """
    Compute traditional RMSE without trajectory alignment.
    
    Args:
        est: Estimated values (N-dimensional array)
        gt: Ground truth values (N-dimensional array)
    
    Returns:
        float: Root mean squared error
    """
    return sqrt(mean_squared_error(gt, est))


def compute_angle_rmse(est_angles, gt_angles):
    """
    Compute RMSE for angular values with proper wrapping.
    
    Handles discontinuities at ±π by computing the smallest angular difference.
    
    Args:
        est_angles: Estimated angles in radians (N-dimensional array)
        gt_angles: Ground truth angles in radians (N-dimensional array)
    
    Returns:
        float: Root mean squared angular error
    """
    # Compute angle difference with wrapping
    diff = gt_angles - est_angles
    # Wrap to [-π, π] using atan2 trick
    diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return sqrt(np.mean(diff_wrapped**2))


def compute_instantaneous_errors(p_est, p_gt):
    """
    Compute instantaneous absolute errors at each timestep.
    
    Args:
        p_est: Estimated trajectory (Nx3 numpy array) [E, N, U]
        p_gt: Ground truth trajectory (Nx3 numpy array) [E, N, U]
    
    Returns:
        dict: Errors in each axis, 2D horizontal error, and 3D total error
    """
    error_E = np.abs(p_gt[:, 0] - p_est[:, 0])
    error_N = np.abs(p_gt[:, 1] - p_est[:, 1])
    error_U = np.abs(p_gt[:, 2] - p_est[:, 2])
    error_pos_2D = np.sqrt(error_E**2 + error_N**2)
    error_pos_3D = np.sqrt(error_E**2 + error_N**2 + error_U**2)
    
    return {
        'E': error_E,
        'N': error_N,
        'U': error_U,
        '2D': error_pos_2D,
        '3D': error_pos_3D
    }


def analyze_outage_segment(errors, start_idx, end_idx):
    """
    Analyze errors during a specific segment (e.g., GNSS outage).
    
    Args:
        errors: Array of errors over time
        start_idx: Start index of segment
        end_idx: End index of segment
    
    Returns:
        dict: Statistics for the segment
    """
    segment_errors = errors[start_idx:end_idx]

    if len(segment_errors) == 0:
        return {'mean': float('nan'), 'max': float('nan'),
                'min': float('nan'), 'std': float('nan')}

    return {
        'mean': np.mean(segment_errors),
        'max': np.max(segment_errors),
        'min': np.min(segment_errors),
        'std': np.std(segment_errors)
    }


def log_evaluation_results(logger, results, log_file):
    """
    Log comprehensive evaluation results.
    
    Args:
        logger: Python logger instance
        results: Dictionary containing all evaluation results
        log_file: Path to log file
    """
    logger.info('='*60)
    logger.info('EKF Error Analysis')
    logger.info('='*60)
    logger.info(f'Dataset: {results["dataset"]}')
    logger.info(f'GNSS Outage: {results["gnss_outage"]["start"]}s to {results["gnss_outage"]["end"]}s '
                f'(duration: {results["gnss_outage"]["duration"]}s)')
    logger.info(f'Total samples: {results["total_samples"]}')
    logger.info('')
    
    # Traditional RMSE
    logger.info('--- Traditional RMSE (no alignment) ---')
    logger.info('Position RMSE (meters):')
    logger.info(f'  East:  {results["position_rmse"]["E"]:.3f} m')
    logger.info(f'  North: {results["position_rmse"]["N"]:.3f} m')
    logger.info(f'  2D Horizontal: {results["position_rmse"]["2D"]:.3f} m')
    logger.info(f'  Up:    {results["position_rmse"]["U"]:.3f} m')
    logger.info(f'  3D Total: {results["position_rmse"]["3D"]:.3f} m')
    logger.info('')
    logger.info('Velocity RMSE (m/s):')
    logger.info(f'  East:  {results["velocity_rmse"]["E"]:.3f} m/s')
    logger.info(f'  North: {results["velocity_rmse"]["N"]:.3f} m/s')
    logger.info(f'  2D Horizontal: {results["velocity_rmse"]["2D"]:.3f} m/s')
    logger.info(f'  Up:    {results["velocity_rmse"]["U"]:.3f} m/s')
    logger.info(f'  3D Total: {results["velocity_rmse"]["3D"]:.3f} m/s')
    logger.info('')
    logger.info('Orientation RMSE (radians):')
    logger.info(f'  Roll:  {results["orientation_rmse"]["roll"]:.3f} rad')
    logger.info(f'  Pitch: {results["orientation_rmse"]["pitch"]:.3f} rad')
    logger.info(f'  Yaw:   {results["orientation_rmse"]["yaw"]:.3f} rad')
    logger.info('')
    
    # ATE
    logger.info('--- Absolute Translation Error (ATE) - no alignment (same ENU frame) ---')
    ate = results["ate"]
    logger.info('ATE RMSE (meters):')
    logger.info(f'  East:  {ate["rmse_E"]:.3f} m')
    logger.info(f'  North: {ate["rmse_N"]:.3f} m')
    logger.info(f'  2D Horizontal: {ate["rmse_2D"]:.3f} m')
    logger.info(f'  Up:    {ate["rmse_U"]:.3f} m')
    logger.info(f'  3D Total: {ate["rmse_3D"]:.3f} m')
    logger.info('')
    logger.info('ATE Statistics (3D):')
    logger.info(f'  Mean:   {ate["mean"]:.3f} m')
    logger.info(f'  Median: {ate["median"]:.3f} m')
    logger.info(f'  Std:    {ate["std"]:.3f} m')
    logger.info(f'  Min:    {ate["min"]:.3f} m')
    logger.info(f'  Max:    {ate["max"]:.3f} m')
    logger.info('')
    
    # RTE
    logger.info('--- Relative Trajectory Error (RTE) - local consistency ---')
    logger.info('RTE over 1s segments (short-term accuracy):')
    rte_1s = results["rte_1s"]
    logger.info(f'  RMSE:   {rte_1s["rmse"]:.3f} m')
    logger.info(f'  Mean:   {rte_1s["mean"]:.3f} m')
    logger.info(f'  Median: {rte_1s["median"]:.3f} m')
    logger.info('')
    logger.info('RTE over 5s segments (medium-term drift):')
    rte_5s = results["rte_5s"]
    logger.info(f'  RMSE:   {rte_5s["rmse"]:.3f} m')
    logger.info(f'  Mean:   {rte_5s["mean"]:.3f} m')
    logger.info(f'  Median: {rte_5s["median"]:.3f} m')
    logger.info('')
    logger.info('RTE over 10s segments (long-term drift):')
    rte_10s = results["rte_10s"]
    logger.info(f'  RMSE:   {rte_10s["rmse"]:.3f} m')
    logger.info(f'  Mean:   {rte_10s["mean"]:.3f} m')
    logger.info(f'  Median: {rte_10s["median"]:.3f} m')
    logger.info('')
    
    # Peak errors
    logger.info('Peak Errors:')
    logger.info(f'  Max 2D position error: {results["peak_errors"]["max_2d"]:.3f} m')
    logger.info(f'  Max yaw error: {results["peak_errors"]["max_yaw"]:.3f} rad '
                f'({np.degrees(results["peak_errors"]["max_yaw"]):.2f}°)')
    
    # Outage segment
    if results["outage_analysis"] is not None:
        logger.info('')
        outage = results["outage_analysis"]
        logger.info(f'GNSS Outage Segment ({outage["start_idx"]} to {outage["end_idx"]}):')
        logger.info(f'  Mean 2D error: {outage["mean"]:.3f} m')
        logger.info(f'  Max 2D error: {outage["max"]:.3f} m')
    
    logger.info('='*60)
    logger.info(f'Log saved to: {log_file}')


def compute_kitti_metrics(p_est, r_est, p_gt, r_gt, lengths=None):
    """
    Compute KITTI odometry benchmark metrics t_rel and r_rel.

    Faithfully reimplements the official KITTI odometry evaluation protocol:
      Geiger et al., "Are we ready for Autonomous Driving? The KITTI Vision
      Benchmark Suite", CVPR 2012.
      Official C++ devkit: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

    Reference Python port (cross-checked):
      Huangying Zhan, kitti-odom-eval,
      https://github.com/Huangying-Zhan/kitti-odom-eval

    For each start frame i and subsequence length L the end frame j is chosen
    such that the GT path length from i to j equals L.  The SE(3) relative
    poses are computed in the body frame of i (T_i^{-1} T_j), and the pose
    error is the composition:

      T_err = (T_ij_gt)^{-1}  *  T_ij_est

    Translational error  = ||translation(T_err)||
                         = ||R_i_est^T (p_j_est-p_i_est)  -  R_i_gt^T (p_j_gt-p_i_gt)||

    Rotational error     = arccos( (trace(R_err) - 1) / 2 )

      t_rel  — mean relative translational error, **percent** of traveled distance.
      r_rel  — mean relative rotational error, **degrees per kilometer**.

    Args:
        p_est   : Estimated positions  (N×3) in ENU [m]
        r_est   : Estimated orientations (N×3) as [roll, pitch, yaw] [rad]
        p_gt    : Ground-truth positions (N×3) in ENU [m]
        r_gt    : Ground-truth orientations (N×3) as [roll, pitch, yaw] [rad]
        lengths : Subsequence lengths in meters. If None, lengths are set
                  automatically to [100, 200, …, N*100] where N*100 is the
                  largest multiple of 100 m that fits within the GT trajectory,
                  matching the official KITTI step size while adapting to
                  sequences shorter or longer than 800 m.

    Returns:
        dict with keys:
          't_rel'           — scalar [%]
          'r_rel'           — scalar [deg/km]
          't_rel_by_length' — {L: mean_t_rel} per length
          'r_rel_by_length' — {L: mean_r_rel} per length
          'n_segments'      — total number of evaluated (i, L) pairs
          'total_dist_m'    — total GT trajectory length [m]
          'lengths_used'    — list of subsequence lengths actually evaluated
    """
    N = len(p_gt)

    # Build rotation matrices from ZYX Euler angles
    def _euler_to_rot(rpy):
        """[roll, pitch, yaw] → 3×3 rotation matrix (body → nav)."""
        cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
        cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
        cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
        return np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,    cp*sr,             cp*cr            ],
        ])

    # Pre-build all rotation matrices (avoids repeated trig inside inner loop)
    Rot_gt  = np.array([_euler_to_rot(r_gt[i])  for i in range(N)])
    Rot_est = np.array([_euler_to_rot(r_est[i]) for i in range(N)])

    # Cumulative distance along GT trajectory
    step_dist = np.linalg.norm(np.diff(p_gt, axis=0), axis=1)  # (N-1,)
    cum_dist  = np.concatenate([[0.0], np.cumsum(step_dist)])   # (N,)
    total_dist = cum_dist[-1]

    if lengths is None:
        max_len = int(total_dist // 100) * 100
        lengths = list(range(100, max_len + 1, 100)) if max_len >= 100 else [100]

    t_errors_all = []
    r_errors_all = []
    t_by_length  = {L: [] for L in lengths}
    r_by_length  = {L: [] for L in lengths}

    for i in range(N):
        for L in lengths:
            target = cum_dist[i] + L
            if target > total_dist:
                continue
            # First j with cum_dist[j] >= cum_dist[i] + L
            j = int(np.searchsorted(cum_dist, target))
            if j >= N:
                continue

            # SE(3) relative translations in the body frame of i (official KITTI method):
            #   t_delta = R_i^T * (p_j - p_i)
            # Error = ||t_delta_est - t_delta_gt||  (rotation-invariant norm)
            t_delta_gt  = Rot_gt[i].T  @ (p_gt[j]  - p_gt[i])
            t_delta_est = Rot_est[i].T @ (p_est[j] - p_est[i])
            t_err = np.linalg.norm(t_delta_est - t_delta_gt)

            # Rotational error: angle of relative-pose error rotation
            # R_ij_gt = R_i_gt^T @ R_j_gt,  R_ij_est = R_i_est^T @ R_j_est
            # R_err = R_ij_gt^T @ R_ij_est
            R_ij_gt  = Rot_gt[i].T  @ Rot_gt[j]
            R_ij_est = Rot_est[i].T @ Rot_est[j]
            R_err    = R_ij_gt.T @ R_ij_est
            trace_val = float(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
            r_err = np.arccos(trace_val)   # [rad]

            t_pct         = t_err / L * 100.0                       # [%]
            r_deg_per_km  = r_err / L * (180.0 / np.pi) * 1000.0   # [deg/km]

            t_errors_all.append(t_pct)
            r_errors_all.append(r_deg_per_km)
            t_by_length[L].append(t_pct)
            r_by_length[L].append(r_deg_per_km)

    if not t_errors_all:
        return {
            't_rel': float('nan'), 'r_rel': float('nan'),
            't_rel_by_length': {L: float('nan') for L in lengths},
            'r_rel_by_length': {L: float('nan') for L in lengths},
            'n_segments': 0,
            'total_dist_m': float(total_dist),
            'lengths_used': lengths,
        }

    return {
        't_rel':           float(np.mean(t_errors_all)),
        'r_rel':           float(np.mean(r_errors_all)),
        't_rel_by_length': {L: float(np.mean(v)) if v else float('nan')
                            for L, v in t_by_length.items()},
        'r_rel_by_length': {L: float(np.mean(v)) if v else float('nan')
                            for L, v in r_by_length.items()},
        'n_segments':      len(t_errors_all),
        'total_dist_m':    float(total_dist),
        'lengths_used':    lengths,
    }


def evaluate_navigation_performance(p_est, v_est, r_est, p_gt, v_gt, r_gt,
                                    dataset_name, gnss_outage_info,
                                    sample_rate=10):
    """
    Complete evaluation of navigation system performance.
    
    Args:
        p_est: Estimated positions (Nx3) [E, N, U]
        v_est: Estimated velocities (Nx3) [E, N, U]
        r_est: Estimated orientations (Nx3) [roll, pitch, yaw]
        p_gt: Ground truth positions (Nx3) [E, N, U]
        v_gt: Ground truth velocities (Nx3) [E, N, U]
        r_gt: Ground truth orientations (Nx3) [roll, pitch, yaw]
        dataset_name: Name of the dataset
        gnss_outage_info: Dict with 'start', 'end', 'duration', 'start_idx', 'end_idx'
        sample_rate: Sampling rate in Hz (default: 10)
    
    Returns:
        dict: Comprehensive evaluation results
    """
    # Traditional RMSE
    position_rmse = {
        'E': compute_traditional_rmse(p_est[:, 0], p_gt[:, 0]),
        'N': compute_traditional_rmse(p_est[:, 1], p_gt[:, 1]),
        '2D': sqrt(mean_squared_error(p_gt[:, :2].ravel(), p_est[:, :2].ravel())),  # Horizontal 2D RMSE
        'U': compute_traditional_rmse(p_est[:, 2], p_gt[:, 2]),
        '3D': sqrt(mean_squared_error(p_gt.ravel(), p_est.ravel()))  # Total 3D RMSE
    }
    
    velocity_rmse = {
        'E': compute_traditional_rmse(v_est[:, 0], v_gt[:, 0]),
        'N': compute_traditional_rmse(v_est[:, 1], v_gt[:, 1]),
        '2D': sqrt(mean_squared_error(v_gt[:, :2].ravel(), v_est[:, :2].ravel())),  # Horizontal 2D RMSE
        'U': compute_traditional_rmse(v_est[:, 2], v_gt[:, 2]),
        '3D': sqrt(mean_squared_error(v_gt.ravel(), v_est.ravel()))  # Total 3D RMSE
    }
    
    orientation_rmse = {
        'roll': compute_traditional_rmse(r_est[:, 0], r_gt[:, 0]),
        'pitch': compute_traditional_rmse(r_est[:, 1], r_gt[:, 1]),
        'yaw': compute_angle_rmse(r_est[:, 2], r_gt[:, 2])  # Use angle-aware RMSE
    }
    
    # Advanced trajectory metrics
    # Both trajectories are in same ENU frame - no alignment needed
    # Alignment would introduce artificial scaling errors
    ate = compute_ate(p_est, p_gt)
    rte_1s = compute_rte(p_est, p_gt, delta=int(1*sample_rate))   # 1 second
    rte_5s = compute_rte(p_est, p_gt, delta=int(5*sample_rate))   # 5 seconds
    rte_10s = compute_rte(p_est, p_gt, delta=int(10*sample_rate)) # 10 seconds
    
    # Instantaneous errors
    inst_errors = compute_instantaneous_errors(p_est, p_gt)
    # Yaw error with proper angle wrapping (handle ±π transitions)
    yaw_diff = r_gt[:, 2] - r_est[:, 2]
    # Wrap to [-π, π] range
    yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
    error_yaw = np.abs(yaw_diff)
    
    # Peak errors
    peak_errors = {
        'max_2d': np.max(inst_errors['2D']),
        'max_yaw': np.max(error_yaw)
    }
    
    # Outage segment analysis
    outage_analysis = None
    if gnss_outage_info['start_idx'] != 0 or gnss_outage_info['end_idx'] != 0:
        outage_stats = analyze_outage_segment(
            inst_errors['2D'], 
            gnss_outage_info['start_idx'], 
            gnss_outage_info['end_idx']
        )
        outage_analysis = {
            'start_idx': gnss_outage_info['start_idx'],
            'end_idx': gnss_outage_info['end_idx'],
            'mean': outage_stats['mean'],
            'max': outage_stats['max']
        }
    
    return {
        'dataset': dataset_name,
        'total_samples': len(p_est),
        'gnss_outage': gnss_outage_info,
        'position_rmse': position_rmse,
        'velocity_rmse': velocity_rmse,
        'orientation_rmse': orientation_rmse,
        'ate': ate,
        'rte_1s': rte_1s,
        'rte_5s': rte_5s,
        'rte_10s': rte_10s,
        'peak_errors': peak_errors,
        'outage_analysis': outage_analysis,
        'instantaneous_errors': inst_errors,
        'yaw_errors': error_yaw
    }
