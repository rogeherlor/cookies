# -*- coding: utf-8 -*-
"""
Genetic Algorithm Optimization for EKF Parameters using scipy.optimize

Optimizes:
- Process noise Q (position, velocity, orientation, biases)
- Measurement noise R
- Initial covariance P
- Gauss-Markov coefficients (beta_acc, beta_gyr)

Fitness metric: ATE RMSE during GPS outage (standard SLAM metric)
"""
import json
import logging
import numpy as np
from math import sin, cos, tan, radians, sqrt
import pymap3d as pm
from pathlib import Path
from datetime import datetime
from scipy.optimize import differential_evolution
import data_loader
import ekf_core
import eskf_core


################### OPTIMIZATION PARAMETERS ###########################

# These are imported from ekf_config.py so the optimizer always runs on the same
# dataset / outage / rotation mode / core that ekf.py is currently configured for.
import ekf_config
NAV_DATA        = ekf_config.NAV_DATA
OUTAGE_START    = ekf_config.OUTAGE_START
OUTAGE_DURATION = ekf_config.OUTAGE_DURATION
USE_3D_ROTATION = ekf_config.USE_3D_ROTATION
CORE_NAME       = ekf_config.CORE_NAME

CORE_MODULES = {
    "ekf": ekf_core,
    "eskf": eskf_core,
}

# Parameter bounds for optimization (log scale)
# Order:
#   Qpos, Qvel, QorientXY, QorientZ, Qacc, QgyrXY, QgyrZ,
#   Rpos, Rvel,
#   beta_acc, beta_gyr,
#   P_pos, P_vel, P_orient, P_acc, P_gyr
BOUNDS = [
    (-2, 2),     # log10(Qpos): 0.01 to 100
    (-2, 2),     # log10(Qvel): 0.01 to 100
    (-5, -1),    # log10(QorientXY): 0.00001 to 0.1
    (-2, 1),     # log10(QorientZ): 0.01 to 10
    (-3, 0),     # log10(Qacc): 0.001 to 1
    (-6, -2),    # log10(QgyrXY): 0.000001 to 0.01
    (-3, 0),     # log10(QgyrZ): 0.001 to 1
    (-1, 2),     # log10(Rpos): 0.1 to 100m
    (-2, 1),     # log10(Rvel): 0.01 to 10 (m/s)^2
    (-8, -5),    # log10(abs(beta_acc)): very small negative
    (-2, 1),     # log10(abs(beta_gyr)): small negative
    (-1, 1.5),   # log10(P_pos_std): 0.1 to 30m
    (-1, 0.5),   # log10(P_vel_std): 0.1 to 3 m/s
    (-2, -0.5),  # log10(P_orient_std): 0.01 to 0.3 rad
    (-3, -1),    # log10(P_acc_std): 0.001 to 0.1 m/s²
    (-4, -2),    # log10(P_gyr_std): 0.0001 to 0.01 rad/s
]

################### PARAMETER DECODING ###########################

def decode_params(x: np.ndarray) -> dict:
    """Convert log-scale parameter vector to actual EKF parameters."""
    return {
        'Qpos': 10**x[0],
        'Qvel': 10**x[1],
        'QorientXY': 10**x[2],
        'QorientZ': 10**x[3],
        'Qacc': 10**x[4],
        'QgyrXY': 10**x[5],
        'QgyrZ': 10**x[6],
        'Rpos': 10**x[7],
        'Rvel': 10**x[8],
        'beta_acc': -10**x[9],   # Negative
        'beta_gyr': -10**x[10],  # Negative
        'P_pos_std': 10**x[11],
        'P_vel_std': 10**x[12],
        'P_orient_std': 10**x[13],
        'P_acc_std': 10**x[14],
        'P_gyr_std': 10**x[15],
    }


################### EKF RUNNER & FITNESS FUNCTION ###########################

def fitness_function(x: np.ndarray, nav_data, t1: float, d: float) -> float:
    """
    Comprehensive fitness function for EKF parameter optimization.

    Combines multiple state-of-the-art navigation metrics into a single
    scalar cost, ensuring the optimizer produces a physically consistent
    filter — not just one that minimises 2D position during the outage.

    Components (weighted sum):
      1. Position ATE RMSE during outage (2D + vertical)
      2. Position RMSE during GPS-aided phase (filter should track well)
      3. Orientation RMSE vs ground truth (roll, pitch, yaw)
      4. Velocity RMSE vs ground truth
      5. ANEES penalty — Average Normalised Estimation Error Squared.
         For a consistent filter ANEES ≈ 1.  If ANEES >> 1 the filter is
         overconfident; if ANEES << 1 it is too conservative.  Both are
         penalised so the optimizer cannot cheat with absurd Q values.

    Lower is better.
    """
    # Initialize counter and logger on first call
    if not hasattr(fitness_function, 'eval_count'):
        fitness_function.eval_count = 0
        fitness_function.logger = None
        fitness_function.best_fitness = float('inf')
    
    fitness_function.eval_count += 1
    
    params = decode_params(x)
    
    # Select EKF core based on shared configuration
    core_module = CORE_MODULES.get(CORE_NAME, ekf_core)
    
    # Run EKF with these parameters
    try:
        outage_config = {'start': t1, 'duration': d}
        ekf_result = core_module.run_ekf(nav_data, params, outage_config, USE_3D_ROTATION)
        
        # Ground truth
        lla0 = nav_data.lla0
        lla = nav_data.lla
        frecIMU = nav_data.sample_rate
        A = int(t1 * frecIMU)
        B = int((t1 + d) * frecIMU)
        
        f = pm.geodetic2enu(lla[:, 0], lla[:, 1], lla[:, 2], lla0[0], lla0[1], lla0[2])
        f_array = np.column_stack([f[0], f[1], f[2]])

        p = ekf_result['p']
        v = ekf_result['v']
        r = ekf_result['r']
        std_pos = ekf_result['std_pos']
        std_vel = ekf_result['std_vel']
        std_orient = ekf_result['std_orient']

        N = len(p)
        orient_gt = nav_data.orient
        vel_gt = nav_data.vel_enu

        # ─── 1. Position error (2D + 3D) ────────────────────────────────
        pos_err = p - f_array

        # Outage phase — 2D + vertical
        err_2d_outage = np.sqrt(pos_err[A:B, 0]**2 + pos_err[A:B, 1]**2)
        ate_2d_outage = np.sqrt(np.mean(err_2d_outage**2))
        ate_up_outage = np.sqrt(np.mean(pos_err[A:B, 2]**2))

        # GPS-aided phase (everything outside the outage)
        gps_mask = np.ones(N, dtype=bool)
        gps_mask[A:B] = False
        err_2d_gps = np.sqrt(pos_err[gps_mask, 0]**2 + pos_err[gps_mask, 1]**2)
        ate_2d_gps = np.sqrt(np.mean(err_2d_gps**2))

        # ─── 2. Orientation error ────────────────────────────────────────
        orient_err = r - orient_gt
        # Wrap to [-pi, pi]
        orient_err = (orient_err + np.pi) % (2 * np.pi) - np.pi
        rmse_roll  = np.sqrt(np.mean(orient_err[:, 0]**2))
        rmse_pitch = np.sqrt(np.mean(orient_err[:, 1]**2))
        rmse_yaw   = np.sqrt(np.mean(orient_err[:, 2]**2))

        # ─── 3. Velocity error ───────────────────────────────────────────
        vel_err = v - vel_gt
        rmse_vel = np.sqrt(np.mean(vel_err[:, 0]**2 + vel_err[:, 1]**2))

        # ─── 4. ANEES — filter consistency ───────────────────────────────
        # ANEES = (1/N) * Σ  (error' * P⁻¹ * error)  for position states
        # For a 3-state (E,N,U) position this should equal 3 when consistent.
        # We normalise to per-dimension: ANEES_dim ≈ 1 when consistent.
        # Use GPS-aided phase only (during outage, errors grow naturally).
        eps = 1e-12
        nees_sum = 0.0
        nees_count = 0
        # Subsample to keep computation light (every 10th sample)
        for k in range(0, N, 10):
            if A <= k < B:
                continue  # Skip outage
            var_p = std_pos[k]**2 + eps
            nees_k = np.sum(pos_err[k]**2 / var_p)
            nees_sum += nees_k
            nees_count += 1
        anees = nees_sum / max(nees_count, 1) / 3.0  # Per-dimension, ideal = 1.0

        # Penalty: deviation from ANEES = 1  (symmetric in log space)
        # anees_penalty = |log10(ANEES)|, so anees=1 → 0, anees=10 → 1, anees=0.1 → 1
        anees_penalty = abs(np.log10(max(anees, 1e-6)))

        # ─── Combine into scalar cost ───────────────────────────────────
        # Weights reflect priorities:
        #   - Primary: 2D outage position (the traditional metric)
        #   - Secondary: vertical, GPS-aided tracking, orientation
        #   - Regulariser: ANEES consistency
        cost = (
            5.0  * ate_2d_outage       # 2D position during outage [m]
          + 5.0  * ate_up_outage       # Vertical during outage [m]
          + 5.0  * ate_2d_gps          # GPS-aided tracking quality [m]
          + 5.0  * rmse_roll           # Roll RMSE [rad] (~57 deg = 1 rad)
          + 5.0  * rmse_pitch          # Pitch RMSE [rad]
          + 2.0  * rmse_yaw            # Yaw RMSE [rad]
          + 1.0  * rmse_vel            # Velocity RMSE [m/s]
          + 3.0  * anees_penalty       # Filter consistency
        )
        
        # Sanity checks
        if np.isnan(cost) or np.isinf(cost) or cost > 10000:
            if fitness_function.logger:
                fitness_function.logger.info(f"Eval {fitness_function.eval_count}: DIVERGED (fitness=1e6)")
            return 1e6
        
        # Log every evaluation with breakdown
        if fitness_function.logger:
            fitness_function.logger.info(
                f"Eval {fitness_function.eval_count}: cost={cost:.3f} "
                f"[pos2d_out={ate_2d_outage:.2f} up_out={ate_up_outage:.2f} "
                f"pos2d_gps={ate_2d_gps:.2f} roll={np.degrees(rmse_roll):.2f}deg "
                f"pitch={np.degrees(rmse_pitch):.2f}deg yaw={np.degrees(rmse_yaw):.1f}deg "
                f"vel={rmse_vel:.2f} ANEES={anees:.2f}]\n"
                + format_params_dict(params)
            )
            # Log new global minimum
            if cost < fitness_function.best_fitness:
                fitness_function.best_fitness = cost
                fitness_function.logger.info(
                    f"*** NEW BEST at eval {fitness_function.eval_count}: cost={cost:.3f} ***\n"
                    + format_params_dict(params)
                )
        
        return cost
        
    except Exception as e:
        if fitness_function.logger:
            fitness_function.logger.warning(f"Eval {fitness_function.eval_count}: ERROR - {str(e)[:50]}")
        return 1e6


################### FORMATTING ###########################

def format_params_dict(params: dict) -> str:
    """Format EKF parameters as a compact one-line copy-pasteable Python dict string."""
    return (
        "ekf_params = {"
        f"'Qpos': {params['Qpos']:.3e}, "
        f"'Qvel': {params['Qvel']:.3e}, "
        f"'QorientXY': {params['QorientXY']:.4e}, "
        f"'QorientZ': {params['QorientZ']:.4e}, "
        f"'Qacc': {params['Qacc']:.4e}, "
        f"'QgyrXY': {params['QgyrXY']:.4e}, "
        f"'QgyrZ': {params['QgyrZ']:.4e}, "
        f"'Rpos': {params['Rpos']:.2f}, "
        f"'Rvel': {params.get('Rvel', 0.0):.3e}, "
        f"'beta_acc': {params['beta_acc']:.3e}, "
        f"'beta_gyr': {params['beta_gyr']:.3e}, "
        f"'P_pos_std': {params['P_pos_std']:.2f}, "
        f"'P_vel_std': {params['P_vel_std']:.2f}, "
        f"'P_orient_std': {params['P_orient_std']:.3f}, "
        f"'P_acc_std': {params['P_acc_std']:.3e}, "
        f"'P_gyr_std': {params['P_gyr_std']:.3e}, "
        "}"
    )


################### OPTIMIZATION ###########################

if __name__ == "__main__":
    # Setup logging
    logs_dir = Path(__file__).parent / '../../../logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'ekf_genetic_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("EKF PARAMETER OPTIMIZATION USING DIFFERENTIAL EVOLUTION")
    logger.info("="*60)
    logger.info(f"Dataset: {NAV_DATA.dataset_name}")
    logger.info(f"Samples: {len(NAV_DATA.lla)}")
    logger.info(f"GNSS outage: {OUTAGE_START}s to {OUTAGE_START+OUTAGE_DURATION}s")
    logger.info(f"Parameters: {len(BOUNDS)}")
    logger.info(f"Fitness metric: Multi-objective (pos + orient + vel + ANEES consistency)")
    logger.info(f"Log file: {log_file}")
    logger.info("="*60)
    
    # Set logger for fitness function to enable per-evaluation logging
    fitness_function.logger = logger
    fitness_function.eval_count = 0
    fitness_function.best_fitness = float('inf')
    
    # Callback to log progress with copy-pasteable dict
    iteration_counter = [0]
    def log_progress(xk, convergence):
        iteration_counter[0] += 1
        fitness = fitness_function(xk, NAV_DATA, OUTAGE_START, OUTAGE_DURATION)
        params = decode_params(xk)
        logger.info(f"\n--- Iteration {iteration_counter[0]}: Best fitness = {fitness:.2f}m (convergence={convergence:.6f}) ---")
        logger.info("\n" + format_params_dict(params))
    
    logger.info("Starting optimization...")
    
    # Run differential evolution optimizer
    result = differential_evolution(
        fitness_function,
        BOUNDS,
        args=(NAV_DATA, OUTAGE_START, OUTAGE_DURATION),
        strategy='best1bin',
        maxiter=40,          # 40 generations
        popsize=15,          # 15 * 15 params = 225 population (efficient)
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=int(datetime.now().timestamp()),
        callback=log_progress,
        disp=True,
        polish=False,
        workers=1            # Set to -1 for parallel (but may have issues with data sharing)
    )
    
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Final cost: {result.fun:.4f}")
    logger.info(f"Evaluations: {result.nfev}")
    logger.info(f"Iterations: {result.nit}")
    logger.info(f"Success: {result.success}")
    if not result.success:
        logger.warning(f"Optimization message: {result.message}")
    
    # Decode best parameters
    best_params = decode_params(result.x)
    logger.info("\nOptimized parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key:20s}: {value:.6e}")
    
    # Print copy-pasteable dict (same format as DEFAULT_EKF_PARAMS)
    dict_str = format_params_dict(best_params)
    logger.info("\n" + "="*60)
    logger.info("COPY-PASTE INTO ekf.py as ekf_params:")
    logger.info("="*60)
    logger.info("\n" + dict_str)
    
    # Also print to stdout cleanly (in case log is noisy)
    print("\n" + "="*60)
    print("COPY-PASTE INTO ekf.py:")
    print("="*60)
    print(dict_str)
    print(f"\n# Cost: {result.fun:.4f} | Dataset: {NAV_DATA.dataset_name} | "
          f"Outage: {OUTAGE_START}s+{OUTAGE_DURATION}s")
    
    # Save to JSON (same keys as DEFAULT_EKF_PARAMS for direct loading)
    output_file = Path(__file__).parent / 'optimized_ekf_params.json'
    params_to_save = {
        'ekf_params': {k: float(v) for k, v in best_params.items()},
        'metadata': {
            'cost': float(result.fun),
            'dataset': NAV_DATA.dataset_name,
            'outage_config': {'start': OUTAGE_START, 'duration': OUTAGE_DURATION},
            'use_3d_rotation': USE_3D_ROTATION,
            'optimization': {
                'method': 'differential_evolution',
                'evaluations': int(result.nfev),
                'iterations': int(result.nit),
                'success': bool(result.success)
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(params_to_save, f, indent=2)
    
    logger.info(f"\nJSON saved to: {output_file}")
    logger.info(f"Log saved to: {log_file}")
    logger.info("\nTo load from JSON in ekf.py:")
    logger.info("  import json")
    logger.info("  with open('optimized_ekf_params.json') as f:")
    logger.info("      ekf_params = json.load(f)['ekf_params']")
