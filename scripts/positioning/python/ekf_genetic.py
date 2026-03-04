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


################### OPTIMIZATION PARAMETERS ###########################

# These are imported from ekf_config.py so the optimizer always runs on the same
# dataset / outage / rotation mode that ekf.py is currently configured for.
import ekf_config
NAV_DATA        = ekf_config.NAV_DATA
OUTAGE_START    = ekf_config.OUTAGE_START
OUTAGE_DURATION = ekf_config.OUTAGE_DURATION
USE_3D_ROTATION = ekf_config.USE_3D_ROTATION

# Parameter bounds for optimization (log scale)
# Order: Qpos, Qvel, QorientXY, QorientZ, Qacc, QgyrXY, QgyrZ, Rpos, beta_acc, beta_gyr, P_pos, P_vel, P_orient, P_acc, P_gyr
BOUNDS = [
    (-2, 2),     # log10(Qpos): 0.01 to 100
    (-2, 2),     # log10(Qvel): 0.01 to 100
    (-5, -1),    # log10(QorientXY): 0.00001 to 0.1
    (-2, 1),     # log10(QorientZ): 0.01 to 10
    (-3, 0),     # log10(Qacc): 0.001 to 1
    (-6, -2),    # log10(QgyrXY): 0.000001 to 0.01
    (-3, 0),     # log10(QgyrZ): 0.001 to 1
    (-1, 2),     # log10(Rpos): 0.1 to 100m
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
        'beta_acc': -10**x[8],  # Negative
        'beta_gyr': -10**x[9],  # Negative
        'P_pos_std': 10**x[10],
        'P_vel_std': 10**x[11],
        'P_orient_std': 10**x[12],
        'P_acc_std': 10**x[13],
        'P_gyr_std': 10**x[14],
    }


################### EKF RUNNER & FITNESS FUNCTION ###########################

def fitness_function(x: np.ndarray, nav_data, t1: float, d: float) -> float:
    """
    Fitness function for optimization: ATE RMSE during GPS outage.
    
    This is the standard metric used in SLAM/navigation evaluation.
    Lower is better.
    
    Args:
        x: Parameter vector (log scale)
        nav_data: Navigation dataset
        t1: GNSS outage start time (seconds)
        d: GNSS outage duration (seconds)
    
    Returns:
        ATE RMSE during outage period (meters)
    """
    # Initialize counter and logger on first call
    if not hasattr(fitness_function, 'eval_count'):
        fitness_function.eval_count = 0
        fitness_function.logger = None
        fitness_function.best_fitness = float('inf')
    
    fitness_function.eval_count += 1
    
    params = decode_params(x)
    
    # Run EKF with these parameters
    try:
        outage_config = {'start': t1, 'duration': d}
        ekf_result = ekf_core.run_ekf(nav_data, params, outage_config, USE_3D_ROTATION)
        
        # Compute fitness: ATE RMSE during outage
        lla0 = nav_data.lla0
        lla = nav_data.lla
        frecIMU = nav_data.sample_rate
        A = int(t1 * frecIMU)
        B = int((t1 + d) * frecIMU)
        
        f = pm.geodetic2enu(lla[:, 0], lla[:, 1], lla[:, 2], lla0[0], lla0[1], lla0[2])
        f_array = np.column_stack([f[0], f[1], f[2]])
        
        # Error during outage segment (standard ATE metric)
        diff = f_array[A:B] - ekf_result['p'][A:B]
        error_2D = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
        ate_rmse = np.sqrt(np.mean(error_2D**2))
        
        # Sanity checks
        if np.isnan(ate_rmse) or np.isinf(ate_rmse) or ate_rmse > 10000:
            if fitness_function.logger:
                fitness_function.logger.info(f"Eval {fitness_function.eval_count}: DIVERGED (fitness=1e6)")
            return 1e6
        
        # Log every evaluation with copy-pasteable dict
        if fitness_function.logger:
            fitness_function.logger.info(
                f"Eval {fitness_function.eval_count}: fitness={ate_rmse:.2f}m\n"
                + format_params_dict(params)
            )
            # Log new global minimum
            if ate_rmse < fitness_function.best_fitness:
                fitness_function.best_fitness = ate_rmse
                fitness_function.logger.info(
                    f"*** NEW BEST at eval {fitness_function.eval_count}: {ate_rmse:.2f}m ***\n"
                    + format_params_dict(params)
                )
        
        return ate_rmse
        
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
        f"'beta_acc': {params['beta_acc']:.3e}, "
        f"'beta_gyr': {params['beta_gyr']:.3e}, "
        f"'P_pos_std': {params['P_pos_std']:.2f}, "
        f"'P_vel_std': {params['P_vel_std']:.2f}, "
        f"'P_orient_std': {params['P_orient_std']:.3f}, "
        f"'P_acc_std': {params['P_acc_std']:.3e}, "
        f"'P_gyr_std': {params['P_gyr_std']:.3e}"
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
    logger.info(f"Fitness metric: ATE RMSE during outage (standard SLAM metric)")
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
        maxiter=20,          # 20 generations
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
    logger.info(f"Final ATE RMSE: {result.fun:.2f} meters")
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
    print(f"\n# ATE RMSE: {result.fun:.2f}m | Dataset: {NAV_DATA.dataset_name} | "
          f"Outage: {OUTAGE_START}s+{OUTAGE_DURATION}s")
    
    # Save to JSON (same keys as DEFAULT_EKF_PARAMS for direct loading)
    output_file = Path(__file__).parent / 'optimized_ekf_params.json'
    params_to_save = {
        'ekf_params': {k: float(v) for k, v in best_params.items()},
        'metadata': {
            'ate_rmse_outage': float(result.fun),
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
