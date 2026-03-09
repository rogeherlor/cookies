# -*- coding: utf-8 -*-
"""
ins_genetic.py — Parameter optimiser for any filter in the INS/GNSS framework.

Uses scipy.optimize.differential_evolution to tune filter noise covariances
and Gauss-Markov coefficients.  Reads the active filter and dataset from
ins_config.py so the optimised parameters apply to the same configuration
used in ins_runner.py and ins_compare.py.

Usage:
    1. Edit ins_config.py:  set FILTER and MODE_3D to the target configuration.
    2. Run:  python ins_genetic.py
    3. Wait for convergence (~minutes to hours depending on machine).
    4. Copy the printed FILTER_PARAMS dict into ins_config.py.

IMPORTANT
---------
Run once per filter variant (6 runs for a fair comparison):
    ekf_vanilla, ekf_enhanced, eskf_vanilla, eskf_enhanced,
    iekf_vanilla, iekf_enhanced.
"imu_only" has no tunable parameters and should not be optimised.

The cost function combines outage accuracy, GPS-aided tracking quality,
orientation RMSE, velocity RMSE, and ANEES filter-consistency.
Lower cost = better overall filter.
"""
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import differential_evolution
import sys

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import pymap3d as pm
import ins_config
import filter_params as fp
from filters import (
    ekf_vanilla, ekf_enhanced,
    eskf_vanilla, eskf_enhanced,
    iekf_vanilla, iekf_enhanced,
)

# ── Read shared configuration ──────────────────────────────────────────────────
NAV_DATA        = ins_config.NAV_DATA
OUTAGE_START    = ins_config.OUTAGE_START
OUTAGE_DURATION = ins_config.OUTAGE_DURATION
USE_3D          = ins_config.MODE_3D
FILTER_NAME     = ins_config.FILTER

# ── Filter dispatch (imu_only excluded — nothing to optimise) ─────────────────
FILTERS = {
    "ekf_vanilla":   ekf_vanilla,
    "ekf_enhanced":  ekf_enhanced,
    "eskf_vanilla":  eskf_vanilla,
    "eskf_enhanced": eskf_enhanced,
    "iekf_vanilla":  iekf_vanilla,
    "iekf_enhanced": iekf_enhanced,
}


# ── Parameter search bounds (log₁₀ scale) ─────────────────────────────────────
# Order:
#   Qpos, Qvel, QorientXY, QorientZ, Qacc, QgyrXY, QgyrZ,
#   Rpos,
#   beta_acc (abs), beta_gyr (abs),
#   P_pos_std, P_vel_std, P_orient_std, P_acc_std, P_gyr_std
BOUNDS = [
    (-2, 2),     # log10(Qpos):      0.01 – 100
    (-2, 2),     # log10(Qvel):      0.01 – 100
    (-5, -1),    # log10(QorientXY): 1e-5 – 0.1
    (-2, 1),     # log10(QorientZ):  0.01 – 10
    (-3, 0),     # log10(Qacc):      0.001 – 1
    (-6, -2),    # log10(QgyrXY):    1e-6 – 0.01
    (-3, 0),     # log10(QgyrZ):     0.001 – 1
    (-1, 2),     # log10(Rpos):      0.1 – 100 m²
    (-8, -5),    # log10(|beta_acc|): very small, negative
    (-2, 1),     # log10(|beta_gyr|): small, negative
    (-1, 1.5),   # log10(P_pos_std):  0.1 – 30 m
    (-1, 0.5),   # log10(P_vel_std):  0.1 – 3 m/s
    (-2, -0.5),  # log10(P_orient_std): 0.01 – 0.3 rad
    (-3, -1),    # log10(P_acc_std):  0.001 – 0.1 m/s²
    (-4, -2),    # log10(P_gyr_std):  0.0001 – 0.01 rad/s
]


def decode_params(x: np.ndarray) -> dict:
    """Convert log₁₀ parameter vector → actual filter parameter dict."""
    return {
        'Qpos':        10**x[0],
        'Qvel':        10**x[1],
        'QorientXY':   10**x[2],
        'QorientZ':    10**x[3],
        'Qacc':        10**x[4],
        'QgyrXY':      10**x[5],
        'QgyrZ':       10**x[6],
        'Rpos':        10**x[7],
        'beta_acc':   -10**x[8],    # negative (stable decay)
        'beta_gyr':   -10**x[9],    # negative
        'P_pos_std':   10**x[10],
        'P_vel_std':   10**x[11],
        'P_orient_std': 10**x[12],
        'P_acc_std':   10**x[13],
        'P_gyr_std':   10**x[14],
    }


def format_params_dict(params: dict) -> str:
    """Return a copy-pasteable Python dict string of the parameters."""
    return (
        "FILTER_PARAMS = {"
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
        f"'P_gyr_std': {params['P_gyr_std']:.3e}, "
        "}"
    )


def fitness_function(x: np.ndarray, nav_data, t1: float, d: float) -> float:
    """
    Multi-objective cost for filter parameter optimisation.

    Components (weighted sum, lower = better):
      1. 2D position ATE during GNSS outage  (primary metric, weight 5)
      2. Vertical position ATE during outage  (weight 5)
      3. 2D position ATE during GPS-aided phase  (weight 5)
      4. Roll RMSE   (weight 5)
      5. Pitch RMSE  (weight 5)
      6. Yaw RMSE    (weight 2)
      7. Velocity RMSE  (weight 1)
      8. ANEES consistency penalty  (weight 3)
    """
    if not hasattr(fitness_function, 'eval_count'):
        fitness_function.eval_count = 0
        fitness_function.logger     = None
        fitness_function.best       = float('inf')

    fitness_function.eval_count += 1

    params        = decode_params(x)
    filter_module = FILTERS.get(FILTER_NAME)

    try:
        result = filter_module.run(
            nav_data=nav_data,
            params=params,
            outage_config={'start': t1, 'duration': d},
            use_3d_rotation=USE_3D,
        )

        lla0    = nav_data.lla0
        lla     = nav_data.lla
        frecIMU = nav_data.sample_rate
        A       = int(t1 * frecIMU)
        B       = int((t1 + d) * frecIMU)

        f       = pm.geodetic2enu(lla[:,0], lla[:,1], lla[:,2], lla0[0], lla0[1], lla0[2])
        p_gt    = np.column_stack([f[0], f[1], f[2]])

        p       = result['p']
        v       = result['v']
        r       = result['r']
        std_pos = result['std_pos']
        N       = len(p)

        pos_err = p - p_gt

        # 1 & 2 — outage errors
        err_2d_out = np.sqrt(pos_err[A:B, 0]**2 + pos_err[A:B, 1]**2)
        ate_2d_out = np.sqrt(np.mean(err_2d_out**2)) if len(err_2d_out) > 0 else 0.0
        ate_up_out = np.sqrt(np.mean(pos_err[A:B, 2]**2)) if B > A else 0.0

        # 3 — GPS-aided errors
        mask_gps   = np.ones(N, dtype=bool)
        mask_gps[A:B] = False
        err_2d_gps = np.sqrt(pos_err[mask_gps, 0]**2 + pos_err[mask_gps, 1]**2)
        ate_2d_gps = np.sqrt(np.mean(err_2d_gps**2))

        # 4–6 — orientation
        orient_err = (r - nav_data.orient + np.pi) % (2*np.pi) - np.pi
        rmse_roll  = np.sqrt(np.mean(orient_err[:, 0]**2))
        rmse_pitch = np.sqrt(np.mean(orient_err[:, 1]**2))
        rmse_yaw   = np.sqrt(np.mean(orient_err[:, 2]**2))

        # 7 — velocity
        vel_err  = v - nav_data.vel_enu
        rmse_vel = np.sqrt(np.mean(vel_err[:, 0]**2 + vel_err[:, 1]**2))

        # 8 — ANEES consistency (GPS-aided phase, sampled every 10th epoch)
        eps = 1e-12
        nees_sum, nees_cnt = 0.0, 0
        for k in range(0, N, 10):
            if A <= k < B:
                continue
            var_p = std_pos[k]**2 + eps
            nees_sum += np.sum(pos_err[k]**2 / var_p)
            nees_cnt += 1
        anees = nees_sum / max(nees_cnt, 1) / 3.0
        anees_penalty = abs(np.log10(max(anees, 1e-6)))

        cost = (
            5.0 * ate_2d_out
          + 5.0 * ate_up_out
          + 5.0 * ate_2d_gps
          + 5.0 * rmse_roll
          + 5.0 * rmse_pitch
          + 2.0 * rmse_yaw
          + 1.0 * rmse_vel
          + 3.0 * anees_penalty
        )

        if np.isnan(cost) or np.isinf(cost) or cost > 10_000:
            return 1e6

        if fitness_function.logger:
            fitness_function.logger.info(
                f"Eval {fitness_function.eval_count:5d}: cost={cost:.3f} "
                f"[2d_out={ate_2d_out:.2f} up_out={ate_up_out:.2f} "
                f"2d_gps={ate_2d_gps:.2f} roll={np.degrees(rmse_roll):.2f}° "
                f"pitch={np.degrees(rmse_pitch):.2f}° yaw={np.degrees(rmse_yaw):.1f}° "
                f"vel={rmse_vel:.2f} ANEES={anees:.2f}]\n"
                + format_params_dict(params)
            )
            if cost < fitness_function.best:
                fitness_function.best = cost
                fitness_function.logger.info(
                    f"*** NEW BEST  eval={fitness_function.eval_count}  cost={cost:.3f} ***\n"
                    + format_params_dict(params)
                )

        return cost

    except Exception as e:
        if fitness_function.logger:
            fitness_function.logger.warning(
                f"Eval {fitness_function.eval_count}: ERROR — {str(e)[:80]}"
            )
        return 1e6


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    if FILTER_NAME == 'imu_only':
        print("imu_only has no tunable parameters.  Nothing to optimise.")
        sys.exit(0)

    if FILTER_NAME not in FILTERS:
        raise ValueError(f"Unknown filter '{FILTER_NAME}'.")

    # ── Logging ────────────────────────────────────────────────────────────────
    logs_dir = _HERE / '../../../logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = logs_dir / f'ins_genetic_{FILTER_NAME}_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 65)
    logger.info("FILTER PARAMETER OPTIMISATION — DIFFERENTIAL EVOLUTION")
    logger.info("=" * 65)
    logger.info(f"Filter     : {FILTER_NAME}")
    logger.info(f"Dataset    : {NAV_DATA.dataset_name}")
    logger.info(f"Samples    : {len(NAV_DATA.lla)}")
    logger.info(f"Outage     : {OUTAGE_START}s – {OUTAGE_START+OUTAGE_DURATION}s")
    logger.info(f"Mode       : {'3D' if USE_3D else '2D'}")
    logger.info(f"Parameters : {len(BOUNDS)}")
    logger.info(f"Log file   : {log_file}")
    logger.info("=" * 65)

    fitness_function.logger = logger
    fitness_function.eval_count = 0
    fitness_function.best = float('inf')

    iteration_counter = [0]

    def log_progress(xk, convergence):
        iteration_counter[0] += 1
        params = decode_params(xk)
        logger.info(
            f"\n--- Generation {iteration_counter[0]}  "
            f"(convergence={convergence:.6f}) ---\n"
            + format_params_dict(params)
        )

    logger.info("Starting optimisation…")

    result = differential_evolution(
        fitness_function,
        BOUNDS,
        args=(NAV_DATA, OUTAGE_START, OUTAGE_DURATION),
        strategy='best1bin',
        maxiter=40,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=int(datetime.now().timestamp()),
        callback=log_progress,
        disp=True,
        polish=False,
        workers=1,
    )

    best_params = decode_params(result.x)

    logger.info("\n" + "=" * 65)
    logger.info("OPTIMISATION COMPLETE")
    logger.info("=" * 65)
    logger.info(f"Final cost   : {result.fun:.4f}")
    logger.info(f"Evaluations  : {result.nfev}")
    logger.info(f"Iterations   : {result.nit}")
    logger.info(f"Success      : {result.success}")

    dict_str = format_params_dict(best_params)
    logger.info("\n" + "=" * 65)
    logger.info("PASTE INTO ins_config.py as FILTER_PARAMS:")
    logger.info("=" * 65)
    logger.info("\n" + dict_str)

    print("\n" + "=" * 65)
    print(f"OPTIMISED PARAMETERS — {FILTER_NAME} | {NAV_DATA.dataset_name}")
    print("=" * 65)
    print(dict_str)
    print(f"\n# Cost: {result.fun:.4f}  |  "
          f"Outage: {OUTAGE_START}s+{OUTAGE_DURATION}s  |  "
          f"Mode: {'3D' if USE_3D else '2D'}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_dir = _HERE / '../../../outputs/genetic'
    out_dir.mkdir(parents=True, exist_ok=True)
    json_file = out_dir / f'{FILTER_NAME}_params.json'

    with open(json_file, 'w') as fh:
        json.dump({
            'filter_params': {k: float(v) for k, v in best_params.items()},
            'metadata': {
                'filter':    FILTER_NAME,
                'cost':      float(result.fun),
                'dataset':   NAV_DATA.dataset_name,
                'outage':    {'start': OUTAGE_START, 'duration': OUTAGE_DURATION},
                'mode_3d':   USE_3D,
                'optimiser': {
                    'method':      'differential_evolution',
                    'evaluations': int(result.nfev),
                    'iterations':  int(result.nit),
                    'success':     bool(result.success),
                },
            },
        }, fh, indent=2)

    logger.info(f"\nJSON saved to: {json_file}")

    # ── Save to central filter_params store ────────────────────────────────────
    prev_cost = fp.get_cost(FILTER_NAME, USE_3D, NAV_DATA.dataset_name)
    if prev_cost is None or result.fun < prev_cost:
        fp.set(
            FILTER_NAME, USE_3D, NAV_DATA.dataset_name,
            params=best_params, cost=float(result.fun),
            metadata={
                'optimiser':   'ins_genetic',
                'evaluations': int(result.nfev),
                'iterations':  int(result.nit),
                'success':     bool(result.success),
                'timestamp':   timestamp,
                'outage':      {'start': OUTAGE_START, 'duration': OUTAGE_DURATION},
            },
        )
        logger.info(f"Saved to filter_params store (filter_params.json)")
    else:
        logger.info(f"Existing stored cost ({prev_cost:.4f}) is better — filter_params not updated.")

    logger.info(f"Log  saved to: {log_file}")
    logger.info("\nTo load from JSON:")
    logger.info("  import json")
    logger.info(f"  with open('{json_file}') as f:")
    logger.info("      FILTER_PARAMS = json.load(f)['filter_params']")
