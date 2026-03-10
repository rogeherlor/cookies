# -*- coding: utf-8 -*-
"""
ins_compare.py — Run all filter variants and generate comparison plots.

Runs all 7 cases with the same dataset and GNSS outage window:
    1. EKF Vanilla    (Groves 2013, GPS only)
    2. EKF Enhanced   (EKF + NHC + ZUPT)
    3. ESKF Vanilla   (Solà 2017, GPS only)
    4. ESKF Enhanced  (ESKF + NHC + ZUPT)
    5. IEKF Vanilla   (Barrau & Bonnabel 2017, GPS only)
    6. IEKF Enhanced  (IEKF + NHC + ZUPT)
    7. IMU Only       (pure dead reckoning — baseline)

Per-filter outputs are saved to outputs/<filter_name>/<dataset>_outage_*/
Comparison plots are saved to outputs/comparison/<dataset>_outage_*/

IMPORTANT — parameter tuning
------------------------------
Default parameters are generic starting points.  Results from a first run
should NOT be used to draw performance conclusions.  Run ins_genetic.py for
each filter variant to obtain optimised FILTER_PARAMS, then paste them into
the TUNED_PARAMS dict below before running comparisons.
"""
import os
import sys
import json
import logging
import numpy as np
import pymap3d as pm
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import ins_config
import filter_params as fp
import metrics
import visualize
import visualize_state
import visualize_compare
from filters import (
    ekf_vanilla, ekf_enhanced,
    eskf_vanilla, eskf_enhanced,
    iekf_vanilla, iekf_enhanced,
    imu_only,
)

# ── Tuned parameters ───────────────────────────────────────────────────────────
# Auto-loaded from filter_params.json (written by ins_genetic.py / ins_genetic_fast.py).
# Any filter without stored params will fall back to its built-in DEFAULT_PARAMS.
# To override a specific filter manually, add its key here:
#   TUNED_PARAMS['ekf_vanilla'] = {'Qpos': ..., ...}
def _load_tuned_params(nav_data, mode_3d):
    result = {}
    for key in ['ekf_vanilla', 'ekf_enhanced', 'eskf_vanilla', 'eskf_enhanced',
                'iekf_vanilla', 'iekf_enhanced']:
        p = fp.get(key, mode_3d, nav_data.dataset_name)
        if p is not None:
            result[key] = p
    return result


# ── Filter configurations ──────────────────────────────────────────────────────
FILTER_CONFIGS = [
    {'name': 'EKF Vanilla',    'key': 'ekf_vanilla',   'module': ekf_vanilla},
    {'name': 'EKF Enhanced',   'key': 'ekf_enhanced',  'module': ekf_enhanced},
    {'name': 'ESKF Vanilla',   'key': 'eskf_vanilla',  'module': eskf_vanilla},
    {'name': 'ESKF Enhanced',  'key': 'eskf_enhanced', 'module': eskf_enhanced},
    {'name': 'IEKF Vanilla',   'key': 'iekf_vanilla',  'module': iekf_vanilla},
    {'name': 'IEKF Enhanced',  'key': 'iekf_enhanced', 'module': iekf_enhanced},
    {'name': 'IMU Only',       'key': 'imu_only',      'module': imu_only},
]


def main():
    # ── Shared configuration ───────────────────────────────────────────────────
    nav_data     = ins_config.NAV_DATA
    use_3d       = ins_config.MODE_3D
    TUNED_PARAMS = _load_tuned_params(nav_data, use_3d)
    t1       = ins_config.OUTAGE_START
    d        = ins_config.OUTAGE_DURATION

    lla      = nav_data.lla
    lla0     = nav_data.lla0
    frecIMU  = nav_data.sample_rate

    f     = pm.geodetic2enu(lla[:,0], lla[:,1], lla[:,2], lla0[0], lla0[1], lla0[2])
    p_gt  = np.column_stack([f[0], f[1], f[2]])

    outage_config = {'start': t1, 'duration': d}
    gnss_outage_info = {
        'start':     t1, 'end': t1 + d, 'duration': d,
        'start_idx': int(t1 * frecIMU),
        'end_idx':   int((t1 + d) * frecIMU),
    }

    # ── Output directories ─────────────────────────────────────────────────────
    traj_subdir   = (f"{nav_data.dataset_name}_outage_{t1}s_{d}s"
                     if (t1 > 0 or d > 0) else f"{nav_data.dataset_name}_no_outage")
    compare_dir   = _HERE / f'../../../outputs/comparison/{traj_subdir}'
    logs_dir      = _HERE / '../../../logs'
    compare_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = logs_dir / f'ins_compare_{timestamp}.log'
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
    logger.info("INS FILTER COMPARISON")
    logger.info("=" * 65)
    logger.info(f"Dataset    : {nav_data.dataset_name}")
    logger.info(f"GNSS outage: {t1}s – {t1+d}s  ({d}s)")
    logger.info(f"Rotation   : {'3D' if use_3d else '2D'}")
    logger.info(f"Filters    : {len(FILTER_CONFIGS)}")
    logger.info("")
    logger.info("NOTE: Results use default parameters unless TUNED_PARAMS is set.")
    logger.info("      Run ins_genetic.py for each filter before comparing.")
    logger.info("=" * 65)

    # ── Run all filters ────────────────────────────────────────────────────────
    all_results = []

    for cfg in FILTER_CONFIGS:
        fname  = cfg['name']
        fkey   = cfg['key']
        fmod   = cfg['module']
        params = TUNED_PARAMS.get(fkey, None)

        logger.info(f"\nRunning {fname}  ({'tuned' if params else 'default params'})…")

        result = fmod.run(
            nav_data=nav_data,
            params=params,
            outage_config=outage_config,
            use_3d_rotation=use_3d,
        )

        p = result['p'];  v = result['v'];  r = result['r']

        # Metrics
        mets = metrics.evaluate_navigation_performance(
            p_est=p, v_est=v, r_est=r,
            p_gt=p_gt, v_gt=nav_data.vel_enu, r_gt=nav_data.orient,
            dataset_name=nav_data.dataset_name,
            gnss_outage_info=gnss_outage_info,
            sample_rate=frecIMU,
        )

        ate_2d = mets.get('ate', {}).get('rmse_2D', float('nan'))
        pos_rmse_2d = mets.get('position_rmse', {}).get('2D', float('nan'))
        logger.info(f"  ATE 2D = {ate_2d:.2f} m  |  pos RMSE 2D = {pos_rmse_2d:.2f} m")

        entry = {
            'name':    fname,
            'key':     fkey,
            'p':       p,
            'v':       v,
            'r':       r,
            'std_pos': result['std_pos'],
            'metrics': mets,
            'result':  result,
        }
        all_results.append(entry)

        # Save individual per-filter outputs
        ind_dir = _HERE / f'../../../outputs/{fkey}/{traj_subdir}'
        ind_dir.mkdir(parents=True, exist_ok=True)
        run_id  = f"outage_{t1}s_{d}s" if (t1 > 0 or d > 0) else 'no_outage'

        np.savez(
            ind_dir / f'{run_id}_trajectories.npz',
            p_est=p, v_est=v, r_est=r,
            p_gt=p_gt, v_gt=nav_data.vel_enu, r_gt=nav_data.orient,
            bias_acc=result['bias_acc'], bias_gyr=result['bias_gyr'],
            std_pos=result['std_pos'], std_vel=result['std_vel'],
            std_orient=result['std_orient'],
            time=np.arange(len(p)) / frecIMU,
        )

        mets_serial = {
            'filter': fkey,
            'dataset': mets['dataset'],
            'total_samples': mets['total_samples'],
            'position_rmse': mets['position_rmse'],
            'velocity_rmse': mets['velocity_rmse'],
            'orientation_rmse': mets['orientation_rmse'],
            'ate':  {k: val for k, val in mets['ate'].items()  if k != 'errors'},
            'rte_1s': {k: val for k, val in mets['rte_1s'].items() if k != 'errors'},
            'outage_analysis': mets['outage_analysis'],
        }
        with open(ind_dir / f'{run_id}_results.json', 'w') as fh:
            json.dump(mets_serial, fh, indent=2)

        if fkey != 'imu_only':
            time_arr = np.arange(len(p)) / frecIMU
            try:
                visualize_state.plot_bias_estimates(
                    time_arr, result['bias_acc'], result['bias_gyr'], gnss_outage_info,
                    save_path=str(ind_dir / f'{run_id}_bias_estimates.png'),
                )
                visualize_state.plot_uncertainty_evolution(
                    time_arr, result['std_pos'], result['std_vel'], result['std_orient'],
                    gnss_outage_info,
                    save_path=str(ind_dir / f'{run_id}_uncertainty.png'),
                )
            except Exception as e:
                logger.warning(f"  State plots failed for {fname}: {e}")

    logger.info("\n" + "=" * 65)
    logger.info("Generating comparison plots…")

    # ── Comparison plots ───────────────────────────────────────────────────────
    generated = visualize_compare.generate_comparison_plots(
        filter_results=all_results,
        p_gt=p_gt,
        gnss_outage_info=gnss_outage_info,
        sample_rate=frecIMU,
        output_dir=str(compare_dir),
    )

    logger.info(f"Generated {len(generated)} comparison plots in: {compare_dir}")
    logger.info(f"Log: {log_file}")

    # ── Summary table to console ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"COMPARISON SUMMARY — {nav_data.dataset_name}  |  "
          f"Outage {t1}s+{d}s  |  {'3D' if use_3d else '2D'}")
    print("=" * 80)
    print(f"{'Filter':<20} {'ATE 2D [m]':>12} {'ATE 3D [m]':>12} {'Pos RMSE 2D':>13} {'Outage max [m]':>16}")
    print("-" * 80)
    for entry in all_results:
        mets   = entry['metrics']
        ate2d  = mets.get('ate', {}).get('rmse_2D',          float('nan'))
        ate3d  = mets.get('ate', {}).get('rmse_3D',          float('nan'))
        prmse  = mets.get('position_rmse', {}).get('2D', float('nan'))
        outagm = mets.get('outage_analysis', {}).get('max', float('nan'))
        print(f"{entry['name']:<20} {ate2d:>12.2f} {ate3d:>12.2f} {prmse:>13.2f} {outagm:>16.2f}")
    print("=" * 80)
    print("NOTE: Tune parameters with ins_genetic.py before drawing conclusions.\n")


if __name__ == '__main__':
    main()
