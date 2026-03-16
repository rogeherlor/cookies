# -*- coding: utf-8 -*-
"""
ins_runner.py — Individual filter runner for INS/GNSS navigation.

Reads configuration from ins_config.py and runs the selected filter.
Evaluates performance metrics, saves results, and generates plots.

Usage:
    python ins_runner.py

Configuration:
    Edit ins_config.py to choose the filter, dataset, outage window,
    rotation mode, and (optionally) pre-tuned parameters.

IMPORTANT — parameter tuning
------------------------------
Default parameters give a rough first look.  Run ins_genetic.py for the
selected filter to optimise parameters before drawing any conclusions.
"""
import os
import sys
import json
import logging
import numpy as np
import pymap3d as pm
from pathlib import Path
from datetime import datetime

# Allow local imports from this directory and the filters sub-package
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import ins_config
import metrics
import visualize
import visualize_state
from filters import (
    ekf_vanilla, ekf_enhanced,
    eskf_vanilla, eskf_enhanced,
    iekf_vanilla, iekf_enhanced,
    imu_only,
)
from dl_filters.deep_iekf  import iekf_ai_imu
from dl_filters.tlio       import tlio_runner
from dl_filters.deep_kf    import deep_kf_runner
from dl_filters.tartan_imu import tartan_runner

# ── Filter dispatch table ──────────────────────────────────────────────────────
FILTERS = {
    "ekf_vanilla":   ekf_vanilla,
    "ekf_enhanced":  ekf_enhanced,
    "eskf_vanilla":  eskf_vanilla,
    "eskf_enhanced": eskf_enhanced,
    "iekf_vanilla":  iekf_vanilla,
    "iekf_enhanced": iekf_enhanced,
    "imu_only":      imu_only,
    # Deep learning filters
    "iekf_ai_imu":   iekf_ai_imu,
    "tlio":          tlio_runner,
    "deep_kf":       deep_kf_runner,
    "tartan_imu":    tartan_runner,
}


def main():
    # ── Load configuration ─────────────────────────────────────────────────────
    filter_name  = ins_config.FILTER
    use_3d       = ins_config.MODE_3D
    t1           = ins_config.OUTAGE_START
    d            = ins_config.OUTAGE_DURATION
    nav_data     = ins_config.NAV_DATA
    filter_params = ins_config.FILTER_PARAMS

    # ── Validate filter name ───────────────────────────────────────────────────
    if filter_name not in FILTERS:
        raise ValueError(
            f"Unknown filter '{filter_name}'. "
            f"Available: {list(FILTERS.keys())}"
        )

    # ── Setup output directories ───────────────────────────────────────────────
    base_dir    = _HERE
    logs_dir    = base_dir / '../../../logs'
    outputs_dir = base_dir / f'../../../outputs/{filter_name}'

    if t1 == 0 and d == 0:
        run_id    = 'no_outage'
        traj_subdir = f"{nav_data.dataset_name}_no_outage"
    else:
        run_id      = f"outage_{t1}s_{d}s"
        traj_subdir = f"{nav_data.dataset_name}_outage_{t1}s_{d}s"

    output_subdir = outputs_dir / traj_subdir
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_subdir.mkdir(parents=True, exist_ok=True)

    # ── Logging ────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = logs_dir / f'ins_{filter_name}_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"INS RUNNER — {filter_name.upper()}")
    logger.info("=" * 60)
    logger.info(f"Dataset    : {nav_data.dataset_name}")
    logger.info(f"Samples    : {len(nav_data.lla)}")
    logger.info(f"Sample rate: {nav_data.sample_rate} Hz")
    logger.info(f"GNSS outage: {t1}s – {t1+d}s  ({d}s)")
    logger.info(f"Rotation   : {'3D (roll/pitch/yaw)' if use_3d else '2D (yaw only)'}")
    logger.info(f"Parameters : {'custom' if filter_params else 'default (run ins_genetic.py to optimise)'}")
    logger.info("=" * 60)

    # ── Run selected filter ────────────────────────────────────────────────────
    filter_module = FILTERS[filter_name]
    outage_config = {'start': t1, 'duration': d}

    result = filter_module.run(
        nav_data=nav_data,
        params=filter_params,
        outage_config=outage_config,
        use_3d_rotation=use_3d,
    )

    p         = result['p']
    v         = result['v']
    r         = result['r']
    bias_acc  = result['bias_acc']
    bias_gyr  = result['bias_gyr']
    std_pos   = result['std_pos']
    std_vel   = result['std_vel']
    std_orient = result['std_orient']
    std_bias_acc = result['std_bias_acc']
    std_bias_gyr = result['std_bias_gyr']

    # ── Ground truth in ENU ────────────────────────────────────────────────────
    lla  = nav_data.lla
    lla0 = nav_data.lla0
    f    = pm.geodetic2enu(lla[:,0], lla[:,1], lla[:,2], lla0[0], lla0[1], lla0[2])
    p_gt = np.column_stack([f[0], f[1], f[2]])

    gnss_outage_info = {
        'start':     t1,
        'end':       t1 + d,
        'duration':  d,
        'start_idx': int(t1 * nav_data.sample_rate),
        'end_idx':   int((t1 + d) * nav_data.sample_rate),
    }

    # ── Evaluate metrics ───────────────────────────────────────────────────────
    results = metrics.evaluate_navigation_performance(
        p_est=p, v_est=v, r_est=r,
        p_gt=p_gt, v_gt=nav_data.vel_enu, r_gt=nav_data.orient,
        dataset_name=nav_data.dataset_name,
        gnss_outage_info=gnss_outage_info,
        sample_rate=nav_data.sample_rate,
    )

    metrics.log_evaluation_results(logger, results, str(log_file))

    # ── Save JSON results ──────────────────────────────────────────────────────
    results_file = output_subdir / f'{run_id}_results.json'
    results_serializable = {
        'filter':         filter_name,
        'dataset':        results['dataset'],
        'total_samples':  results['total_samples'],
        'gnss_outage':    results['gnss_outage'],
        'position_rmse':  results['position_rmse'],
        'velocity_rmse':  results['velocity_rmse'],
        'orientation_rmse': results['orientation_rmse'],
        'ate':     {k: v for k, v in results['ate'].items()     if k != 'errors'},
        'rte_1s':  {k: v for k, v in results['rte_1s'].items()  if k != 'errors'},
        'rte_5s':  {k: v for k, v in results['rte_5s'].items()  if k != 'errors'},
        'rte_10s': {k: v for k, v in results['rte_10s'].items() if k != 'errors'},
        'peak_errors':    results['peak_errors'],
        'outage_analysis': results['outage_analysis'],
    }
    with open(results_file, 'w') as fh:
        json.dump(results_serializable, fh, indent=2)
    logger.info(f'Results saved to: {results_file}')

    # ── Save trajectory NPZ ────────────────────────────────────────────────────
    np.savez(
        output_subdir / f'{run_id}_trajectories.npz',
        p_est=p, v_est=v, r_est=r,
        p_gt=p_gt, v_gt=nav_data.vel_enu, r_gt=nav_data.orient,
        bias_acc=bias_acc, bias_gyr=bias_gyr,
        std_pos=std_pos, std_vel=std_vel, std_orient=std_orient,
        std_bias_acc=std_bias_acc, std_bias_gyr=std_bias_gyr,
        time=np.arange(len(p)) / nav_data.sample_rate,
    )

    # ── Trajectory and error plots ─────────────────────────────────────────────
    generated = visualize.generate_all_plots(
        results=results,
        p_est=p, v_est=v, r_est=r,
        p_gt=p_gt, v_gt=nav_data.vel_enu, r_gt=nav_data.orient,
        sample_rate=nav_data.sample_rate,
        output_dir=str(output_subdir),
        run_id=run_id,
        accel_flu=nav_data.accel_flu,
        gyro_flu=nav_data.gyro_flu,
        lla0=lla0,
        gps_available=nav_data.gps_available,
    )
    logger.info(f'Generated {len(generated)} trajectory charts.')

    # ── Filter-state plots (skip for imu_only — no covariance) ────────────────
    time_arr     = np.arange(len(p)) / nav_data.sample_rate
    errors_pos   = p_gt - p

    if filter_name != 'imu_only':
        visualize_state.plot_bias_estimates(
            time_arr, bias_acc, bias_gyr, gnss_outage_info,
            save_path=str(output_subdir / f'{run_id}_bias_estimates.png'),
        )
        visualize_state.plot_uncertainty_evolution(
            time_arr, std_pos, std_vel, std_orient, gnss_outage_info,
            save_path=str(output_subdir / f'{run_id}_uncertainty.png'),
        )
        visualize_state.plot_bias_uncertainty(
            time_arr, bias_acc, bias_gyr, std_bias_acc, std_bias_gyr, gnss_outage_info,
            save_path=str(output_subdir / f'{run_id}_bias_uncertainty.png'),
        )
        visualize_state.plot_filter_consistency(
            time_arr, errors_pos, std_pos, gnss_outage_info,
            save_path=str(output_subdir / f'{run_id}_filter_consistency.png'),
        )
        visualize_state.plot_convergence_dashboard(
            time_arr, p, p_gt, v, nav_data.vel_enu, r, nav_data.orient,
            std_pos, std_vel, std_orient,
            bias_acc, bias_gyr, std_bias_acc, std_bias_gyr,
            gnss_outage_info, gps_available=nav_data.gps_available,
            save_path=str(output_subdir / f'{run_id}_convergence_dashboard.png'),
        )
        logger.info('Generated filter-state charts.')

    # ── Interactive trajectory plot ────────────────────────────────────────────
    visualize.show_interactive_plot(
        p, p_gt, gnss_outage_info, nav_data.dataset_name,
        lla0=lla0, gps_available=nav_data.gps_available,
    )

    logger.info(f'Run complete. Output directory: {output_subdir}')
    logger.info(f'Log file: {log_file}')


if __name__ == '__main__':
    main()
