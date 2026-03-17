# -*- coding: utf-8 -*-
"""
ins_compare.py — Run all filter variants and generate comparison plots.

Runs all 8 cases with the same dataset and GNSS outage window:
    1. EKF Vanilla    (Groves 2013, GPS only)
    2. EKF Enhanced   (EKF + NHC + ZUPT)
    3. ESKF Vanilla   (Solà 2017, GPS only)
    4. ESKF Enhanced  (ESKF + NHC + ZUPT)
    5. IEKF Vanilla   (Barrau & Bonnabel 2017, GPS only)
    6. IEKF Enhanced  (IEKF + NHC + ZUPT)
    7. IMU Only       (pure dead reckoning — baseline)
    8. IEKF AI-IMU    (Brossard et al. 2020, IMU dead-reckoning + CNN covariance adapter)

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
import argparse
import numpy as np
import pymap3d as pm
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import ins_config
import data_loader
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
from smoothers import rts_smoother, isam2_runner
from dl_filters.deep_iekf  import iekf_ai_imu
from dl_filters.tlio       import tlio_runner
from dl_filters.deep_kf    import deep_kf_runner
from dl_filters.tartan_imu import tartan_runner

# ── Tuned parameters ───────────────────────────────────────────────────────────
# Auto-loaded from filter_params.json (written by ins_genetic.py / ins_genetic_cv.py).
# Any filter without stored params will fall back to its built-in DEFAULT_PARAMS.
def _load_tuned_params(nav_data, mode_3d):
    """
    Load tuned params for the current dataset.

    Priority:
    1. LOO key "__loo_held_<dataset_name>__" — from ins_genetic_cv.py --held-out
    2. Per-dataset key "<dataset_name>" — from ins_genetic.py
    3. CV aggregate key "__cv_kitti__" — from ins_genetic_cv.py without --held-out

    DL filters (tlio, deep_kf, tartan_imu) share the same classical-filter
    parameter names (Qpos, Qvel, Rpos, P_pos_std, …) as the ESKF/EKF.  When
    no DL-specific tuned params exist, derive them from the best available
    classical filter (eskf_enhanced → eskf_vanilla → ekf_enhanced → ekf_vanilla).
    """
    result = {}
    loo_key = f'__loo_held_{nav_data.dataset_name}__'
    cv_key  = '__cv_kitti__'

    for key in ['ekf_vanilla', 'ekf_enhanced', 'eskf_vanilla', 'eskf_enhanced',
                'iekf_vanilla', 'iekf_enhanced']:
        p = (fp.get(key, mode_3d, loo_key)
             or fp.get(key, mode_3d, nav_data.dataset_name)
             or fp.get(key, mode_3d, cv_key))
        if p is not None:
            result[key] = p

    # ── Derive DL-filter classical params from best tuned classical filter ────
    # TLIO / Deep-KF / Tartan-IMU all have a traditional KF layer whose
    # parameters (Qpos, Qvel, Rpos, P_pos_std …) are identical in name to
    # the ESKF/EKF tuned set.  Use the best available tuned classical filter
    # so DL filters benefit from genetic optimisation without a separate run.
    _best_classical = (result.get('eskf_enhanced')
                       or result.get('eskf_vanilla')
                       or result.get('ekf_enhanced')
                       or result.get('ekf_vanilla'))

    if _best_classical is not None:
        # Keys shared directly between classical filters and DL filter params
        _shared = ['Qpos', 'Qvel', 'Qacc', 'Rpos',
                   'beta_acc', 'beta_gyr',
                   'P_pos_std', 'P_vel_std', 'P_orient_std',
                   'P_acc_std', 'P_gyr_std']
        _dl_base = {k: _best_classical[k] for k in _shared if k in _best_classical}

        # Anisotropic orient / gyro noise: average X/Y and Z components
        if 'QorientXY' in _best_classical:
            _dl_base['Qorient'] = (
                _best_classical.get('QorientXY', 1e-5) +
                _best_classical.get('QorientZ', 1e-5)) / 2.0
        if 'QgyrXY' in _best_classical:
            _dl_base['Qgyr'] = (
                _best_classical.get('QgyrXY', 1e-7) +
                _best_classical.get('QgyrZ', 1e-7)) / 2.0

        for dl_key in ['tlio', 'deep_kf', 'tartan_imu']:
            if dl_key not in result:   # don't overwrite if already tuned directly
                result[dl_key] = _dl_base.copy()

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
    # Deep learning filters
    {'name': 'IEKF AI-IMU',  'key': 'iekf_ai_imu', 'module': iekf_ai_imu},
    {'name': 'TLIO',         'key': 'tlio',         'module': tlio_runner},
    {'name': 'Deep KF',      'key': 'deep_kf',      'module': deep_kf_runner},
    {'name': 'Tartan IMU',   'key': 'tartan_imu',   'module': tartan_runner},
    # Online smoothers
    {'name': 'iSAM2',        'key': 'isam2',        'module': isam2_runner},
]


def main():
    # ── CLI arguments ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Run all filter variants and generate comparison plots.')
    parser.add_argument('--test-seq', type=str, default=None,
                        help='Override dataset: KITTI seq ID ("01"–"10") or full drive name')
    parser.add_argument('--dr-mode', action='store_true', default=False,
                        help='Dead-reckoning mode: disable GPS for all filters '
                             '(trel/rrel comparable to Brossard et al. 2020 Table I)')
    parser.add_argument('--ai-imu-weights', type=str, default=None,
                        help='Path to fold-specific AI-IMU weights (iekfnets_held_*.p). '
                             'Auto-detected from --test-seq when not provided.')
    args = parser.parse_args()

    # ── Shared configuration ───────────────────────────────────────────────────
    if args.test_seq is not None:
        nav_data = data_loader.get_kitti_dataset(args.test_seq)
    else:
        nav_data = ins_config.NAV_DATA

    use_3d        = ins_config.MODE_3D
    dr_mode       = args.dr_mode or getattr(ins_config, 'DR_MODE', False)
    use_rts_as_gt = ins_config.USE_RTS_AS_GT and not dr_mode
    TUNED_PARAMS  = _load_tuned_params(nav_data, use_3d)
    t1       = ins_config.OUTAGE_START
    d        = ins_config.OUTAGE_DURATION

    lla      = nav_data.lla
    lla0     = nav_data.lla0
    frecIMU  = nav_data.sample_rate

    f        = pm.geodetic2enu(lla[:,0], lla[:,1], lla[:,2], lla0[0], lla0[1], lla0[2])
    p_kitti  = np.column_stack([f[0], f[1], f[2]])

    # In DR_MODE: disable GPS for all filters; outage simulation is irrelevant
    if dr_mode:
        import dataclasses
        nav_data = dataclasses.replace(
            nav_data,
            gps_available=np.zeros(len(nav_data.lla), dtype=bool),
        )
        outage_config    = None
        gnss_outage_info = {'start': 0, 'end': 0, 'duration': 0,
                            'start_idx': 0, 'end_idx': 0}
    else:
        outage_config = {'start': t1, 'duration': d}
        gnss_outage_info = {
            'start':     t1, 'end': t1 + d, 'duration': d,
            'start_idx': int(t1 * frecIMU),
            'end_idx':   int((t1 + d) * frecIMU),
        }

    # ── AI-IEKF weights path (LOO fold-specific) ───────────────────────────────
    _repo_root = _HERE / '../../../..'
    if args.ai_imu_weights:
        _ai_weights = Path(args.ai_imu_weights)
    else:
        _drive = data_loader.KITTI_SEQ_TO_DRIVE.get(
            nav_data.dataset_name, nav_data.dataset_name)
        _candidate = _repo_root / f'artifacts/deep_iekf/iekfnets_held_{_drive}.p'
        _ai_weights = _candidate if _candidate.exists() else None

    # ── Output directories ─────────────────────────────────────────────────────
    if dr_mode:
        traj_subdir = f"{nav_data.dataset_name}_dr_mode"
    elif t1 > 0 or d > 0:
        traj_subdir = f"{nav_data.dataset_name}_outage_{t1}s_{d}s"
    else:
        traj_subdir = f"{nav_data.dataset_name}_no_outage"
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
    logger.info(f"Mode       : {'DEAD-RECKONING (DR_MODE)' if dr_mode else f'GPS-aided, outage {t1}s–{t1+d}s ({d}s)'}")
    logger.info(f"Rotation   : {'3D' if use_3d else '2D'}")
    logger.info(f"Filters    : {len(FILTER_CONFIGS)}")
    if _ai_weights:
        logger.info(f"AI weights : {_ai_weights}")
    logger.info("")
    logger.info("NOTE: Results use default parameters unless TUNED_PARAMS is set.")
    logger.info("      Run ins_genetic.py for each filter before comparing.")
    logger.info("=" * 65)

    # ── RTS Smoother — skipped in DR_MODE (no GPS available) ──────────────────
    if dr_mode:
        p_rts     = None
        p_rts_vis = None
        p_gt      = p_kitti
        gt_label  = 'KITTI GPS GT'
        logger.info("DR_MODE: skipping RTS smoother; ground truth = KITTI GPS")
    else:
        # ── RTS Smoother (best-achievable reference, always full GPS) ──────────
        logger.info("Running RTS smoother (best-achievable reference, no outage)…")
        rts_params = TUNED_PARAMS.get('ekf_enhanced', None)
        rts_result = rts_smoother.run(
            nav_data=nav_data,
            params=rts_params,
            use_3d_rotation=use_3d,
        )
        p_rts = rts_result['p']
        logger.info("  RTS smoother done.")

        # ── Select ground truth source ─────────────────────────────────────────
        # p_gt        : array used for metric evaluation and as the GT line in plots
        # p_rts_vis   : optional second reference overlaid in plots (purple); None
        #               when RTS is already the primary GT
        if use_rts_as_gt:
            p_gt       = p_rts
            p_rts_vis  = None
            gt_label   = 'Ground Truth (RTS)'
            logger.info("Ground truth: RTS smoother")
        else:
            p_gt       = p_kitti
            p_rts_vis  = p_rts
            gt_label   = 'KITTI GPS GT'
            logger.info("Ground truth: KITTI GPS")

    # ── Run all filters ────────────────────────────────────────────────────────
    all_results = []

    for cfg in FILTER_CONFIGS:
        fname  = cfg['name']
        fkey   = cfg['key']
        fmod   = cfg['module']
        params = TUNED_PARAMS.get(fkey, None)

        logger.info(f"\nRunning {fname}  ({'tuned' if params else 'default params'})…")

        # For AI-IEKF: inject fold-specific weights via env var if available
        if fkey == 'iekf_ai_imu' and _ai_weights is not None:
            os.environ['AI_IMU_WEIGHTS'] = str(_ai_weights)
        elif fkey == 'iekf_ai_imu':
            os.environ.pop('AI_IMU_WEIGHTS', None)

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

        # KITTI t_rel / r_rel — always vs raw KITTI GPS, full sequence (no outage)
        kitti_mets = metrics.compute_kitti_metrics(
            p_est=p, r_est=r,
            p_gt=p_kitti, r_gt=nav_data.orient,
        )
        logger.info(f"  t_rel = {kitti_mets['t_rel']:.2f} %  |  r_rel = {kitti_mets['r_rel']:.2f} deg/km"
                    f"  (n_seg={kitti_mets['n_segments']})")

        entry = {
            'name':        fname,
            'key':         fkey,
            'p':           p,
            'v':           v,
            'r':           r,
            'std_pos':     result['std_pos'],
            'metrics':     mets,
            'kitti_mets':  kitti_mets,
            'result':      result,
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
            'kitti_metrics': kitti_mets,
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
        p_rts=p_rts_vis,
        gt_label=gt_label,
        gnss_outage_info=gnss_outage_info,
        sample_rate=frecIMU,
        output_dir=str(compare_dir),
        lla0=nav_data.lla0,
    )

    logger.info(f"Generated {len(generated)} comparison plots in: {compare_dir}")
    logger.info(f"Log: {log_file}")

    # ── Summary table to console ───────────────────────────────────────────────
    _mode_str = 'DR mode (no GPS)' if dr_mode else f'Outage {t1}s+{d}s'
    print("\n" + "=" * 100)
    print(f"COMPARISON SUMMARY — {nav_data.dataset_name}  |  "
          f"{_mode_str}  |  {'3D' if use_3d else '2D'}")
    print("=" * 100)
    print(f"{'Filter':<20} {'ATE 2D [m]':>12} {'ATE 3D [m]':>12} {'Pos RMSE 2D':>13} "
          f"{'Outage max [m]':>16} {'t_rel [%]':>11} {'r_rel [°/km]':>13}")
    print("-" * 100)
    for entry in all_results:
        mets   = entry['metrics']
        km     = entry['kitti_mets']
        ate2d  = mets.get('ate', {}).get('rmse_2D',      float('nan'))
        ate3d  = mets.get('ate', {}).get('rmse_3D',      float('nan'))
        prmse  = mets.get('position_rmse', {}).get('2D', float('nan'))
        outagm = mets.get('outage_analysis', {}).get('max', float('nan'))
        trel   = km.get('t_rel', float('nan'))
        rrel   = km.get('r_rel', float('nan'))
        print(f"{entry['name']:<20} {ate2d:>12.2f} {ate3d:>12.2f} {prmse:>13.2f} "
              f"{outagm:>16.2f} {trel:>11.2f} {rrel:>13.2f}")
    print("=" * 100)
    print("NOTE: Tune parameters with ins_genetic.py before drawing conclusions.\n"
          "      t_rel / r_rel are KITTI odometry metrics (full sequence, vs raw GPS).\n")


if __name__ == '__main__':
    main()
