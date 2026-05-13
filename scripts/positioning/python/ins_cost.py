# -*- coding: utf-8 -*-
"""
ins_cost.py — Shared cost function and parameter decoding for the GA tuners.

The single GA driver in this directory (`ins_genetic_cv.py`) imports its
parameter decoding, search bounds, and cost function from here. Centralising
these keeps the journal-grade definition of "cost" in one place; the GA
driver only orchestrates LOO splits, parallel workers, and result I/O.

Journal-grade cost design
-------------------------

Each per-(filter, dataset) cost is a sum of three normalised metrics, each of
which is reported in the results tables:

    J = ATE_outage / ATE_REF  +  t_rel / T_REL_REF  +  r_rel / R_REL_REF

The normalisers are NOT hand-tuned weights — they are operational thresholds:

  * ATE_REF    = 1.0  m       — NHTSA / ISO lane-keeping operational accuracy
  * T_REL_REF  = 1.0  %       — KITTI leaderboard "elite" cutoff (Geiger 2012)
  * R_REL_REF  = 1.0  deg/km  — KITTI leaderboard "elite" cutoff

Each component value of 1 in J corresponds to "one unit of acceptable
degradation" — a reviewer can disagree with the threshold but cannot accuse
us of arbitrary scalarisation.

Filter-consistency constraint
-----------------------------

A filter that minimises J at the cost of grossly underestimated covariance
will fail in real-world deployment (downstream consumers receive bad
uncertainty bounds). To prevent the GA from gaming the benchmark this way,
the cost is capped to 1e6 (rejection) when the Average Normalized Estimation
Error Squared (ANEES) lies outside [0.1, 10] — a 10× consistency band around
the chi-squared expectation of 1.0. Reference: Bar-Shalom Ch. 5; Chen et al.,
*Kalman Filter Auto-tuning through Enforcing Chi-Squared Normalized Error
Distributions with Bayesian Optimization*, arXiv:2306.07225, 2023.

Multi-window outage averaging
-----------------------------

The fitness for one (filter, dataset) pair is the mean of J across a small
set of GNSS-outage windows (default: starts at {40, 100, 160} s, all 60 s
long). This prevents the GA from over-tuning to a single dead-reckoning
window inside a specific trajectory — important for real-world transfer.
Windows that exceed the trajectory length are skipped automatically.

Parameter-space conventions
---------------------------

Filter parameters are searched in log10 space; `decode_params(x)` converts
the 15-D vector back into the dict each filter's `run()` expects.
"""
from __future__ import annotations

import numpy as np
import pymap3d as pm

# ── Search bounds (log10) — shared across all GA drivers ──────────────────────
BOUNDS = [
    (-2, 2),     # log10(Qpos)
    (-2, 2),     # log10(Qvel)
    (-5, -1),    # log10(QorientXY)
    (-2, 1),     # log10(QorientZ)
    (-3, 0),     # log10(Qacc)
    (-6, -2),    # log10(QgyrXY)
    (-3, 0),     # log10(QgyrZ)
    (-1, 2),     # log10(Rpos)
    (-8, -5),    # log10(|beta_acc|)  — stored negative
    (-2, 1),     # log10(|beta_gyr|)  — stored negative
    (-1, 1.5),   # log10(P_pos_std)
    (-1, 0.5),   # log10(P_vel_std)
    (-2, -0.5),  # log10(P_orient_std)
    (-3, -1),    # log10(P_acc_std)
    (-4, -2),    # log10(P_gyr_std)
]

# ── Cost-function constants (KITTI-leaderboard operational thresholds) ────────
ATE_REF   = 1.0   # m       — NHTSA / ISO lane-keeping threshold
T_REL_REF = 1.0   # %       — KITTI elite cutoff (Geiger 2012)
R_REL_REF = 1.0   # deg/km  — KITTI elite cutoff

# ── Filter-consistency band for ANEES (chi² mean = 1, allow 10× spread) ──────
ANEES_LO  = 0.1
ANEES_HI  = 10.0

# ── Default outage-window set for fitness eval (multi-window averaging) ──────
DEFAULT_OUTAGE_STARTS = (40.0, 100.0, 160.0)
DEFAULT_OUTAGE_DURATION = 60.0

# ── Maximum admissible cost before rejection ─────────────────────────────────
COST_REJECT = 1e6


def decode_params(x):
    """Decode the 15-D log10 vector into the filter `params` dict."""
    return {
        'Qpos':         10**x[0],
        'Qvel':         10**x[1],
        'QorientXY':    10**x[2],
        'QorientZ':     10**x[3],
        'Qacc':         10**x[4],
        'QgyrXY':       10**x[5],
        'QgyrZ':        10**x[6],
        'Rpos':         10**x[7],
        'beta_acc':    -10**x[8],
        'beta_gyr':    -10**x[9],
        'P_pos_std':    10**x[10],
        'P_vel_std':    10**x[11],
        'P_orient_std': 10**x[12],
        'P_acc_std':    10**x[13],
        'P_gyr_std':    10**x[14],
    }


def _enu_ground_truth(nav_data):
    """Convert geodetic ground truth to ENU."""
    f = pm.geodetic2enu(nav_data.lla[:, 0], nav_data.lla[:, 1], nav_data.lla[:, 2],
                        nav_data.lla0[0], nav_data.lla0[1], nav_data.lla0[2])
    return np.column_stack([f[0], f[1], f[2]])


def _kitti_metrics(p_est, r_est, p_gt, r_gt):
    """Wrapper around metrics.compute_kitti_metrics that returns scalars only."""
    # Imported lazily to avoid a hard dependency at module import time.
    from metrics import compute_kitti_metrics
    out = compute_kitti_metrics(p_est, r_est, p_gt, r_gt)
    return float(out.get('t_rel', float('nan'))), float(out.get('r_rel', float('nan')))


def _anees(pos_err, std_pos, A, B, N, stride=10, eps=1e-12):
    """Average NEES over the GPS-aided phase (epochs outside [A:B])."""
    ns, nc = 0.0, 0
    for k in range(0, N, stride):
        if A <= k < B:
            continue
        var_p = std_pos[k]**2 + eps
        ns += float(np.sum(pos_err[k]**2 / var_p))
        nc += 1
    return ns / max(nc, 1) / 3.0    # 3 DOF for position


def single_window_cost(filter_module, nav_data, params, t1, d, use_3d,
                      *, return_components=False):
    """
    Run one filter on one (dataset, outage-window) pair and return the cost.

    Cost = ATE_outage / ATE_REF + t_rel / T_REL_REF + r_rel / R_REL_REF
    Reject (return COST_REJECT) when:
      * filter throws an exception
      * cost is non-finite or > COST_REJECT/100
      * ANEES outside [ANEES_LO, ANEES_HI]   ← consistency constraint

    If `return_components=True` returns a dict with raw component values
    (used by loggers / post-tune diagnostic reports).
    """
    try:
        frecIMU = nav_data.sample_rate
        N_total = len(nav_data.lla)

        A = int(t1 * frecIMU)
        B = int((t1 + d) * frecIMU)

        # Skip the window if it doesn't fit in the trajectory.
        if B >= N_total or A < 0 or A >= B:
            if return_components:
                return {'cost': COST_REJECT, 'reason': 'window_too_long'}
            return COST_REJECT

        p_gt = _enu_ground_truth(nav_data)

        res = filter_module.run(nav_data, params,
                                {'start': t1, 'duration': d}, use_3d)

        p = res['p']
        r = res['r']
        std_pos = res['std_pos']

        pos_err = p - p_gt

        # ATE during the outage window only — operationally the most
        # important number (filter is dead-reckoning here).
        err_out = np.sqrt(pos_err[A:B, 0]**2
                        + pos_err[A:B, 1]**2
                        + pos_err[A:B, 2]**2)
        ate_outage = float(np.sqrt(np.mean(err_out**2)))

        # KITTI relative metrics (whole trajectory, official KITTI segments)
        t_rel, r_rel = _kitti_metrics(p, r, p_gt, nav_data.orient)

        # Filter-consistency check (ANEES on the GPS-aided phase only)
        anees = _anees(pos_err, std_pos, A, B, len(p_gt))
        in_band = ANEES_LO <= anees <= ANEES_HI

        cost = (ate_outage / ATE_REF
              + t_rel      / T_REL_REF
              + r_rel      / R_REL_REF)

        if not in_band:
            if return_components:
                return {
                    'cost': COST_REJECT,
                    'reason': 'anees_out_of_band',
                    'anees': anees,
                    'ate_outage': ate_outage,
                    't_rel': t_rel, 'r_rel': r_rel,
                }
            return COST_REJECT

        if not np.isfinite(cost) or cost > COST_REJECT / 100:
            if return_components:
                return {'cost': COST_REJECT, 'reason': 'non_finite_or_huge'}
            return COST_REJECT

        if return_components:
            return {
                'cost':       float(cost),
                'ate_outage': ate_outage,
                't_rel':      t_rel,
                'r_rel':      r_rel,
                'anees':      anees,
            }
        return float(cost)

    except Exception as exc:
        if return_components:
            return {'cost': COST_REJECT, 'reason': f'exception:{exc}'}
        return COST_REJECT


def fitness(filter_module, nav_data, params, use_3d,
            outage_starts=DEFAULT_OUTAGE_STARTS,
            outage_duration=DEFAULT_OUTAGE_DURATION):
    """
    Multi-window fitness for one (filter, dataset) pair.

    Returns the mean of `single_window_cost` across the valid outage windows.
    A window is "valid" when it fits within the trajectory; when fewer than
    one window fits the function returns COST_REJECT.
    """
    costs = []
    for t1 in outage_starts:
        c = single_window_cost(filter_module, nav_data, params, t1,
                               outage_duration, use_3d)
        if c >= COST_REJECT:
            # Hard rejection on any window keeps the filter consistent across
            # outage placements — better to reject than to mask a bad window.
            return COST_REJECT
        costs.append(c)
    if not costs:
        return COST_REJECT
    return float(np.mean(costs))
