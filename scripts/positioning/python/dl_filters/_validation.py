"""
_validation.py — Shared journal-metric validation hook for DL training scripts.

The classical-filter genetic pipeline selects parameters by minimising the
three-component normalised cost defined in ins_cost.py:

    J = ATE_outage / 1 m + t_rel / 1 % + r_rel / 1 deg/km

For like-for-like comparison with the classical filters at journal-review
time, each DL training script reports this same J on the held-out sequence
during training. The hook is DISPLAY-ONLY — it never enters backprop. Each
filter keeps its original published loss as the optimisation objective.

Usage in a training script
--------------------------

    from dl_filters._validation import validate_with_journal_metric, format_val_line

    # ... at end-of-epoch ...
    if val_every and (epoch % val_every == 0 or epoch == n_epochs - 1):
        model.eval()
        with torch.no_grad():
            metrics = validate_with_journal_metric(
                filter_module=tlio_runner,        # or deep_kf_runner / tartan_runner / iekf_ai_imu
                val_seq=args.val_seq,             # '01' .. '10'
                outage_start=40.0,                # match tab:kitti_benchmark_60
                outage_duration=60.0,
            )
        print(format_val_line(epoch, metrics))
        model.train()
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure positioning/python/ is on sys.path so ins_cost / data_loader / metrics import.
_HERE = Path(__file__).resolve().parent
_POSITIONING_PY = _HERE.parent
if str(_POSITIONING_PY) not in sys.path:
    sys.path.insert(0, str(_POSITIONING_PY))


def validate_with_journal_metric(
    filter_module,
    val_seq: str,
    *,
    outage_start: float = 40.0,
    outage_duration: float = 60.0,
    params: dict | None = None,
    use_3d: bool = True,
) -> dict:
    """
    Run one inference pass on the held-out KITTI sequence with the requested
    GNSS outage applied and return the three-component journal cost plus its
    components and ATE_no_outage as diagnostics.

    Parameters
    ----------
    filter_module : module
        Any of the DL runners — must expose
        run(nav_data, params, outage_config, use_3d_rotation) -> dict with
        keys 'p', 'r', 'std_pos'.
    val_seq : str
        KITTI sequence id ('01', '04', '06', '07', '08', '09', '10').
    outage_start, outage_duration : float
        Seconds. Defaults match tab:kitti_benchmark_60 (40 s start, 60 s window).
    params : dict, optional
        Filter parameters passed through to filter_module.run().
    use_3d : bool
        Forwarded to filter_module.run() as use_3d_rotation.

    Returns
    -------
    dict with keys:
        'J'                 : float  — three-component normalised cost (outage window)
        'ate_outage_m'      : float  — RMSE position error inside the outage [m]
        't_rel_pct'         : float  — KITTI relative translation error [%]
        'r_rel_deg_per_km'  : float  — KITTI relative rotation error [deg/km]
        'anees'             : float  — average NEES on the GNSS-aided phase
        'ate_no_outage_m'   : float  — RMSE position error on the GNSS-aided phase [m]
        'val_seq'           : str
        'outage_start_s'    : float
        'outage_duration_s' : float

    On any failure, 'J' is set to float('inf') and a 'reason' key is included.
    """
    import numpy as np
    import data_loader
    import ins_cost

    nav_data = data_loader.get_kitti_dataset(val_seq)

    components = ins_cost.single_window_cost(
        filter_module, nav_data, params or {},
        float(outage_start), float(outage_duration), use_3d,
        return_components=True, gate_anees=False,
    )

    if components.get('cost', ins_cost.COST_REJECT) >= ins_cost.COST_REJECT:
        return {
            'J':                 float('inf'),
            'ate_outage_m':      float('nan'),
            't_rel_pct':         float('nan'),
            'r_rel_deg_per_km':  float('nan'),
            'anees':             float('nan'),
            'ate_no_outage_m':   float('nan'),
            'val_seq':           val_seq,
            'outage_start_s':    float(outage_start),
            'outage_duration_s': float(outage_duration),
            'reason':            components.get('reason', 'unknown'),
        }

    # ATE on the GPS-aided phase (i.e. outside the outage window) — useful
    # diagnostic that distinguishes "good baseline, bad DR" from "bad both".
    sr = float(nav_data.sample_rate)
    A = int(outage_start * sr)
    B = int((outage_start + outage_duration) * sr)

    p_gt = ins_cost._enu_ground_truth(nav_data)
    res = filter_module.run(nav_data, params or {},
                            {'start': float(outage_start),
                             'duration': float(outage_duration)},
                            use_3d)
    p = res['p']
    pos_err = p - p_gt
    mask = np.ones(len(p_gt), dtype=bool)
    mask[A:B] = False
    err_in = np.sqrt(pos_err[mask, 0]**2 + pos_err[mask, 1]**2 + pos_err[mask, 2]**2)
    ate_no_outage = float(np.sqrt(np.mean(err_in**2))) if err_in.size else float('nan')

    return {
        'J':                 float(components['cost']),
        'ate_outage_m':      float(components['ate_outage']),
        't_rel_pct':         float(components['t_rel']),
        'r_rel_deg_per_km':  float(components['r_rel']),
        'anees':             float(components['anees']),
        'ate_no_outage_m':   ate_no_outage,
        'val_seq':           val_seq,
        'outage_start_s':    float(outage_start),
        'outage_duration_s': float(outage_duration),
    }


def format_val_line(epoch: int, metrics: dict) -> str:
    """One-line formatter for the per-epoch [val] log line."""
    if metrics.get('reason'):
        return (f"[val] epoch={epoch}  seq={metrics['val_seq']}  J=inf  "
                f"reason={metrics['reason']}")
    return (
        f"[val] epoch={epoch}  seq={metrics['val_seq']}  "
        f"J={metrics['J']:.3f}  "
        f"ATE_out={metrics['ate_outage_m']:.2f}m  "
        f"t_rel={metrics['t_rel_pct']:.2f}%  "
        f"r_rel={metrics['r_rel_deg_per_km']:.2f}deg/km  "
        f"ATE_nooutage={metrics['ate_no_outage_m']:.2f}m  "
        f"ANEES={metrics['anees']:.2f}"
    )
