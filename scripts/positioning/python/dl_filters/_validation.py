"""
_validation.py — Shared journal-metric validation hook for DL training scripts.

The classical-filter genetic pipeline selects parameters by minimising the
three-component normalised cost defined in ins_cost.py:

    J = ATE_outage / 1 m + t_rel / 1 % + r_rel / 1 deg/km

For like-for-like comparison with the classical filters at journal-review
time, each DL training script reports this same J on the held-out sequence
during training. The hook is DISPLAY-ONLY — it never enters backprop. Each
filter keeps its original published loss as the optimisation objective.

Ground truth for J is FGO-Batch (via `ins_cost.get_fgo_batch_gt`), matching
what the final benchmark (ins_compare.py, GT_SOURCE='batch') scores against —
not raw KITTI OXTS.

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


# Sequences too short to be a meaningful inner validation set. Measured on the
# KITTI clean drives: seq 04 is 2967 samples / 29.7 s, every other clean drive is
# 11363-53663 samples / 113.6-536.6 s. Still trained on — just never the validator.
SHORT_SEQS = frozenset({'04'})


def inner_split(clean_seqs: list, held_out: str) -> tuple:
    """Split the LOO training pool into (train_seqs, inner_val_seq).

    Why this exists
    ---------------
    In a leave-one-out study the held-out sequence is the TEST set. Using it to
    pick the epoch to deploy — `if val_loss < best_val: save()` — or to drive
    ReduceLROnPlateau is model selection on the test set, and it measurably
    inflated the DL rows: TLIO's folds were being frozen at epochs 10/11/12 of
    50 and Tartan's fold 08 at epoch 3, whichever epoch happened to score best
    on the very sequence the paper then reports. The seven classical filters do
    not get that: ins_genetic_cv.py excludes the held-out drive from the
    objective entirely (`list_kitti_datasets(held_out=...)`), and deep_iekf
    deploys its final epoch. So the comparison was not like-for-like.

    The fix is a nested split: one sequence is carved out of the six TRAINING
    sequences to serve as the inner validation set, and every fitting decision
    (best-checkpoint, LR schedule, early stopping) is made against that. The
    held-out sequence is then never loaded during training at all.

    The rule
    --------
    Deterministic rotation: the inner validation sequence is the first sequence
    FOLLOWING the held-out one in the canonical clean-sequence list (cyclically)
    that is not in SHORT_SEQS. It has to be reproducible and stateable in the
    paper, and rotating spreads the role across drives instead of always
    spending the same one:

        held-out 01 -> inner val 06     held-out 08 -> inner val 09
        held-out 04 -> inner val 06     held-out 09 -> inner val 10
        held-out 06 -> inner val 07     held-out 10 -> inner val 01
        held-out 07 -> inner val 08

    SHORT_SEQS exists because KITTI seq 04 is 29.7 s against 113.6-536.6 s for
    every other clean drive. Plain rotation would have made it fold 01's inner
    validation set, i.e. selected TLIO's deployed epoch on 574 windows — swapping
    a biased selection signal for a merely noisy one. It stays a training
    sequence throughout; only its turn as validator is skipped.

    Returns (train_seqs, inner_val_seq).
    """
    if held_out not in clean_seqs:
        raise ValueError(f"held_out {held_out!r} not in {clean_seqs}")
    if len(clean_seqs) < 3:
        raise ValueError(
            f"need >=3 sequences to hold one out AND keep an inner validation "
            f"sequence, got {len(clean_seqs)}: {clean_seqs}")
    n = len(clean_seqs)
    i = clean_seqs.index(held_out)
    inner_val = None
    for k in range(1, n):
        cand = clean_seqs[(i + k) % n]
        if cand not in SHORT_SEQS:
            inner_val = cand
            break
    if inner_val is None:      # every other sequence is short — take the next one
        inner_val = clean_seqs[(i + 1) % n]
    train_seqs = [s for s in clean_seqs if s not in (held_out, inner_val)]
    return train_seqs, inner_val


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
        'val_seq'           : str
        'outage_start_s'    : float
        'outage_duration_s' : float

    On any failure, 'J' is set to float('inf') and a 'reason' key is included.
    """
    import data_loader
    import ins_cost

    nav_data = data_loader.get_kitti_dataset(val_seq)
    gt = ins_cost.get_fgo_batch_gt(nav_data)

    components = ins_cost.single_window_cost(
        filter_module, nav_data, params or {},
        float(outage_start), float(outage_duration), use_3d,
        return_components=True, gate_anees=False, gt=gt,
    )

    if components.get('cost', ins_cost.COST_REJECT) >= ins_cost.COST_REJECT:
        return {
            'J':                 float('inf'),
            'ate_outage_m':      float('nan'),
            't_rel_pct':         float('nan'),
            'r_rel_deg_per_km':  float('nan'),
            'anees':             float('nan'),
            'val_seq':           val_seq,
            'outage_start_s':    float(outage_start),
            'outage_duration_s': float(outage_duration),
            'reason':            components.get('reason', 'unknown'),
        }

    return {
        'J':                 float(components['cost']),
        'ate_outage_m':      float(components['ate_outage']),
        't_rel_pct':         float(components['t_rel']),
        'r_rel_deg_per_km':  float(components['r_rel']),
        'anees':             float(components['anees']),
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
        f"ANEES={metrics['anees']:.2f}"
    )
