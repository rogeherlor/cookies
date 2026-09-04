# -*- coding: utf-8 -*-
"""
Incremental vs. Fixed-Lag iSAM2 — timing / compute study
========================================================
Runs the full-history incremental smoother (isam2_runner) and the 2-minute
fixed-lag smoother (isam2_fixedlag_runner) on the KITTI clean sequences and
records, per GPS epoch:

    update_ms : wall-time of the update()+calculateEstimate() block [ms].
    n_vars    : number of variables in the estimate (memory / compute proxy).

For each sequence it writes a CSV and a two-panel figure (update time and
variable count vs. epoch) with the 1 Hz real-time budget line (1000 ms), and
prints a summary table.

The variable-count curve is the robust, implementation-independent evidence:
the incremental estimate grows linearly with sequence length while the fixed-lag
one stays bounded by the window.  Update time is secondary (the incremental
runner performs three solver iterations per epoch vs. the fixed-lag's one).

Run (conda cookies env):
    PYTHONPATH=scripts/positioning/python \
    /home/ceiur/miniconda3/envs/cookies/bin/python \
    scripts/positioning/python/smoothers/bench_fixedlag_timing.py
"""

import sys
import argparse
import csv
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import data_loader
from smoothers import isam2_runner, isam2_fixedlag_runner

RT_BUDGET_MS = 1000.0          # 1 Hz GPS-rate update budget
CLEAN_SEQS   = ['01', '04', '06', '07', '08', '09', '10']


def _run_one(mod, nav, outage):
    out = mod.run(nav, params=None, outage_config=outage, use_3d_rotation=True)
    um = np.asarray(out['update_ms'])
    nv = np.asarray(out['n_vars'])
    mask = nv > 0                       # keep only GPS-update epochs
    idx = np.nonzero(mask)[0]
    return idx, um[mask], nv[mask]


def _summ(name, um, nv):
    return (f"  {name:16s} updates={len(um):5d}  "
            f"upd_ms mean={um.mean():6.2f} med={np.median(um):6.2f} "
            f"max={um.max():7.2f}  n_vars final={int(nv[-1]):5d} max={int(nv.max()):5d}  "
            f"over_budget={int((um > RT_BUDGET_MS).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', nargs='+', default=CLEAN_SEQS)
    ap.add_argument('--outage-start', type=float, default=0.0,
                    help='0 disables the outage (cleaner growth curve)')
    ap.add_argument('--outage-duration', type=float, default=0.0)
    ap.add_argument('--outdir', default=str(Path(__file__).parent / 'timing_study'))
    args = ap.parse_args()

    outage = None
    if args.outage_start > 0 and args.outage_duration > 0:
        outage = {'start': args.outage_start, 'duration': args.outage_duration}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        have_mpl = True
    except Exception as e:
        print(f"[warn] matplotlib unavailable ({e}); writing CSV only")
        have_mpl = False

    print(f"Outage: {outage}   outdir: {outdir}\n")
    for seq in args.seqs:
        nav = data_loader.get_kitti_dataset(seq)
        dur = nav.accel_flu.shape[0] / nav.sample_rate
        print(f"KITTI {seq}  ({dur:.0f}s / {dur/60:.1f} min)")

        i_inc, um_inc, nv_inc = _run_one(isam2_runner, nav, outage)
        i_fl,  um_fl,  nv_fl  = _run_one(isam2_fixedlag_runner, nav, outage)
        print(_summ("incremental", um_inc, nv_inc))
        print(_summ("fixed-lag 2min", um_fl, nv_fl))

        # CSV
        csv_path = outdir / f'timing_seq{seq}.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch_inc', 'update_ms_inc', 'n_vars_inc'])
            for e, u, n in zip(i_inc, um_inc, nv_inc):
                w.writerow([int(e), f'{u:.4f}', int(n)])
            w.writerow([])
            w.writerow(['epoch_fl', 'update_ms_fl', 'n_vars_fl'])
            for e, u, n in zip(i_fl, um_fl, nv_fl):
                w.writerow([int(e), f'{u:.4f}', int(n)])

        # Figure
        if have_mpl:
            t_inc = i_inc / nav.sample_rate
            t_fl  = i_fl / nav.sample_rate
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
            a1.plot(t_inc, um_inc, label='incremental', lw=1.0)
            a1.plot(t_fl,  um_fl,  label='fixed-lag 2 min', lw=1.0)
            a1.axhline(RT_BUDGET_MS, color='r', ls='--', lw=0.8,
                       label='1 Hz budget (1000 ms)')
            a1.set_xlabel('time [s]'); a1.set_ylabel('update time [ms]')
            a1.set_title(f'KITTI {seq}: per-update time'); a1.legend(); a1.grid(alpha=.3)
            a2.plot(t_inc, nv_inc, label='incremental', lw=1.2)
            a2.plot(t_fl,  nv_fl,  label='fixed-lag 2 min', lw=1.2)
            a2.set_xlabel('time [s]'); a2.set_ylabel('variables in estimate')
            a2.set_title(f'KITTI {seq}: state size'); a2.legend(); a2.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(outdir / f'timing_seq{seq}.png', dpi=130)
            plt.close(fig)
        print()

    print(f"Wrote CSVs/figures to {outdir}")


if __name__ == '__main__':
    main()
