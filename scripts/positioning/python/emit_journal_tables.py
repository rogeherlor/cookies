# -*- coding: utf-8 -*-
"""
emit_journal_tables.py — Aggregate per-(filter, seq, regime) JSONs into the
two .tex tables that go into 3.tex.

Reads the result JSONs produced by `ins_compare.py` (one per filter, sequence,
and regime), pulls (ATE [m], t_rel [%], r_rel [deg/km]) from each, and writes:

    outputs/tables/no_outage_<dataset>.tex
    outputs/tables/outage_60s_<dataset>.tex

Each .tex file is a tabular body that can be `\\input{}`'d into 3.tex.

Usage
-----
  # After running ./run_genetic_loo.sh && ./run_journal_evaluation.sh:
  python emit_journal_tables.py                       # KITTI by default
  python emit_journal_tables.py --dataset cookies     # cookies tables
  python emit_journal_tables.py --no-outage-start 0 --outage-start 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE      = Path(__file__).parent
_REPO_ROOT = _HERE / '../../..'

# Filter keys reported in the journal tables (order matters: it dictates row order).
DEFAULT_FILTERS = [
    'esekfg_vanilla',
    'esekfg_enhanced',
    'esekfs_vanilla',
    'esekfs_enhanced',
    'iekf_vanilla',
    'iekf_enhanced',
    'imu_only',
    'iekf_ai_imu',
    'tlio',
    'deep_kf',
    'tartan_imu',
]

KITTI_SEQS    = ['01', '04', '06', '07', '08', '09', '10']
COOKIES_SEQS  = ['c01', 'c02', 'c03', 'c04', 'c05']


def _result_json_path(filter_key: str, seq_id: str,
                      outage_start: float, outage_duration: float) -> Path:
    """
    Locate the JSON file ins_compare.py wrote for a given run.

    `ins_compare.py` writes to outputs/<fkey>/<traj_subdir>/<run_id>_results.json
    where:
        traj_subdir = "<dataset_name>_no_outage"          if no outage
                      "<dataset_name>_outage_<t1>s_<d>s"  otherwise
        run_id      = "no_outage"                          if no outage
                      "outage_<t1>s_<d>s"                  otherwise
    The sequence ID (01, c03, etc.) is part of the dataset_name.
    """
    if outage_start == 0 and outage_duration == 0:
        subdir = f'{seq_id}_no_outage'
        run_id = 'no_outage'
    else:
        t1 = int(outage_start)
        d  = int(outage_duration)
        subdir = f'{seq_id}_outage_{t1}s_{d}s'
        run_id = f'outage_{t1}s_{d}s'
    return _REPO_ROOT / f'outputs/{filter_key}/{subdir}/{run_id}_results.json'


def _read_metrics(json_path: Path) -> dict:
    """Return {'ate': float, 't_rel': float, 'r_rel': float} or NaNs if missing."""
    nan_row = {'ate': float('nan'), 't_rel': float('nan'), 'r_rel': float('nan')}
    if not json_path.exists():
        return nan_row
    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        print(f"  [WARN] could not parse {json_path}: {exc}", file=sys.stderr)
        return nan_row
    ate_block = data.get('ate', {}) or {}
    kitti     = data.get('kitti_metrics', {}) or {}
    return {
        'ate':   float(ate_block.get('rmse', float('nan'))),
        't_rel': float(kitti.get('t_rel', float('nan'))),
        'r_rel': float(kitti.get('r_rel', float('nan'))),
    }


def _filter_label(filter_key: str) -> str:
    """LaTeX-friendly display name for a filter."""
    pretty = {
        'esekfg_vanilla':  r'ES-EKF Groves',
        'esekfg_enhanced': r'ES-EKF Groves +',
        'esekfs_vanilla':  r'ES-EKF Sol\`{a}',
        'esekfs_enhanced': r'ES-EKF Sol\`{a} +',
        'iekf_vanilla':    r'IEKF',
        'iekf_enhanced':   r'IEKF +',
        'imu_only':        r'IMU-only (AI-IMU, fixed $N$)',
        'iekf_ai_imu':     r'Deep IEKF',
        'tlio':            r'TLIO',
        'deep_kf':         r'DKF',
        'tartan_imu':      r'Tartan IMU',
    }
    return pretty.get(filter_key, filter_key.replace('_', r'\_'))


def _format_cell(val: float, kind: str) -> str:
    """Format ATE, t_rel, or r_rel for a LaTeX cell."""
    if not np.isfinite(val):
        return r'$-$'
    if kind == 'ate':
        return f'{val:.2f}'
    if kind == 't_rel':
        return f'{val:.2f}'
    if kind == 'r_rel':
        return f'{val:.2f}'
    return f'{val:.3g}'


def _emit_tabular(rows: list, seqs: list, regime_label: str,
                  caption: str, label: str) -> str:
    """
    Build a LaTeX longtable-style body from rows.

    rows is a list of (filter_label, [{'ate', 't_rel', 'r_rel'} per seq]).
    """
    # Three metric columns per sequence: ATE, t_rel, r_rel
    n_seq = len(seqs)
    col_spec = '@{}l' + ('|ccc' * n_seq) + '|ccc@{}'  # last block = mean ± std

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(rf'\caption{{{caption}}}')
    lines.append(rf'\label{{{label}}}')
    lines.append(r'\centering')
    lines.append(r'\setlength{\tabcolsep}{3pt}')
    lines.append(r'\scriptsize')
    lines.append(rf'\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\hline')

    # Header row 1 — sequence IDs
    seq_headers = ' & '.join([rf'\multicolumn{{3}}{{c|}}{{{s}}}' for s in seqs])
    lines.append(rf'Filter & {seq_headers} & '
                 rf'\multicolumn{{3}}{{c}}{{Mean $\pm$ Std}} \\')

    # Header row 2 — column names
    sub = r' & ATE & $t_{rel}$ & $r_{rel}$' * (n_seq + 1)
    lines.append(sub + r' \\')
    lines.append(r'\hline')

    # Data rows
    for label_text, cells in rows:
        ates  = np.array([c['ate']   for c in cells])
        trels = np.array([c['t_rel'] for c in cells])
        rrels = np.array([c['r_rel'] for c in cells])
        line  = label_text
        for c in cells:
            line += (' & ' + _format_cell(c['ate'],   'ate')
                   + ' & ' + _format_cell(c['t_rel'], 't_rel')
                   + ' & ' + _format_cell(c['r_rel'], 'r_rel'))
        # mean ± std summary
        def _mp(vals: np.ndarray, kind: str) -> str:
            v = vals[np.isfinite(vals)]
            if v.size == 0:
                return r'$-$'
            return f'{np.mean(v):.2f}$\\pm${np.std(v):.2f}'
        line += (' & ' + _mp(ates,  'ate')
              +  ' & ' + _mp(trels, 't_rel')
              +  ' & ' + _mp(rrels, 'r_rel')
              +  r' \\')
        lines.append(line)

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\\[2pt]')
    lines.append(rf'\textit{{{regime_label}}} '
                 rf'(ATE in m, $t_{{rel}}$ in \%, $r_{{rel}}$ in $^\circ$/km)')
    lines.append(r'\end{table}')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(
        description='Emit no-outage and 60s-outage LaTeX tables from ins_compare result JSONs.')
    parser.add_argument('--dataset', choices=['kitti', 'cookies'], default='kitti')
    parser.add_argument('--seqs', nargs='+', default=None,
                        help='Sequence IDs (default: all clean seqs for the dataset)')
    parser.add_argument('--filters', nargs='+', default=None,
                        help=f'Filter keys (default: {DEFAULT_FILTERS})')
    parser.add_argument('--outage-start',    type=float, default=40.0,
                        help='Outage start time for the 60s table (default 40 s)')
    parser.add_argument('--outage-duration', type=float, default=60.0,
                        help='Outage duration for the 60s table (default 60 s)')
    parser.add_argument('--out-dir', type=Path,
                        default=_REPO_ROOT / 'outputs/tables',
                        help='Where to write the .tex files')
    args = parser.parse_args()

    seqs    = args.seqs    or (KITTI_SEQS if args.dataset == 'kitti' else COOKIES_SEQS)
    filters = args.filters or DEFAULT_FILTERS

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating {len(seqs)} sequences × {len(filters)} filters × 2 regimes")
    print(f"Dataset    : {args.dataset}")
    print(f"Output dir : {args.out_dir}\n")

    # Build rows for each regime
    no_outage_rows = []
    outage_rows    = []

    for fkey in filters:
        no_cells, ou_cells = [], []
        for seq in seqs:
            no_cells.append(_read_metrics(_result_json_path(fkey, seq, 0, 0)))
            ou_cells.append(_read_metrics(_result_json_path(
                fkey, seq, args.outage_start, args.outage_duration)))
        no_outage_rows.append((_filter_label(fkey), no_cells))
        outage_rows   .append((_filter_label(fkey), ou_cells))

    # Emit
    no_tex = _emit_tabular(
        no_outage_rows, seqs,
        regime_label='No GNSS outage (full GPS aiding)',
        caption=f'KITTI Benchmarking Without Outage --- {args.dataset.upper()}',
        label=f'tab:journal_no_outage_{args.dataset}')
    ou_tex = _emit_tabular(
        outage_rows, seqs,
        regime_label=f'GNSS outage: start {int(args.outage_start)}\\,s, '
                     f'duration {int(args.outage_duration)}\\,s',
        caption=(f'KITTI Benchmarking During '
                 f'{int(args.outage_duration)}\\,s GNSS Outage --- {args.dataset.upper()}'),
        label=f'tab:journal_outage_{int(args.outage_duration)}s_{args.dataset}')

    no_path = args.out_dir / f'no_outage_{args.dataset}.tex'
    ou_path = args.out_dir / f'outage_{int(args.outage_duration)}s_{args.dataset}.tex'
    no_path.write_text(no_tex)
    ou_path.write_text(ou_tex)

    print(f"Wrote {no_path}")
    print(f"Wrote {ou_path}\n")

    # Brief console summary so the user can sanity-check before \\input{}'ing
    print("Mean ± std across sequences (sanity check):\n")
    for label_, no_cells, ou_cells in zip(
        [r[0] for r in no_outage_rows],
        [r[1] for r in no_outage_rows],
        [r[1] for r in outage_rows],
    ):
        def _mp(cells, key):
            v = np.array([c[key] for c in cells], dtype=float)
            v = v[np.isfinite(v)]
            return (f'{np.mean(v):.2f}±{np.std(v):.2f}'
                    if v.size else '   -   ')
        print(f"  {label_:<28} no-outage  ATE={_mp(no_cells, 'ate'):>10}"
              f"  t_rel={_mp(no_cells, 't_rel'):>9}"
              f"  r_rel={_mp(no_cells, 'r_rel'):>9}")
        print(f"  {' ':<28} 60s-outage ATE={_mp(ou_cells, 'ate'):>10}"
              f"  t_rel={_mp(ou_cells, 't_rel'):>9}"
              f"  r_rel={_mp(ou_cells, 'r_rel'):>9}")

    # Per-cell LaTeX rows ready to be pasted into the existing inline tables in
    # 3.tex (tab:kitti_benchmark and tab:kitti_benchmark_60). One \\-terminated
    # line per filter, six metric columns per sequence.
    def _row_text(cells: list) -> str:
        parts = []
        for c in cells:
            parts.append(f"{c['ate']:.2f}"   if np.isfinite(c['ate'])   else '-')
            parts.append(f"{c['t_rel']:.2f}" if np.isfinite(c['t_rel']) else '-')
            parts.append(f"{c['r_rel']:.2f}" if np.isfinite(c['r_rel']) else '-')
        return ' & '.join(parts) + r' \\'

    print(f"\n--- Paste-ready rows for tab:kitti_benchmark (no outage) ---")
    for fkey, (label_, cells) in zip(filters, no_outage_rows):
        print(f"  \\textbf{{{label_}}} & {_row_text(cells)}")
    print(f"\n--- Paste-ready rows for tab:kitti_benchmark_60 (60 s outage) ---")
    for fkey, (label_, cells) in zip(filters, outage_rows):
        print(f"  \\textbf{{{label_}}} & {_row_text(cells)}")


if __name__ == '__main__':
    main()
