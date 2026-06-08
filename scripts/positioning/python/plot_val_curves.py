# -*- coding: utf-8 -*-
"""
plot_val_curves.py — Parse journal-cost validation lines from ins_train logs
and plot J(epoch) per (filter, fold).

Reads `logs/ins_train_*.log` files written by ins_train.py and the per-K-epoch
validation hook from dl_filters/_validation.py. Each fold prints lines like:

    [val] epoch=49  seq=01  J=8.884  ATE_out=4.87m  t_rel=1.38%  r_rel=2.63deg/km  ANEES=0.55

The script:
  1. Walks one or more log files.
  2. Tracks which filter and held-out sequence is currently being trained via
     `[N/28] START <Filter Label> fold=<SEQ>` lines (also `mode=all`).
  3. Builds {(filter_key, seq) → [(epoch, J, ATE_out, t_rel, r_rel, ANEES), …]}.
  4. Writes one PDF figure per filter (one curve per fold) showing J on a log
     y-axis vs epoch. ATE_out / t_rel / r_rel get their own optional sub-axes.
  5. Writes a CSV `outputs/figures/dl_val_curves.csv` for the chapter.

Usage
-----
    # Plot every log found under logs/
    python3 scripts/positioning/python/plot_val_curves.py

    # Plot specific logs only
    python3 scripts/positioning/python/plot_val_curves.py logs/ins_train_20260601_*.log

    # Just check the parse without writing figures
    python3 scripts/positioning/python/plot_val_curves.py --dry-run

Outputs
-------
    outputs/figures/dl_val_curves_<filter>.pdf
    outputs/figures/dl_val_curves.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE / '../../..'


# ── Regexes ───────────────────────────────────────────────────────────────────
# Filter label → filter key. The label is what ins_train.py prints in the
# START line; the key is what artifacts/, outputs/ etc. use.
_LABEL_TO_KEY = {
    'TLIO (ResNet1D displacement)':  'tlio',
    'Deep KF (LSTM state prediction)': 'deep_kf',
    'Tartan IMU (LoRA fine-tuning)':  'tartan_imu',
    'AI-IMU (Deep IEKF CNN)':         'iekf_ai_imu',
}

_START_RE = re.compile(
    r'START\s+(?P<label>.+?)\s+(?:fold=(?P<seq>\S+)|mode=(?P<mode>\S+))'
)

# Validation line from dl_filters/_validation.format_val_line()
_VAL_RE = re.compile(
    r'\[val\]\s+epoch=(?P<epoch>-?\d+)\s+seq=(?P<seq>\S+)\s+'
    r'J=(?P<J>[\d.eE+-]+|inf|nan)\s+'
    r'ATE_out=(?P<ate>[\d.eE+-]+|nan)m\s+'
    r't_rel=(?P<trel>[\d.eE+-]+|nan)%\s+'
    r'r_rel=(?P<rrel>[\d.eE+-]+|nan)deg/km\s+'
    r'ANEES=(?P<anees>[\d.eE+-]+|nan)'
)


def _parse_float(s: str) -> float:
    if s in ('inf', 'Infinity'):
        return float('inf')
    if s == 'nan':
        return float('nan')
    try:
        return float(s)
    except ValueError:
        return float('nan')


def parse_logs(log_paths: list[Path]) -> dict[tuple[str, str], list[dict]]:
    """Return {(filter_key, seq) → [{'epoch': …, 'J': …, …}, …]}."""
    series: dict[tuple[str, str], list[dict]] = {}
    for p in log_paths:
        cur_filter: Optional[str] = None
        cur_seq: Optional[str] = None
        try:
            text = p.read_text(errors='replace')
        except FileNotFoundError:
            print(f"  [skip] {p} not found", file=sys.stderr)
            continue
        for line in text.splitlines():
            m_start = _START_RE.search(line)
            if m_start:
                label = m_start.group('label').strip()
                cur_filter = _LABEL_TO_KEY.get(label)
                cur_seq = m_start.group('seq') or m_start.group('mode')
                continue
            m_val = _VAL_RE.search(line)
            if m_val and cur_filter:
                # The val line carries its own seq; trust it over the START
                # line because mode=all runs still print seq=<val_seq>.
                key = (cur_filter, m_val.group('seq'))
                series.setdefault(key, []).append({
                    'epoch': int(m_val.group('epoch')),
                    'J':     _parse_float(m_val.group('J')),
                    'ate':   _parse_float(m_val.group('ate')),
                    't_rel': _parse_float(m_val.group('trel')),
                    'r_rel': _parse_float(m_val.group('rrel')),
                    'anees': _parse_float(m_val.group('anees')),
                })
    # Sort each series by epoch
    for k, rows in series.items():
        rows.sort(key=lambda r: r['epoch'])
    return series


def write_csv(series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['filter', 'seq', 'epoch',
                    'J', 'ate_out_m', 't_rel_pct', 'r_rel_deg_km', 'anees'])
        for (fkey, seq), rows in sorted(series.items()):
            for r in rows:
                w.writerow([fkey, seq, r['epoch'],
                            r['J'], r['ate'], r['t_rel'], r['r_rel'],
                            r['anees']])


def plot_per_filter(series, out_dir: Path, suffix: str) -> list[Path]:
    """One PDF per filter; one J(epoch) line per held-out fold."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not installed — figures not written",
              file=sys.stderr)
        return []
    written: list[Path] = []
    # Group by filter
    by_filter: dict[str, list[tuple[str, list[dict]]]] = {}
    for (fkey, seq), rows in series.items():
        by_filter.setdefault(fkey, []).append((seq, rows))
    for fkey, fold_series in sorted(by_filter.items()):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        fold_series.sort(key=lambda t: t[0])
        for seq, rows in fold_series:
            epochs = [r['epoch'] for r in rows]
            Js     = [r['J']     for r in rows]
            if not epochs:
                continue
            ax.plot(epochs, Js, marker='o', linewidth=1.0, markersize=3,
                    label=f"seq {seq}  (n={len(rows)})")
        ax.set_yscale('log')
        ax.set_xlabel('Training epoch')
        ax.set_ylabel(r'Journal cost $J$ '
                      r'$=\,$ATE$_{\rm out}/1\,$m$\,+\,t_{rel}/1\,\%\,'
                      r'+\,r_{rel}/1\,^\circ$/km')
        ax.set_title(f'{fkey} — journal cost vs training epoch '
                     '(held-out sequence, 40 s × 60 s outage)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8, loc='best', ncol=2)
        out_path = out_dir / f'dl_val_curves_{fkey}{suffix}.pdf'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        written.append(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', nargs='*', default=None,
                        help='Specific log files to parse. Defaults to '
                             'logs/ins_train_*.log under the repo root.')
    parser.add_argument('--out-dir', type=Path,
                        default=_REPO_ROOT / 'outputs/figures',
                        help='Where to write the PDF figures and CSV.')
    parser.add_argument('--suffix', default='_kitti',
                        help='Filename suffix appended to each PDF '
                             '(default: _kitti).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse and summarise; do not write figures '
                             'or CSV.')
    args = parser.parse_args()

    if args.logs:
        log_paths = [Path(p) for p in args.logs]
    else:
        log_paths = sorted((_REPO_ROOT / 'logs').glob('ins_train_*.log'))
        if not log_paths:
            print('No ins_train_*.log files found under logs/.', file=sys.stderr)
            return 1

    print(f'Parsing {len(log_paths)} log file(s) ...')
    series = parse_logs(log_paths)

    if not series:
        print('No [val] lines found. Make sure ins_train.py was run with '
              '--val-metric-every > 0 (default 10 for tlio/deep_kf/tartan, '
              '50 for ai_imu).', file=sys.stderr)
        return 1

    print(f'\nParsed {len(series)} (filter, seq) pairs:')
    for (fkey, seq), rows in sorted(series.items()):
        if not rows:
            print(f'  {fkey:12s}  seq {seq:>4s}  n_points=0')
            continue
        finite_Js = [r['J'] for r in rows if r['J'] != float('inf')]
        if finite_Js:
            print(f'  {fkey:12s}  seq {seq:>4s}  n_points={len(rows):3d}  '
                  f'J range [{min(finite_Js):.2f}, {max(finite_Js):.2f}]  '
                  f'final J={rows[-1]["J"]:.2f} '
                  f'(epoch {rows[-1]["epoch"]})')
        else:
            print(f'  {fkey:12s}  seq {seq:>4s}  n_points={len(rows):3d}  '
                  f'all J non-finite')

    if args.dry_run:
        print('\nDry run — no files written.')
        return 0

    csv_path = args.out_dir / 'dl_val_curves.csv'
    write_csv(series, csv_path)
    print(f'\nWrote {csv_path}')

    pdfs = plot_per_filter(series, args.out_dir, args.suffix)
    for p in pdfs:
        print(f'Wrote {p}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
