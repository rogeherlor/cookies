#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the 4 requested LaTeX tables:

  1. outputs/tables_hailo/accuracy_no_outage.tex  — ATE/t_rel/r_rel, all 13
     filters x 7 KITTI seqs, CPU. Reads this repo's already-cached,
     LOO-tuned outputs/<filter_key>/... results (same source data as
     emit_journal_tables.py — see that module for the run pipeline that
     produced them).
  2. outputs/tables_hailo/accuracy_outage.tex     — same, with the 40s
     start / 60s duration outage window (matches emit_journal_tables.py's
     default; seq 04 is too short for this window and shows "$-$" for
     every filter, matching that script's own known gap).
  3. outputs/tables_hailo/timing_cpu.tex          — wall time (s) and
     real-time factor per filter x seq, CPU, all 13 filters. Reads the
     FRESH results from run_full_benchmark.py (real hardware timing,
     measured on this Pi — not from the cached accuracy JSONs, which don't
     carry reliable per-sample timing).
  4. outputs/tables_hailo/timing_hailo.tex        — Hailo wall time, RTF,
     and speedup vs. CPU, only for the 4 filters with a Hailo backend
     (deep_kf, tlio, tartan_imu, deep_iekf).

Usage:
    python3 scripts/positioning/hailo/build_latex_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_OUT_DIR = _REPO_ROOT / "outputs/tables_hailo"

# ── Shared filter metadata ───────────────────────────────────────────────────
CLASSICAL = ['esekfg_vanilla', 'esekfg_enhanced', 'esekfs_vanilla', 'esekfs_enhanced',
             'iekf_vanilla', 'iekf_enhanced', 'imu_only']
DL        = ['iekf_ai_imu_online', 'tlio', 'deep_kf', 'tartan_imu']
SMOOTHERS = ['isam2', 'isam2_fixedlag']
ALL_FILTERS = CLASSICAL + DL + SMOOTHERS   # row order = table row order

# key used by run_full_benchmark.py's approach names, which differ for the
# DL group ('deep_iekf' there == 'iekf_ai_imu_online' here — see
# _full_eval_worker.py's DL_TUNED_KEY comment).
BENCH_APPROACH_KEY = {
    'iekf_ai_imu_online': 'deep_iekf', 'tlio': 'tlio',
    'deep_kf': 'deep_kf', 'tartan_imu': 'tartan_imu',
}

LABELS = {
    'esekfg_vanilla':  r'ES-EKF Groves',
    'esekfg_enhanced': r'ES-EKF Groves +',
    'esekfs_vanilla':  r'ES-EKF Sol\`{a}',
    'esekfs_enhanced': r'ES-EKF Sol\`{a} +',
    'iekf_vanilla':    r'IEKF',
    'iekf_enhanced':   r'IEKF +',
    'imu_only':        r'IMU-only',
    'iekf_ai_imu_online': r'Deep IEKF',
    'tlio':            r'TLIO',
    'deep_kf':         r'DKF',
    'tartan_imu':      r'Tartan IMU',
    'isam2':           r'iSAM2',
    'isam2_fixedlag':  r'iSAM2 FL',
}

KITTI_SEQS = ['01', '04', '06', '07', '08', '09', '10']
KITTI_TERRAIN = {'01': 'High.', '04': 'Cntry', '06': 'Urban', '07': 'Urban',
                 '08': 'U+C', '09': 'U+C', '10': 'U+C'}
_KITTI_SEQ_TO_DRIVE = {
    '01': '2011_10_03_drive_0042_extract',
    '04': '2011_09_30_drive_0016_extract',
    '06': '2011_09_30_drive_0020_extract',
    '07': '2011_09_30_drive_0027_extract',
    '08': '2011_09_30_drive_0028_extract',
    '09': '2011_09_30_drive_0033_extract',
    '10': '2011_09_30_drive_0034_extract',
}

OUTAGE_START = 40.0
OUTAGE_DURATION = 60.0

# Interval between successive invocations of each filter's own update step —
# NOT the 10ms/100Hz IMU sample period uniformly, since several filters only
# act on a subset of samples (see scripts/positioning/hailo/README.md's
# "Update rate in the real filter loop" table and each *_runner.py's own
# stride constant, cited per row below). Classical EKF/IEKF filters and
# IMU-only propagate+update every IMU sample by construction (100Hz).
# Deep IEKF's causal MesNet is likewise a per-step sliding-window call in
# the true online algorithm (100Hz) -- the compiled Hailo .hef batches a
# whole sequence into one call as an export/benchmarking convenience, not
# because the algorithm's own decision rate is lower (see hailo/README.md,
# "deep_iekf doesn't fit the same framing").
UPDATE_INTERVAL_MS = {
    'esekfg_vanilla': 10.0, 'esekfg_enhanced': 10.0,
    'esekfs_vanilla': 10.0, 'esekfs_enhanced': 10.0,
    'iekf_vanilla': 10.0, 'iekf_enhanced': 10.0, 'imu_only': 10.0,
    'iekf_ai_imu_online': 10.0,   # causal MesNet, per-step (see note above)
    'tlio': 50.0,                 # tlio_runner.py: _UPDATE_STRIDE=5 -> 20Hz
    'deep_kf': 10.0,               # every IMU sample, 100Hz
    'tartan_imu': 1000.0,          # tartan_runner.py: tartan_interval=100 -> 1Hz
    'isam2': 1000.0,               # isam2_runner.py: factor graph @ GPS rate, 1Hz
    'isam2_fixedlag': 1000.0,      # same GPS-rate trigger as isam2
}


# ── Accuracy: read cached ins_compare.py results ─────────────────────────────

def _result_json_path(filter_key, seq_id, outage_start, outage_duration):
    dataset_name = _KITTI_SEQ_TO_DRIVE[seq_id]
    if outage_start == 0 and outage_duration == 0:
        subdir, run_id = f'{dataset_name}_no_outage', 'no_outage'
    else:
        tag = f'outage_{float(outage_start)}s_{float(outage_duration)}s'
        subdir, run_id = f'{dataset_name}_{tag}', tag
    return _REPO_ROOT / f'outputs/{filter_key}/{subdir}/{run_id}_results.json'


def _read_accuracy(json_path):
    nan_row = {'ate': float('nan'), 't_rel': float('nan'), 'r_rel': float('nan')}
    if not json_path.exists():
        return nan_row
    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        print(f"  [WARN] could not parse {json_path}: {exc}", file=sys.stderr)
        return nan_row
    ate_block = data.get('ate', {}) or {}
    kitti = data.get('kitti_metrics', {}) or {}
    return {
        'ate':   float(ate_block.get('rmse', float('nan'))),
        't_rel': float(kitti.get('t_rel', float('nan'))),
        'r_rel': float(kitti.get('r_rel', float('nan'))),
    }


def _fmt_num(val):
    return r'$-$' if not np.isfinite(val) else f'{val:.2f}'


def _get_accuracy_cell(idx, bkey, backend, scenario, seq):
    """Reads {'ate','t_rel','r_rel'} from a fresh run_full_benchmark.py /
    _full_eval_worker.py record (real hardware, this Pi), NaN if missing
    (e.g. a KITTI sequence too short for the outage window, or — for
    deep_iekf's Hailo row — too short for the HEF's fixed input length)."""
    nan_row = {'ate': float('nan'), 't_rel': float('nan'), 'r_rel': float('nan')}
    rec = idx.get((bkey, backend, scenario, seq))
    acc = rec.get('accuracy') if rec else None
    if not acc:
        return nan_row
    return {
        'ate':   float(acc['ate']['rmse']),
        't_rel': float(acc['kitti_metrics']['t_rel']),
        'r_rel': float(acc['kitti_metrics']['r_rel']),
    }


def build_accuracy_table(idx, scenario, caption, label, out_path):
    """CPU accuracy, all 13 filters — reads fresh results from
    run_full_benchmark.py (measured on this Pi), NOT the historical cached
    ins_compare.py JSONs (see _read_accuracy/_result_json_path, kept below
    for reference but no longer used by default)."""
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'    \centering')
    lines.append(rf'    \caption{{{caption}}}')
    lines.append(r'    \renewcommand{\arraystretch}{1.3}')
    lines.append(r'    \setlength{\tabcolsep}{3pt} ')
    lines.append(r'    ')
    lines.append(r'    \resizebox{\textwidth}{!}{')
    col_spec = 'l | ' + ' | '.join(['rrr'] * len(KITTI_SEQS))
    lines.append(rf'    \begin{{tabular}}{{{col_spec}}} ')
    lines.append(r'    \Xhline{1.2pt}')

    seq_headers = ' & '.join(
        rf'\multicolumn{{3}}{{c{"|" if i < len(KITTI_SEQS) - 1 else ""}}}'
        rf'{{\textbf{{Seq. {s} ({KITTI_TERRAIN[s]})}}}}'
        for i, s in enumerate(KITTI_SEQS)
    )
    lines.append(rf'    \multirow{{2}}{{*}}{{\textbf{{Algorithm}}}} & {seq_headers} \\')
    sub = ' & '.join([r'\textbf{ATE} & $\bm{t_{rel}}$ & $\bm{r_{rel}}$'] * len(KITTI_SEQS))
    lines.append(rf'    & {sub} \\')
    lines.append(r'    \Xhline{1.2pt}')
    lines.append(r'')

    for fkey in ALL_FILTERS:
        bkey = BENCH_APPROACH_KEY.get(fkey, fkey)
        cells = [_get_accuracy_cell(idx, bkey, 'cpu', scenario, s) for s in KITTI_SEQS]
        row = rf'  \textbf{{{LABELS[fkey]}}}'
        for c in cells:
            row += f" & {_fmt_num(c['ate'])} & {_fmt_num(c['t_rel'])} & {_fmt_num(c['r_rel'])}"
        row += r' \\'
        lines.append(row)

    lines.append(r'    \Xhline{1.2pt}')
    lines.append(r'    \end{tabular}')
    lines.append(r'    }')
    lines.append(rf'    \label{{{label}}}')
    lines.append(r'\end{table*}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f"Wrote {out_path}")


# ── Timing: read fresh run_full_benchmark.py results ─────────────────────────

def _load_bench_results():
    """Returns {(approach, backend, scenario, seq): full_record}, where
    full_record has 'timing' always and 'accuracy' for Hailo/DL rows (see
    _full_eval_worker.py — accuracy is computed there only for backend=
    'hailo', since CPU accuracy is already the cached ins_compare.py data)."""
    p = _HERE / "full_benchmark_results" / "all_results.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    idx = {}
    for r in data:
        idx[(r['approach'], r['backend'], r['scenario'], r['seq'])] = r
    return idx


def _fmt_time(val):
    return r'$-$' if val is None else f'{val:.2f}'


def build_timing_cpu_table(idx, scenario, caption, label, out_path):
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'    \centering')
    lines.append(rf'    \caption{{{caption}}}')
    lines.append(r'    \renewcommand{\arraystretch}{1.3}')
    lines.append(r'    \setlength{\tabcolsep}{3pt} ')
    lines.append(r'    \resizebox{\textwidth}{!}{')
    col_spec = 'l | ' + ' | '.join(['rr'] * len(KITTI_SEQS))
    lines.append(rf'    \begin{{tabular}}{{{col_spec}}} ')
    lines.append(r'    \Xhline{1.2pt}')

    seq_headers = ' & '.join(
        rf'\multicolumn{{2}}{{c{"|" if i < len(KITTI_SEQS) - 1 else ""}}}'
        rf'{{\textbf{{Seq. {s} ({KITTI_TERRAIN[s]})}}}}'
        for i, s in enumerate(KITTI_SEQS)
    )
    lines.append(rf'    \multirow{{2}}{{*}}{{\textbf{{Algorithm}}}} & {seq_headers} \\')
    sub = ' & '.join([r'\textbf{Wall (s)} & $\bm{RTF}$'] * len(KITTI_SEQS))
    lines.append(rf'    & {sub} \\')
    lines.append(r'    \Xhline{1.2pt}')
    lines.append(r'')

    for fkey in ALL_FILTERS:
        bkey = BENCH_APPROACH_KEY.get(fkey, fkey)
        row = rf'  \textbf{{{LABELS[fkey]}}}'
        for s in KITTI_SEQS:
            rec = idx.get((bkey, 'cpu', scenario, s))
            t = rec['timing'] if rec else None
            wall = t['wall_s'] if t else None
            rtf = t['real_time_factor'] if t else None
            row += f" & {_fmt_time(wall)} & {_fmt_time(rtf)}"
        row += r' \\'
        lines.append(row)

    lines.append(r'    \Xhline{1.2pt}')
    lines.append(r'    \end{tabular}')
    lines.append(r'    }')
    lines.append(rf'    \label{{{label}}}')
    lines.append(r'\end{table*}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f"Wrote {out_path}")


def build_timing_hailo_table(idx, scenario, caption, label, out_path):
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'    \centering')
    lines.append(rf'    \caption{{{caption}}}')
    lines.append(r'    \renewcommand{\arraystretch}{1.3}')
    lines.append(r'    \setlength{\tabcolsep}{3pt} ')
    lines.append(r'    \resizebox{\textwidth}{!}{')
    col_spec = 'l | ' + ' | '.join(['rrr'] * len(KITTI_SEQS))
    lines.append(rf'    \begin{{tabular}}{{{col_spec}}} ')
    lines.append(r'    \Xhline{1.2pt}')

    seq_headers = ' & '.join(
        rf'\multicolumn{{3}}{{c{"|" if i < len(KITTI_SEQS) - 1 else ""}}}'
        rf'{{\textbf{{Seq. {s} ({KITTI_TERRAIN[s]})}}}}'
        for i, s in enumerate(KITTI_SEQS)
    )
    lines.append(rf'    \multirow{{2}}{{*}}{{\textbf{{Algorithm}}}} & {seq_headers} \\')
    sub = ' & '.join([r'\textbf{Hailo (s)} & $\bm{RTF}$ & \textbf{Speedup}'] * len(KITTI_SEQS))
    lines.append(rf'    & {sub} \\')
    lines.append(r'    \Xhline{1.2pt}')
    lines.append(r'')

    for fkey in DL:
        bkey = BENCH_APPROACH_KEY[fkey]
        row = rf'  \textbf{{{LABELS[fkey]}}}'
        for s in KITTI_SEQS:
            rec_cpu = idx.get((bkey, 'cpu', scenario, s))
            rec_hailo = idx.get((bkey, 'hailo', scenario, s))
            t_cpu = rec_cpu['timing'] if rec_cpu else None
            t_hailo = rec_hailo['timing'] if rec_hailo else None
            wall = t_hailo['wall_s'] if t_hailo else None
            rtf = t_hailo['real_time_factor'] if t_hailo else None
            speedup = None
            if t_cpu and t_hailo and t_hailo['wall_s'] > 0:
                speedup = t_cpu['wall_s'] / t_hailo['wall_s']
            speedup_str = r'$-$' if speedup is None else f'{speedup:.2f}' + r'$\times$'
            row += f" & {_fmt_time(wall)} & {_fmt_time(rtf)}"
            row += f" & {speedup_str}"
        row += r' \\'
        lines.append(row)

    lines.append(r'    \Xhline{1.2pt}')
    lines.append(r'    \end{tabular}')
    lines.append(r'    }')
    lines.append(rf'    \label{{{label}}}')
    lines.append(r'\end{table*}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f"Wrote {out_path}")


def _fmt_rtf(val):
    """Real-time factor, with a dagger marking real-time FAILURE (RTF<1x —
    the filter cannot keep up with a live 100Hz IMU stream)."""
    if val is None:
        return r'$-$'
    return f'{val:.2f}' + (r'$^\dagger$' if val < 1.0 else '')


def build_accuracy_hailo_table(idx, scenario, caption, label, out_path):
    """Same structure/rows as build_accuracy_table, but for the 4 filters
    with a Hailo backend, substitutes Hailo-measured accuracy (both from
    fresh run_full_benchmark.py / _full_eval_worker.py results, this Pi,
    same evaluate_navigation_performance / compute_kitti_metrics
    methodology) in place of CPU. Classical filters and smoothers have no
    Hailo variant, so they keep showing the same CPU numbers as Table 1 —
    there is nothing to substitute."""
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'    \centering')
    lines.append(rf'    \caption{{{caption}}}')
    lines.append(r'    \renewcommand{\arraystretch}{1.3}')
    lines.append(r'    \setlength{\tabcolsep}{3pt} ')
    lines.append(r'    ')
    lines.append(r'    \resizebox{\textwidth}{!}{')
    col_spec = 'l | ' + ' | '.join(['rrr'] * len(KITTI_SEQS))
    lines.append(rf'    \begin{{tabular}}{{{col_spec}}} ')
    lines.append(r'    \Xhline{1.2pt}')

    seq_headers = ' & '.join(
        rf'\multicolumn{{3}}{{c{"|" if i < len(KITTI_SEQS) - 1 else ""}}}'
        rf'{{\textbf{{Seq. {s} ({KITTI_TERRAIN[s]})}}}}'
        for i, s in enumerate(KITTI_SEQS)
    )
    lines.append(rf'    \multirow{{2}}{{*}}{{\textbf{{Algorithm}}}} & {seq_headers} \\')
    sub = ' & '.join([r'\textbf{ATE} & $\bm{t_{rel}}$ & $\bm{r_{rel}}$'] * len(KITTI_SEQS))
    lines.append(rf'    & {sub} \\')
    lines.append(r'    \Xhline{1.2pt}')
    lines.append(r'')

    for fkey in ALL_FILTERS:
        bkey = BENCH_APPROACH_KEY.get(fkey, fkey)
        backend = 'hailo' if fkey in DL else 'cpu'
        cells = [_get_accuracy_cell(idx, bkey, backend, scenario, s) for s in KITTI_SEQS]
        row = rf'  \textbf{{{LABELS[fkey]}}}'
        for c in cells:
            row += f" & {_fmt_num(c['ate'])} & {_fmt_num(c['t_rel'])} & {_fmt_num(c['r_rel'])}"
        row += r' \\'
        lines.append(row)

    lines.append(r'    \Xhline{1.2pt}')
    lines.append(r'    \end{tabular}')
    lines.append(r'    }')
    lines.append(rf'    \label{{{label}}}')
    lines.append(r'\end{table*}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f"Wrote {out_path}")


def _fmt_stat(values, suffix=''):
    """Mean +/- std over the KITTI sequences that had a valid record for
    this (filter, scenario) cell; '$-$' if none did (e.g. Seq 04 is too
    short for the 40s+60s outage window and contributes nothing there)."""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return r'$-$'
    if len(vals) == 1:
        return f'{vals[0]:.2f}' + suffix
    return f'{np.mean(vals):.2f}$\\pm${np.std(vals, ddof=1):.2f}' + suffix


def _sample_latency_ms(t):
    """Per-IMU-sample wall time in ms, amortised over the whole run
    (wall_s * 1000 / n_samples, n_samples = duration_s * 100Hz) — matches
    the convention 4.tex's tab:pos_hailo_summary already uses. Directly
    comparable to the 10ms/sample real-time budget, and does not conflate
    filter speed with how long a given KITTI drive happens to be, unlike
    raw wall-clock seconds (which range 30-862s across sequences purely
    from sequence duration, not filter cost)."""
    if t is None or t['dataset_duration_s'] <= 0:
        return None
    n_samples = t['dataset_duration_s'] * 100.0
    return t['wall_s'] * 1000.0 / n_samples


def _sample_routine_ms(t):
    """Per-sample cost of the baseline work present on EVERY sample
    (propagation, state/covariance bookkeeping) — the event's own cost
    (the isolated network/GPS/solve call, from _event_stats_ms) is an
    ADDITION on top of this on event samples, not a replacement of it:
    physically, a filter's propagate step still runs on the sample where
    its measurement update also fires. So the leftover after subtracting
    the event's own total time is spread over ALL samples, not just the
    non-event ones — dividing by only non-event samples breaks down (and
    for a filter whose event fires on ~every sample, like Deep KF's
    100Hz network call, divides by ~zero) since there may be few or no
    'pure routine, nothing else happened' samples to begin with. Only
    available where the event itself is separately instrumented
    (currently the 4 DL filters' network calls); None otherwise."""
    if t is None or t['dataset_duration_s'] <= 0 or 'mean_ms' not in t:
        return None
    n_samples = t['dataset_duration_s'] * 100.0
    event_total_ms = t['mean_ms'] * t['n_calls']
    routine_total_ms = t['wall_s'] * 1000.0 - event_total_ms
    return routine_total_ms / n_samples


def _event_stats_ms(t):
    """(mean, worst) cost of a single EVENT sample (the rare, expensive
    one — network call, GPS correction, factor-graph solve), NOT amortised
    over the routine samples around it. This is the number that matters
    for a hard-real-time verdict: a causal loop fed one new IMU sample
    every 10ms must finish EVERY sample, including the expensive one, before
    the next arrives — averaging the expensive event's cost down across
    the cheap samples around it (as the old 'Latency'/'Update' columns
    did) hides exactly the failure mode a critical application cares
    about. mean_ms/p95_ms are already per-event (not per-sample-amortised)
    in the source JSON — see _full_eval_worker.py. Only available for the
    4 DL filters today (see _sample_routine_ms)."""
    if t is None or 'mean_ms' not in t or 'p95_ms' not in t:
        return None, None
    return t['mean_ms'], t['p95_ms']


def _fmt_interval(ms):
    return f'{ms:.0f}'


# (section header, event-trigger description, list of filter keys)
_TIMING_GROUPS = [
    ('Classical Filters \\& IMU-only --- CPU',
     [('esekfg_vanilla',  'cpu', 'GPS fix, 1\\,Hz'),
      ('esekfg_enhanced', 'cpu', 'GPS fix, 1\\,Hz + \\ac{nhc}/\\ac{zupt} every sample'),
      ('esekfs_vanilla',  'cpu', 'GPS fix, 1\\,Hz'),
      ('esekfs_enhanced', 'cpu', 'GPS fix, 1\\,Hz + \\ac{nhc}/\\ac{zupt} every sample'),
      ('iekf_vanilla',    'cpu', 'GPS fix, 1\\,Hz'),
      ('iekf_enhanced',   'cpu', 'GPS fix, 1\\,Hz + \\ac{nhc}/\\ac{zupt} every sample'),
      ('imu_only',        'cpu', 'none --- pure dead reckoning')]),
    ('Deep Learning Filters --- CPU and Hailo-8L',
     [('iekf_ai_imu_online', 'cpu',   'network inference, 100\\,Hz (17-sample window)'),
      ('iekf_ai_imu_online', 'hailo', 'network call, batched (see $\\ddagger$)'),
      ('tlio',               'cpu',   'network call, 20\\,Hz'),
      ('tlio',               'hailo', 'network call, 20\\,Hz'),
      ('deep_kf',            'cpu',   'network call, 100\\,Hz'),
      ('deep_kf',            'hailo', 'network call, 100\\,Hz'),
      ('tartan_imu',         'cpu',   'network call, 1\\,Hz'),
      ('tartan_imu',         'hailo', 'network call, 1\\,Hz')]),
    ('Smoothers --- CPU',
     [('isam2',          'cpu', 'factor-graph solve, GPS rate (1\\,Hz)'),
      ('isam2_fixedlag', 'cpu', 'factor-graph solve, GPS rate (1\\,Hz)')]),
]


def build_timing_merged_summary_table(idx, caption, label, out_path):
    """Real-time feasibility report, not just a speed comparison: for
    every filter, separates the cheap ROUTINE sample (no event firing)
    from the rare, expensive EVENT sample (GPS correction / network call
    / factor-graph solve), and verdicts each filter against the worst
    EVENT observed, not the sample-amortised average — because a causal
    loop fed one new IMU sample every 10ms must finish EVERY sample,
    including its rare expensive ones, before the next arrives; averaging
    that expensive event down across the cheap samples around it (as a
    plain mean-latency table would) can hide a real deadline miss. Example
    this table catches that an averaged table would not: Tartan~IMU's
    amortised latency looks fine (~1\\,ms/sample) but its actual per-event
    network call peaks at ~138\\,ms (see \\_event\\_stats\\_ms) — 13+ IMU
    periods, a real worst-case stall invisible in the average.

    Grouped into 3 sections (classical+IMU-only / DL / smoothers) with the
    event trigger described per row instead of a single 'update interval'
    number — classical VANILLA filters' one expensive step is the sparse
    (1Hz) GPS correction, not a uniform 10ms cost as an earlier version of
    this table implied; ENHANCED variants add a second, denser NHC/ZUPT
    tier on top of that. Real Routine/Event numbers exist only for the 4
    DL filters today (their runners already time the network call in
    isolation); the 9 classical/smoother filters need new per-step timers
    (GPS-correction and factor-graph-solve timers specifically, not just
    a whole-loop timer) before this table can carry real numbers for them
    too — shown as TBD, not silently backfilled with the old combined
    average."""
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'    \centering')
    lines.append(rf'    \caption{{{caption}}}')
    lines.append(r'    \renewcommand{\arraystretch}{1.3}')
    lines.append(r'    \resizebox{\textwidth}{!}{')
    lines.append(r'    \begin{tabular}{l l c c c c}')
    lines.append(r'    \Xhline{1.2pt}')
    lines.append(r'    \textbf{Algorithm} & \textbf{Event Trigger} & \textbf{Routine (ms)} & '
                 r'\textbf{Event Mean (ms)} & \textbf{Event Worst (ms)} & \textbf{10\,ms Verdict} \\')
    lines.append(r'    \Xhline{1.2pt}')

    for section_title, rows in _TIMING_GROUPS:
        lines.append(r'')
        lines.append(rf'    \multicolumn{{6}}{{l}}{{\textit{{{section_title}}}}} \\')
        lines.append(r'    \hline')
        for fkey, backend, trigger_desc in rows:
            bkey = BENCH_APPROACH_KEY.get(fkey, fkey)
            routine, ev_mean, ev_worst = [], [], []
            for scenario in ('no_outage', 'outage'):
                for s in KITTI_SEQS:
                    rec = idx.get((bkey, backend, scenario, s))
                    t = rec['timing'] if rec else None
                    r_ms = _sample_routine_ms(t)
                    if r_ms is not None:
                        routine.append(r_ms)
                    m, w = _event_stats_ms(t)
                    if m is not None:
                        ev_mean.append(m)
                        ev_worst.append(w)

            if fkey == 'imu_only':
                verdict = r'\textit{N/A}'
                routine_cell, mean_cell, worst_cell = _fmt_stat(routine), r'$-$', r'$-$'
            elif fkey == 'iekf_ai_imu_online' and backend == 'hailo':
                # Batched: the whole (truncated) sequence is one Hailo
                # call, not a per-sample streaming event — see caveat. The
                # CPU row, by contrast, now streams genuinely (one real
                # network call per IMU sample) and gets a real verdict
                # below, same as every other row.
                verdict = r'\textit{N/A}$^\ddagger$'
                routine_cell = _fmt_stat(routine)
                mean_cell = _fmt_stat(ev_mean) if ev_mean else 'TBD'
                worst_cell = _fmt_stat(ev_worst) if ev_worst else 'TBD'
            elif not ev_worst:
                verdict = 'TBD'
                routine_cell, mean_cell, worst_cell = 'TBD', 'TBD', 'TBD'
            else:
                # Worst realistic single-sample cost on an event sample is
                # baseline work (Routine, still runs every sample) PLUS
                # the worst observed event cost on top of it — using the
                # event's worst-case alone would understate the true
                # worst sample by dropping the baseline work that also
                # has to happen on that same sample.
                worst_combined = np.mean(routine) + max(ev_worst)
                verdict = (r'\textcolor{red!70!black}{\textbf{FAIL}}' if worst_combined > 10.0
                          else r'\textbf{PASS}')
                routine_cell = _fmt_stat(routine)
                mean_cell = _fmt_stat(ev_mean)
                worst_cell = _fmt_stat(ev_worst)

            backend_suffix = f' ({"Hailo-8L" if backend == "hailo" else "CPU"})' if fkey in DL else ''
            row = (rf'  \textbf{{{LABELS[fkey]}}}{backend_suffix} & {trigger_desc} & {routine_cell} '
                   rf'& {mean_cell} & {worst_cell} & {verdict} \\')
            lines.append(row)

    lines.append(r'    \Xhline{1.2pt}')
    lines.append(r'    \end{tabular}')
    lines.append(r'    }')
    lines.append(r'    \\[2pt]')
    lines.append(r'    \footnotesize \textbf{Routine}: baseline per-sample cost present on every sample '
                 r'(propagation, state/covariance bookkeeping), regardless of whether the event also fires on that '
                 r'sample. \textbf{Event Mean/Worst}: mean and worst-case (95th percentile) cost of the event '
                 r'itself (network call / \ac{gps} correction / factor-graph solve) in isolation, \emph{not} '
                 r'amortised over the cheap samples around it. \textbf{10\,ms Verdict}: does Routine $+$ the worst '
                 r'observed Event still complete before the next \ac{imu} sample arrives 10\,ms later — the '
                 r'deadline that matters for genuinely causal, hard real-time operation, independent of how rarely '
                 r'the event fires. '
                 r"$^\ddagger$ Deep IEKF's Hailo \texttt{.hef} processes a whole (truncated) sequence in one batched "
                 r'call rather than streaming sample by sample, so no single-sample event deadline applies to it as '
                 r'deployed here (Section~\ref{sec:4_positioning}); its CPU row, by contrast, streams a genuine '
                 r'network call per \ac{imu} sample and carries a real verdict. '
                 r"Classical/smoother TBD cells need new per-step timers (around each filter's GPS-correction or "
                 r'factor-graph-solve call specifically, not just the whole loop) and a rerun of the benchmark '
                 r'sweep; Routine/Event figures pool over both outage and no-outage runs and all 7 LOO sequences '
                 r'(Seq.\ 04 excluded from outage runs, too short for the window).')
    lines.append(rf'    \label{{{label}}}')
    lines.append(r'\end{table*}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f"Wrote {out_path}")


def main():
    idx = _load_bench_results()
    if idx is None:
        print("full_benchmark_results/all_results.json not found — "
              "run run_full_benchmark.py first.")
        return

    build_accuracy_table(
        idx, 'no_outage', "KITTI Benchmarking Without Outage",
        "tab:kitti_benchmark_no_outage", _OUT_DIR / "accuracy_no_outage.tex")
    build_accuracy_table(
        idx, 'outage',
        f"KITTI Benchmarking With Outage ({int(OUTAGE_START)}s start, {int(OUTAGE_DURATION)}s duration)",
        "tab:kitti_benchmark_outage", _OUT_DIR / "accuracy_outage.tex")

    build_timing_cpu_table(
        idx, 'no_outage', "KITTI Execution Time, CPU, Without Outage",
        "tab:kitti_timing_cpu_no_outage", _OUT_DIR / "timing_cpu_no_outage.tex")
    build_timing_cpu_table(
        idx, 'outage', "KITTI Execution Time, CPU, With Outage",
        "tab:kitti_timing_cpu_outage", _OUT_DIR / "timing_cpu_outage.tex")
    build_timing_hailo_table(
        idx, 'no_outage', "KITTI Execution Time on Hailo-8L, Without Outage",
        "tab:kitti_timing_hailo_no_outage", _OUT_DIR / "timing_hailo_no_outage.tex")
    build_timing_hailo_table(
        idx, 'outage', "KITTI Execution Time on Hailo-8L, With Outage",
        "tab:kitti_timing_hailo_outage", _OUT_DIR / "timing_hailo_outage.tex")

    build_accuracy_hailo_table(
        idx, 'no_outage',
        "KITTI Benchmarking Without Outage (Hailo-8L where available)",
        "tab:kitti_benchmark_hailo_no_outage", _OUT_DIR / "accuracy_hailo_no_outage.tex")
    build_accuracy_hailo_table(
        idx, 'outage',
        "KITTI Benchmarking With Outage (Hailo-8L where available)",
        "tab:kitti_benchmark_hailo_outage", _OUT_DIR / "accuracy_hailo_outage.tex")

    build_timing_merged_summary_table(
        idx, "Per-Step Execution Time by Backend, Mean $\\pm$ Std over 7 LOO Sequences",
        "tab:kitti_timing_merged", _OUT_DIR / "timing_merged.tex")


if __name__ == "__main__":
    main()
