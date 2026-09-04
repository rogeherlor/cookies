#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrates _full_eval_worker.py across (approach x backend x scenario x
KITTI sequence), each in its own subprocess (see _full_eval_worker.py's
docstring for why). Timing only — accuracy is read from this repo's
already-cached, LOO-tuned outputs/<filter_key>/... results (see
build_latex_tables.py).

Usage (inside the arm64 Hailo docker image, /dev/hailo0 required for the
Hailo-backed rows):
    python3 scripts/positioning/hailo/run_full_benchmark.py
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_WORKER = _HERE / "_full_eval_worker.py"

CLASSICAL = ['esekfg_vanilla', 'esekfg_enhanced', 'esekfs_vanilla', 'esekfs_enhanced',
             'iekf_vanilla', 'iekf_enhanced', 'imu_only']
DL         = ['deep_kf', 'tlio', 'tartan_imu', 'deep_iekf']
SMOOTHERS  = ['isam2', 'isam2_fixedlag']
ALL_APPROACHES = CLASSICAL + DL + SMOOTHERS

SEQS = ['01', '04', '06', '07', '08', '09', '10']
SCENARIOS = ['no_outage', 'outage']

OUT_DIR = _HERE / "full_benchmark_results"


def _taskset_prefix():
    """CPU pinning applied UNIFORMLY to every job, from $BENCH_TASKSET.

    Every row in a timing table has to be measured under the same conditions or
    the columns cannot be compared. This used to be split: _hailo_rerun.sh ran
    its jobs under `taskset -c 1,2,3` while run_full_benchmark.py ran the CPU
    jobs unpinned across every core, so the Hailo and CPU halves of the same
    table came from different machines in all but name. Setting BENCH_TASKSET
    once here applies it to both; leaving it unset pins nothing, which is fine
    as long as it is nothing for everyone. The chosen affinity is recorded in
    each result JSON (`n_cpus_affinity`) either way.

    On the Pi:  BENCH_TASKSET=1,2,3 python3 run_full_benchmark.py
    """
    cores = os.environ.get("BENCH_TASKSET", "").strip()
    if not cores:
        return []
    if shutil.which("taskset") is None:
        print(f"  [WARN] BENCH_TASKSET={cores} set but `taskset` is not "
              f"installed — running unpinned.")
        return []
    return ["taskset", "-c", cores]


_THERM_ZONE  = "/sys/class/thermal/thermal_zone0/temp"
_CPUFREQ     = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq"
_CPUFREQ_MAX = "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq"


def _gate_cores():
    """The cores a job actually runs on, so the sampler watches the right ones."""
    cores = os.environ.get("BENCH_TASKSET", "").strip()
    if cores:
        got = [int(c) for c in cores.split(",") if c.strip().isdigit()]
        if got:
            return got
    return [1]


def _read_int(path):
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except Exception:
        return None


def _temp_c():
    v = _read_int(_THERM_ZONE)
    return None if v is None else v / 1000.0


def _min_cur_freq(cores):
    vals = [v for v in (_read_int(_CPUFREQ.format(c)) for c in cores) if v]
    return min(vals) if vals else None


def _max_rated_freq(cores):
    vals = [v for v in (_read_int(_CPUFREQ_MAX.format(c)) for c in cores) if v]
    return max(vals) if vals else None


_THROTTLE_WORD = "/sys/devices/platform/soc/soc:firmware/get_throttled"
_LIVE_THROTTLE_BITS = 0xF   # 0x1 undervolt, 0x2 arm capped, 0x4 throttled, 0x8 soft limit


def _throttled_word():
    """Live throttle bits from the firmware.

    This sysfs node reports the CURRENT state only, unlike `vcgencmd
    get_throttled`, whose sticky "has occurred" bits (0x?0000) stay latched
    until the next reboot and so cannot distinguish a board that is throttling
    now from one that throttled hours ago. The live bits are what tells us
    whether THIS job was measured on a capped clock.
    """
    try:
        with open(_THROTTLE_WORD) as fh:
            raw = fh.read().strip()
        return int(raw, 16) if raw.lower().startswith("0x") else int(raw)
    except Exception:
        return None


def _thermal_gate(cores):
    """Between jobs ONLY: let a passively cooled board shed heat before the
    next measurement starts.

    This never interrupts a running job. Once a measurement has begun it is
    allowed to finish, because stopping it midway would leave a partial,
    incomparable timing. The gate only delays the START of the next one.

    Why it exists: the Pi 5 in this rig has no fan. Under a sustained 3-core
    load it reaches the soft temperature limit (~82 C), which caps the ARM
    clock (2.40 -> 2.20 GHz observed) and silently inflates every timing taken
    afterwards. Waiting a little between jobs keeps each measurement on an
    unthrottled clock without changing how any single job is measured.

    Tunables (env): BENCH_TEMP_HIGH, BENCH_TEMP_RESUME, BENCH_GATE_MAXWAIT,
    BENCH_INTERJOB_SLEEP; BENCH_THERMAL_GATE=0 disables it entirely.
    """
    if os.environ.get("BENCH_THERMAL_GATE", "1") == "0":
        return None
    high    = float(os.environ.get("BENCH_TEMP_HIGH", "78"))
    resume  = float(os.environ.get("BENCH_TEMP_RESUME", "72"))
    maxwait = float(os.environ.get("BENCH_GATE_MAXWAIT", "300"))
    pause   = float(os.environ.get("BENCH_INTERJOB_SLEEP", "2"))
    if pause > 0:
        time.sleep(pause)
    t = _temp_c()
    if t is None or t < high:
        return None
    t0 = time.time()
    print(f"    [thermal] {t:.1f}C >= {high:.0f}C -- pausing for <= {resume:.0f}C",
          flush=True)
    while time.time() - t0 < maxwait:
        time.sleep(5)
        t = _temp_c()
        if t is not None and t <= resume:
            break
    waited = time.time() - t0
    print(f"    [thermal] resuming at {t:.1f}C after {waited:.0f}s", flush=True)
    return {'gate_wait_s': round(waited, 1), 'gate_release_temp_c': t}


def run_one(approach, backend, scenario, seq, cores=None, gate_info=None):
    out_path = OUT_DIR / f"{approach}_{backend}_{scenario}_{seq}.json"
    cmd = _taskset_prefix() + [sys.executable, str(_WORKER),
          "--approach", approach, "--backend", backend,
          "--scenario", scenario, "--seq", seq, "--out", str(out_path)]

    # Watch temperature and the actual core clock FOR THE DURATION of this job,
    # so a row measured on a throttled clock can be identified and re-measured
    # individually instead of casting doubt over the whole sweep. The sampler
    # only reads two sysfs files twice a second and is not pinned to the
    # benchmark cores, so it does not perturb what it measures.
    cores = cores or _gate_cores()
    fmax = _max_rated_freq(cores)
    samp = {'temp_max': _temp_c() or 0.0,
            'freq_min': _min_cur_freq(cores) or (fmax or 0),
            'throttle': (_throttled_word() or 0) & _LIVE_THROTTLE_BITS,
            'n': 0}
    _stop = threading.Event()

    def _sample():
        while not _stop.is_set():
            t, f, w = _temp_c(), _min_cur_freq(cores), _throttled_word()
            if t is not None:
                samp['temp_max'] = max(samp['temp_max'], t)
            if f:
                samp['freq_min'] = min(samp['freq_min'] or f, f)
            if w:
                samp['throttle'] |= (w & _LIVE_THROTTLE_BITS)
            samp['n'] += 1
            _stop.wait(0.5)

    _th = threading.Thread(target=_sample, daemon=True)
    t0 = time.time()
    _th.start()
    try:
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
    finally:
        _stop.set()
        _th.join(timeout=2)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"    FAILED {approach}/{backend}/{scenario}/{seq} ({dt:.1f}s): "
              f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else '?'}")
        return None
    with open(out_path) as fh:
        r = json.load(fh)

    # The flag comes from the firmware's live throttle bits. Minimum observed
    # frequency is recorded for context but deliberately NOT used as the
    # criterion: the governor idles the cores down to 1.8 GHz between jobs, so
    # a min-frequency test reports throttling on a perfectly cool board.
    capped = bool(samp['throttle'])
    r['thermal'] = {
        'temp_c_max_during': round(samp['temp_max'], 1),
        'cpu_khz_min_during': samp['freq_min'],
        'cpu_khz_rated': fmax,
        'throttle_bits_during': samp['throttle'],
        'throttled_during': capped,
        'samples': samp['n'],
    }
    if gate_info:
        r['thermal'].update(gate_info)
    out_path.write_text(json.dumps(r, indent=2))

    flag = ''
    if capped:
        flag = f"  [THROTTLED bits=0x{samp['throttle']:x} — re-measure]"
    print(f"    ok {approach:16s} {backend:5s} {scenario:10s} seq{seq}  "
          f"wall={r['timing']['wall_s']:.2f}s  ({dt:.1f}s incl. subprocess overhead)"
          f"  {samp['temp_max']:.0f}C{flag}")
    return r


def main():
    cpu_only = '--cpu-only' in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {OUT_DIR}\n")

    jobs = []
    for approach in ALL_APPROACHES:
        backends = (['cpu', 'hailo'] if approach in DL else ['cpu'])
        if cpu_only:
            backends = ['cpu']
        for backend in backends:
            for scenario in SCENARIOS:
                for seq in SEQS:
                    jobs.append((approach, backend, scenario, seq))

    print(f"Total jobs: {len(jobs)}\n")

    results = []
    cores = _gate_cores()
    for i, (approach, backend, scenario, seq) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {approach}/{backend}/{scenario}/seq{seq}")
        gate = _thermal_gate(cores)
        r = run_one(approach, backend, scenario, seq, cores, gate)
        if r is not None:
            results.append(r)

    print(f"\nDone. {len(results)}/{len(jobs)} succeeded.")
    capped = [r for r in results if (r.get('thermal') or {}).get('throttled_during')]
    if capped:
        print(f"[THERMAL] {len(capped)} row(s) measured on a capped clock — "
              f"re-measure these before reporting:")
        for r in capped:
            print(f"    {r['approach']}/{r['backend']}/{r['scenario']}/seq{r['seq']}"
                  f"  {r['thermal']['temp_c_max_during']}C")
    else:
        print("[THERMAL] no row was measured on a capped clock.")

    # Rebuild the aggregate from EVERY *.json on disk, not just this run's
    # results — a partial (e.g. --cpu-only) rerun must not drop previously
    # collected results (e.g. the Hailo rows) from the aggregate.
    all_results = []
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name == "all_results.json":
            continue
        try:
            all_results.append(json.loads(p.read_text()))
        except Exception as exc:
            print(f"  [WARN] could not parse {p}: {exc}")
    (OUT_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"Aggregate rebuilt: {len(all_results)} total results -> {OUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
