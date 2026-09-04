#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build ONE .hef PER LOO FOLD for tlio, deep_kf and tartan_imu.

WHY THIS EXISTS
---------------
The CPU backends load per-fold weights (artifacts/<model>/fold_<seq>.pt, or
lora_fold_<seq>.pt for tartan_imu) and REFUSE to fall back to any other
checkpoint — see tlio_runner.py::_find_weights, which raises with "both would
train/test-leak for held-out sequence". The Hailo side had no such guarantee:
a single tlio.hef / deep_kf.hef / tartan_imu.hef was used for all 7 sequences.

That single HEF was traced to fold_01. Measured on seq 06:

    Hailo (the .hef)            ATE = 5.3955
    CPU forced to fold_01       ATE = 5.4504   <- ~1% apart => same weights
    CPU with fold_06 (correct)  ATE = 7.2033

So on 6 of the 7 sequences the Hailo row was being evaluated on data the model
had trained on. The signature is visible across the table: Hailo "beats" CPU on
5 of 6 leaked sequences, and is WORSE on seq 01 (1.369 vs 1.156) — the one
sequence where fold_01 is legitimately held out and only quantisation differs.

deep_iekf is already correct: deep_iekf_stream/ holds 7 per-fold HEFs and
_full_eval_worker.py::_deep_iekf_stream_paths picks the right one per sequence.
This script brings the other three models up to that standard.

WHERE IT MUST RUN
-----------------
An x86_64 host with the Hailo Dataflow Compiler installed. The DFC is x86-only;
the Raspberry Pi has HailoRT (inference) but cannot compile. Nothing in this
script needs a Hailo device attached — only the compiler.

HOW IT WORKS
------------
Stages 1-3 hardcode canonical filenames in their own directory (e.g.
tlio/tlio.onnx -> tlio/tlio_hailo_model.har -> tlio/tlio.hef), so they cannot
be pointed at a fold-specific name directly. Rather than modify three
already-validated stage scripts, this driver runs the normal pipeline once per
fold and moves the result aside afterwards:

    for each fold F:
        stage 0  --artifact artifacts/<model>/fold_F.pt   (writes <model>.onnx)
        stage 1                                            (-> _hailo_model.har)
        stage 2                                            (-> _quantized.har)
        stage 3                                            (-> <model>.hef)
        mv <model>.hef  <model>_fold_F.hef

The pre-existing single-fold artefacts are backed up to <model>/_pre_per_fold/
before the first build and restored if nothing succeeds.

USAGE
-----
    python3 build_per_fold_hefs.py                    # all 3 models, all 7 folds
    python3 build_per_fold_hefs.py --models tlio      # one model
    python3 build_per_fold_hefs.py --folds 01 06      # subset of folds
    python3 build_per_fold_hefs.py --dry-run          # print the plan only

AFTER IT FINISHES
-----------------
Copy the <model>/<model>_fold_*.hef files to the Pi (same paths), then apply
the loader change described in todo.md so _full_eval_worker.py selects the
per-fold HEF, and re-run the Hailo half of the sweep.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent

SEQS = ['01', '04', '06', '07', '08', '09', '10']

# model -> (subdir, stage0 filename, artifact dir, per-fold checkpoint pattern)
MODELS = {
    'tlio': {
        'dir': _HERE / 'tlio',
        'stage0': '0_onnx_converter.py',
        'artifacts': _REPO_ROOT / 'artifacts/tlio',
        'ckpt': 'fold_{seq}.pt',
        'hef': 'tlio.hef',
        'takes_artifact': True,
        'postproc': 'tlio_postproc.pt',
    },
    'deep_kf': {
        'dir': _HERE / 'deep_kf',
        'stage0': '0_onxx_converter.py',      # note: upstream spelling
        'artifacts': _REPO_ROOT / 'artifacts/deep_kf',
        'ckpt': 'fold_{seq}.pt',
        'hef': 'deep_kf.hef',
        'takes_artifact': True,
        'postproc': None,   # whole net is on-device; no host-side head
    },
    'tartan_imu': {
        'dir': _HERE / 'tartan_imu',
        'stage0': '0_onnx_converter.py',
        'artifacts': _REPO_ROOT / 'artifacts/tartan_imu',
        'ckpt': 'lora_fold_{seq}.pt',
        'hef': 'tartan_imu.hef',
        # Verified on the x86 box, as todo.md asked. Stage 0 did NOT honour a
        # fold env var: the driver set TARTAN_LORA while tartan_runner reads
        # TARTAN_IMU_LORA, and _find_lora_adapter() with no seq_id falls back to
        # the all-sequences adapter — which does not exist here, so it would have
        # dropped to ZERO-SHOT and produced seven identical, LoRA-less "per-fold"
        # HEFs without erroring. Fixed the way todo.md specified: stage 0 (and
        # stage 2) now take --artifact, mirroring tlio's.
        'takes_artifact': True,
        'postproc': 'tartan_imu_postproc.pt',
    },
}

# Stage 2 takes --artifact too, and it matters for more than the printout:
#   * tlio   — its default is artifacts/tlio/tlio_resnet.pt, which does not exist
#              in this repo, so stage 2 dies on torch.load before quantising.
#   * deep_kf— the INT8 calibration set is CAPTURED from a live CPU run driven by
#              this checkpoint (DEEP_KF_WEIGHTS), so the wrong artifact calibrates
#              fold F's graph on fold_01's error-state distribution.
#   * tartan — selects the LoRA fold behind the 'MAE vs PyTorch' check.
STAGES = ['1_parsing.py', '2_optimisation.py', '3_compilation.py']
STAGES_TAKING_ARTIFACT = {'2_optimisation.py'}

# deep_kf's INT8 ranges are calibrated from states captured by running the CPU
# filter. Which sequences those come from matters: the model runs
# autoregressively during an outage, so a single drive does not cover the range
# it actually visits, and an under-covered range shows up as a systematic bias
# rather than as noise. Calibrate on this fold's FIVE TRAINING sequences —
# broad coverage, and still leak-free because the held-out sequence is excluded.
MODELS_TAKING_CALIB_SEQS = {'deep_kf'}

# Removed before each fold so no stage can silently consume the previous fold's
# output when an earlier stage fails. Without this, one failed export leaves a
# stale .onnx/.har on disk and the next stage happily builds a wrong-fold HEF.
_INTERMEDIATE_GLOBS = ('*.onnx', '*_hailo_model.har', '*_quantized_model.har',
                       '*_compiled_model.har', '*.hef', '*_postproc.pt')


def _run(cmd, cwd, env=None, dry=False):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    if dry:
        return True
    p = subprocess.run(cmd, cwd=str(cwd), env=env)
    return p.returncode == 0


def build_model(name, cfg, folds, dry=False):
    mdir = cfg['dir']
    if not mdir.is_dir():
        print(f"  SKIP {name}: {mdir} not found")
        return {}
    backup = mdir / '_pre_per_fold'
    if not backup.exists() and not dry:
        backup.mkdir(parents=True, exist_ok=True)
        for pat in ('*.hef', '*.har', '*.onnx'):
            for f in mdir.glob(pat):
                shutil.copy2(f, backup / f.name)
        print(f"  backed up existing artefacts -> {backup}")

    import os
    results = {}
    for seq in folds:
        ckpt = cfg['artifacts'] / cfg['ckpt'].format(seq=seq)
        if not ckpt.exists():
            print(f"  {name} fold {seq}: MISSING {ckpt} — skipped")
            results[seq] = False
            continue
        print(f"  --- {name} fold {seq} ({ckpt.name}) ---")

        # Wipe intermediates so a stage that fails cannot leave the previous
        # fold's artefact for the next stage to pick up. OTHER folds' finished
        # outputs (<model>_fold_<other>.hef) are preserved, but THIS fold's are
        # deleted up front: on a rebuild they already exist, and if this run then
        # fails the old files would survive and be loaded by the evaluator as
        # though they were current — silently pinning the results to a previous,
        # superseded checkpoint. Absent files fail loudly; stale files do not.
        if not dry:
            for pat in _INTERMEDIATE_GLOBS:
                for f in mdir.glob(pat):
                    if '_fold_' in f.name and f'_fold_{seq}.' not in f.name:
                        continue
                    f.unlink()

        env = dict(os.environ)
        cmd0 = [sys.executable, cfg['stage0'], '--artifact', str(ckpt)]

        ok = _run(cmd0, mdir, env=env, dry=dry)
        for stage in STAGES:
            if not ok:
                break
            cmd = [sys.executable, stage]
            if stage in STAGES_TAKING_ARTIFACT:
                cmd += ['--artifact', str(ckpt)]
                if name in MODELS_TAKING_CALIB_SEQS:
                    cmd += ['--calib-seqs'] + [q for q in SEQS if q != seq]
            ok = _run(cmd, mdir, env=env, dry=dry)

        if not ok:
            print(f"  {name} fold {seq}: FAILED")
            results[seq] = False
            continue

        # The .hef is only half the model for tlio and tartan_imu: the partitioning
        # leaves a host-side head on the CPU (tlio's bn1+fc1/2/3, tartan's
        # LSTM+Trunk+robot head) whose weights stage 0 writes to <model>_postproc.pt
        # STRAIGHT FROM THE FOLD CHECKPOINT. Those are as fold-specific as the
        # backbone, so they have to travel with it — otherwise fold F's accelerated
        # backbone runs into whichever fold's head was exported last, which is a
        # different kind of leak from the one this script was written to fix, and a
        # louder one. deep_iekf_stream already pairs them this way.
        moves = [(mdir / cfg['hef'],
                  mdir / cfg['hef'].replace('.hef', f'_fold_{seq}.hef'))]
        if cfg.get('postproc'):
            pp = cfg['postproc']
            stem, ext = pp.rsplit('.', 1)
            moves.append((mdir / pp, mdir / f'{stem}_fold_{seq}.{ext}'))

        if not dry:
            missing = [s.name for s, _ in moves if not s.exists()]
            if missing:
                # Only reachable if a stage exits 0 without writing — the wipe
                # above means it can never be a stale file from another fold.
                print(f"  {name} fold {seq}: pipeline produced no {', '.join(missing)}")
                results[seq] = False
                continue
            for s, d in moves:
                shutil.move(str(s), str(d))
        print(f"  {name} fold {seq}: -> {', '.join(d.name for _, d in moves)}")
        results[seq] = True
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=list(MODELS),
                    choices=list(MODELS))
    ap.add_argument('--folds', nargs='+', default=SEQS, choices=SEQS)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    try:
        import hailo_sdk_client  # noqa: F401
    except Exception:
        print("WARNING: hailo_sdk_client not importable — this host probably has\n"
              "         no Dataflow Compiler. The DFC is x86-only; the Pi cannot\n"
              "         compile. Re-run this on the x86 build machine.\n")
        if not args.dry_run:
            return 1

    summary = {}
    for name in args.models:
        print(f"\n=== {name} ===")
        summary[name] = build_model(name, MODELS[name], args.folds, dry=args.dry_run)

    print("\n=== SUMMARY ===")
    rc = 0
    for name, res in summary.items():
        good = sum(1 for v in res.values() if v)
        print(f"  {name}: {good}/{len(res)} folds built")
        if good != len(res):
            rc = 1
    return rc


if __name__ == '__main__':
    sys.exit(main())
