"""
train_ai_imu.py — Training script for the AI-IMU deep IEKF.

Usage
-----
  # Full training on raw KITTI data (recommended, matches paper):
  python dl_filters/deep_iekf/train_ai_imu.py \\
      --mode kitti \\
      --kitti-raw-dir /path/to/kitti/raw \\
      --epochs 400 \\
      --output artifacts/deep_iekf

  # Single-sequence fine-tuning / sanity check on our KITTI .mat file:
  python dl_filters/deep_iekf/train_ai_imu.py \\
      --mode single \\
      --epochs 50 \\
      --output artifacts/deep_iekf

  # CAUSAL / online model for Hailo (strictly no future samples):
  #   swaps MesNet → CausalMesNet (left-padded), saves to artifacts/deep_iekf_online/.
  #   Trains FROM SCRATCH by default so a causal-vs-acausal comparison is fair
  #   (add "--warm-start" only if you want the best deployable model fast).
  #   Run it with iekf_ai_imu_online.py (key 'iekf_ai_imu_online').
  python dl_filters/deep_iekf/train_ai_imu.py \\
      --mode loo --causal --held-out 2011_09_30_drive_0033_extract --epochs 400

NOTE on 10 Hz vs 100 Hz
-----------------------
--mode kitti  : Uses raw KITTI OXTS data at 100 Hz, matching the paper's setup.
                Requires the full KITTI raw dataset directory containing date folders
                (2011_09_26/, 2011_09_30/, 2011_10_03/, …).
                Download: http://www.cvlibs.net/datasets/kitti/raw_data.php

--mode single : Uses our single 10 Hz sequence (10_03_0027 from .mat).
                The model will train/overfit to one sequence — useful only for
                debugging the pipeline.  Not suitable for production.

After training, the weights file (iekfnets.p) and normalization factors
(iekfnets_norm.p) are saved to the output directory.  iekf_ai_imu.py picks them
up automatically.

Credit / License
----------------
Training infrastructure (train_torch_filter.py, utils_torch_filter.py) and
KITTIDataset / KITTIParameters are the work of Martin Brossard, Axel Barrau, and
Silvère Bonnabel, distributed under the MIT License (external/ai-imu-dr/LICENSE).
Original repo: https://github.com/mbrossar/ai-imu-dr
This script is a thin driver that calls their training loop with our dataset adapter.
"""

import sys
import os
import argparse
import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
_REPO_ROOT   = _HERE.parent.parent.parent.parent.parent
_AI_IMU_SRC  = _REPO_ROOT / 'external/ai-imu-dr/src'
_SCRIPT_DIR  = _HERE.parent.parent     # scripts/positioning/python/
_ARTIFACTS   = _REPO_ROOT / 'artifacts/deep_iekf'
_ARTIFACTS_ONLINE = _REPO_ROOT / 'artifacts/deep_iekf_online'   # causal weights


def _add_paths():
    for p in [str(_AI_IMU_SRC), str(_SCRIPT_DIR), str(_HERE)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _set_seed(seed):
    """
    Seed Python/NumPy/Torch RNGs for reproducible DL training. Upstream AI-IMU sets
    no seed, so training is otherwise non-deterministic (random init, dropout, batch
    order). With a fixed seed, MesNet and CausalMesNet also get identical conv init
    (pad layers consume no RNG, param-creation order matches) → controlled comparison.
    No-op when seed is None (preserves the original non-deterministic behaviour).
    """
    if seed is None:
        return
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"  RNG seed = {seed} (reproducible training)")


def _resolve_warm_start(warm_start, held_out):
    """
    Resolve the warm-start source for causal training.

    warm_start is None    → DO NOT warm-start (train from scratch). This is the
                            default, so a causal-vs-acausal comparison isolates the
                            effect of causality and is not confounded by inheriting
                            acausal-learned features.
    warm_start == 'auto'  → prefer the SAME-fold acausal weights
                            (artifacts/deep_iekf/fold_<seq>.p — same LOO split, no
                            held-out leak), else the generic iekfnets.p.
    warm_start == <path>  → use that file.

    Use 'auto'/<path> only when the goal is the best deployable causal model fast,
    not a controlled comparison. Returns a path string or None.
    """
    if warm_start is None:
        return None
    if warm_start != 'auto':
        return warm_start
    try:
        from kitti_sequences import KITTI_SEQ_TO_DRIVE
        _d2s = {v: k for k, v in KITTI_SEQ_TO_DRIVE.items()}
    except Exception:
        _d2s = {}
    if held_out:
        seq = _d2s.get(held_out, held_out)
        cand = _ARTIFACTS / f'fold_{seq}.p'
        if cand.exists():
            return str(cand)
    generic = _ARTIFACTS / 'iekfnets.p'
    return str(generic) if generic.exists() else None


def _maybe_make_causal(iekf, causal, warm_start=None):
    """
    If `causal`, replace iekf.mes_net with a CausalMesNet (left-padded, strictly
    online).  Optionally warm-start its conv/linear weights from an existing
    acausal weights file (`warm_start` path) for faster convergence.
    Must be called AFTER prepare_filter and BEFORE set_optimizer.
    """
    if not causal:
        return iekf
    from causal_mesnet import attach_causal_mesnet
    ws = None
    if warm_start is not None and Path(warm_start).exists():
        ws = torch.load(warm_start, map_location='cpu')
        print(f"  Causal MesNet: warm-starting from {warm_start}")
    else:
        print("  Causal MesNet: training from scratch (no warm-start)")
    attach_causal_mesnet(iekf, warm_start_state_dict=ws)
    iekf.train()
    return iekf


# ── Journal-metric validation hook (display-only) ─────────────────────────────

def _build_aiimu_val_hook(output_dir: Path, val_drive, val_metric_every: int,
                          causal: bool = False):
    """
    Return a callable hook(epoch) -> J that runs the journal three-component
    metric on VAL_DRIVE using the just-saved weights, and returns the resulting
    cost (inf on failure). Returns None if validation should be skipped.

    val_drive is the INNER-VALIDATION drive, never the LOO held-out one. This
    hook previously scored the held-out (test) drive: harmless for the deployed
    weights, since nothing selected on it, but it printed the test metric every
    K epochs, which is the same selection-on-test problem the other three models
    had — just routed through whoever was reading the log rather than through
    code. It now also drives checkpoint selection, so pointing it at the test
    drive would be an outright leak.

    When `causal`, validation uses the online runner (iekf_ai_imu_online) loading
    the CausalMesNet weights via AI_IMU_ONLINE_WEIGHTS; otherwise the batch runner.
    """
    if not val_drive or not val_metric_every:
        return None
    try:
        from kitti_sequences import KITTI_SEQ_TO_DRIVE
    except Exception as e:
        print(f"  [val] AI-IMU hook unavailable ({e}) — skipping journal metric")
        return None
    _drive_to_seq = {v: k for k, v in KITTI_SEQ_TO_DRIVE.items()}
    val_seq = _drive_to_seq.get(val_drive, val_drive)
    if val_seq not in {'01', '04', '06', '07', '08', '09', '10'}:
        return None
    runner_mod = ('dl_filters.deep_iekf.iekf_ai_imu_online' if causal
                  else 'dl_filters.deep_iekf.iekf_ai_imu')
    env_var = 'AI_IMU_ONLINE_WEIGHTS' if causal else 'AI_IMU_WEIGHTS'
    try:
        from dl_filters._validation import (validate_with_journal_metric,
                                            format_val_line)
        import importlib
        runner = importlib.import_module(runner_mod)
    except Exception as e:
        print(f"  [val] AI-IMU hook unavailable ({e}) — skipping journal metric")
        return None
    weights_canonical = Path(output_dir) / 'iekfnets.p'

    def _hook(epoch):
        import os as _os
        if not weights_canonical.exists():
            return float('inf')
        _prev = _os.environ.get(env_var)
        _os.environ[env_var] = str(weights_canonical)
        _J = float('inf')
        try:
            m = validate_with_journal_metric(filter_module=runner, val_seq=val_seq)
            print(format_val_line(epoch, m))
            _J = float(m.get('J', float('inf')))
        except Exception as exc:
            print(f"  [val] journal-metric hook failed: {exc}")
        finally:
            if _prev is None:
                _os.environ.pop(env_var, None)
            else:
                _os.environ[env_var] = _prev
        return _J
    return _hook


# ── KITTI mode ─────────────────────────────────────────────────────────────────

def train_kitti(kitti_raw_dir, output_dir, epochs, continue_training,
                held_out=None, val_metric_every=0, causal=False, warm_start=None,
                seed=None):
    """
    Train using the AI-IMU's own KITTIDataset on raw 100 Hz KITTI data.
    This matches the paper's training setup exactly.

    held_out : drive name to exclude from training (LOO protocol).
               Format: '2011_09_30_drive_0033_extract' (with or without _extract).
               When set, weights are saved as iekfnets_held_<drive>.p.
    """
    _add_paths()
    _set_seed(seed)
    from main_kitti import KITTIParameters, KITTIDataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import types
    args = types.SimpleNamespace()
    args.path_data_base    = str(kitti_raw_dir)
    args.path_temp         = str(output_dir)
    args.dataset_class     = KITTIDataset
    args.parameter_class   = KITTIParameters
    args.epochs            = epochs
    args.seq_dim           = 60 * 100       # 60 s batches at 100 Hz
    args.continue_training = continue_training
    args.read_data         = True
    args.train_filter      = True
    args.test_filter       = False
    args.results_filter    = False

    print(f"Starting KITTI training: {epochs} epochs")
    print(f"  Raw data dir : {kitti_raw_dir}")
    print(f"  Output dir   : {output_dir}")
    if held_out:
        print(f"  LOO held-out : {held_out}")

    if held_out is None:
        # Standard full training via launch()
        if causal:
            raise SystemExit("--causal requires a manual training loop; use "
                             "--mode loo (recommended) or pass --held-out.")
        from main_kitti import launch
        launch(args)
    else:
        # LOO: create dataset, remove held-out sequence, train manually
        from train_torch_filter import (prepare_filter, prepare_loss_data,
                                        set_optimizer, train_loop, save_iekf)
        dataset = KITTIDataset(args)

        # Drive names in KITTIDataset may omit the _extract suffix
        held_variants = {held_out, held_out.replace('_extract', '')}
        removed = False
        for key in list(dataset.datasets_train_filter.keys()):
            if any(v in key for v in held_variants):
                del dataset.datasets_train_filter[key]
                print(f"  Removed '{key}' from training sequences.")
                removed = True
        if not removed:
            print(f"  WARNING: held-out '{held_out}' not found in "
                  f"datasets_train_filter keys: "
                  f"{list(dataset.datasets_train_filter.keys())}")

        iekf      = prepare_filter(args, dataset)
        iekf      = _maybe_make_causal(iekf, causal, _resolve_warm_start(warm_start, held_out))
        prepare_loss_data(args, dataset)
        optimizer = set_optimizer(iekf)
        _val_hook = _build_aiimu_val_hook(output_dir, held_out, val_metric_every,
                                          causal=causal)
        for epoch in range(1, epochs + 1):
            train_loop(args, dataset, epoch, iekf, optimizer, args.seq_dim)
            save_iekf(args, iekf)
            if epoch % 50 == 0 or epoch == epochs:
                print(f"  Epoch {epoch}/{epochs} done")
            if _val_hook is not None and (
                    epoch % val_metric_every == 0 or epoch == epochs):
                _val_hook(epoch)

        # Copy default iekfnets.p → fold-specific filename
        import shutil
        src  = output_dir / 'iekfnets.p'
        dst  = output_dir / f'iekfnets_held_{held_out}.p'
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Fold weights saved to {dst}")

    # Save normalization factors alongside the weights
    _save_norm_factors(output_dir, args)

    # Export to ONNX (standard weights only)
    _export_onnx(output_dir)


def _save_norm_factors(output_dir, args):
    """Save u_loc and u_std so the inference wrapper can normalize inputs."""
    _add_paths()
    try:
        from main_kitti import KITTIParameters, KITTIDataset
        from utils_torch_filter import TORCHIEKF

        dataset = KITTIDataset(args)
        torch_iekf = TORCHIEKF()
        torch_iekf.filter_parameters = args.parameter_class()
        torch_iekf.set_param_attr()
        torch_iekf.get_normalize_u(dataset)

        norm = {
            'u_loc': torch_iekf.u_loc,
            'u_std': torch_iekf.u_std,
        }
        norm_path = Path(output_dir) / 'iekfnets_norm.p'
        torch.save(norm, norm_path)
        print(f"Normalization factors saved to {norm_path}")
    except Exception as e:
        print(f"Warning: could not save normalization factors: {e}")


# ── LOO mode (data_loader pickle files, no raw KITTI needed) ─────────────────

def train_loo(output_dir, epochs, held_out=None, val_metric_every=0,
              causal=False, warm_start=None, seed=None):
    """
    Train using our preprocessed KITTI pickle files (datasets/raw_kitti/<drive>.p)
    at 100 Hz — same frequency as the AI-IMU paper — without needing the full raw
    KITTI dataset directory.

    held_out : full drive name excluded from training and used as validation,
               e.g. '2011_09_30_drive_0028_extract'.
               When set, weights are also saved as iekfnets_held_<drive>.p.
    """
    import types
    import shutil
    _add_paths()
    _set_seed(seed)
    from kitti_sequences import MultiSequenceDataset, KITTI_CLEAN_SEQS
    from main_kitti import KITTIParameters
    from train_torch_filter import (prepare_filter, prepare_loss_data,
                                    set_optimizer, train_loop, save_iekf)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"LOO training: {epochs} epochs")
    if held_out:
        print(f"  LOO held-out : {held_out}")
    print(f"  Output dir   : {output_dir}")

    dataset = MultiSequenceDataset(seqs=KITTI_CLEAN_SEQS, held_out=held_out,
                                   sample_rate=100.0)

    # NESTED split, matching TLIO / Deep KF / Tartan IMU. MultiSequenceDataset
    # puts only the held-out drive aside; one more drive is carved out here as
    # the INNER VALIDATION set, and it drives both the journal metric and
    # checkpoint selection below.
    #
    # This departs from the upstream protocol, which trains on everything except
    # the held-out drive and deploys the final epoch. The trade is deliberate:
    # the other three estimators each spend one drive on validation, so leaving
    # this one training on all of them gave it ~19% more data on average, and
    # deploying its last epoch meant it was the only model with no guard against
    # a bad final epoch. Consistency across the four is worth more here than
    # fidelity to a training-loop detail.
    from kitti_sequences import KITTI_SEQ_TO_DRIVE
    _d2s = {v: k for k, v in KITTI_SEQ_TO_DRIVE.items()}
    inner_val_drive = None
    if held_out:
        from dl_filters._validation import inner_split
        _held_seq = _d2s.get(held_out, held_out)
        if _held_seq in KITTI_CLEAN_SEQS:
            _, _inner_seq = inner_split(list(KITTI_CLEAN_SEQS), _held_seq)
            inner_val_drive = KITTI_SEQ_TO_DRIVE[_inner_seq]
            for key in list(dataset.datasets_train_filter.keys()):
                if inner_val_drive in key or inner_val_drive.replace('_extract', '') in key:
                    del dataset.datasets_train_filter[key]
                    print(f"  Inner validation: seq {_inner_seq} ({key}) "
                          f"removed from training.")
            print(f"  Training drives  : {list(dataset.datasets_train_filter)}")

    args = types.SimpleNamespace()
    args.path_temp         = str(output_dir)
    args.parameter_class   = KITTIParameters
    args.epochs            = epochs
    args.seq_dim           = 60 * 100   # 60 s batches at 100 Hz (matches kitti mode)
    args.continue_training = False

    iekf      = prepare_filter(args, dataset)
    iekf      = _maybe_make_causal(iekf, causal, _resolve_warm_start(warm_start, held_out))
    prepare_loss_data(args, dataset)
    optimizer = set_optimizer(iekf)
    _val_hook = _build_aiimu_val_hook(output_dir, inner_val_drive, val_metric_every,
                                      causal=causal)

    seq_id = _d2s.get(held_out, held_out) if held_out else None
    stem = f'fold_{seq_id}' if held_out else 'iekfnets'
    fold_dst = output_dir / f'{stem}.p'
    best_J, best_J_epoch = float('inf'), None

    for epoch in range(1, epochs + 1):
        train_loop(args, dataset, epoch, iekf, optimizer, args.seq_dim)
        save_iekf(args, iekf)          # canonical iekfnets.p, read by the hook
        if epoch % 50 == 0 or epoch == epochs:
            print(f"  Epoch {epoch}/{epochs} done")
        if _val_hook is not None and (
                epoch % val_metric_every == 0 or epoch == epochs):
            _J = _val_hook(epoch)
            # Keep the BEST epoch by the journal metric on the inner-validation
            # drive, as the other three models do, rather than whatever the last
            # epoch happened to produce.
            if held_out and _J < best_J:
                best_J, best_J_epoch = _J, epoch
                shutil.copy2(output_dir / 'iekfnets.p', fold_dst)
                print(f"  -> saved best ({fold_dst.name})  J={_J:.4f} @ epoch {epoch}")

    if held_out:
        if _val_hook is None:
            # No validation available: fall back to the upstream behaviour, but
            # say so, because the resulting file is NOT a selected checkpoint.
            src = output_dir / 'iekfnets.p'
            if src.exists():
                shutil.copy2(src, fold_dst)
                print(f"  [warn] no validation hook — deployed FINAL epoch to {fold_dst}")
        elif best_J_epoch is None:
            raise RuntimeError(
                f"No checkpoint selected for {stem}: the journal metric was inf "
                f"at every evaluated epoch on inner-validation drive "
                f"{inner_val_drive}. Refusing to deploy an unselected checkpoint.")
        else:
            print(f"  Selected epoch {best_J_epoch} (J={best_J:.4f} on "
                  f"inner-val {inner_val_drive}) -> {fold_dst}")

    norm_path = output_dir / f'{stem}_norm.p'
    torch.save(dataset.normalize_factors, norm_path)
    print(f"  Normalization factors saved to {norm_path}")

    # Convenience export only. It is NOT on the deployment path: the Hailo
    # binaries come from deep_iekf_stream/build_stream.py, which exports its own
    # ONNX from these weights. It currently fails on some opset/onnx combinations
    # ("No Adapter To Version 17 for Pad"), and a failure here must not mark the
    # fold as failed and discard a checkpoint that was already written.
    try:
        _export_onnx(output_dir, weights_stem=stem)
    except Exception as exc:
        print(f"  [warn] optional ONNX export failed ({type(exc).__name__}: {exc}); "
              f"weights at {fold_dst if held_out else output_dir} are unaffected.")


# ── Single-sequence mode ───────────────────────────────────────────────────────

def train_single(output_dir, epochs):
    """
    Train / fine-tune on a single 10 Hz sequence loaded from our .mat file.
    This will overfit — useful only for pipeline validation.
    """
    _add_paths()
    import ins_config
    from kitti_sequences import SingleSequenceDataset
    from utils_torch_filter import TORCHIEKF
    from train_torch_filter import (train_filter as _train_filter, prepare_filter,
                                    prepare_loss_data, set_optimizer, train_loop,
                                    save_iekf)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset from our NavigationData
    nav_data = ins_config.NAV_DATA
    dataset  = SingleSequenceDataset(nav_data, name=nav_data.dataset_name)

    # Build args namespace
    import types
    args = types.SimpleNamespace()
    args.path_temp         = str(output_dir)
    args.continue_training = False
    args.epochs            = epochs
    args.seq_dim           = min(60 * int(nav_data.sample_rate),
                                  len(nav_data.accel_flu) - 1)

    # KITTIParameters gives better starting point than base Parameters
    try:
        from main_kitti import KITTIParameters
        args.parameter_class = KITTIParameters
    except Exception:
        from utils_numpy_filter import NUMPYIEKF
        args.parameter_class = NUMPYIEKF.Parameters

    print(f"Single-sequence training: {epochs} epochs on '{nav_data.dataset_name}'")
    print(f"  Sample rate  : {nav_data.sample_rate} Hz (paper uses 100 Hz)")
    print(f"  Output dir   : {output_dir}")
    print("  WARNING: single-sequence training will overfit. Use --mode kitti for real results.")

    iekf = prepare_filter(args, dataset)
    prepare_loss_data(args, dataset)
    save_iekf(args, iekf)
    optimizer = set_optimizer(iekf)

    import time
    for epoch in range(1, epochs + 1):
        train_loop(args, dataset, epoch, iekf, optimizer, args.seq_dim)
        save_iekf(args, iekf)
        print(f"Epoch {epoch}/{epochs} done")

    # Save normalization factors
    norm = {
        'u_loc': iekf.u_loc,
        'u_std': iekf.u_std,
    }
    norm_path = output_dir / 'iekfnets_norm.p'
    torch.save(norm, norm_path)
    print(f"Weights  : {output_dir / 'iekfnets.p'}")
    print(f"Norm     : {norm_path}")

    # Export to ONNX
    _export_onnx(output_dir)


# ── ONNX export ────────────────────────────────────────────────────────────────

def _export_onnx(output_dir, weights_stem='iekfnets'):
    """Export the trained MesNet CNN to ONNX after training completes.

    weights_stem : filename stem (without extension) for both the source .p
                   weights and the output .onnx, e.g. 'iekfnets_held_<drive>'.
    """
    try:
        from export_onnx import export as _export
        weights_path = Path(output_dir) / f'{weights_stem}.p'
        onnx_path    = Path(output_dir) / f'{weights_stem}.onnx'
        _export(weights_path, onnx_path)
    except Exception as e:
        print(f"Warning: ONNX export failed ({e}). "
              "Run export_onnx.py manually after installing onnx and onnxruntime.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Train the AI-IMU deep IEKF (Brossard et al. 2020)')
    p.add_argument('--mode', choices=['kitti', 'single', 'loo'], default='single',
                   help='loo: LOO training via data_loader pickle files at 100 Hz (default for ins_train.py); '
                        'kitti: full training on raw KITTI 100 Hz data (requires --kitti-raw-dir); '
                        'single: quick test on one 10 Hz sequence')
    p.add_argument('--kitti-raw-dir', type=str, default=None,
                   help='[kitti mode] Path to KITTI raw data root directory '
                        '(contains 2011_09_26/, etc.)')
    p.add_argument('--epochs', type=int, default=400,
                   help='Number of training epochs (paper uses 400)')
    p.add_argument('--output', type=str,
                   default=None,
                   help='Output directory for weights and normalization factors '
                        '(default: artifacts/deep_iekf/, or artifacts/deep_iekf_online/ '
                        'when --causal)')
    p.add_argument('--causal', action=argparse.BooleanOptionalAction, default=True,
                   help='Train a strictly causal (left-padded) CausalMesNet for online '
                        'use (no future samples). DEFAULT — causal online is the primary '
                        'AI-IMU. Output folder: artifacts/deep_iekf_online/. Pass '
                        '--no-causal to train the acausal batch model instead '
                        '(artifacts/deep_iekf/), kept only as a diagnostic reference. '
                        'Requires --mode loo or --held-out.')
    p.add_argument('--warm-start', dest='warm_start', nargs='?', const='auto',
                   default=None,
                   help='[--causal] Warm-start the causal conv layers from acausal '
                        'weights. DEFAULT: OFF — train from scratch, so a causal-vs-'
                        'acausal comparison is not confounded. Pass "--warm-start" '
                        '(no value) to auto-pick the same-fold acausal weights, or '
                        '"--warm-start <path>" for a specific file. Use only when you '
                        'want the best deployable causal model fast, not a fair study.')
    p.add_argument('--continue', dest='continue_training', action='store_true',
                   help='[kitti mode] Resume from existing iekfnets.p in output dir')
    p.add_argument('--held-out', dest='held_out', default=None,
                   help='[kitti mode] LOO: drive name to exclude from training '
                        '(e.g. 2011_09_30_drive_0033_extract). Weights saved as '
                        'iekfnets_held_<drive>.p. Use run_loo_evaluation.py to '
                        'run all 7 folds automatically.')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for reproducible training (default: 42, matching '
                        'the genetic optimiser and the other DL filters). With a fixed '
                        'seed the acausal and causal models share identical conv init, '
                        'giving a controlled "cost of causality" comparison.')
    p.add_argument('--val-metric-every', type=int, default=20,
                   help='[kitti/loo modes] Every K epochs (and at final epoch), '
                        'run a display-only validation on the held-out sequence '
                        'using the journal three-component metric '
                        'J = ATE_outage + t_rel + r_rel (default 50; 0 disables). '
                        'Original Brossard covariance-regression loss is unchanged.')
    args = p.parse_args()

    # Resolve default output folder (causal weights live in a separate folder).
    if args.output is None:
        args.output = str(_ARTIFACTS_ONLINE if args.causal else _ARTIFACTS)
    # Warm-start is resolved per-fold inside train_loo/train_kitti (fold-aware).
    warm_start = args.warm_start

    if args.mode == 'kitti':
        if args.kitti_raw_dir is None:
            p.error('--kitti-raw-dir is required for --mode kitti')
        train_kitti(
            kitti_raw_dir=args.kitti_raw_dir,
            output_dir=args.output,
            epochs=args.epochs,
            continue_training=args.continue_training,
            held_out=args.held_out,
            val_metric_every=args.val_metric_every,
            causal=args.causal,
            warm_start=warm_start,
            seed=args.seed,
        )
    elif args.mode == 'loo':
        train_loo(
            output_dir=args.output,
            epochs=args.epochs,
            held_out=args.held_out,
            val_metric_every=args.val_metric_every,
            causal=args.causal,
            warm_start=warm_start,
            seed=args.seed,
        )
    else:
        # --mode single is a smoke path only and does not support causal training;
        # fall back to the acausal model (and its folder) even though causal is the
        # default elsewhere.
        if args.causal:
            print("  NOTE: --mode single does not support causal training; using the "
                  "acausal batch model for this smoke run. Use --mode loo for causal.")
            if args.output == str(_ARTIFACTS_ONLINE):
                args.output = str(_ARTIFACTS)
        train_single(
            output_dir=args.output,
            epochs=args.epochs,
        )


if __name__ == '__main__':
    main()
