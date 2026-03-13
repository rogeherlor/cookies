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
_HERE        = Path(__file__).parent
_REPO_ROOT   = _HERE.parent.parent.parent.parent.parent
_AI_IMU_SRC  = _REPO_ROOT / 'external/ai-imu-dr/src'
_SCRIPT_DIR  = _HERE.parent.parent     # scripts/positioning/python/
_ARTIFACTS   = _REPO_ROOT / 'artifacts/deep_iekf'


def _add_paths():
    for p in [str(_AI_IMU_SRC), str(_SCRIPT_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ── KITTI mode ─────────────────────────────────────────────────────────────────

def train_kitti(kitti_raw_dir, output_dir, epochs, continue_training):
    """
    Train using the AI-IMU's own KITTIDataset on raw 100 Hz KITTI data.
    This matches the paper's training setup exactly.
    """
    _add_paths()
    from main_kitti import KITTIParameters, KITTIDataset, launch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build args namespace compatible with AI-IMU's launch()
    import types
    args = types.SimpleNamespace()
    args.path_data_base  = str(kitti_raw_dir)
    args.path_temp       = str(output_dir)
    args.dataset_class   = KITTIDataset
    args.parameter_class = KITTIParameters
    args.epochs          = epochs
    args.seq_dim         = 60 * 100       # 60 s batches at 100 Hz
    args.continue_training = continue_training
    args.read_data       = True           # parse raw KITTI on first run
    args.train_filter    = True
    args.test_filter     = False
    args.results_filter  = False

    print(f"Starting KITTI training: {epochs} epochs")
    print(f"  Raw data dir : {kitti_raw_dir}")
    print(f"  Output dir   : {output_dir}")
    launch(args)

    # Save normalization factors alongside the weights
    _save_norm_factors(output_dir, args)

    # Export to ONNX
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

def _export_onnx(output_dir):
    """Export the trained MesNet CNN to ONNX after training completes."""
    try:
        from export_onnx import export as _export
        weights_path = Path(output_dir) / 'iekfnets.p'
        onnx_path    = Path(output_dir) / 'iekfnets.onnx'
        _export(weights_path, onnx_path)
    except Exception as e:
        print(f"Warning: ONNX export failed ({e}). "
              "Run export_onnx.py manually after installing onnx and onnxruntime.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Train the AI-IMU deep IEKF (Brossard et al. 2020)')
    p.add_argument('--mode', choices=['kitti', 'single'], default='single',
                   help='kitti: full training on raw KITTI 100 Hz data (recommended); '
                        'single: quick test on one 10 Hz sequence')
    p.add_argument('--kitti-raw-dir', type=str, default=None,
                   help='[kitti mode] Path to KITTI raw data root directory '
                        '(contains 2011_09_26/, etc.)')
    p.add_argument('--epochs', type=int, default=400,
                   help='Number of training epochs (paper uses 400)')
    p.add_argument('--output', type=str,
                   default=str(_ARTIFACTS),
                   help='Output directory for weights and normalization factors '
                        '(default: artifacts/deep_iekf/ at repo root)')
    p.add_argument('--continue', dest='continue_training', action='store_true',
                   help='[kitti mode] Resume from existing iekfnets.p in output dir')
    args = p.parse_args()

    if args.mode == 'kitti':
        if args.kitti_raw_dir is None:
            p.error('--kitti-raw-dir is required for --mode kitti')
        train_kitti(
            kitti_raw_dir=args.kitti_raw_dir,
            output_dir=args.output,
            epochs=args.epochs,
            continue_training=args.continue_training,
        )
    else:
        train_single(
            output_dir=args.output,
            epochs=args.epochs,
        )


if __name__ == '__main__':
    main()
