"""
export_onnx.py — Export the AI-IMU MesNet CNN adapter to ONNX.

Only the CNN covariance adapter (MesNet) is exported — the IEKF filter loop is
NumPy-based and cannot be represented in ONNX.  At inference time the ONNX model
produces per-timestep measurement covariances (N, 2) which are fed into NUMPYIEKF.

Usage
-----
  python dl_filters/deep_iekf/export_onnx.py
  python dl_filters/deep_iekf/export_onnx.py --weights artifacts/deep_iekf/iekfnets.p
  python dl_filters/deep_iekf/export_onnx.py --output  artifacts/deep_iekf/iekfnets.onnx

The exported model:
  Input  : u               float64  (N, 6)   — raw IMU [gyro_flu, accel_flu]
  Output : measurements_covs float64 (N, 2)  — [cov_lat, cov_up] per timestep

Credit / License
----------------
MesNet CNN and KITTIParameters are the work of Martin Brossard, Axel Barrau, and
Silvère Bonnabel, MIT License (external/ai-imu-dr/LICENSE).
Original repo: https://github.com/mbrossar/ai-imu-dr
"""

import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path

_HERE       = Path(__file__).resolve().parent
_REPO_ROOT  = _HERE.parent.parent.parent.parent.parent
_AI_IMU_SRC = _REPO_ROOT / 'external/ai-imu-dr/src'
_ARTIFACTS  = _REPO_ROOT / 'artifacts/deep_iekf'


def _add_aimu_path():
    src = str(_AI_IMU_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


class MesNetWrapper(nn.Module):
    """
    Self-contained ONNX-exportable wrapper around MesNet.

    Embeds IMU normalization (u_loc, u_std) and the baseline covariance
    (cov0_measurement, beta_measurement) as buffers so the model is fully
    portable without needing the original TORCHIEKF instance at runtime.

    Input  : u  (N, 6)  float64 — raw IMU [gyro_flu(3), accel_flu(3)]
    Output : measurements_covs (N, 2) float64 — [cov_lat, cov_up]
    """

    def __init__(self, torch_iekf):
        super().__init__()
        self.cov_net = torch_iekf.mes_net.cov_net
        self.cov_lin = torch_iekf.mes_net.cov_lin
        self.register_buffer('u_loc',            torch_iekf.u_loc.clone())
        self.register_buffer('u_std',            torch_iekf.u_std.clone())
        self.register_buffer('cov0_measurement', torch_iekf.cov0_measurement.clone())
        self.register_buffer('beta_measurement', torch_iekf.mes_net.beta_measurement.clone())

    def forward(self, u):
        # u: (N, 6) → normalize → (1, 6, N) for Conv1d
        u_n = ((u - self.u_loc) / self.u_std).t().unsqueeze(0)   # (1, 6, N)
        # CNN: (1, 6, N) → (1, 32, N)
        features = self.cov_net(u_n).squeeze(0).t()               # (N, 32)
        z_cov = self.cov_lin(features)                            # (N, 2)
        z_cov_net = self.beta_measurement.unsqueeze(0) * z_cov
        return self.cov0_measurement.unsqueeze(0) * (10 ** z_cov_net)  # (N, 2)


def export(weights_path, output_path, seq_len=4544):
    """
    Load trained weights and export MesNetWrapper to ONNX.

    Parameters
    ----------
    weights_path : str or Path   — iekfnets.p produced by train_ai_imu.py
    output_path  : str or Path   — destination .onnx file
    seq_len      : int           — representative sequence length for tracing
                                   (dynamic axes allow any length at runtime)
    """
    _add_aimu_path()
    from utils_torch_filter import TORCHIEKF
    try:
        from main_kitti import KITTIParameters
        torch_iekf = TORCHIEKF(KITTIParameters)
    except Exception:
        torch_iekf = TORCHIEKF()

    # Load weights
    mondict = torch.load(weights_path, map_location='cpu')
    torch_iekf.load_state_dict(mondict)
    torch_iekf.eval()

    # Load normalization factors
    norm_path = Path(weights_path).parent / 'iekfnets_norm.p'
    if norm_path.exists():
        norm = torch.load(norm_path, map_location='cpu')
        torch_iekf.u_loc = norm['u_loc'].double()
        torch_iekf.u_std = norm['u_std'].double()
    else:
        raise FileNotFoundError(
            f"Normalization factors not found at {norm_path}. "
            "Re-run train_ai_imu.py which saves iekfnets_norm.p alongside weights."
        )

    # Ensure cov0_measurement is set
    if torch_iekf.cov0_measurement is None:
        torch_iekf.set_param_attr()

    # Build wrapper
    wrapper = MesNetWrapper(torch_iekf)
    wrapper.eval()

    # Dummy input: (seq_len, 6) float64
    dummy_u = torch.randn(seq_len, 6).double()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_u,),
        str(output_path),
        input_names=['u'],
        output_names=['measurements_covs'],
        dynamic_axes={
            'u':                 {0: 'seq_len'},
            'measurements_covs': {0: 'seq_len'},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"ONNX model exported to {output_path}")

    # Quick validation with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(output_path))
        out = sess.run(None, {'u': dummy_u.numpy()})
        assert out[0].shape == (seq_len, 2), f"Unexpected output shape {out[0].shape}"
        print(f"ONNX validation passed: output shape {out[0].shape}")
    except ImportError:
        print("onnxruntime not installed — skipping validation.")
    except Exception as e:
        print(f"ONNX validation warning: {e}")


def main():
    p = argparse.ArgumentParser(
        description='Export AI-IMU MesNet CNN adapter to ONNX')
    p.add_argument('--weights', type=str,
                   default=str(_ARTIFACTS / 'iekfnets.p'),
                   help='Path to trained iekfnets.p weights file')
    p.add_argument('--output', type=str,
                   default=str(_ARTIFACTS / 'iekfnets.onnx'),
                   help='Output .onnx file path')
    p.add_argument('--seq-len', type=int, default=4544,
                   help='Representative sequence length for ONNX tracing (default: 4544)')
    args = p.parse_args()

    if not Path(args.weights).exists():
        p.error(f"Weights file not found: {args.weights}\n"
                "Train first with: python dl_filters/deep_iekf/train_ai_imu.py")

    export(args.weights, args.output, args.seq_len)


if __name__ == '__main__':
    main()
