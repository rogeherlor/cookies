"""
Head-to-head accuracy of the two Hailo Deep-IEKF solutions, both via the DFC
SDK_QUANTIZED emulator (faithful proxy for the real HEF: for the 4544 model the
emulator gave 4.99 vs 5.01 on-device), against ONE identical PyTorch float64
full-sequence reference:

  Solution 1  whole-sequence  W=4544  (../deep_iekf/deep_iekf_quantized_model.har)
              -> feed the sequence (right zero-padded to 4544), keep first N outputs.
  Solution 2  streaming       W=32    (deep_iekf_stream_quantized_model.har)
              -> block-16 driver: stride-16, 32-wide windows, keep last 16 per call,
                 reconstruct the whole sequence.  Also per-tick (keep last 1).

Reports steady-state MAE (excluding the first 16 warmup samples) per channel and
overall, plus the number of HEF calls each solution needs to cover the sequence.
Runs inside the DFC container.
"""
import sys
from pathlib import Path
import numpy as np
import torch

_HERE   = Path(__file__).resolve().parent
_REPO   = _HERE.parent.parent.parent.parent
_DEEP   = _REPO / "scripts/positioning/hailo/deep_iekf"
for _p in [str(_REPO / "scripts/positioning/python"),
           str(_REPO / "scripts/positioning/python/dl_filters/deep_iekf"),
           str(_REPO / "external/ai-imu-dr/src"), str(_DEEP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_loader
from hailo_sdk_client import ClientRunner, InferenceContext

W_STREAM = 32
CARRY    = 16          # left context carried between blocks
FRESH    = W_STREAM - CARRY
W_FULL   = 4544
COV_DIM  = 2
SKIP     = 16          # whole-stream ZeroPad warmup, excluded from steady MAE

Q_STREAM = _HERE / "deep_iekf_stream_quantized_model.har"
Q_FULL   = _DEEP / "deep_iekf_quantized_model.har"
FOLD     = _REPO / "artifacts/deep_iekf_online/fold_01.p"


def build_wrapper():
    from utils_torch_filter import TORCHIEKF
    from causal_mesnet import attach_causal_mesnet
    from iekf_ai_imu import _find_norm_factors
    from kitti_params import get_kitti_parameters
    import importlib.util
    spec = importlib.util.spec_from_file_location("oc", str(_DEEP / "0_onnx_converter.py"))
    oc = importlib.util.module_from_spec(spec); spec.loader.exec_module(oc)
    ti = TORCHIEKF(get_kitti_parameters())
    if ti.cov0_measurement is None:
        ti.cov0_measurement = torch.tensor([1.0, 10.0]).double()
    attach_causal_mesnet(ti)
    ti.load_state_dict(torch.load(str(FOLD), map_location="cpu", weights_only=False))
    ti.eval()
    norm = _find_norm_factors(FOLD)
    ti.u_loc = norm["u_loc"].double(); ti.u_std = norm["u_std"].double()
    w = oc.MesNetFullHailo(ti).eval()
    return w, ti.u_loc.numpy().astype(np.float32), ti.u_std.numpy().astype(np.float32)


def run_wrapper(w, u_norm):
    x = torch.from_numpy(u_norm.T[None, :, None, :].astype(np.float32))
    with torch.no_grad():
        return w(x).detach().numpy()[0, :, 0, :].T          # (N,2)


def emu_infer(har_path, u_nhwc_batch):
    """Run a batch (B,1,Win,6) NHWC through SDK_QUANTIZED, return (B,Win,2)."""
    runner = ClientRunner(har=str(har_path))
    name = [l.name for l in runner._hn.get_input_layers()][0]
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        res = runner.infer(ctx, {name: u_nhwc_batch})
    Win = u_nhwc_batch.shape[2]
    return np.asarray(res).reshape(-1, Win, COV_DIM)


def stream_block16(har_path, u_norm):
    """Reconstruct whole-sequence covs from the W=32 model via stride-16 blocks."""
    N = len(u_norm)
    starts = list(range(0, N, FRESH))
    wins = np.zeros((len(starts), W_STREAM, 6), np.float32)
    for j, t in enumerate(starts):
        lo = t - CARRY
        src_lo = max(lo, 0); pad = src_lo - lo
        seg = u_norm[src_lo: lo + W_STREAM]
        wins[j, pad:pad + len(seg)] = seg
    out_win = emu_infer(har_path, wins[:, None, :, :])          # (B,32,2)
    out = np.zeros((N, 2), np.float32)
    for j, t in enumerate(starts):
        n = min(FRESH, N - t)
        out[t:t + n] = out_win[j, CARRY:CARRY + n]
    return out, len(starts)


def full4544(har_path, u_norm):
    """Feed sequence right-zero-padded to 4544 through the W=4544 model; keep first N."""
    N = len(u_norm)
    buf = np.zeros((W_FULL, 6), np.float32)
    buf[:min(N, W_FULL)] = u_norm[:W_FULL]
    out = emu_infer(har_path, buf[None, None, :, :])           # (1,4544,2)
    return out[0][:N], 1


def mae(a, b):
    d = np.abs(a[SKIP:] - b[SKIP:])
    return d.mean(0), float(d.mean())


def main():
    w, u_loc, u_std = build_wrapper()
    for seq in ["04", "01"]:
        nav = data_loader.get_kitti_dataset(seq)
        imu = np.concatenate([nav.gyro_flu, nav.accel_flu], axis=1).astype(np.float32)
        u_norm = (imu - u_loc) / u_std
        N = len(u_norm)
        ref = run_wrapper(w, u_norm)                            # PyTorch f64 full-pass

        s2, calls2 = stream_block16(Q_STREAM, u_norm)
        pc2, m2 = mae(ref, s2)

        # 4544 model only covers the first min(N,4544) samples
        cov = min(N, W_FULL)
        s1, calls1 = full4544(Q_FULL, u_norm)
        pc1, m1 = mae(ref[:cov], s1[:cov])

        print(f"\n=== seq {seq}  N={N} ===")
        print(f"  ref cov scale (median |cov_lat|,|cov_up|): "
              f"{np.median(np.abs(ref[:,0])):.3f}, {np.median(np.abs(ref[:,1])):.3f}")
        print(f"  Solution 2  stream W=32 block-16   MAE overall={m2:.4f}  "
              f"[cov_lat={pc2[0]:.4f}, cov_up={pc2[1]:.4f}]  "
              f"calls={calls2}  covers N={N}")
        print(f"  Solution 1  whole-seq W=4544       MAE overall={m1:.4f}  "
              f"[cov_lat={pc1[0]:.4f}, cov_up={pc1[1]:.4f}]  "
              f"calls={calls1}  covers first {cov} of {N}"
              + ("  <-- TRUNCATED" if cov < N else ""))


if __name__ == "__main__":
    main()
