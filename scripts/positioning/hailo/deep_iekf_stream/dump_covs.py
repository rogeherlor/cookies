"""
Dump per-sequence measurement covariances for the two Hailo deployment paths,
so the REAL IEKF can be driven with each on the host and ATE compared.

Per sequence writes covs_<seq>.npz with:
  covs_ref    (N,2) float64 : ReplicationPad CausalMesNet, whole-sequence torch
                              (the true Deep-IEKF covariances the CPU benchmark uses)
  covs_stream (N,2) float32 : streaming W=32 QUANTIZED emulator, block-16 driver
                              (faithful proxy for deep_iekf_stream.hef on device)
  covs_full   (M,2) float32 : whole-seq W=4544 QUANTIZED emulator, M=min(N,4544)
                              (faithful proxy for deep_iekf.hef; can't exceed 4544)
All three use the SAME fold_01 weights (the HEFs are baked from fold_01), so the
comparison isolates deployment path, not fold.
"""
import sys, os
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent.parent
_DEEP = _REPO / "scripts/positioning/hailo/deep_iekf"
for _p in [str(_REPO / "scripts/positioning/python"),
           str(_REPO / "scripts/positioning/python/dl_filters/deep_iekf"),
           str(_REPO / "external/ai-imu-dr/src"), str(_DEEP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_loader
from hailo_sdk_client import ClientRunner, InferenceContext

W_STREAM, CARRY, FRESH, W_FULL, COV_DIM = 32, 16, 16, 4544, 2
Q_STREAM = _HERE / "deep_iekf_stream_quantized_model.har"
Q_FULL   = _DEEP / "deep_iekf_quantized_model.har"
FOLD     = _REPO / "artifacts/deep_iekf_online/fold_01.p"
OUT      = _HERE / "covs_dump"
SEQS     = ["04", "06", "07", "01", "10", "09", "00", "08"]


def load_ti():
    from utils_torch_filter import TORCHIEKF
    from causal_mesnet import attach_causal_mesnet
    from iekf_ai_imu import _find_norm_factors
    from kitti_params import get_kitti_parameters
    ti = TORCHIEKF(get_kitti_parameters())
    if ti.cov0_measurement is None:
        ti.cov0_measurement = torch.tensor([1.0, 10.0]).double()
    attach_causal_mesnet(ti)
    ti.load_state_dict(torch.load(str(FOLD), map_location="cpu", weights_only=False))
    ti.eval()
    norm = _find_norm_factors(FOLD)
    ti.u_loc = norm["u_loc"].double(); ti.u_std = norm["u_std"].double()
    u_loc = ti.u_loc.numpy().astype(np.float32); u_std = ti.u_std.numpy().astype(np.float32)
    return ti, u_loc, u_std


def covs_ref(ti, imu):
    """ReplicationPad CausalMesNet, float64, whole sequence (real reference)."""
    u_loc = ti.u_loc.numpy(); u_std = ti.u_std.numpy()
    u_norm = (imu.astype(np.float64) - u_loc) / u_std
    u = torch.from_numpy(u_norm).T.unsqueeze(0)          # (1,6,N)
    with torch.no_grad():
        return ti.mes_net(u, ti).numpy()                 # (N,2)


def emu(har, batch_nhwc):
    r = ClientRunner(har=str(har))
    name = [l.name for l in r._hn.get_input_layers()][0]
    with r.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        res = r.infer(ctx, {name: batch_nhwc})
    return np.asarray(res).reshape(-1, batch_nhwc.shape[2], COV_DIM)


def covs_stream(u_norm):
    N = len(u_norm); starts = list(range(0, N, FRESH))
    wins = np.zeros((len(starts), W_STREAM, 6), np.float32)
    for j, t in enumerate(starts):
        lo = t - CARRY; src = max(lo, 0); pad = src - lo
        seg = u_norm[src: lo + W_STREAM]; wins[j, pad:pad + len(seg)] = seg
    ow = emu(Q_STREAM, wins[:, None, :, :])              # (B,32,2)
    out = np.zeros((N, 2), np.float32)
    for j, t in enumerate(starts):
        n = min(FRESH, N - t); out[t:t + n] = ow[j, CARRY:CARRY + n]
    return out


def covs_full(u_norm):
    N = len(u_norm); M = min(N, W_FULL)
    buf = np.zeros((W_FULL, 6), np.float32); buf[:M] = u_norm[:M]
    o = emu(Q_FULL, buf[None, None, :, :])               # (1,4544,2)
    return o[0][:M]


def main():
    OUT.mkdir(exist_ok=True)
    ti, u_loc, u_std = load_ti()
    for seq in SEQS:
        nav = data_loader.get_kitti_dataset(seq)
        imu = np.concatenate([nav.gyro_flu, nav.accel_flu], axis=1).astype(np.float32)
        u_norm = (imu - u_loc) / u_std
        N = len(u_norm)
        cr = covs_ref(ti, imu); cs = covs_stream(u_norm); cf = covs_full(u_norm)
        np.savez(str(OUT / f"covs_{seq}.npz"),
                 covs_ref=cr, covs_stream=cs, covs_full=cf, N=N)
        print(f"seq {seq}: N={N}  ref{cr.shape} stream{cs.shape} full{cf.shape}(<=4544)")
    print("DONE dump ->", OUT)


if __name__ == "__main__":
    main()
