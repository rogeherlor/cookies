"""
Per-fold covariance dump: each benchmark sequence uses ITS OWN held-out fold's
streaming HEF (quantized HAR) — the correct LOO setup, unlike dump_covs.py which
used fold_01 for everything.  Writes covs_perfold_<seq>.npz with covs_ref (that
fold's float64 CausalMesNet) and covs_stream (that fold's W=32 quantized emulator,
block-16).  Resolves whether the seq08 / fold_06 concerns are real under the
correct fold.
"""
import sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent.parent
_DEEP = _REPO / "scripts/positioning/hailo/deep_iekf"
_ONLINE = _REPO / "artifacts/deep_iekf_online"
for _p in [str(_REPO / "scripts/positioning/python"),
           str(_REPO / "scripts/positioning/python/dl_filters/deep_iekf"),
           str(_REPO / "external/ai-imu-dr/src"), str(_DEEP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_loader
from hailo_sdk_client import ClientRunner, InferenceContext

W, CARRY, FRESH, COV_DIM = 32, 16, 16, 2
SEQ_FOLD = {"01": "01", "04": "04", "06": "06", "07": "07",
            "08": "08", "09": "09", "10": "10"}
OUT = _HERE / "covs_dump"


def load_ti(fold):
    from utils_torch_filter import TORCHIEKF
    from causal_mesnet import attach_causal_mesnet
    from iekf_ai_imu import _find_norm_factors
    from kitti_params import get_kitti_parameters
    f = _ONLINE / f"fold_{fold}.p"
    ti = TORCHIEKF(get_kitti_parameters())
    if ti.cov0_measurement is None:
        ti.cov0_measurement = torch.tensor([1.0, 10.0]).double()
    attach_causal_mesnet(ti)
    ti.load_state_dict(torch.load(str(f), map_location="cpu", weights_only=False))
    ti.eval()
    nm = _find_norm_factors(f)
    ti.u_loc = nm["u_loc"].double(); ti.u_std = nm["u_std"].double()
    return ti


def covs_ref(ti, imu):
    un = (imu.astype(np.float64) - ti.u_loc.numpy()) / ti.u_std.numpy()
    u = torch.from_numpy(un).T.unsqueeze(0)
    with torch.no_grad():
        return ti.mes_net(u, ti).numpy()


def stream_covs(qhar, u_norm, beta, cov0):
    """Emulate the z-output HEF over the sequence (block-16) and reconstruct
    cov = cov0 * 10**(beta * z) on the host."""
    N = len(u_norm); starts = list(range(0, N, FRESH))
    wins = np.zeros((len(starts), W, 6), np.float32)
    for j, t in enumerate(starts):
        lo = t - CARRY; src = max(lo, 0); pad = src - lo
        seg = u_norm[src: lo + W]; wins[j, pad:pad + len(seg)] = seg
    r = ClientRunner(har=str(qhar))
    name = [l.name for l in r._hn.get_input_layers()][0]
    with r.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        res = r.infer(ctx, {name: wins[:, None, :, :]})
    oz = np.asarray(res).reshape(-1, W, COV_DIM)                 # z windows
    z = np.zeros((N, 2), np.float32)
    for j, t in enumerate(starts):
        n = min(FRESH, N - t); z[t:t + n] = oz[j, CARRY:CARRY + n]
    return (cov0 * np.power(10.0, beta * z)).astype(np.float32)  # -> cov


def main():
    OUT.mkdir(exist_ok=True)
    for seq, fold in SEQ_FOLD.items():
        qhar = _HERE / f"deep_iekf_stream_fold_{fold}_quantized_model.har"
        if not qhar.exists():
            print(f"seq {seq}: MISSING {qhar.name} — skip"); continue
        pp = np.load(_HERE / f"deep_iekf_stream_fold_{fold}_postproc.npz")
        beta, cov0 = pp["beta"], pp["cov0"]
        ti = load_ti(fold)
        nav = data_loader.get_kitti_dataset(seq)
        imu = np.concatenate([nav.gyro_flu, nav.accel_flu], 1).astype(np.float32)
        un = (imu - ti.u_loc.numpy()) / ti.u_std.numpy()
        cr = covs_ref(ti, imu); cs = stream_covs(qhar, un, beta, cov0)
        np.savez(str(OUT / f"covs_perfold_{seq}.npz"),
                 covs_ref=cr, covs_stream=cs, N=len(imu), fold=fold)
        print(f"seq {seq} (fold_{fold}): N={len(imu)}  ref/stream dumped")
    print("DONE perfold dump")


if __name__ == "__main__":
    main()
