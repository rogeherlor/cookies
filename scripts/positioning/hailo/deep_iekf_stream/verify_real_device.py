"""
Real-hardware validation of the streaming Deep-IEKF HEFs — runs the physical
Hailo-8L (via hailo_backend.HailoDeepIEKFStream, genuine HailoRT calls, not the
DFC SDK_QUANTIZED emulator) over full real KITTI sequences, per fold, and
compares against the float64 CausalMesNet reference. Reports accuracy AND
genuine per-call device latency (real, not the DFC's static estimate).

Run inside the hailo_ai_sw_suite container WITH device passthrough
(--privileged, -v /dev:/dev, ...).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent.parent
for _p in [str(_REPO / "scripts/positioning/python"),
           str(_REPO / "scripts/positioning/python/dl_filters/deep_iekf"),
           str(_REPO / "external/ai-imu-dr/src"),
           str(_REPO / "scripts/positioning/hailo")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_loader
import hailo_backend

SEQ_FOLD = {"04": "04", "06": "06", "07": "07", "01": "01",
            "10": "10", "09": "09", "08": "08"}
MODE = "block"
SKIP = 16   # ZeroPad warmup, excluded from steady-state MAE
import os
DUMP = Path(os.environ.get("REALDEVICE_DUMP_DIR", str(_HERE / "covs_dump")))


def load_ti(fold):
    from utils_torch_filter import TORCHIEKF
    from causal_mesnet import attach_causal_mesnet
    from iekf_ai_imu import _find_norm_factors
    from kitti_params import get_kitti_parameters
    f = _REPO / f"artifacts/deep_iekf_online/fold_{fold}.p"
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


def main():
    print(f"{'seq':>4} {'fold':>5} {'N':>6} | {'real MAE (steady)':>18} {'MAE cov_lat':>12} "
          f"{'MAE cov_up':>11} | {'calls':>6} {'mean ms/call':>13} {'total ms':>9} "
          f"{'seq dur ms':>10} {'realtime×':>10}")
    for seq, fold in SEQ_FOLD.items():
        hef = _HERE / f"deep_iekf_stream_fold_{fold}.hef"
        pp = _HERE / f"deep_iekf_stream_fold_{fold}_postproc.npz"
        ti = load_ti(fold)
        nav = data_loader.get_kitti_dataset(seq)
        imu = np.concatenate([nav.gyro_flu, nav.accel_flu], axis=1).astype(np.float32)
        ref = covs_ref(ti, imu)
        N = len(imu)

        with hailo_backend.HailoDeepIEKFStream(hef, pp, mode=MODE) as net:
            t0 = time.perf_counter()
            covs, lat, _cpu = net.run_stream(imu)
            wall = time.perf_counter() - t0

        d = np.abs(covs[SKIP:] - ref[SKIP:])
        mae = float(d.mean()); mae_lat = float(d[:, 0].mean()); mae_up = float(d[:, 1].mean())
        seq_dur_ms = N / 100 * 1000
        rt = seq_dur_ms / (wall * 1000)
        print(f"{seq:>4} {fold:>5} {N:>6} | {mae:>17.4f} {mae_lat:>12.4f} {mae_up:>11.4f} | "
              f"{len(lat):>6} {lat.mean()*1000:>12.4f} {wall*1000:>9.1f} {seq_dur_ms:>10.0f} {rt:>9.0f}x")

        DUMP.mkdir(exist_ok=True)
        np.savez(str(DUMP / f"covs_realdevice_{seq}.npz"),
                 covs_ref=ref, covs_realdevice=covs.astype(np.float32), N=N, fold=fold)
    print("\nDONE real-device validation.")


if __name__ == "__main__":
    main()
