"""
Real-hardware validation of the unified Deep-KF Hailo path.

Runs deep_kf_runner.run(nav, backend='hailo', hailo_net=HailoDeepKF(...)) on
the ACTUAL physical Hailo-8L (via hailo_backend, genuine HailoRT calls) for
both no-outage and the standard 40s/60s-outage scenario, and compares against
the CPU (torch) equivalent — same corrected control-flow, only the LSTM
forward pass differs by backend. Reports ATE and genuine per-call latency.
"""
import sys, time
from pathlib import Path
import numpy as np
import pymap3d as pm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent.parent
for _p in [str(_REPO / "scripts/positioning/python"),
           str(_REPO / "scripts/positioning/python/dl_filters/deep_kf"),
           str(_REPO / "scripts/positioning/hailo")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_loader
import deep_kf_runner as dkr
import hailo_backend

SEQ = "01"
FOLD = "01"
HEF = _HERE / "deep_kf.hef"


def ate(p, gt, lo=0, hi=None):
    hi = len(gt) if hi is None else hi
    L = min(len(p), hi)
    d = p[lo:L, :2] - gt[lo:L, :2]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def main():
    import os
    os.environ['DEEP_KF_WEIGHTS'] = str(_REPO / f"artifacts/deep_kf/fold_{FOLD}.pt")
    nav = data_loader.get_kitti_dataset(SEQ)
    lla = nav.lla
    e_, n_, u_ = pm.geodetic2enu(lla[:, 0], lla[:, 1], lla[:, 2], lla[0, 0], lla[0, 1], lla[0, 2])
    gt = np.column_stack([e_, n_, u_])

    for label, outage_cfg in [("no-outage", {'start': 0., 'duration': 0.}),
                              ("outage 40s/60s", {'start': 40., 'duration': 60.})]:
        print(f"\n=== {label} ===")
        t0 = time.perf_counter()
        r_cpu = dkr.run(nav, backend='cpu', outage_config=outage_cfg)
        t_cpu = time.perf_counter() - t0

        with hailo_backend.HailoDeepKF(HEF) as hnet:
            t0 = time.perf_counter()
            r_hw = dkr.run(nav, backend='hailo', outage_config=outage_cfg, hailo_net=hnet)
            t_hw = time.perf_counter() - t0
            lat = hnet.total_s / hnet.n_calls if hnet.n_calls else 0.0
            n_calls = hnet.n_calls

        L = min(len(r_cpu['p']), len(r_hw['p']), len(gt))
        print(f"ATE CPU  vs KITTI: {ate(r_cpu['p'], gt):.3f} m  (wall {t_cpu:.2f}s)")
        print(f"ATE Hailo vs KITTI: {ate(r_hw['p'], gt):.3f} m  (wall {t_hw:.2f}s)")
        d = r_hw['p'][:L, :2] - r_cpu['p'][:L, :2]
        print(f"ATE Hailo vs CPU (agreement): {np.sqrt(np.mean(np.sum(d*d,axis=1))):.4f} m")
        print(f"Real device: {n_calls} calls, mean latency {lat*1000:.4f} ms/call")


if __name__ == "__main__":
    main()
