"""
kitti_params.py — dependency-light access to AI-IMU's KITTIParameters.

WHY THIS EXISTS
---------------
The genuine KITTIParameters lives in external/ai-imu-dr/src/main_kitti.py, but
that module imports navpy, matplotlib, dataset, train_torch_filter, utils_plot,
... at load time. Several of those are absent on lean deploy targets — e.g. the
Raspberry Pi runtime container has no `navpy` — so `from main_kitti import
KITTIParameters` raises ModuleNotFoundError there.

Every Deep-IEKF inference path wrapped that import in a bare `except Exception`
and, on failure, silently fell back to a WRONG hardcoded covariance base
([0.2, 300.0], the upstream NUMPYIEKF.Parameters defaults) instead of the
values the network was actually trained with (KITTIParameters: cov_lat=1,
cov_up=10, plus a full set of process-noise / initial-covariance overrides).
That made the Pi run the filter with the wrong parameters while the PC (which
has navpy) ran it correctly — a silent, machine-dependent divergence that threw
the Deep IEKF trajectory off by ~40x. See get_kitti_parameters() below.

get_kitti_parameters() returns the genuine class when main_kitti is importable
(keeping PC/training behaviour byte-identical) and this faithful local mirror
otherwise, so every machine uses the SAME parameters — no navpy required.

KEEPING IT IN SYNC
------------------
`KITTIParameters` below is an exact copy of the class in
external/ai-imu-dr/src/main_kitti.py (submodule pin ai-imu-dr @ 50eff65). If
that submodule's KITTIParameters ever changes, update this mirror to match.
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_AI_IMU_SRC = _HERE.parents[4] / 'external/ai-imu-dr/src'   # <repo>/external/ai-imu-dr/src


def _base_parameters_cls():
    """The NUMPYIEKF.Parameters base class (imports cleanly on every target —
    utils_numpy_filter needs only numpy/scipy/termcolor, all present)."""
    if str(_AI_IMU_SRC) not in sys.path:
        sys.path.insert(0, str(_AI_IMU_SRC))
    from utils_numpy_filter import NUMPYIEKF
    return NUMPYIEKF.Parameters


# Faithful mirror of external/ai-imu-dr/src/main_kitti.py::KITTIParameters.
class KITTIParameters(_base_parameters_cls()):
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8
    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10


_warned = False


def get_kitti_parameters():
    """Return the KITTIParameters class the Deep-IEKF filters were trained with.

    Prefers the genuine external/ai-imu-dr/src/main_kitti.py class (so behaviour
    stays byte-identical to training wherever main_kitti's full dependency stack
    is installed), and falls back to the local mirror above — with a one-time
    stderr warning — when main_kitti cannot be imported (e.g. no navpy on the
    Pi). Either way the covariance parameters are the same; the fallback never
    silently substitutes wrong defaults the way the old inline `except` did.
    """
    global _warned
    if str(_AI_IMU_SRC) not in sys.path:
        sys.path.insert(0, str(_AI_IMU_SRC))
    try:
        from main_kitti import KITTIParameters as _KP
        return _KP
    except Exception as exc:
        if not _warned:
            print(f"[kitti_params] main_kitti unavailable ({type(exc).__name__}: "
                  f"{exc}); using bundled KITTIParameters mirror. Filter behaviour "
                  f"is unchanged (same trained covariance parameters).",
                  file=sys.stderr)
            _warned = True
        return KITTIParameters
