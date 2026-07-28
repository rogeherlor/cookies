# -*- coding: utf-8 -*-
"""
Shared subprocess bridge to /opt/conda-gtsam/bin/python3, used by every
smoother whose native run() needs `import gtsam` (fgo_batch_runner,
isam2_runner, isam2_fixedlag_runner, isam2_map_runner).

GTSAM has no aarch64 PyPI wheel, so on this repo's Raspberry Pi target it
only exists in the isolated /opt/conda-gtsam micromamba env (see
docker/Dockerfile's GTSAM section); the main interpreter has torch/scipy
for the other filters instead and can never `import gtsam` directly. On a
PC with a native GTSAM install, callers never reach this module at all —
each runner's run() tries the native in-process import first.

Extracted from fgo_batch_runner.py's original inlined version so
isam2_runner / isam2_fixedlag_runner / isam2_map_runner can reuse it
instead of duplicating the same ~50 lines three more times.
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent
CONDA_PYTHON = "/opt/conda-gtsam/bin/python3"
WORKER = _REPO_ROOT / "scripts/positioning/hailo/_isam2_conda_worker.py"


def run_via_conda_subprocess(runner_key, nav_data, params, outage_config=None, use_3d_rotation=True):
    """Bridge one smoother's run() call to the isolated conda-gtsam
    interpreter. Returns the same schema as the native run(), except
    std_pos/std_vel/std_orient/std_bias_acc/std_bias_gyr, which the worker
    doesn't serialise (nothing in this repo consumes a smoother's own std
    from the subprocess path) — left as zeros."""
    if not Path(CONDA_PYTHON).exists() or not WORKER.exists():
        raise ImportError(
            "GTSAM is not importable in this process, and the "
            f"/opt/conda-gtsam subprocess bridge is unavailable "
            f"(conda_python exists={Path(CONDA_PYTHON).exists()}, "
            f"worker exists={WORKER.exists()}). "
            "Install with: conda install -c conda-forge gtsam"
        )

    outage_config = outage_config or {'start': 0.0, 'duration': 0.0}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        nav_npz = td / "nav.npz"
        params_json = td / "params.json"
        out_npz = td / "out.npz"

        save_kwargs = {}
        for f in ("accel_flu", "gyro_flu", "vel_enu", "lla", "orient",
                  "gps_available", "gps_speed_mps", "gps_cog_rad", "time"):
            v = getattr(nav_data, f, None)
            if v is not None:
                save_kwargs[f] = v
        save_kwargs["sample_rate"]  = np.array(nav_data.sample_rate)
        save_kwargs["gps_rate"]     = np.array(getattr(nav_data, "gps_rate", 1.0))
        save_kwargs["lla0"]         = nav_data.lla0
        save_kwargs["dataset_name"] = np.array(nav_data.dataset_name)
        np.savez(nav_npz, **save_kwargs)
        params_json.write_text(json.dumps(params) if params else "")

        cmd = [CONDA_PYTHON, str(WORKER),
              "--runner", runner_key,
              "--nav-npz", str(nav_npz),
              "--params-json", str(params_json),
              "--outage-start", str(outage_config.get('start', 0.0)),
              "--outage-duration", str(outage_config.get('duration', 0.0)),
              "--use-3d", "1" if use_3d_rotation else "0",
              "--out-npz", str(out_npz)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{runner_key} conda-gtsam subprocess failed:\n{proc.stdout}\n{proc.stderr}")

        out = np.load(out_npz)
        N = out['p'].shape[0]
        zeros = np.zeros((N, 3))
        return {
            'p': out['p'], 'v': out['v'], 'r': out['r'],
            'bias_acc': out['bias_acc'], 'bias_gyr': out['bias_gyr'],
            'std_pos': zeros, 'std_vel': zeros, 'std_orient': zeros,
            'std_bias_acc': zeros, 'std_bias_gyr': zeros,
        }
