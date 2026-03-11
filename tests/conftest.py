"""
Shared fixtures and helpers for all INS/GNSS filter tests.

Adds scripts/positioning/python to sys.path and provides:
  - make_nav_data()     : factory for synthetic NavigationData
  - tight_params        : low-noise Kalman params for deterministic short runs
  - stationary_nav_data : 100-sample stationary fixture
  - kitti_data          : session-scoped KITTI loader (auto-skips if file absent)
"""
import sys
import os
from pathlib import Path

import numpy as np
import pytest
import pymap3d as pm

# ---------------------------------------------------------------------------
# sys.path injection
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "scripts" / "positioning" / "python")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data_loader import NavigationData  # noqa: E402  (must come after sys.path)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLA0_DEFAULT = np.array([48.0, 11.0, 515.0])   # Munich — KITTI-like reference
KITTI_PATH   = str(Path(__file__).resolve().parent.parent /
                   "datasets" / "raw_kitti" / "10_03_0027.mat")

# ---------------------------------------------------------------------------
# make_nav_data — core synthetic builder
# ---------------------------------------------------------------------------

def make_nav_data(
    N: int,
    sample_rate: float = 100.0,
    accel_flu: np.ndarray = None,
    gyro_flu: np.ndarray = None,
    vel_enu: np.ndarray = None,
    orient: np.ndarray = None,
    gps_available: np.ndarray = None,
    lla0: np.ndarray = None,
) -> NavigationData:
    """
    Build a synthetic NavigationData suitable for filter unit tests.

    * accel_flu defaults to [0, 0, +9.81] (stationary FLU, gravity reaction).
    * gyro_flu  defaults to [0, 0, 0].
    * vel_enu   defaults to [0, 0, 0].
    * orient    defaults to [0, 0, 0] (roll=pitch=yaw=0, i.e. yaw=0 → Forward≡East).
    * gps_available defaults to all-False (pure-IMU mode).
    * lla0 defaults to LLA0_DEFAULT (Munich).

    LLA positions are derived by integrating vel_enu so they are self-consistent:
    lla[0] == lla0, and geodetic2enu(lla[i], lla0) returns the integrated ENU position.
    gps_speed_mps and gps_cog_rad are set to NaN where GPS is unavailable.
    """
    if lla0 is None:
        lla0 = LLA0_DEFAULT.copy()

    if accel_flu is None:
        accel_flu = np.tile([0.0, 0.0, 9.81], (N, 1))
    if gyro_flu is None:
        gyro_flu = np.zeros((N, 3))
    if vel_enu is None:
        vel_enu = np.zeros((N, 3))
    if orient is None:
        orient = np.zeros((N, 3))
    if gps_available is None:
        gps_available = np.zeros(N, dtype=bool)

    accel_flu    = np.asarray(accel_flu,    dtype=np.float64)
    gyro_flu     = np.asarray(gyro_flu,     dtype=np.float64)
    vel_enu      = np.asarray(vel_enu,      dtype=np.float64)
    orient       = np.asarray(orient,       dtype=np.float64)
    gps_available = np.asarray(gps_available, dtype=bool)

    # Integrate vel_enu → ENU positions → LLA
    Ts = 1.0 / sample_rate
    enu = np.zeros((N, 3))
    for i in range(1, N):
        enu[i] = enu[i-1] + Ts * vel_enu[i-1]

    lla = np.zeros((N, 3))
    for i in range(N):
        lat, lon, alt = pm.enu2geodetic(
            enu[i, 0], enu[i, 1], enu[i, 2],
            lla0[0], lla0[1], lla0[2]
        )
        lla[i] = [lat, lon, alt]

    # GPS observables: NaN when GPS unavailable
    speed = np.sqrt(vel_enu[:, 0]**2 + vel_enu[:, 1]**2)
    cog   = np.arctan2(vel_enu[:, 0], vel_enu[:, 1])   # atan2(E, N)
    gps_speed_mps = np.where(gps_available, speed, np.nan)
    gps_cog_rad   = np.where(gps_available, cog,   np.nan)

    nd = NavigationData(
        accel_flu     = accel_flu,
        gyro_flu      = gyro_flu,
        vel_enu       = vel_enu,
        lla           = lla,
        orient        = orient,
        gps_available = gps_available,
        sample_rate   = float(sample_rate),
        dataset_name  = "synthetic_test",
        gps_rate      = 1.0,
        lla0          = lla0.copy(),
        gps_speed_mps = gps_speed_mps,
        gps_cog_rad   = gps_cog_rad,
    )
    nd.validate()
    return nd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_nav():
    """Return the make_nav_data factory as a fixture."""
    return make_nav_data


@pytest.fixture
def tight_params():
    """Low-noise Kalman parameters for short synthetic test runs."""
    return {
        "Qpos": 1e-8, "Qvel": 1e-8,
        "QorientXY": 1e-6, "QorientZ": 1e-6,
        "Qacc": 1e-6, "QgyrXY": 1e-7, "QgyrZ": 1e-7,
        "Rpos": 0.1,
        "beta_acc": -0.1, "beta_gyr": -0.1,
        "P_pos_std": 0.01, "P_vel_std": 0.01, "P_orient_std": 0.01,
        "P_acc_std": 1e-4, "P_gyr_std": 1e-5,
        # Enhanced-filter extras
        "Rnhc": 0.01, "Rzupt": 0.001,
        "zupt_accel_threshold": 0.5, "zupt_gyro_threshold": 0.1,
    }


@pytest.fixture
def stationary_nav_data():
    """100-sample stationary NavigationData at 100 Hz."""
    return make_nav_data(N=100, sample_rate=100.0)


@pytest.fixture(scope="session")
def kitti_data():
    """Load the KITTI test dataset; skip if file is not present."""
    if not os.path.exists(KITTI_PATH):
        pytest.skip(f"KITTI dataset not available at: {KITTI_PATH}")
    from data_loader import load_kitti_mat
    return load_kitti_mat(KITTI_PATH, sample_rate=10.0)
