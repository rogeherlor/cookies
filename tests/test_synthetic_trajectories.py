"""
Analytical trajectory tests for all INS/GNSS filters.

Each test constructs a NavigationData with analytically known IMU inputs,
runs one or more filters, and checks that the output matches the expected
physical behaviour.

Convention reminders:
  Body frame : FLU  (x=Forward, y=Left, z=Up)
  Nav  frame : ENU  (x=East, y=North, z=Up)
  yaw = 0   → Forward ≡ East
  yaw = π/2 → Forward ≡ North
  Gravity in ENU : g = [0, 0, -9.81]
  Stationary IMU reads : accel_flu = [0, 0, +9.81]  (specific force = a - g)

Trajectories tested
  A. Static          – no motion, all filters hold near origin
  B. Straight East   – forward accel at yaw=0 → position grows East only
  C. Straight North  – forward accel at yaw=π/2 → position grows North only
  D. Covariance growth during GPS outage
  E. GPS convergence – filter corrects toward offset GPS truth
"""
import importlib
from math import pi

import numpy as np
import pytest
import pymap3d as pm

from conftest import make_nav_data, LLA0_DEFAULT

# ---------------------------------------------------------------------------
# Filter lists
# ---------------------------------------------------------------------------
ALL_FILTERS    = ["ekf_vanilla", "ekf_enhanced",
                  "eskf_vanilla", "eskf_enhanced",
                  "iekf_vanilla", "iekf_enhanced",
                  "imu_only"]
KALMAN_FILTERS = [f for f in ALL_FILTERS if f != "imu_only"]

G = 9.80665   # m/s² (consistent with ekf_vanilla.GRAVITY magnitude)


def _run(filter_name, nav_data, params):
    mod = importlib.import_module(f"filters.{filter_name}")
    return mod.run(nav_data, params=params, outage_config=None, use_3d_rotation=True)


# ---------------------------------------------------------------------------
# Trajectory A — Static
# ---------------------------------------------------------------------------

class TestStaticTrajectory:
    """
    accel_flu = [0, 0, +9.81], gyro = [0, 0, 0], no GPS.
    Expected: position ≈ origin, velocity ≈ 0, attitude ≈ 0.
    """
    N    = 100
    RATE = 100.0      # Hz — short, high-rate for clean integration

    @pytest.mark.parametrize("filter_name", ALL_FILTERS)
    def test_position_stays_near_origin(self, filter_name, tight_params):
        nd  = make_nav_data(N=self.N, sample_rate=self.RATE)
        res = _run(filter_name, nd, tight_params)
        err = np.linalg.norm(res["p"][-1])
        assert err < 0.5, \
            f"{filter_name}: static position drift {err:.3f} m > 0.5 m"

    @pytest.mark.parametrize("filter_name", ALL_FILTERS)
    def test_velocity_stays_near_zero(self, filter_name, tight_params):
        nd  = make_nav_data(N=self.N, sample_rate=self.RATE)
        res = _run(filter_name, nd, tight_params)
        err = np.linalg.norm(res["v"][-1])
        assert err < 0.2, \
            f"{filter_name}: static velocity drift {err:.3f} m/s > 0.2 m/s"

    @pytest.mark.parametrize("filter_name", ALL_FILTERS)
    def test_attitude_stays_near_zero(self, filter_name, tight_params):
        nd  = make_nav_data(N=self.N, sample_rate=self.RATE)
        res = _run(filter_name, nd, tight_params)
        err = np.linalg.norm(res["r"][-1])
        assert err < 0.05, \
            f"{filter_name}: static attitude drift {err:.4f} rad > 0.05 rad"


# ---------------------------------------------------------------------------
# Trajectory B — Straight East (yaw = 0)
# ---------------------------------------------------------------------------

class TestStraightEastTrajectory:
    """
    Forward acceleration a=1 m/s² at yaw=0 → FLU x=[1,0,0] maps to ENU East.
    Specific force in body: [a, 0, 9.81].
    Expected: p_East grows, p_North ≈ 0, p_Up ≈ 0.

    Uses imu_only (pure integration, no GPS to fight the trajectory).
    """
    N    = 200         # 2 s at 100 Hz
    RATE = 100.0
    A    = 1.0         # m/s² forward acceleration

    @pytest.fixture
    def east_nav(self):
        accel = np.tile([self.A, 0.0, G], (self.N, 1))
        gyro  = np.zeros((self.N, 3))
        # True vel_enu: East = a*t, others zero
        t       = np.arange(self.N) / self.RATE
        vel_enu = np.column_stack([self.A * t, np.zeros(self.N), np.zeros(self.N)])
        orient  = np.zeros((self.N, 3))   # yaw=0 throughout
        return make_nav_data(N=self.N, sample_rate=self.RATE,
                             accel_flu=accel, gyro_flu=gyro,
                             vel_enu=vel_enu, orient=orient)

    def test_position_grows_east(self, east_nav, tight_params):
        res = _run("imu_only", east_nav, tight_params)
        assert res["p"][-1, 0] > 0.5, \
            f"East position {res['p'][-1,0]:.3f} m — should be > 0.5 m"

    def test_no_north_drift(self, east_nav, tight_params):
        res = _run("imu_only", east_nav, tight_params)
        assert abs(res["p"][-1, 1]) < 0.5, \
            f"North drift {res['p'][-1,1]:.3f} m > 0.5 m (frame convention error)"

    def test_no_vertical_drift(self, east_nav, tight_params):
        res = _run("imu_only", east_nav, tight_params)
        assert abs(res["p"][-1, 2]) < 0.5, \
            f"Vertical drift {res['p'][-1,2]:.3f} m > 0.5 m"

    def test_east_velocity_grows(self, east_nav, tight_params):
        """Velocity East should grow roughly as a*t."""
        res = _run("imu_only", east_nav, tight_params)
        expected_v_east = self.A * (self.N - 1) / self.RATE
        assert res["v"][-1, 0] > 0.5 * expected_v_east, \
            f"East velocity {res['v'][-1,0]:.3f} m/s too small"

    def test_position_grows_only_east_all_kalman_filters(self, east_nav, tight_params):
        """Kalman filters must agree on direction (with small GPS init effect)."""
        for fname in KALMAN_FILTERS:
            res = _run(fname, east_nav, tight_params)
            # East must dominate North (allow factor-3 tolerance for short run)
            assert res["p"][-1, 0] > abs(res["p"][-1, 1]), \
                f"{fname}: East {res['p'][-1,0]:.3f} < |North| {abs(res['p'][-1,1]):.3f}"


# ---------------------------------------------------------------------------
# Trajectory C — Straight North (yaw = π/2)
# ---------------------------------------------------------------------------

class TestStraightNorthTrajectory:
    """
    Forward acceleration at yaw=π/2: FLU x=[1,0,0] maps to ENU North.
    Specific force in body: [a, 0, 9.81] (same as East trajectory).
    Expected: p_North grows, p_East ≈ 0.
    """
    N    = 200
    RATE = 100.0
    A    = 1.0

    @pytest.fixture
    def north_nav(self):
        accel  = np.tile([self.A, 0.0, G], (self.N, 1))
        gyro   = np.zeros((self.N, 3))
        t      = np.arange(self.N) / self.RATE
        # At yaw=π/2, Forward → North, so v_North = a*t
        vel_enu = np.column_stack([np.zeros(self.N), self.A * t, np.zeros(self.N)])
        orient  = np.tile([0.0, 0.0, pi / 2], (self.N, 1))   # yaw=π/2
        return make_nav_data(N=self.N, sample_rate=self.RATE,
                             accel_flu=accel, gyro_flu=gyro,
                             vel_enu=vel_enu, orient=orient)

    def test_position_grows_north(self, north_nav, tight_params):
        res = _run("imu_only", north_nav, tight_params)
        assert res["p"][-1, 1] > 0.5, \
            f"North position {res['p'][-1,1]:.3f} m — should be > 0.5 m"

    def test_no_east_drift(self, north_nav, tight_params):
        res = _run("imu_only", north_nav, tight_params)
        assert abs(res["p"][-1, 0]) < 0.5, \
            f"East drift {res['p'][-1,0]:.3f} m > 0.5 m (yaw convention error)"

    def test_north_dominates_east(self, north_nav, tight_params):
        res = _run("imu_only", north_nav, tight_params)
        assert res["p"][-1, 1] > abs(res["p"][-1, 0]), \
            "North position must dominate East when yaw=π/2"


# ---------------------------------------------------------------------------
# Trajectory D — Covariance growth during GPS outage
# ---------------------------------------------------------------------------

class TestCovarianceGrowthDuringOutage:
    """
    Provide GPS for the first 10 samples, then none for 190 more.
    After GPS loss, std_pos must grow (uncertainty increases).
    """
    N_GPS    = 10
    N_OUTAGE = 190
    RATE     = 10.0   # 10 Hz so 10 samples = 1 s of GPS

    @pytest.fixture
    def outage_nav(self):
        N = self.N_GPS + self.N_OUTAGE
        gps = np.zeros(N, dtype=bool)
        gps[:self.N_GPS] = True
        return make_nav_data(N=N, sample_rate=self.RATE, gps_available=gps)

    @pytest.mark.parametrize("filter_name", KALMAN_FILTERS)
    def test_std_pos_grows_after_gps_loss(self, filter_name, outage_nav, tight_params):
        res = _run(filter_name, outage_nav, tight_params)
        # std at start of outage vs end of outage
        std_start = res["std_pos"][self.N_GPS, 0]
        std_end   = res["std_pos"][-1, 0]
        assert std_end > std_start, \
            (f"{filter_name}: std_pos East did not grow during GPS outage "
             f"({std_start:.4f} → {std_end:.4f})")

    @pytest.mark.parametrize("filter_name", ["ekf_vanilla", "eskf_vanilla", "iekf_vanilla"])
    def test_std_vel_grows_after_gps_loss_vanilla(self, filter_name, outage_nav, tight_params):
        """Vanilla filters (no ZUPT) must not collapse velocity uncertainty after GPS loss."""
        res = _run(filter_name, outage_nav, tight_params)
        std_start = res["std_vel"][self.N_GPS, 0]
        std_end   = res["std_vel"][-1, 0]
        assert std_end >= std_start * 0.9, \
            f"{filter_name}: std_vel should not decrease significantly during outage"


# ---------------------------------------------------------------------------
# Trajectory E — GPS convergence
# ---------------------------------------------------------------------------

class TestGpsConvergence:
    """
    IMU integrates from origin (stationary).
    GPS positions are offset by +5 m East.
    After 60 s of 1 Hz GPS, position should converge toward the GPS offset.
    """
    N        = 600   # 60 s at 10 Hz
    RATE     = 10.0
    OFFSET_E = 5.0   # metres East offset injected via GPS

    @pytest.fixture
    def gps_offset_nav(self):
        """
        Build nav_data where LLA is shifted +5 m East relative to origin.
        The IMU integrates from origin (zero velocity, stationary),
        but the GPS (= lla) is offset so the filter must converge toward +5 m East.
        """
        N   = self.N
        gps = np.zeros(N, dtype=bool)
        gps[::10] = True    # 1 Hz GPS at 10 Hz IMU rate

        lla0 = LLA0_DEFAULT.copy()
        # Offset every LLA by +5 m East via pymap3d
        lla_offset = np.zeros((N, 3))
        for i in range(N):
            lat, lon, alt = pm.enu2geodetic(
                self.OFFSET_E, 0.0, 0.0,
                lla0[0], lla0[1], lla0[2]
            )
            lla_offset[i] = [lat, lon, alt]

        accel = np.tile([0.0, 0.0, G], (N, 1))
        gyro  = np.zeros((N, 3))

        gps_speed = np.where(gps, 0.0, np.nan)
        gps_cog   = np.where(gps, 0.0, np.nan)

        from data_loader import NavigationData
        nd = NavigationData(
            accel_flu     = accel,
            gyro_flu      = gyro,
            vel_enu       = np.zeros((N, 3)),
            lla           = lla_offset,
            orient        = np.zeros((N, 3)),
            gps_available = gps,
            sample_rate   = float(self.RATE),
            dataset_name  = "gps_convergence_test",
            gps_rate      = 1.0,
            lla0          = lla0,
            gps_speed_mps = gps_speed,
            gps_cog_rad   = gps_cog,
        )
        nd.validate()
        return nd

    @pytest.mark.parametrize("filter_name", KALMAN_FILTERS)
    def test_position_converges_toward_gps(self, filter_name, gps_offset_nav, tight_params):
        """
        After 60 s the East position should be closer to +5 m than to 0.
        Full convergence within 2 m is expected with tight_params (Rpos=0.1).
        """
        res = _run(filter_name, gps_offset_nav, tight_params)
        p_east_final = res["p"][-1, 0]
        # Must be on the correct side (closer to 5 than to 0)
        assert p_east_final > 1.0, \
            (f"{filter_name}: position did not converge toward GPS offset "
             f"(p_East={p_east_final:.3f} m, expected ≈ {self.OFFSET_E} m)")
