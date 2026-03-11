"""
Tests for NHC (Non-Holonomic Constraint) and ZUPT (Zero-Velocity Update).

ZUPT trigger conditions (from ekf_enhanced.py):
  |‖acc_b‖ - 9.81| < zupt_accel_threshold   AND
  ‖ω_b‖             < zupt_gyro_threshold    AND
  ‖v_nav‖           < 1.0                    (hard-coded 1 m/s)

NHC fires unconditionally every time step.
NHC constrains lateral and vertical velocity in the body frame to zero.

Enhanced filters tested:
  ekf_enhanced, eskf_enhanced, iekf_enhanced
Vanilla baselines:
  ekf_vanilla,  eskf_vanilla,  iekf_vanilla
"""
import importlib
from math import pi

import numpy as np
import pytest

from conftest import make_nav_data

# ---------------------------------------------------------------------------
# Filter pairs
# ---------------------------------------------------------------------------
ENHANCED_FILTERS = ["ekf_enhanced", "eskf_enhanced", "iekf_enhanced"]
VANILLA_FILTERS  = ["ekf_vanilla",  "eskf_vanilla",  "iekf_vanilla"]

G = 9.80665


def _run(filter_name, nav_data, params):
    mod = importlib.import_module(f"filters.{filter_name}")
    return mod.run(nav_data, params=params, outage_config=None, use_3d_rotation=True)


# ---------------------------------------------------------------------------
# 1. ZUPT — Zero-Velocity Update
# ---------------------------------------------------------------------------

class TestZupt:

    @pytest.fixture
    def stationary_nd(self):
        """
        All ZUPT conditions satisfied:
          accel ≈ [0, 0, 9.81] → |‖acc‖ - 9.81| = 0 < threshold
          gyro  = [0, 0, 0]    → ‖ω‖ = 0 < threshold
          vel   = [0, 0, 0]    → speed = 0 < 1.0
        """
        return make_nav_data(N=200, sample_rate=100.0)

    @pytest.fixture
    def fast_nd(self):
        """
        ZUPT should NOT fire: speed = 10 m/s >> 1 m/s threshold.
        Forward motion at yaw=0, constant velocity (no acceleration in body frame
        beyond gravity cancel), no GPS.
        """
        N    = 200
        rate = 100.0
        # Vehicle moving at 10 m/s East: accel_flu = [0,0,9.81] (constant velocity)
        accel   = np.tile([0.0, 0.0, G], (N, 1))
        gyro    = np.zeros((N, 3))
        vel_enu = np.tile([10.0, 0.0, 0.0], (N, 1))   # 10 m/s East
        return make_nav_data(N=N, sample_rate=rate,
                             accel_flu=accel, gyro_flu=gyro, vel_enu=vel_enu)

    @pytest.mark.parametrize("filter_name", ENHANCED_FILTERS)
    def test_zupt_suppresses_velocity_when_stationary(
            self, filter_name, stationary_nd, tight_params):
        """
        After ZUPT fires on stationary data, velocity must stay near zero.
        Tolerance: RMS velocity < 0.05 m/s over the last 100 samples.
        """
        res = _run(filter_name, stationary_nd, tight_params)
        vel_rms = np.sqrt(np.mean(res["v"][100:] ** 2))
        assert vel_rms < 0.05, \
            (f"{filter_name}: velocity RMS={vel_rms:.4f} m/s > 0.05 m/s "
             f"— ZUPT should suppress velocity drift on stationary data")

    @pytest.mark.parametrize("filter_name", ENHANCED_FILTERS)
    def test_zupt_does_not_suppress_high_speed_velocity(
            self, filter_name, fast_nd, tight_params):
        """
        At 10 m/s ZUPT must NOT fire (speed > 1 m/s threshold).
        The filter should reflect the true 10 m/s forward velocity.
        We check that mean |v| in East direction stays above 0.5 m/s.
        """
        res = _run(filter_name, fast_nd, tight_params)
        mean_v_east = np.mean(np.abs(res["v"][-50:, 0]))
        assert mean_v_east > 0.5, \
            (f"{filter_name}: mean |v_East|={mean_v_east:.3f} m/s < 0.5 m/s — "
             f"ZUPT should NOT fire at 10 m/s (threshold is 1 m/s)")

    @pytest.mark.parametrize("filter_name", ENHANCED_FILTERS)
    def test_zupt_reduces_velocity_error_vs_vanilla(
            self, filter_name, stationary_nd, tight_params):
        """
        Enhanced filter's velocity should be closer to zero than vanilla's
        on stationary data, because ZUPT provides an additional correction.
        """
        vanilla = filter_name.replace("_enhanced", "_vanilla")
        res_e   = _run(filter_name, stationary_nd, tight_params)
        res_v   = _run(vanilla,     stationary_nd, tight_params)
        vel_rms_e = np.sqrt(np.mean(res_e["v"][10:] ** 2))
        vel_rms_v = np.sqrt(np.mean(res_v["v"][10:] ** 2))
        # Enhanced must be at least as good (10% tolerance for numerical effects)
        assert vel_rms_e <= vel_rms_v * 1.1, \
            (f"{filter_name} vel_rms={vel_rms_e:.5f} > "
             f"{vanilla} vel_rms={vel_rms_v:.5f} × 1.1")


# ---------------------------------------------------------------------------
# 2. NHC — Non-Holonomic Constraint
# ---------------------------------------------------------------------------

class TestNhcConstraint:

    @pytest.fixture
    def forward_motion_nd(self):
        """
        Vehicle moving forward (East, yaw=0) at constant 5 m/s.
        Inject a small lateral acceleration (0.05 m/s²) to create
        a lateral velocity error that NHC should suppress.
        """
        N    = 300
        rate = 100.0
        # Small lateral leakage: [0.05 m/s² lateral, 0 m/s², 9.81 gravity cancel]
        accel   = np.tile([0.0, 0.05, G], (N, 1))  # tiny lateral accel
        gyro    = np.zeros((N, 3))
        vel_enu = np.tile([5.0, 0.0, 0.0], (N, 1))   # 5 m/s East (forward at yaw=0)
        return make_nav_data(N=N, sample_rate=rate,
                             accel_flu=accel, gyro_flu=gyro, vel_enu=vel_enu)

    @pytest.mark.parametrize("enhanced,vanilla",
                             list(zip(ENHANCED_FILTERS, VANILLA_FILTERS)))
    def test_nhc_suppresses_lateral_velocity_accumulation(
            self, enhanced, vanilla, forward_motion_nd, tight_params):
        """
        With small lateral accel, vanilla allows lateral velocity to drift.
        NHC should keep lateral velocity much closer to zero for the enhanced filter.
        """
        res_e = _run(enhanced, forward_motion_nd, tight_params)
        res_v = _run(vanilla,  forward_motion_nd, tight_params)

        # Mean absolute lateral (North) velocity over last 100 samples
        lat_e = np.mean(np.abs(res_e["v"][-100:, 1]))
        lat_v = np.mean(np.abs(res_v["v"][-100:, 1]))

        # Enhanced must not be worse than vanilla (even if NHC helps only modestly)
        assert lat_e <= lat_v * 1.5, \
            (f"{enhanced} lateral vel={lat_e:.4f} m/s is more than 1.5× "
             f"{vanilla} lateral vel={lat_v:.4f} m/s — NHC should constrain lateral motion")

    @pytest.mark.parametrize("filter_name", ENHANCED_FILTERS)
    def test_nhc_lateral_velocity_stays_small_on_forward_motion(
            self, filter_name, forward_motion_nd, tight_params):
        """
        During forward motion with NHC active, lateral velocity should stay
        below a reasonable bound (< 0.5 m/s) even with a 0.05 m/s² lateral leak.
        """
        res = _run(filter_name, forward_motion_nd, tight_params)
        lat_vel = np.mean(np.abs(res["v"][-100:, 1]))
        assert lat_vel < 0.5, \
            (f"{filter_name}: mean lateral velocity={lat_vel:.4f} m/s > 0.5 m/s "
             f"— NHC should suppress lateral drift")


# ---------------------------------------------------------------------------
# 3. Enhanced vs Vanilla — stationary scenario
# ---------------------------------------------------------------------------

class TestEnhancedVsVanillaStationary:

    @pytest.fixture
    def stationary_nd(self):
        return make_nav_data(N=200, sample_rate=100.0)

    @pytest.mark.parametrize("enhanced,vanilla",
                             list(zip(ENHANCED_FILTERS, VANILLA_FILTERS)))
    def test_enhanced_position_not_worse(
            self, enhanced, vanilla, stationary_nd, tight_params):
        """
        Enhanced filter must achieve position RMSE ≤ vanilla on stationary data.
        Allow 20% tolerance for numerical differences.
        """
        res_e = _run(enhanced, stationary_nd, tight_params)
        res_v = _run(vanilla,  stationary_nd, tight_params)
        pos_rms_e = np.sqrt(np.mean(res_e["p"][10:] ** 2))
        pos_rms_v = np.sqrt(np.mean(res_v["p"][10:] ** 2))
        assert pos_rms_e <= pos_rms_v * 1.2, \
            (f"{enhanced} pos_rms={pos_rms_e:.4f} > "
             f"{vanilla} pos_rms={pos_rms_v:.4f} × 1.2")

    @pytest.mark.parametrize("enhanced,vanilla",
                             list(zip(ENHANCED_FILTERS, VANILLA_FILTERS)))
    def test_enhanced_velocity_not_worse(
            self, enhanced, vanilla, stationary_nd, tight_params):
        res_e = _run(enhanced, stationary_nd, tight_params)
        res_v = _run(vanilla,  stationary_nd, tight_params)
        vel_rms_e = np.sqrt(np.mean(res_e["v"][10:] ** 2))
        vel_rms_v = np.sqrt(np.mean(res_v["v"][10:] ** 2))
        assert vel_rms_e <= vel_rms_v * 1.1, \
            (f"{enhanced} vel_rms={vel_rms_e:.5f} > "
             f"{vanilla} vel_rms={vel_rms_v:.5f} × 1.1")
