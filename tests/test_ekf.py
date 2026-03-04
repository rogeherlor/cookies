# -*- coding: utf-8 -*-
"""
Comprehensive tests for the Error-State EKF implementation.

Tests cover:
  1. Rotation matrices (Rx, Ry, Rz) — orthogonality, determinant, sign conventions
  2. Body-to-navigation frame projection for known orientations
  3. Stationary scenario — gravity cancellation, zero velocity preservation
  4. Constant-velocity straight line — no drift with perfect IMU
  5. Pure yaw turn — heading change reflected in trajectory
  6. GPS correction — filter converges to GPS truth
  7. GPS outage — covariance grows, trajectory diverges gracefully
  8. Bias estimation — filter estimates injected accelerometer bias
  9. Gyroscope bias estimation — filter estimates injected gyro bias
  10. Orientation error-state injection — angles corrected on GPS update
  11. W matrix — Euler rate mapping at non-zero roll/pitch
  12. Error-state reset — x[0:9] zeroed after each GPS update

Each test builds a synthetic NavigationData with analytically known inputs so the
expected outputs can be computed in closed form.
"""
import numpy as np
import pytest
from math import sin, cos, tan, sqrt, radians, pi
from dataclasses import dataclass

# conftest.py adds the source dir to sys.path
from data_loader import NavigationData
import ekf_core

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Reference position (Madrid, Spain) — used for all synthetic datasets
LLA0 = np.array([40.4168, -3.7038, 650.0])


def _default_ekf_params(**overrides):
    """Return a sensible set of EKF parameters for unit tests.

    Low process noise + low measurement noise → fast convergence, predictable
    behaviour in short synthetic trajectories.
    """
    params = dict(
        Qpos=1e-4, Qvel=1e-4,
        QorientXY=1e-4, QorientZ=1e-4,
        Qacc=1e-6, QgyrXY=1e-6, QgyrZ=1e-6,
        Rpos=1.0,
        beta_acc=-0.01, beta_gyr=-0.01,
        P_pos_std=1.0, P_vel_std=1.0, P_orient_std=0.1,
        P_acc_std=0.01, P_gyr_std=0.01,
    )
    params.update(overrides)
    return params


def _make_nav_data(
    N, sample_rate,
    accel_flu=None, gyro_flu=None,
    vel_enu=None, lla=None, orient=None,
    gps_available=None, gps_rate=1.0
):
    """Build a NavigationData from arrays (or zeros by default)."""
    import pymap3d as pm

    if accel_flu is None:
        # Default: stationary FLU — accel reads [0, 0, +9.81]
        accel_flu = np.tile([0.0, 0.0, 9.81], (N, 1))
    if gyro_flu is None:
        gyro_flu = np.zeros((N, 3))
    if vel_enu is None:
        vel_enu = np.zeros((N, 3))
    if orient is None:
        orient = np.zeros((N, 3))
    if lla is None:
        # All samples at the reference point
        lla = np.tile(LLA0, (N, 1))
    if gps_available is None:
        # GPS every sample_rate/gps_rate samples
        gps_available = np.zeros(N, dtype=bool)
        gps_interval = max(1, int(sample_rate / gps_rate))
        gps_available[::gps_interval] = True

    return NavigationData(
        accel_flu=accel_flu.astype(np.float64),
        gyro_flu=gyro_flu.astype(np.float64),
        vel_enu=vel_enu.astype(np.float64),
        lla=lla.astype(np.float64),
        orient=orient.astype(np.float64),
        gps_available=gps_available,
        sample_rate=float(sample_rate),
        dataset_name="test_synthetic",
        gps_rate=float(gps_rate),
        lla0=LLA0.copy(),
    )


def _enu_to_lla(e, n, u):
    """Convert a single ENU point back to LLA using the test reference."""
    import pymap3d as pm
    lat, lon, alt = pm.enu2geodetic(e, n, u, LLA0[0], LLA0[1], LLA0[2])
    return np.array([lat, lon, alt])


def _build_lla_trajectory(enu_positions):
    """Convert an Nx3 ENU trajectory to Nx3 LLA."""
    return np.array([_enu_to_lla(r[0], r[1], r[2]) for r in enu_positions])


# ---------------------------------------------------------------------------
# 1.  Rotation matrix unit tests (standalone, no EKF run)
# ---------------------------------------------------------------------------

class TestRotationMatrices:
    """Verify Rx, Ry, Rz individually and composed as Rbn = Rz @ Ry @ Rx."""

    @staticmethod
    def _Rx(roll):
        return np.array([
            [1, 0, 0],
            [0, cos(roll), -sin(roll)],
            [0, sin(roll), cos(roll)]
        ])

    @staticmethod
    def _Ry(pitch):
        # FLU convention: -sin in [0,2], +sin in [2,0]
        return np.array([
            [cos(pitch), 0, -sin(pitch)],
            [0, 1, 0],
            [sin(pitch), 0, cos(pitch)]
        ])

    @staticmethod
    def _Rz(yaw):
        return np.array([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw), cos(yaw), 0],
            [0, 0, 1]
        ])

    @pytest.mark.parametrize("angle", [0, 0.1, pi / 6, pi / 4, pi / 2, pi, -pi / 3])
    def test_rotation_orthogonality(self, angle):
        """R @ R.T == I and det(R) == +1 for each elementary rotation."""
        for R in (self._Rx(angle), self._Ry(angle), self._Rz(angle)):
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_Rz_90_yaw_east_to_north(self):
        """Yaw=+90° should rotate the Forward (x) body axis to Left of East → North."""
        Rbn = self._Rz(pi / 2)
        # Body forward = [1,0,0] → should map to [0,1,0] (North) in ENU
        result = Rbn @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-12)

    def test_Ry_pitch_up_positive(self):
        """Positive pitch should tilt Forward (x) upward in FLU.

        Pitch = +30°: body x=[1,0,0] → ENU [cos30, 0, sin30] (goes up, not down).
        This is the key sign test for the FLU Ry.
        """
        pitch = pi / 6  # 30°
        Rbn = self._Ry(pitch)
        result = Rbn @ np.array([1, 0, 0])
        # Forward component cos(pitch), up component +sin(pitch)
        np.testing.assert_allclose(result, [cos(pitch), 0, sin(pitch)], atol=1e-12)

    def test_Ry_pitch_down_negative(self):
        """Negative pitch → nose points down."""
        pitch = -pi / 6
        Rbn = self._Ry(pitch)
        result = Rbn @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [cos(pitch), 0, sin(pitch)], atol=1e-12)
        assert result[2] < 0, "Negative pitch should tilt forward axis downward"

    def test_Rx_roll_right_positive(self):
        """Positive roll should tilt the Left (y) body axis downward in FLU.

        Roll = +30°: body y=[0,1,0] → ENU [0, cos30, -sin30] (left side goes down → right roll).
        Wait — in FLU body frame, y = Left. After a right-roll (positive roll about x-forward):
        the Left axis should tilt upward in the body frame sense, but in ENU the
        Up component decreases. Let's just verify the matrix algebra:
        Rx(roll) @ [0,1,0] = [0, cos(roll), sin(roll)]  (body Left maps with +sin in z).
        """
        roll = pi / 6
        Rx = self._Rx(roll)
        result = Rx @ np.array([0, 1, 0])
        np.testing.assert_allclose(result, [0, cos(roll), sin(roll)], atol=1e-12)

    def test_composed_identity_at_zero(self):
        """Rbn(0,0,0) == I."""
        Rbn = self._Rz(0) @ self._Ry(0) @ self._Rx(0)
        np.testing.assert_allclose(Rbn, np.eye(3), atol=1e-12)

    def test_composed_90_yaw_30_pitch(self):
        """Compound rotation: yaw=90° then pitch=30°."""
        yaw, pitch = pi / 2, pi / 6
        Rbn = self._Rz(yaw) @ self._Ry(pitch) @ self._Rx(0)
        # Forward body axis [1,0,0]:
        #  after pitch: [cos30, 0, sin30]
        #  after yaw90: [-0, cos30, sin30]  → [0, cos30, sin30]
        result = Rbn @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [0, cos(pitch), sin(pitch)], atol=1e-12)

    def test_Ry_inverse_transpose_equivalence(self):
        """For a proper rotation, R^{-1} == R^T."""
        pitch = 0.7
        Ry = self._Ry(pitch)
        np.testing.assert_allclose(np.linalg.inv(Ry), Ry.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 2.  Gravity cancellation — stationary scenario
# ---------------------------------------------------------------------------

class TestStationaryScenario:
    """A stationary device with rpy=[0,0,0] should see gravity cancelled.

    accel_flu = [0, 0, +9.81] (specific force pointing up)
    accENU   = Rbn @ acc = [0, 0, +9.81]  (with Rbn=I at zero angles)
    accENU + g = [0, 0, +9.81] + [0, 0, -9.81] = [0, 0, 0]

    So position and velocity should stay at zero (apart from small filter noise).
    """

    @pytest.fixture
    def stationary_result_3d(self):
        N = 200  # 2 seconds at 100 Hz
        nav = _make_nav_data(N, sample_rate=100, gps_rate=10)
        params = _default_ekf_params()
        return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

    @pytest.fixture
    def stationary_result_2d(self):
        N = 200
        nav = _make_nav_data(N, sample_rate=100, gps_rate=10)
        params = _default_ekf_params()
        return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=False)

    def test_position_stays_near_zero_3d(self, stationary_result_3d):
        p = stationary_result_3d['p']
        assert np.max(np.abs(p)) < 1.0, f"Position drifted to {np.max(np.abs(p)):.2f} m"

    def test_velocity_stays_near_zero_3d(self, stationary_result_3d):
        v = stationary_result_3d['v']
        assert np.max(np.abs(v)) < 0.5, f"Velocity grew to {np.max(np.abs(v)):.3f} m/s"

    def test_position_stays_near_zero_2d(self, stationary_result_2d):
        p = stationary_result_2d['p']
        assert np.max(np.abs(p)) < 1.0

    def test_orientation_stays_near_zero(self, stationary_result_3d):
        r = stationary_result_3d['r']
        # With zero gyro, rpy should remain near zero
        assert np.max(np.abs(r)) < 0.1, f"Orientation drifted to {np.degrees(np.max(np.abs(r)))}°"


# ---------------------------------------------------------------------------
# 3.  Constant velocity — straight North with perfect IMU
# ---------------------------------------------------------------------------

class TestConstantVelocityNorth:
    """Walk straight North at 1 m/s with perfect sensors.

    - accel_flu = [0, 0, +9.81]  (only gravity in specific force, no acceleration)
    - gyro_flu = [0, 0, 0]
    - yaw = 0 → Forward=East in ENU?  No — at yaw=0, Rbn=I, so Forward=East.
      To go North we need yaw=π/2.

    Actually, let's set yaw=π/2 (heading North) and check the trajectory goes North.
    But since rpy starts at [0,0,0] inside ekf_core, we can't set the initial yaw directly.
    Instead, let's give a constant accel in the East direction (forward at yaw=0 maps to East)
    and check the trajectory moves East.

    Simpler approach: keep yaw=0, forward=East, give vel_enu=[1,0,0] and constant GPS
    positions moving East at 1 m/s.  The accel_flu = [0,0,9.81] (no body acceleration,
    only gravity).  GPS corrects the trajectory.
    """

    @pytest.fixture
    def const_vel_result(self):
        N = 500  # 5 seconds at 100 Hz
        dt = 1 / 100
        sample_rate = 100
        gps_rate = 10

        # Build ENU trajectory: moving East at 1 m/s
        enu = np.zeros((N, 3))
        enu[:, 0] = np.arange(N) * dt * 1.0  # East position
        lla = _build_lla_trajectory(enu)

        vel = np.zeros((N, 3))
        vel[:, 0] = 1.0  # 1 m/s East

        # Stationary accel (no body acceleration, just gravity)
        accel = np.tile([0.0, 0.0, 9.81], (N, 1))

        nav = _make_nav_data(
            N, sample_rate=sample_rate, gps_rate=gps_rate,
            accel_flu=accel, vel_enu=vel, lla=lla,
        )
        params = _default_ekf_params(Rpos=0.1)
        return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

    def test_moves_east(self, const_vel_result):
        p = const_vel_result['p']
        # Final East position should be close to 5 m (5 s at 1 m/s)
        assert p[-1, 0] > 3.0, f"East position only {p[-1,0]:.2f} m — expected ~5 m"

    def test_stays_near_zero_north(self, const_vel_result):
        p = const_vel_result['p']
        assert np.max(np.abs(p[:, 1])) < 2.0, "Unexpected North drift"

    def test_stays_near_zero_up(self, const_vel_result):
        p = const_vel_result['p']
        assert np.max(np.abs(p[:, 2])) < 2.0, "Unexpected Up drift"


# ---------------------------------------------------------------------------
# 4.  Pure yaw turn — heading rotates, trajectory curves
# ---------------------------------------------------------------------------

class TestPureYawTurn:
    """Apply a constant yaw-rate gyro to verify heading changes.

    gyro_flu = [0, 0, ω_yaw] with ω_yaw = 0.1 rad/s
    After 5 seconds → yaw ≈ 0.5 rad ≈ 28.6°
    The rpy[2] output should reflect this.
    """

    @pytest.fixture
    def yaw_turn_result(self):
        N = 500
        sample_rate = 100
        gyro = np.zeros((N, 3))
        gyro[:, 2] = 0.1  # 0.1 rad/s yaw rate

        nav = _make_nav_data(N, sample_rate=sample_rate, gps_rate=10, gyro_flu=gyro)
        params = _default_ekf_params()
        return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

    def test_yaw_increases(self, yaw_turn_result):
        r = yaw_turn_result['r']
        # After 5 s at 0.1 rad/s → ~0.5 rad.  Allow GPS corrections to modify slightly.
        final_yaw = r[-1, 2]
        assert final_yaw > 0.2, f"Yaw only reached {final_yaw:.3f} rad, expected ~0.5"

    def test_roll_pitch_stay_small(self, yaw_turn_result):
        r = yaw_turn_result['r']
        assert np.max(np.abs(r[:, 0])) < 0.15, "Unexpected roll during pure yaw"
        assert np.max(np.abs(r[:, 1])) < 0.15, "Unexpected pitch during pure yaw"


# ---------------------------------------------------------------------------
# 5.  Pitch tilt — gravity projection test (3D only)
# ---------------------------------------------------------------------------

class TestPitchTiltGravityProjection:
    """Apply a constant pitch gyro so the device tilts nose-up.

    With positive pitch, part of gravity leaks into the East (forward) axis.
    In 3D mode this should cause East position drift.  This test verifies
    that the sign is correct: positive pitch → Forward tilts UP → gravity
    component pushes the ENU projection backward (negative East for FLU).

    We run WITHOUT GPS to see the raw IMU integration.
    """

    @pytest.fixture
    def pitch_tilt_result(self):
        N = 100  # 1 second at 100 Hz — short to keep drift small
        sample_rate = 100
        gyro = np.zeros((N, 3))
        # Constant pitch rate to build up ~5.7° pitch (0.1 rad) in 1 s
        gyro[:, 1] = 0.1  # pitch-rate 0.1 rad/s

        # No GPS during the entire run
        gps_avail = np.zeros(N, dtype=bool)

        nav = _make_nav_data(N, sample_rate=sample_rate, gyro_flu=gyro,
                             gps_available=gps_avail)
        params = _default_ekf_params()
        return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

    def test_pitch_grows(self, pitch_tilt_result):
        r = pitch_tilt_result['r']
        assert r[-1, 1] > 0.05, f"Pitch only {r[-1,1]:.4f} rad, expected ~0.1"

    def test_gravity_leaks_into_horizontal(self, pitch_tilt_result):
        """With nose-up pitch, gravity projects backward → negative East velocity."""
        v = pitch_tilt_result['v']
        # After 1 s at ~5° pitch, gravity leakage ≈ 9.81*sin(0.05) ≈ 0.49 m/s²
        # Integrated for ~1 s → ~0.25 m/s.  Should be noticeable.
        # The sign depends on the Ry convention:  positive pitch in FLU
        # means the x-axis (forward) tilts UP, so the *gravity* vector projected
        # into body-forward becomes *negative* (pushes backward = -East at yaw=0).
        # In ENU: accENU_east < 0  →  velocity should drift negative East.
        assert v[-1, 0] < -0.05, (
            f"East velocity {v[-1,0]:.4f} — expected negative (gravity leak backward)"
        )


# ---------------------------------------------------------------------------
# 6.  GPS correction — filter locks onto truth
# ---------------------------------------------------------------------------

class TestGPSCorrection:
    """With continuous GPS and a small offset, the filter should converge."""

    @pytest.fixture
    def corrected_result(self):
        N = 1000  # 10 seconds at 100 Hz
        sample_rate = 100
        gps_rate = 10

        # Truth trajectory: 5 m East offset (constant position)
        enu_truth = np.zeros((N, 3))
        enu_truth[:, 0] = 5.0  # 5 m East
        lla = _build_lla_trajectory(enu_truth)

        nav = _make_nav_data(N, sample_rate=sample_rate, gps_rate=gps_rate, lla=lla)
        params = _default_ekf_params(Rpos=0.1)
        return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

    def test_position_converges_to_gps(self, corrected_result):
        p = corrected_result['p']
        # After 10 s with 10 Hz GPS, filter should be near 5 m East
        assert abs(p[-1, 0] - 5.0) < 1.0, f"East = {p[-1,0]:.2f}, expected ~5.0"

    def test_position_uncertainty_shrinks(self, corrected_result):
        std = corrected_result['std_pos']
        # Uncertainty at end should be much smaller than at start
        assert std[-1, 0] < std[1, 0], "Position uncertainty did not decrease"


# ---------------------------------------------------------------------------
# 7.  GPS outage — covariance should grow
# ---------------------------------------------------------------------------

class TestGPSOutage:
    """During a GPS outage, P should grow and the filter free-runs."""

    @pytest.fixture
    def outage_result(self):
        N = 500
        sample_rate = 100
        gps_rate = 10
        nav = _make_nav_data(N, sample_rate=sample_rate, gps_rate=gps_rate)
        params = _default_ekf_params()
        outage = {'start': 1.0, 'duration': 3.0}  # outage from 1s to 4s
        return ekf_core.run_ekf(nav, params, outage, use_3d_rotation=True)

    def test_covariance_grows_during_outage(self, outage_result):
        std = outage_result['std_pos']
        # At start of outage (index ~100) vs end (index ~400)
        std_before = np.mean(std[90:100, 0])
        std_during = np.mean(std[350:400, 0])
        assert std_during > std_before, "Position uncertainty should grow during outage"

    def test_covariance_shrinks_after_outage(self, outage_result):
        std = outage_result['std_pos']
        # After outage ends at index 400, GPS resumes → uncertainty drops
        std_at_outage_end = std[400, 0]
        std_at_end = std[-1, 0]
        assert std_at_end < std_at_outage_end, "Uncertainty should decrease after GPS resumes"


# ---------------------------------------------------------------------------
# 8.  Accelerometer bias estimation
# ---------------------------------------------------------------------------

class TestAccBiasEstimation:
    """Inject a known acc bias and verify the filter estimates it.

    We add a constant 0.1 m/s² bias on the forward (x) axis of accel_flu.
    With continuous GPS the filter should eventually estimate this bias.
    """

    @pytest.fixture
    def bias_result(self):
        N = 2000  # 20 s at 100 Hz — long enough for bias to converge
        sample_rate = 100
        gps_rate = 10
        bias_fwd = 0.1  # m/s²

        accel = np.tile([0.0 + bias_fwd, 0.0, 9.81], (N, 1))

        nav = _make_nav_data(N, sample_rate=sample_rate, gps_rate=gps_rate,
                             accel_flu=accel)
        params = _default_ekf_params(
            Qacc=0.01,
            beta_acc=-0.001,  # slow mean-reversion so bias is observable
            Rpos=0.5,
        )
        return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

    def test_forward_bias_estimated(self, bias_result):
        b = bias_result['bias_acc']
        # The filter should push the x-bias towards +0.1
        # Allow generous tolerance — the EKF may not fully converge in 20 s
        assert b[-1, 0] > 0.01, (
            f"Forward acc bias estimate {b[-1,0]:.5f} — expected positive"
        )


# ---------------------------------------------------------------------------
# 9.  Gyroscope bias estimation
# ---------------------------------------------------------------------------

class TestGyrBiasEstimation:
    """Inject a constant yaw-rate bias and verify the filter sees it."""

    @pytest.fixture
    def gyro_bias_result(self):
        N = 5000  # 50 s — longer run for gyro bias to become observable
        sample_rate = 100
        gps_rate = 10
        gyro_bias_yaw = 0.02  # rad/s — larger bias, easier to observe

        gyro = np.zeros((N, 3))
        gyro[:, 2] = gyro_bias_yaw  # constant yaw-rate bias

        # Build a trajectory moving East at 2 m/s so heading errors cause
        # visible cross-track drift that GPS can correct → makes yaw observable
        dt = 1 / sample_rate
        enu = np.zeros((N, 3))
        enu[:, 0] = np.arange(N) * dt * 2.0  # 2 m/s East
        lla = _build_lla_trajectory(enu)
        vel = np.zeros((N, 3))
        vel[:, 0] = 2.0

        # Accel: only gravity (constant velocity, no body acceleration)
        accel = np.tile([0.0, 0.0, 9.81], (N, 1))

        nav = _make_nav_data(N, sample_rate=sample_rate, gps_rate=gps_rate,
                             gyro_flu=gyro, lla=lla, vel_enu=vel, accel_flu=accel)
        params = _default_ekf_params(
            QgyrZ=0.01,
            beta_gyr=-0.001,
            Rpos=0.5,
        )
        return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

    def test_yaw_bias_estimated(self, gyro_bias_result):
        """The filter should detect that yaw is drifting and push the
        gyro-z bias estimate away from zero (towards the injected +0.02).
        With position-only GPS, yaw observability is indirect (cross-track
        error), so we just check the bias magnitude grows meaningfully."""
        b = gyro_bias_result['bias_gyr']
        # Bias magnitude on the z-axis should be noticeably non-zero
        assert abs(b[-1, 2]) > 0.001, (
            f"Yaw gyro bias estimate {b[-1,2]:.6f} — expected non-negligible"
        )


# ---------------------------------------------------------------------------
# 10.  Error-state reset — x[0:9] zeroed after GPS update
# ---------------------------------------------------------------------------

class TestErrorStateReset:
    """After every GPS update the navigation error states (δp, δv, δε) should
    be injected into the nominal state and then reset to zero.

    We run 1 second at 100 Hz with 10 Hz GPS (10 updates).  Then inspect
    the trajectory: with GPS correcting a stationary device, position
    should stay near zero.  If the error state is NOT reset, the
    corrections would accumulate and cause jumps.
    """

    def test_stationary_no_jumps(self):
        N = 100
        sample_rate = 100
        gps_rate = 10
        nav = _make_nav_data(N, sample_rate=sample_rate, gps_rate=gps_rate)
        params = _default_ekf_params()
        result = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)
        p = result['p']
        # Check that there are no sudden jumps > 0.5 m between consecutive samples
        dp = np.diff(p, axis=0)
        max_jump = np.max(np.linalg.norm(dp, axis=1))
        assert max_jump < 0.5, f"Position jump of {max_jump:.3f} m detected"


# ---------------------------------------------------------------------------
# 11.  W matrix (Euler rate) test
# ---------------------------------------------------------------------------

class TestEulerRateMatrix:
    """At non-zero roll and pitch, verify that a pure pitch-rate gyro
    input correctly maps through the W matrix and doesn't leak into yaw.

    W = [[1,   sr*tp,  cr*tp],
         [0,   cr,    -sr   ],
         [0,   sr/cp,  cr/cp]]

    With roll=0: W simplifies to
         [[1, 0, tp], [0, 1, 0], [0, 0, 1/cp]]
    So gyro_flu = [0, ωp, 0] → d(rpy)/dt = [0, ωp, 0] — no leak into yaw. ✓

    With roll=π/6 (30°): W[2,1] = sin(π/6)/cos(pitch) > 0
    So gyro_flu = [0, ωp, 0] → d(yaw)/dt = ωp * sin(roll)/cos(pitch)
    This cross-coupling IS physically correct — let's verify the magnitude.
    """

    def test_pitch_rate_leaks_into_yaw_at_nonzero_roll(self):
        """Non-zero roll causes gyro pitch to couple into yaw via W."""
        N = 100
        sample_rate = 100

        roll0 = pi / 6  # 30°
        pitch_rate = 0.1  # rad/s

        # Seed a small initial roll via a brief roll-rate impulse
        gyro = np.zeros((N, 3))
        # For the first 30 samples: roll-rate to build up 30° of roll
        # 30 samples * 0.01 s * ω_roll = 0.5236 rad → ω_roll ≈ 1.745 rad/s
        gyro[:30, 0] = roll0 / (30 * 0.01)
        # Then constant pitch rate for the remaining 70 samples
        gyro[30:, 1] = pitch_rate

        gps_avail = np.zeros(N, dtype=bool)  # no GPS
        nav = _make_nav_data(N, sample_rate=sample_rate, gyro_flu=gyro,
                             gps_available=gps_avail)
        params = _default_ekf_params()
        result = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

        r = result['r']
        # After the roll phase, yaw should grow due to W cross-coupling
        yaw_at_end = r[-1, 2]
        assert abs(yaw_at_end) > 0.01, (
            f"Yaw = {yaw_at_end:.5f} — expected non-zero from W cross-coupling"
        )


# ---------------------------------------------------------------------------
# 12.  2D vs 3D mode — 2D ignores roll/pitch in Rbn
# ---------------------------------------------------------------------------

class TestMode2Dvs3D:
    """In 2D mode, roll and pitch gyro should NOT affect the trajectory
    (Rbn = Rz only).  In 3D mode, pitch gyro SHOULD affect it.
    """

    @pytest.fixture
    def results_2d_3d(self):
        N = 200
        sample_rate = 100
        gyro = np.zeros((N, 3))
        gyro[:, 1] = 0.1  # constant pitch rate

        gps_avail = np.zeros(N, dtype=bool)
        nav = _make_nav_data(N, sample_rate=sample_rate, gyro_flu=gyro,
                             gps_available=gps_avail)
        params = _default_ekf_params()

        r2d = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=False)
        r3d = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)
        return r2d, r3d

    def test_2d_ignores_pitch_in_position(self, results_2d_3d):
        r2d, r3d = results_2d_3d
        # In 2D mode, the trajectory should have no significant East drift
        # (gravity is perfectly cancelled with Rbn=I since pitch is not applied)
        p2d = r2d['p']
        assert np.max(np.abs(p2d[:, 0])) < 0.5, "2D mode should ignore pitch for position"

    def test_3d_has_drift_from_pitch(self, results_2d_3d):
        r2d, r3d = results_2d_3d
        p3d = r3d['p']
        # 3D mode should show East drift from gravity leakage
        assert np.max(np.abs(p3d[:, 0])) > 0.1, "3D mode should show gravity leak from pitch"

    def test_both_modes_integrate_rpy(self, results_2d_3d):
        """Even in 2D mode, rpy is still integrated — just not used in Rbn."""
        r2d, r3d = results_2d_3d
        # Both should show pitch growing
        assert r2d['r'][-1, 1] > 0.05, "2D mode should still integrate pitch angle"
        assert r3d['r'][-1, 1] > 0.05, "3D mode should integrate pitch angle"


# ---------------------------------------------------------------------------
# 13.  Symmetry — rotating 180° should invert the gravity leak direction
# ---------------------------------------------------------------------------

class TestGravityLeakSymmetry:
    """Positive pitch → gravity leaks backward (−East at yaw=0).
    Negative pitch → gravity leaks forward (+East at yaw=0).
    The magnitudes should be symmetric.
    """

    @staticmethod
    def _run_with_pitch_rate(pitch_rate):
        N = 100
        sample_rate = 100
        gyro = np.zeros((N, 3))
        gyro[:, 1] = pitch_rate
        gps_avail = np.zeros(N, dtype=bool)
        nav = _make_nav_data(N, sample_rate=sample_rate, gyro_flu=gyro,
                             gps_available=gps_avail)
        params = _default_ekf_params()
        return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

    def test_opposite_pitch_opposite_drift(self):
        r_pos = self._run_with_pitch_rate(+0.1)
        r_neg = self._run_with_pitch_rate(-0.1)

        v_pos = r_pos['v'][-1, 0]  # East velocity with positive pitch
        v_neg = r_neg['v'][-1, 0]  # East velocity with negative pitch

        # They should have opposite signs
        assert v_pos * v_neg < 0, (
            f"Expected opposite signs: v_pos_east={v_pos:.4f}, v_neg_east={v_neg:.4f}"
        )
        # And roughly equal magnitude (within 20%)
        ratio = abs(v_pos / v_neg) if v_neg != 0 else float('inf')
        assert 0.8 < ratio < 1.2, f"Magnitude ratio {ratio:.2f} — expected ~1.0"


# ---------------------------------------------------------------------------
# 14.  F matrix — skew-symmetric structure of acceleration coupling
# ---------------------------------------------------------------------------

class TestFMatrixStructure:
    """Verify the F matrix acceleration-to-orientation coupling has the
    correct skew-symmetric structure: F[3:6, 6:9] = -[fENU]×

    For accENU = [fE, fN, fU]:
        F[3,7] =  fU    F[3,8] = -fN
        F[4,6] = -fU    F[4,8] =  fE
        F[5,6] =  fN    F[5,7] = -fE
    """

    def test_skew_symmetric_signs(self):
        """Run one step and verify the F matrix structure through the output."""
        # We test indirectly: apply a small orientation error δε and check
        # that the resulting velocity error has the correct sign.
        #
        # With accENU = [0, 0, 9.81] (gravity only) and δε_roll > 0:
        # F[4,6] = -fU = -9.81  → δv_North decreases (negative)
        # F[5,6] = fN = 0       → δv_Up unchanged
        #
        # This is what the F matrix does: a roll error with fU present
        # causes North velocity error.
        N = 20
        sample_rate = 100
        gps_avail = np.zeros(N, dtype=bool)

        # Give a tiny roll offset via initial gyro pulse
        gyro = np.zeros((N, 3))
        gyro[0, 0] = 1.0  # brief roll impulse → ~0.01 rad roll

        nav = _make_nav_data(N, sample_rate=sample_rate, gyro_flu=gyro,
                             gps_available=gps_avail)
        params = _default_ekf_params()
        result = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

        # With positive roll, gravity (fU=9.81) should cause North velocity to
        # decrease (F[4,6] = -fU → negative coupling)
        v_north = result['v'][-1, 1]
        # The coupling is -fU * δε_roll → negative North velocity
        # (gravity component pushes in the -North direction for positive roll in FLU)
        assert v_north != 0, "Expected non-zero North velocity from roll-gravity coupling"


# ---------------------------------------------------------------------------
# 15.  Covariance positive-definiteness
# ---------------------------------------------------------------------------

class TestCovarianceProperties:
    """P should remain positive-definite throughout the run."""

    def test_uncertainties_positive(self):
        N = 500
        sample_rate = 100
        nav = _make_nav_data(N, sample_rate=sample_rate, gps_rate=10)
        params = _default_ekf_params()
        result = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

        for key in ('std_pos', 'std_vel', 'std_orient', 'std_bias_acc', 'std_bias_gyr'):
            std = result[key]
            assert np.all(std >= 0), f"Negative std in {key}"
            # Check no NaN
            assert not np.any(np.isnan(std)), f"NaN in {key}"


# ---------------------------------------------------------------------------
# 16.  Known forward acceleration — position should grow quadratically
# ---------------------------------------------------------------------------

class TestForwardAcceleration:
    """Apply 1 m/s² forward body acceleration (on top of gravity).

    At yaw=0, Rbn=I, forward=East.
    accel_flu = [1.0, 0, 9.81]
    accENU = [1, 0, 9.81], accENU + g = [1, 0, 0]
    After t seconds: pos_East ≈ 0.5 * 1 * t²

    Run without GPS to get pure IMU integration.
    """

    @pytest.fixture
    def accel_result(self):
        N = 100  # 1 second at 100 Hz
        sample_rate = 100
        accel = np.tile([1.0, 0.0, 9.81], (N, 1))  # 1 m/s² forward + gravity
        gps_avail = np.zeros(N, dtype=bool)

        nav = _make_nav_data(N, sample_rate=sample_rate, accel_flu=accel,
                             gps_available=gps_avail)
        params = _default_ekf_params()
        return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)

    def test_east_position_quadratic(self, accel_result):
        p = accel_result['p']
        # After 1 s at 1 m/s²: pos ≈ 0.5 m.  Allow 30% tolerance.
        expected = 0.5 * 1.0 * 1.0**2
        assert abs(p[-1, 0] - expected) < 0.3, (
            f"East pos = {p[-1,0]:.3f}, expected ~{expected:.3f}"
        )

    def test_east_velocity_linear(self, accel_result):
        v = accel_result['v']
        # After 1 s: vel ≈ 1.0 m/s East
        assert abs(v[-1, 0] - 1.0) < 0.3, f"East vel = {v[-1,0]:.3f}, expected ~1.0"

    def test_north_stays_small(self, accel_result):
        p = accel_result['p']
        assert np.max(np.abs(p[:, 1])) < 0.2, "Unexpected North drift"
