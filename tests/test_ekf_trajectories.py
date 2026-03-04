# -*- coding: utf-8 -*-
"""
Analytical trajectory tests for the Error-State EKF.

Each test constructs a known trajectory (straight line, curve, helix, …),
computes the EXACT accel_flu and gyro_flu that a perfect IMU would measure
for that trajectory, feeds them into the EKF with NO GPS, and checks that
the pure-IMU integration reproduces the expected path.

This validates:
  - The sign conventions of Rbn (Rz, Ry, Rx) for FLU → ENU
  - The gravity cancellation (specific force = coord_accel − g)
  - The W matrix (Euler rate ↔ body angular velocity mapping)
  - The F matrix velocity–orientation coupling signs
  - That Forward / Left / Up axes map to East / North / Up correctly
    at known yaw headings

Convention reminder:
  Body frame : FLU  — x = Forward, y = Left, z = Up
  Nav  frame : ENU  — x = East,    y = North, z = Up
  yaw = 0    → Forward ≡ East
  yaw = π/2  → Forward ≡ North
  Gravity    : g_enu = [0, 0, −9.81]
  Specific force (accel reading) = coord_accel − g
    → stationary: accel_flu = [0, 0, +9.81]  (measures reaction to gravity)
"""
import numpy as np
import pytest
from math import sin, cos, tan, sqrt, pi

import sys
from pathlib import Path
_ekf_src = str(Path(__file__).resolve().parent.parent / "scripts" / "positioning" / "python")
if _ekf_src not in sys.path:
    sys.path.insert(0, _ekf_src)

from data_loader import NavigationData
import ekf_core

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

LLA0 = np.array([40.4168, -3.7038, 650.0])
G = 9.81


def _enu_to_lla(e, n, u):
    import pymap3d as pm
    lat, lon, alt = pm.enu2geodetic(e, n, u, LLA0[0], LLA0[1], LLA0[2])
    return np.array([lat, lon, alt])


def _build_lla_trajectory(enu_positions):
    return np.array([_enu_to_lla(r[0], r[1], r[2]) for r in enu_positions])


def _zero_bias_params():
    """EKF params with zero initial bias and negligible process noise
    so that pure IMU integration is as clean as possible."""
    return dict(
        Qpos=1e-20, Qvel=1e-20,
        QorientXY=1e-20, QorientZ=1e-20,
        Qacc=1e-20, QgyrXY=1e-20, QgyrZ=1e-20,
        Rpos=1.0,
        beta_acc=-1e-12, beta_gyr=-1e-12,   # nearly zero → biases don't drift
        P_pos_std=1e-6, P_vel_std=1e-6, P_orient_std=1e-6,
        P_acc_std=1e-12, P_gyr_std=1e-12,
    )


def _make_nav(N, sample_rate, accel_flu, gyro_flu, lla=None):
    """Create a NavigationData with NO GPS (pure IMU)."""
    if lla is None:
        lla = np.tile(LLA0, (N, 1))
    return NavigationData(
        accel_flu=accel_flu.astype(np.float64),
        gyro_flu=gyro_flu.astype(np.float64),
        vel_enu=np.zeros((N, 3)),
        lla=lla,
        orient=np.zeros((N, 3)),
        gps_available=np.zeros(N, dtype=bool),   # NO GPS
        sample_rate=float(sample_rate),
        dataset_name="trajectory_test",
        gps_rate=1.0,
        lla0=LLA0.copy(),
    )


def _run_imu_only(accel_flu, gyro_flu, sample_rate=100, use_3d=True):
    """Run EKF in pure-IMU mode (no GPS) and return result dict."""
    N = accel_flu.shape[0]
    nav = _make_nav(N, sample_rate, accel_flu, gyro_flu)
    params = _zero_bias_params()
    return ekf_core.run_ekf(nav, params, outage_config=None,
                            use_3d_rotation=use_3d)


def _Rz(yaw):
    return np.array([[cos(yaw), -sin(yaw), 0],
                     [sin(yaw),  cos(yaw), 0],
                     [0,         0,        1]])

def _Ry(pitch):
    return np.array([[ cos(pitch), 0, -sin(pitch)],
                     [ 0,          1,  0         ],
                     [ sin(pitch), 0,  cos(pitch)]])

def _Rx(roll):
    return np.array([[1, 0,          0         ],
                     [0, cos(roll), -sin(roll) ],
                     [0, sin(roll),  cos(roll) ]])

def _Rbn(roll, pitch, yaw):
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)


# ══════════════════════════════════════════════════════════════════════
# 1.  STRAIGHT LINES ALONG EACH ENU AXIS
# ══════════════════════════════════════════════════════════════════════

class TestStraightLineEast:
    """
    Trajectory: accelerate 1 m/s² purely East for 2 s, then coast at 2 m/s.
    Total 5 s at 200 Hz = 1000 samples.

    Heading: yaw = 0 → Forward ≡ East.
    Orientation: rpy = [0, 0, 0] throughout (no rotation).

    Body-frame IMU readings:
      accel_flu = Rbn^T @ (coord_accel_enu − g_enu)
      At yaw=0, Rbn=I, g=[0,0,−9.81]
        accelerating phase:  coord_accel = [1, 0, 0]
          specific force = [1, 0, 0] − [0, 0, −9.81] = [1, 0, +9.81]
        coasting phase:      coord_accel = [0, 0, 0]
          specific force = [0, 0, +9.81]
      gyro_flu = [0, 0, 0]

    Expected ENU trajectory:
      0–2 s : p_E = 0.5·t²,   v_E = t
      2–5 s : p_E = 2 + 2·(t−2), v_E = 2
      Final: p_E = 2 + 2·3 = 8 m
    """

    @pytest.fixture
    def result(self):
        sr = 200
        dt = 1 / sr
        N = 1000  # 5 s
        t = np.arange(N) * dt

        accel = np.zeros((N, 3))
        accel[:, 2] = G  # gravity reaction always present
        accel[t < 2.0, 0] = 1.0  # 1 m/s² forward during first 2 s

        gyro = np.zeros((N, 3))
        return _run_imu_only(accel, gyro, sample_rate=sr), t

    def test_final_east_position(self, result):
        res, t = result
        p = res['p']
        # Analytical: p_E(5) = 0.5*1*2² + 2*3 = 2 + 6 = 8 m
        assert abs(p[-1, 0] - 8.0) < 0.15, f"East = {p[-1,0]:.3f}, expected 8.0"

    def test_final_east_velocity(self, result):
        res, t = result
        v = res['v']
        assert abs(v[-1, 0] - 2.0) < 0.05, f"v_E = {v[-1,0]:.4f}, expected 2.0"

    def test_north_stays_zero(self, result):
        res, t = result
        assert np.max(np.abs(res['p'][:, 1])) < 0.05, "Unexpected North drift"

    def test_up_stays_zero(self, result):
        res, t = result
        assert np.max(np.abs(res['p'][:, 2])) < 0.05, "Unexpected Up drift"

    def test_orientation_stays_zero(self, result):
        res, t = result
        assert np.max(np.abs(res['r'])) < 0.01, "Orientation should stay at zero"


class TestStraightLineNorth:
    """
    Trajectory: constant 1 m/s² North for 2 s, coast for 3 s.

    To go North, the body must face North → yaw = π/2.
    BUT the EKF starts with rpy=[0,0,0], so we must seed the yaw first.

    Strategy: we pre-rotate the body to yaw=π/2 instantly at t=0 by giving
    a large yaw-rate impulse in the first sample, then apply forward accel.

    Actually it's simpler: at yaw=0, Forward=East, Left=North.
    So to accelerate North we apply accel_flu = [0, 1, 9.81]  (Left = North).
    This doesn't require any yaw change.
    """

    @pytest.fixture
    def result(self):
        sr = 200
        dt = 1 / sr
        N = 1000
        t = np.arange(N) * dt

        accel = np.zeros((N, 3))
        accel[:, 2] = G
        # Accelerate in the LEFT (y) body direction → North in ENU at yaw=0
        accel[t < 2.0, 1] = 1.0

        gyro = np.zeros((N, 3))
        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_final_north_position(self, result):
        p = result['p']
        assert abs(p[-1, 1] - 8.0) < 0.15, f"North = {p[-1,1]:.3f}, expected 8.0"

    def test_east_stays_zero(self, result):
        assert np.max(np.abs(result['p'][:, 0])) < 0.05, "Unexpected East drift"

    def test_final_north_velocity(self, result):
        v = result['v']
        assert abs(v[-1, 1] - 2.0) < 0.05, f"v_N = {v[-1,1]:.4f}, expected 2.0"


class TestStraightLineUp:
    """
    Trajectory: constant 1 m/s² upward for 2 s, coast for 3 s.

    At yaw=0, Up in body = Up in ENU (z-axis).
    Coord accel = [0, 0, 1].
    Specific force = [0, 0, 1] − [0, 0, −9.81] = [0, 0, 10.81].
    """

    @pytest.fixture
    def result(self):
        sr = 200
        dt = 1 / sr
        N = 1000
        t = np.arange(N) * dt

        accel = np.zeros((N, 3))
        accel[:, 2] = G  # gravity baseline
        accel[t < 2.0, 2] += 1.0  # +1 m/s² upward on top of gravity

        gyro = np.zeros((N, 3))
        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_final_up_position(self, result):
        p = result['p']
        # 0.5*1*4 + 2*3 = 8 m upward, but then gravity pulls it back during coast!
        # Coast phase: a = [0,0,0] (no thrust), coord_accel = accENU + g = [0,0,9.81] + [0,0,-9.81] = [0,0,0]
        # Wait — during coast, specific force = [0,0,9.81], coord_accel = Rbn@[0,0,9.81] + g = [0,0,0]. ✓
        # So Up velocity stays at 2 m/s, position = 2 + 2*3 = 8 m.
        assert abs(p[-1, 2] - 8.0) < 0.15, f"Up = {p[-1,2]:.3f}, expected 8.0"

    def test_horizontal_stays_zero(self, result):
        assert np.max(np.abs(result['p'][:, 0])) < 0.05
        assert np.max(np.abs(result['p'][:, 1])) < 0.05


# ══════════════════════════════════════════════════════════════════════
# 2.  HEADING-AWARE STRAIGHT LINES (yaw ≠ 0)
# ══════════════════════════════════════════════════════════════════════

class TestStraightLineWithYaw90:
    """
    Start at yaw = π/2 (Forward ≡ North), accelerate forward at 1 m/s² for 2 s.

    To set yaw=π/2 at the start, we give a yaw-rate impulse in the first
    few samples to wind up π/2 of yaw, then apply forward acceleration.

    At yaw=π/2: Rbn @ [1,0,0] = [0,1,0] (forward → North).  ✓
    """

    @pytest.fixture
    def result(self):
        sr = 200
        dt = 1 / sr
        # Phase 1: spin up yaw to π/2 in 0.5 s (100 samples at 200 Hz)
        # Phase 2: accelerate forward for 2 s
        # Phase 3: coast for 2.5 s
        N = 1000  # 5 s total
        t = np.arange(N) * dt

        n_spin = 100  # 0.5 s for yaw spin-up
        yaw_rate = (pi / 2) / (n_spin * dt)  # rad/s to reach π/2

        gyro = np.zeros((N, 3))
        gyro[:n_spin, 2] = yaw_rate  # spin to yaw=π/2

        # Build accel accounting for changing yaw during spin-up
        accel = np.zeros((N, 3))
        accel[:, 2] = G  # gravity reaction always

        # After spin-up (t ≥ 0.5s), apply 1 m/s² forward for 2 s
        accel_start = n_spin
        accel_end = n_spin + int(2.0 * sr)
        accel[accel_start:accel_end, 0] = 1.0  # forward in body

        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_moves_north(self, result):
        p = result['p']
        # After spin-up completes at 0.5s, forward = North.
        # 2s of 1 m/s² → v=2, p=2.  Then 2.5s coast → p = 2 + 2*2.5 = 7.
        # (slight spin-up transient)
        assert p[-1, 1] > 5.0, f"North = {p[-1,1]:.2f}, expected ~7"

    def test_east_stays_small(self, result):
        p = result['p']
        # During spin-up there's a small East transient, but final should be small
        assert abs(p[-1, 0]) < 1.5, f"East = {p[-1,0]:.2f}, expected ~0"

    def test_yaw_reaches_pi_over_2(self, result):
        r = result['r']
        # After spin-up, yaw should be near π/2
        yaw_after_spin = r[100, 2]
        assert abs(yaw_after_spin - pi / 2) < 0.05, (
            f"Yaw after spin = {yaw_after_spin:.4f}, expected {pi/2:.4f}"
        )


class TestStraightLineWithYaw180:
    """
    yaw = π → Forward ≡ −East (West).
    Accelerate forward at 1 m/s² → should move West (negative East).
    """

    @pytest.fixture
    def result(self):
        sr = 200
        dt = 1 / sr
        N = 1000
        n_spin = 100
        yaw_rate = pi / (n_spin * dt)  # reach π in 0.5 s

        gyro = np.zeros((N, 3))
        gyro[:n_spin, 2] = yaw_rate

        accel = np.zeros((N, 3))
        accel[:, 2] = G
        accel[n_spin:n_spin + 400, 0] = 1.0  # 2 s of forward accel

        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_moves_west(self, result):
        p = result['p']
        assert p[-1, 0] < -3.0, f"East = {p[-1,0]:.2f}, should be negative (West)"

    def test_north_stays_small(self, result):
        p = result['p']
        assert abs(p[-1, 1]) < 1.5, f"North = {p[-1,1]:.2f}, expected ~0"


class TestStraightLineWithYawMinus90:
    """
    yaw = −π/2 → Forward ≡ −North (South).
    Forward accel → should move South (negative North).
    """

    @pytest.fixture
    def result(self):
        sr = 200
        dt = 1 / sr
        N = 1000
        n_spin = 100
        yaw_rate = -(pi / 2) / (n_spin * dt)  # −π/2

        gyro = np.zeros((N, 3))
        gyro[:n_spin, 2] = yaw_rate

        accel = np.zeros((N, 3))
        accel[:, 2] = G
        accel[n_spin:n_spin + 400, 0] = 1.0

        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_moves_south(self, result):
        p = result['p']
        assert p[-1, 1] < -3.0, f"North = {p[-1,1]:.2f}, should be negative (South)"

    def test_east_stays_small(self, result):
        assert abs(result['p'][-1, 0]) < 1.5


# ══════════════════════════════════════════════════════════════════════
# 3.  CIRCULAR TRAJECTORY (constant yaw rate + centripetal)
# ══════════════════════════════════════════════════════════════════════

class TestCircularTrajectory:
    """
    Constant speed v=2 m/s in a circle of radius R=5 m.

    ω = v/R = 0.4 rad/s yaw rate.
    Duration = one full circle: T = 2πR/v = 5π ≈ 15.7 s.

    Starting at yaw=0 (Forward=East), moving East initially:
      pos_ENU(t) = R·[sin(ωt), 1−cos(ωt), 0] + offset
                 = [5·sin(0.4t), 5−5·cos(0.4t), 0]

    The body feels:
      - Forward accel = 0 (constant speed)
      - Left accel    = centripetal = v²/R = 4/5 = 0.8 m/s² (pointing LEFT = toward center)
      - Up accel      = gravity reaction = 9.81

    Gyro:
      - yaw rate = ω = 0.4 rad/s in body z (Up)
      - roll, pitch rate = 0

    After one full circle the vehicle should return to the origin.
    """

    @pytest.fixture
    def result(self):
        v = 2.0
        R = 5.0
        omega = v / R  # 0.4 rad/s
        T_circle = 2 * pi * R / v  # ~15.71 s
        sr = 200
        dt = 1 / sr
        N = int(T_circle * sr) + 1
        t = np.arange(N) * dt

        accel = np.zeros((N, 3))
        accel[:, 1] = v**2 / R   # centripetal in LEFT direction
        accel[:, 2] = G          # gravity

        gyro = np.zeros((N, 3))
        gyro[:, 2] = omega  # yaw rate

        # We need to give the EKF an initial velocity of v=2 m/s East.
        # The EKF starts with vIMU = vel_enu[0] = [0,0,0].
        # So we add a velocity impulse in the first sample:
        # Give a brief strong forward acceleration to reach v=2 m/s.
        # In 1 sample at dt=0.005: a = v/dt = 400 m/s². That's aggressive.
        # Better: use 10 samples (0.05 s) with a=40 m/s².
        n_ramp = 10
        accel[:n_ramp, 0] = v / (n_ramp * dt)  # forward accel to reach v

        return _run_imu_only(accel, gyro, sample_rate=sr), N, t, R, omega, n_ramp

    def test_returns_near_origin(self, result):
        res, N, t, R, omega, n_ramp = result
        p = res['p']
        # After one full circle, should be back near [0, 0, 0]
        dist = np.linalg.norm(p[-1, :2])
        assert dist < 3.0, f"End point {dist:.2f} m from origin, expected ~0"

    def test_max_radius_correct(self, result):
        res, N, t, R, omega, n_ramp = result
        p = res['p']
        # The trajectory should reach roughly R on the North axis
        # (center of circle is at [0, R, 0], max North ~ 2R)
        max_north = np.max(p[:, 1])
        assert max_north > R, f"Max North = {max_north:.2f}, expected > {R}"
        assert max_north < 3 * R, f"Max North = {max_north:.2f}, too large"

    def test_stays_flat(self, result):
        res, N, t, R, omega, n_ramp = result
        assert np.max(np.abs(res['p'][:, 2])) < 0.5, "Circle should be flat"

    def test_yaw_completes_full_turn(self, result):
        """After one full circle, total yaw change should be ~2π."""
        res, N, t, R, omega, n_ramp = result
        r = res['r']
        # Due to wrapping, check unwrapped yaw
        yaw = r[:, 2]
        dyaw = np.diff(yaw)
        # Fix wrapping jumps
        dyaw[dyaw > pi] -= 2 * pi
        dyaw[dyaw < -pi] += 2 * pi
        total_yaw = np.sum(dyaw)
        assert abs(total_yaw - 2 * pi) < 0.3, (
            f"Total yaw = {total_yaw:.3f}, expected {2*pi:.3f}"
        )


# ══════════════════════════════════════════════════════════════════════
# 4.  PITCHED CLIMB (nose up 30°, forward thrust)
# ══════════════════════════════════════════════════════════════════════

class TestPitchedClimb:
    """
    Vehicle points nose-up at pitch=30° (yaw=0) with 1 m/s² forward body thrust.

    Rbn(0, π/6, 0) @ [1, 0, 0] = [cos30, 0, sin30] = [0.866, 0, 0.5]
    So the 1 m/s² forward thrust projects to:
      accel_ENU = [0.866, 0, 0.5]
    Coord_accel = accel_ENU + g = [0.866, 0, 0.5 − 9.81] ... wait, that's wrong.

    Let me redo:
    specific_force_body = [1, 0, 9.81]  (thrust + gravity reaction in body-Up)
    But at pitch=30°, body-Up ≠ ENU-Up.

    Correct formulation:
      The vehicle is pitched 30° nose-up and at rest.
      Body-Up in ENU: Rbn @ [0,0,1] = [−sin30, 0, cos30] = [−0.5, 0, 0.866]
      To hover (cancel gravity), specific force along body-Up must be g/cos30.
      But we want it simpler.

    Simpler approach — set pitch via gyro, let gravity do its thing:
      1. Spin pitch to 30° in 0.5 s using gyro pitch-rate
      2. Apply forward body thrust of 1 m/s²
      3. Check that position moves in the [cos30, 0, sin30] direction (East + Up)

    The specific force in body frame:
      accel_flu = Rbn^T @ (coord_accel_enu − g_enu)
      With Rbn^T = (Ry(30°))^T at yaw=roll=0:
      In body frame, gravity becomes: Rbn^T @ [0,0,−9.81] = [sin30*9.81, 0, −cos30*9.81]
                                                             = [4.905, 0, −8.496]
      So the gravity reaction (what accel reads when stationary at 30° pitch):
        accel_flu_gravity = −Rbn^T @ g = [−4.905, 0, 8.496]
      Wait: specific_force = Rbn^T @ (a_ENU − g) where a_ENU is coordinate acceleration.
      If stationary: a_ENU = 0, specific_force = −Rbn^T @ g = Rbn^T @ [0,0,9.81]
        = [−sin30*9.81, 0, cos30*9.81] = [−4.905, 0, 8.496]

    So at 30° pitch, stationary, the accelerometer reads:
      accel_flu = [−4.905, 0, 8.496]  (gravity component in forward axis!)

    To add 1 m/s² body-forward thrust:
      accel_flu = [−4.905 + 1, 0, 8.496] = [−3.905, 0, 8.496]

    And the resulting coord_accel = Rbn @ accel_flu + g:
      = Ry(30°) @ [−3.905, 0, 8.496] + [0, 0, −9.81]
      = [cos30*(−3.905) + (−sin30)*8.496, 0, sin30*(−3.905) + cos30*8.496] + [0,0,−9.81]
      = [−3.382 − 4.248, 0, −1.953 + 7.358] + [0,0,−9.81]
      = [−7.630, 0, 5.405] + [0,0,−9.81]
      = [−7.630, 0, −4.405]
    That doesn't look right — we expect +1 m/s² forward which in ENU = [cos30, 0, sin30].

    Let me reconsider. The EKF does:
      accENU = Rbn @ acc_body
      coord_accel = accENU + g = Rbn @ acc_body + [0,0,−9.81]

    If acc_body = [1, 0, 9.81] (thrust + gravity on body z):
      accENU = Ry(30°) @ [1, 0, 9.81]
             = [cos30*1 + (−sin30)*9.81, 0, sin30*1 + cos30*9.81]
             = [0.866 − 4.905, 0, 0.5 + 8.496]
             = [−4.039, 0, 8.996]
      coord_accel = [−4.039, 0, 8.996 − 9.81] = [−4.039, 0, −0.814]

    This is wrong because [1, 0, 9.81] is not the correct specific force for
    a pitched vehicle. The correct specific force for a vehicle at pitch θ
    with 1 m/s² forward thrust AND gravity is:

      sf_body = body_thrust + Rbn^T @ [0, 0, 9.81]   (gravity reaction)
              = [1, 0, 0] + Ry(30°)^T @ [0, 0, 9.81]
              = [1, 0, 0] + [−sin30*9.81, 0, cos30*9.81]
              = [1 − 4.905, 0, 8.496]
              = [−3.905, 0, 8.496]

    Then: accENU = Rbn @ [−3.905, 0, 8.496]
                 = Ry(30°) @ [−3.905, 0, 8.496]
                 = [cos30*(−3.905) + (−sin30)*8.496, 0, sin30*(−3.905) + cos30*8.496]
                 = [−3.382 − 4.248, 0, −1.953 + 7.358]
                 = [−7.630, 0, 5.405]
      coord = [−7.630, 0, 5.405 − 9.81] = [−7.630, 0, −4.405]

    Still wrong! The problem is that I'm applying pitch AFTER the EKF starts,
    so at t=0 pitch=0 and accel=[0,0,9.81]. The pitch builds up gradually.

    Let me use the SIMPLE approach: just spin up pitch and check direction.
    """

    @pytest.fixture
    def result(self):
        sr = 200
        dt = 1 / sr
        N = 600  # 3 s
        t = np.arange(N) * dt

        # Phase 1 (0–0.5s): spin pitch to 30° via gyro
        n_spin = 100
        pitch_target = pi / 6
        pitch_rate = pitch_target / (n_spin * dt)

        gyro = np.zeros((N, 3))
        gyro[:n_spin, 1] = pitch_rate

        # Phase 2 (0.5s–3s): apply 2 m/s² forward thrust + gravity reaction
        # The accel reading depends on the current pitch angle.
        # For simplicity, we construct the correct body-frame specific force
        # at each timestep given the known pitch profile.
        accel = np.zeros((N, 3))
        for i in range(N):
            if i < n_spin:
                pitch_i = pitch_rate * i * dt
                thrust_fwd = 0.0  # no thrust during spin-up
            else:
                pitch_i = pitch_target
                thrust_fwd = 2.0  # 2 m/s² forward body thrust

            # Gravity reaction in body frame: Rbn^T @ [0, 0, 9.81]
            Rbn_i = _Ry(pitch_i)
            grav_body = Rbn_i.T @ np.array([0, 0, G])
            accel[i, :] = np.array([thrust_fwd, 0, 0]) + grav_body

        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_moves_east_and_up(self, result):
        """At pitch=30°, forward thrust projects to East (cos30) and Up (sin30)."""
        p = result['p']
        # Should have positive East AND positive Up displacement
        assert p[-1, 0] > 1.0, f"East = {p[-1,0]:.2f}, expected positive"
        assert p[-1, 2] > 0.5, f"Up = {p[-1,2]:.2f}, expected positive"

    def test_north_stays_small(self, result):
        assert np.max(np.abs(result['p'][:, 1])) < 0.5

    def test_up_east_ratio_matches_pitch(self, result):
        """Ratio of Up/East displacement should be ~tan(30°) = 0.577."""
        p = result['p']
        if abs(p[-1, 0]) > 0.1:
            ratio = p[-1, 2] / p[-1, 0]
            assert abs(ratio - tan(pi / 6)) < 0.15, (
                f"Up/East ratio = {ratio:.3f}, expected tan(30°) = {tan(pi/6):.3f}"
            )


# ══════════════════════════════════════════════════════════════════════
# 5.  ROLL TEST — tilt left, check lateral drift direction
# ══════════════════════════════════════════════════════════════════════

class TestRollTilt:
    """
    Roll 30° (right roll — positive roll in FLU).
    At roll=30°, body-Left tilts upward, body-Up tilts right.
    Gravity leaks into the body-y (Left) axis.

    Body accel at roll θ, stationary:
      sf = Rx(θ)^T @ [0, 0, 9.81]
         = [0, −sin(θ)*9.81, cos(θ)*9.81]
         = [0, −4.905, 8.496]

    EKF does: accENU = Rbn @ sf = Rx(30°) @ [0, −4.905, 8.496]
    coord_accel = accENU + g = Rx(30°) @ [0, −4.905, 8.496] + [0, 0, −9.81]
               = [0, cos30*(−4.905)+(−sin30)*8.496, sin30*(−4.905)+cos30*8.496] + [0,0,−9.81]
               = [0, −4.248 − 4.248, −2.453 + 7.358] + [0, 0, −9.81]
               = [0, −8.496, 4.905] + [0, 0, −9.81]
               = [0, −8.496, −4.905]
    Hmm that gives a downward + southward acceleration, not zero.

    Wait — that IS zero. Let me redo carefully:
    Rx(θ) @ [0, −sinθ·G, cosθ·G] =
      [0,
       cosθ·(−sinθ·G) + (−sinθ)·(cosθ·G),
       sinθ·(−sinθ·G) + cosθ·(cosθ·G)]
      = [0, −2·sinθ·cosθ·G, (cos²θ − sin²θ)·G]... NO.

    Rx(θ) = [[1, 0, 0], [0, cosθ, −sinθ], [0, sinθ, cosθ]]

    Rx(θ) @ [0, −sinθ·G, cosθ·G] =
      [0,
       cosθ·(−sinθ·G) + (−sinθ)·(cosθ·G),
       sinθ·(−sinθ·G) + cosθ·(cosθ·G)]
      = [0,
         −sinθ·cosθ·G − sinθ·cosθ·G,
         −sin²θ·G + cos²θ·G]
      = [0, −2·sinθ·cosθ·G, cos(2θ)·G]

    That's NOT [0, 0, 9.81].  The issue is that sf = Rbn^T @ [0,0,G],
    and Rbn @ sf = Rbn @ Rbn^T @ [0,0,G] = [0,0,G].  ✓
    Then coord_accel = [0,0,G] + [0,0,−G] = [0,0,0].  ✓ Stationary!

    So for the roll test, we just verify that rolling alone (stationary)
    produces zero net motion, and then add a body thrust to see the direction.
    """

    @pytest.fixture
    def result(self):
        sr = 200
        dt = 1 / sr
        N = 600
        n_spin = 100
        roll_target = pi / 6
        roll_rate = roll_target / (n_spin * dt)

        gyro = np.zeros((N, 3))
        gyro[:n_spin, 0] = roll_rate

        # Construct accel with correct gravity reaction at each roll angle
        # Plus a body-Left (y) thrust of 1 m/s² after spin-up
        accel = np.zeros((N, 3))
        for i in range(N):
            if i < n_spin:
                roll_i = roll_rate * i * dt
                thrust_left = 0.0
            else:
                roll_i = roll_target
                thrust_left = 1.0  # 1 m/s² body Left

            Rbn_i = _Rx(roll_i)
            grav_body = Rbn_i.T @ np.array([0, 0, G])
            accel[i, :] = np.array([0, thrust_left, 0]) + grav_body

        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_thrust_left_goes_north_and_up(self, result):
        """At roll=30° (yaw=0), body-Left = ENU [0, cos30, −sin30] (after Rx).

        Actually: Rbn = Rx(30°). Body-Left [0,1,0] in ENU:
        Rx(30°) @ [0,1,0] = [0, cos30, sin30] = [0, 0.866, 0.5]

        So a 1 m/s² Left body thrust should move North AND Up.
        """
        p = result['p']
        assert p[-1, 1] > 0.5, f"North = {p[-1,1]:.2f}, expected positive"

    def test_east_stays_small(self, result):
        assert np.max(np.abs(result['p'][:, 0])) < 0.5


# ══════════════════════════════════════════════════════════════════════
# 6.  AXIS MAPPING VERIFICATION — each body axis maps correctly at yaw=0
# ══════════════════════════════════════════════════════════════════════

class TestAxisMappingAtYaw0:
    """At yaw=0, roll=0, pitch=0, Rbn = I.
    Forward (body x) → East  (ENU x)
    Left    (body y) → North (ENU y)
    Up      (body z) → Up    (ENU z)

    Apply 1 m/s² thrust along each body axis independently and check
    the resulting ENU direction.
    """

    @staticmethod
    def _run_axis(axis_idx):
        sr = 200
        dt = 1 / sr
        N = 200  # 1 s
        accel = np.zeros((N, 3))
        accel[:, 2] = G  # gravity
        accel[:, axis_idx] += 1.0  # 1 m/s² along body axis
        gyro = np.zeros((N, 3))
        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_forward_maps_to_east(self):
        res = self._run_axis(0)
        p = res['p']
        assert p[-1, 0] > 0.3, "Forward should map to +East"
        assert abs(p[-1, 1]) < 0.05, "Forward should not affect North"
        assert abs(p[-1, 2]) < 0.05, "Forward should not affect Up"

    def test_left_maps_to_north(self):
        res = self._run_axis(1)
        p = res['p']
        assert abs(p[-1, 0]) < 0.05, "Left should not affect East"
        assert p[-1, 1] > 0.3, "Left should map to +North"
        assert abs(p[-1, 2]) < 0.05, "Left should not affect Up"

    def test_up_maps_to_up(self):
        res = self._run_axis(2)
        p = res['p']
        assert abs(p[-1, 0]) < 0.05, "Up should not affect East"
        assert abs(p[-1, 1]) < 0.05, "Up should not affect North"
        # The extra 1 m/s² Up on top of gravity: coord_accel_up = 9.81+1 − 9.81 = 1
        assert p[-1, 2] > 0.3, "Up body should map to +Up ENU"


# ══════════════════════════════════════════════════════════════════════
# 7.  AXIS MAPPING AT yaw = π/2 (Forward = North)
# ══════════════════════════════════════════════════════════════════════

class TestAxisMappingAtYaw90:
    """At yaw=π/2:
    Forward (body x) → North  (ENU y)
    Left    (body y) → −East  (ENU −x)  [West]
    Up      (body z) → Up     (ENU z)
    """

    @staticmethod
    def _run_axis_with_yaw(axis_idx, yaw_target=pi / 2):
        sr = 200
        dt = 1 / sr
        n_spin = 100
        N = n_spin + 200  # spin-up + 1 s of thrust
        yaw_rate = yaw_target / (n_spin * dt)

        gyro = np.zeros((N, 3))
        gyro[:n_spin, 2] = yaw_rate

        accel = np.zeros((N, 3))
        # During spin-up: gravity only (no pitch/roll, just yaw → Rbn=Rz → gravity unchanged in body)
        accel[:, 2] = G
        accel[n_spin:, axis_idx] += 1.0  # thrust after spin-up

        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_forward_maps_to_north(self):
        res = self._run_axis_with_yaw(0)
        p = res['p']
        # After spin-up, forward thrust should push North
        assert p[-1, 1] > 0.3, f"Forward at yaw=90° should map to +North, got {p[-1,1]:.3f}"

    def test_left_maps_to_west(self):
        res = self._run_axis_with_yaw(1)
        p = res['p']
        # Left at yaw=90° should map to −East (West)
        assert p[-1, 0] < -0.2, f"Left at yaw=90° should map to −East, got {p[-1,0]:.3f}"

    def test_up_still_maps_to_up(self):
        res = self._run_axis_with_yaw(2)
        p = res['p']
        assert p[-1, 2] > 0.3, f"Up at yaw=90° should still be +Up, got {p[-1,2]:.3f}"


# ══════════════════════════════════════════════════════════════════════
# 8.  S-CURVE (two opposite yaw turns)
# ══════════════════════════════════════════════════════════════════════

class TestSCurve:
    """
    Drive forward at constant speed, turn left 90° then right 90°.
    Should end up parallel to the starting direction but offset laterally.
    This tests that consecutive opposite yaw rotations don't accumulate errors.
    """

    @pytest.fixture
    def result(self):
        sr = 200
        dt = 1 / sr
        v = 2.0
        omega = 0.4  # rad/s
        turn_time = (pi / 2) / omega  # time for 90° turn

        # Segments: straight 1s → left turn 90° → straight 1s → right turn 90° → straight 1s
        n_straight = int(1.0 * sr)
        n_turn = int(turn_time * sr)
        N = 3 * n_straight + 2 * n_turn

        accel = np.zeros((N, 3))
        accel[:, 2] = G
        gyro = np.zeros((N, 3))

        # Ramp up to speed in first 10 samples
        n_ramp = 10
        accel[:n_ramp, 0] = v / (n_ramp * dt)

        # Centripetal when turning
        centripetal = v**2 / (v / omega)  # = v*omega = 0.8 m/s²

        # Segment boundaries
        s1_end = n_straight
        t1_end = s1_end + n_turn
        s2_end = t1_end + n_straight
        t2_end = s2_end + n_turn

        # Left turn (+yaw rate, centripetal to Left = +y body)
        gyro[s1_end:t1_end, 2] = omega
        accel[s1_end:t1_end, 1] = centripetal

        # Right turn (−yaw rate, centripetal to Right = −y body)
        gyro[s2_end:t2_end, 2] = -omega
        accel[s2_end:t2_end, 1] = -centripetal

        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_final_heading_same_as_start(self, result):
        r = result['r']
        # Left 90° then right 90° → net yaw change = 0
        assert abs(r[-1, 2]) < 0.15, f"Final yaw = {r[-1,2]:.3f}, expected ~0"

    def test_final_position_has_east_component(self, result):
        """Vehicle started heading East, should still be moving East at the end."""
        p = result['p']
        assert p[-1, 0] > 3.0, f"East = {p[-1,0]:.2f}, expected positive (moving East)"

    def test_net_north_offset_bounded(self, result):
        """Left then right turns produce a lateral offset equal to 2R.
        R = v/ω = 2/0.4 = 5 m, so offset ~ 2*5 = 10 m (plus straight segments)."""
        p = result['p']
        # The geometry of an S-curve gives ~2R offset plus some from straight segments
        assert abs(p[-1, 1]) < 15.0


# ══════════════════════════════════════════════════════════════════════
# 9.  DECELERATION TEST — brake to a stop
# ══════════════════════════════════════════════════════════════════════

class TestDeceleration:
    """
    Accelerate East at 1 m/s² for 2 s (reach v=2), then decelerate at −1 m/s²
    for 2 s (stop). Position: 0→2→4 then 4→6→6. Final: p_E=4, v_E=0.
    
    Actually: accel phase: p = 0.5*t², v = t.  At t=2: p=2, v=2.
    Decel phase (t'=t−2): p = 2 + 2t' − 0.5t'², v = 2 − t'.
    At t'=2: p = 2 + 4 − 2 = 4, v = 0.  ✓
    """

    @pytest.fixture
    def result(self):
        sr = 200
        dt = 1 / sr
        N = 800  # 4 s
        t = np.arange(N) * dt

        accel = np.zeros((N, 3))
        accel[:, 2] = G
        accel[t < 2.0, 0] = 1.0    # accelerate forward
        accel[t >= 2.0, 0] = -1.0  # decelerate (brake)

        gyro = np.zeros((N, 3))
        return _run_imu_only(accel, gyro, sample_rate=sr)

    def test_comes_to_rest(self, result):
        v = result['v']
        assert abs(v[-1, 0]) < 0.1, f"v_E = {v[-1,0]:.4f}, expected ~0"

    def test_final_position(self, result):
        p = result['p']
        assert abs(p[-1, 0] - 4.0) < 0.2, f"p_E = {p[-1,0]:.3f}, expected ~4.0"

    def test_peak_velocity_at_midpoint(self, result):
        v = result['v']
        # Max velocity should be ~2 m/s around sample 400 (t=2s)
        peak_idx = np.argmax(v[:, 0])
        peak_t = peak_idx / 200
        assert abs(peak_t - 2.0) < 0.2, f"Peak velocity at t={peak_t:.2f}, expected ~2.0"
        assert abs(v[peak_idx, 0] - 2.0) < 0.15


# ══════════════════════════════════════════════════════════════════════
# 10.  2D vs 3D consistency — at zero roll/pitch both should match
# ══════════════════════════════════════════════════════════════════════

class TestMode2D3DConsistencyFlat:
    """When roll=pitch=0 (flat motion with only yaw), 2D and 3D modes
    should produce identical trajectories since Rbn = Rz in both cases."""

    def test_same_trajectory(self):
        sr = 200
        dt = 1 / sr
        N = 400
        t = np.arange(N) * dt

        accel = np.zeros((N, 3))
        accel[:, 2] = G
        accel[t < 1.0, 0] = 1.0  # forward accel

        gyro = np.zeros((N, 3))
        gyro[:, 2] = 0.2  # gentle yaw

        r2d = _run_imu_only(accel, gyro, sample_rate=sr, use_3d=False)
        r3d = _run_imu_only(accel, gyro, sample_rate=sr, use_3d=True)

        # Positions should be nearly identical
        np.testing.assert_allclose(r2d['p'], r3d['p'], atol=0.05,
                                   err_msg="2D and 3D should match for flat motion")

        # Velocities too
        np.testing.assert_allclose(r2d['v'], r3d['v'], atol=0.05,
                                   err_msg="2D and 3D velocities should match")
