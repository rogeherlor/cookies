# -*- coding: utf-8 -*-
"""
Stress-test suite for the Error-State EKF.

Goes far beyond the basic unit tests: every scenario has a closed-form
analytical solution that is compared against the EKF output.  Covers
three regimes:

  A. PURE INS (no GPS) -- verifies the mechanisation equations alone.
     If these fail, no amount of tuning can fix the filter.

  B. GPS-AIDED -- verifies the Kalman update loop.  Complex trajectories
     with GPS corrections: the filter must track position, velocity AND
     orientation.

  C. GPS OUTAGE -- verifies the filter degrades gracefully.  After
     removing GPS, the INS drifts, then GPS returns and the filter
     re-converges.

Usage:
    python stress_test_ekf.py                   # run all
    python stress_test_ekf.py figure8            # run one
    python stress_test_ekf.py --list             # list scenarios
    python stress_test_ekf.py --group ins        # run a group
    python stress_test_ekf.py --group gps_aided
    python stress_test_ekf.py --group outage
"""
import sys
import numpy as np
from math import sin, cos, tan, sqrt, pi, atan2
from pathlib import Path

# Add EKF source to path
_ekf_src = str(Path(__file__).resolve().parent.parent / "scripts" / "positioning" / "python")
if _ekf_src not in sys.path:
    sys.path.insert(0, _ekf_src)

from data_loader import NavigationData
import ekf_core
import pymap3d as pm

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)


# =====================================================================
#  Constants & Helpers
# =====================================================================

LLA0 = np.array([40.4168, -3.7038, 650.0])
G = 9.81


def _enu_to_lla(e, n, u):
    lat, lon, alt = pm.enu2geodetic(e, n, u, LLA0[0], LLA0[1], LLA0[2])
    return np.array([lat, lon, alt])


def _build_lla_trajectory(enu):
    return np.array([_enu_to_lla(r[0], r[1], r[2]) for r in enu])


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


# ---------------------------------------------------------------------------
# EKF parameter sets
# ---------------------------------------------------------------------------

def _zero_noise_params():
    """Near-zero noise: pure mechanisation test. Any deviation = equation bug."""
    return dict(
        Qpos=1e-20, Qvel=1e-20,
        QorientXY=1e-20, QorientZ=1e-20,
        Qacc=1e-20, QgyrXY=1e-20, QgyrZ=1e-20,
        Rpos=1.0,
        beta_acc=-1e-12, beta_gyr=-1e-12,
        P_pos_std=1e-6, P_vel_std=1e-6, P_orient_std=1e-6,
        P_acc_std=1e-12, P_gyr_std=1e-12,
        enable_nhc=False, enable_zupt=False, enable_level=False,
    )


def _gps_aided_params(**overrides):
    """Reasonable params for GPS-aided tests."""
    p = dict(
        Qpos=1e-3, Qvel=1e-2,
        QorientXY=1e-4, QorientZ=1e-3,
        Qacc=1e-6, QgyrXY=1e-6, QgyrZ=1e-6,
        Rpos=1.0,
        beta_acc=-0.001, beta_gyr=-0.001,
        P_pos_std=1.0, P_vel_std=1.0, P_orient_std=0.1,
        P_acc_std=0.01, P_gyr_std=0.01,
        enable_nhc=False, enable_zupt=False, enable_level=False,
    )
    p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# NavData builder
# ---------------------------------------------------------------------------

def _make_nav(N, sr, accel, gyro, lla=None, vel_enu=None, orient=None,
              gps_available=None, gps_rate=1.0):
    if lla is None:
        lla = np.tile(LLA0, (N, 1))
    if vel_enu is None:
        vel_enu = np.zeros((N, 3))
    if orient is None:
        orient = np.zeros((N, 3))
    if gps_available is None:
        gps_available = np.zeros(N, dtype=bool)
    return NavigationData(
        accel_flu=accel.astype(np.float64),
        gyro_flu=gyro.astype(np.float64),
        vel_enu=vel_enu.astype(np.float64),
        lla=lla.astype(np.float64),
        orient=orient.astype(np.float64),
        gps_available=gps_available,
        sample_rate=float(sr),
        dataset_name="stress_test",
        gps_rate=float(gps_rate),
        lla0=LLA0.copy(),
    )


def _run_ins(accel, gyro, sr=200, use_3d=True, params=None):
    """Pure INS run (no GPS)."""
    N = accel.shape[0]
    if params is None:
        params = _zero_noise_params()
    nav = _make_nav(N, sr, accel, gyro)
    return ekf_core.run_ekf(nav, params, outage_config=None,
                            use_3d_rotation=use_3d)


def _generate_truth(accel, gyro, sr=200, use_3d=True):
    """Run a perfect zero-noise INS pass to produce numerically-exact truth.

    The EKF uses forward-Euler integration, so a hand-written analytical
    trajectory will not match sample-for-sample.  This helper runs the
    same EKF code with no process noise, no measurement noise, and no
    GPS -- producing positions/velocities/orientations that are perfectly
    consistent with the IMU inputs.  These are then used as the GPS truth
    for aided tests, ensuring the innovation is zero when the filter is
    correct (no systematic integration-mismatch bias).
    """
    res = _run_ins(accel, gyro, sr=sr, use_3d=use_3d, params=_zero_noise_params())
    truth_enu = res['p']
    truth_vel = res['v']
    truth_rpy = res['r']
    # Convert ENU to LLA for GPS pseudo-measurements
    truth_lla = _build_lla_trajectory(truth_enu)
    return truth_enu, truth_vel, truth_rpy, truth_lla


def _run_gps_aided(accel, gyro, lla_traj, vel_enu, sr=200, gps_rate=1.0,
                   use_3d=True, params=None, outage_config=None):
    """GPS-aided run."""
    N = accel.shape[0]
    if params is None:
        params = _gps_aided_params()

    gps_available = np.zeros(N, dtype=bool)
    gps_interval = max(1, int(sr / gps_rate))
    gps_available[::gps_interval] = True

    nav = _make_nav(N, sr, accel, gyro, lla=lla_traj, vel_enu=vel_enu,
                    gps_available=gps_available, gps_rate=gps_rate)
    return ekf_core.run_ekf(nav, params, outage_config=outage_config,
                            use_3d_rotation=use_3d)


def _make_expected(pos, vel=None, orient=None, dt=None):
    N = pos.shape[0]
    if vel is None and dt is not None:
        vel = np.zeros_like(pos)
        vel[1:] = np.diff(pos, axis=0) / dt
    if orient is None:
        orient = np.zeros((N, 3))
    return dict(pos=pos, vel=vel, orient=orient)


# =====================================================================
#  GROUP A: PURE INS SCENARIOS (no GPS -- verify mechanisation)
# =====================================================================

def ins_figure8():
    """Figure-8 trajectory: two circles joined, testing yaw reversal.

    The vehicle drives a figure-8 at constant speed. This requires
    alternating left/right turns, stressing yaw integration and the
    centripetal acceleration handling.

    Analytical: parametric lemniscate-like path built from two semicircles.
    """
    v = 5.0          # m/s
    R = 20.0         # radius of each lobe
    omega = v / R    # rad/s turn rate
    sr = 200
    dt = 1.0 / sr

    # Phase 1: straight ramp-up (0.1s)
    # Phase 2: left circle (full loop)
    # Phase 3: right circle (full loop)
    T_circle = 2 * pi / omega
    n_ramp = int(0.1 * sr)
    n_circle = int(T_circle * sr)
    N = n_ramp + 2 * n_circle

    t = np.arange(N) * dt
    accel = np.zeros((N, 3))
    accel[:, 2] = G  # gravity
    gyro = np.zeros((N, 3))

    # Ramp-up: accelerate to v
    accel[:n_ramp, 0] = v / (n_ramp * dt)

    # Left circle: positive yaw rate, centripetal left
    i1, i2 = n_ramp, n_ramp + n_circle
    gyro[i1:i2, 2] = omega
    accel[i1:i2, 1] = v * omega  # centripetal = v^2/R = v*omega, pointing left

    # Right circle: negative yaw rate, centripetal right
    i3, i4 = i2, i2 + n_circle
    gyro[i3:i4, 2] = -omega
    accel[i3:i4, 1] = -v * omega  # centripetal pointing right

    res = _run_ins(accel, gyro, sr)

    # Analytical trajectory
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    exp_o = np.zeros((N, 3))
    yaw = 0.0
    vx, vy = 0.0, 0.0
    px, py = 0.0, 0.0

    for i in range(N):
        if i < n_ramp:
            a_fwd = v / (n_ramp * dt)
            vx += a_fwd * cos(yaw) * dt
            vy += a_fwd * sin(yaw) * dt
        elif i < i2:
            yaw += omega * dt
            vx = v * cos(yaw)
            vy = v * sin(yaw)
        else:
            yaw -= omega * dt
            vx = v * cos(yaw)
            vy = v * sin(yaw)

        px += vx * dt
        py += vy * dt
        exp_p[i] = [px, py, 0]
        exp_v[i] = [vx, vy, 0]
        exp_o[i] = [0, 0, yaw]

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "Figure-8 (INS only)",
            "Two full circles in opposite directions at 5 m/s, R=20m. "
            "Tests yaw reversal and centripetal acceleration.")


def ins_helix():
    """Helical climb: circle + constant climb rate.

    Tests simultaneous yaw rotation, centripetal acceleration, and
    vertical motion. The body frame must correctly project gravity
    while climbing.

    Analytical: circular motion in E-N + linear climb in U.
    """
    v_horiz = 3.0    # horizontal speed m/s
    v_up = 1.0       # climb rate m/s
    R = 15.0
    omega = v_horiz / R
    sr = 200
    dt = 1.0 / sr

    T = 2 * pi / omega  # one full loop
    N = int(T * sr) + 1
    t = np.arange(N) * dt
    n_ramp = 10

    accel = np.zeros((N, 3))
    accel[:, 2] = G  # gravity
    gyro = np.zeros((N, 3))

    # Ramp-up forward speed
    accel[:n_ramp, 0] = v_horiz / (n_ramp * dt)
    # Ramp-up vertical speed (body Up axis)
    accel[:n_ramp, 2] += v_up / (n_ramp * dt)

    # Steady circular + climb
    gyro[:, 2] = omega
    accel[:, 1] = v_horiz * omega  # centripetal left

    res = _run_ins(accel, gyro, sr)

    # Analytical
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    exp_o = np.zeros((N, 3))
    for i, ti in enumerate(t):
        if i < n_ramp:
            frac = (i + 1) / n_ramp
            spd = v_horiz * frac
            spd_u = v_up * frac
            yaw_a = omega * ti
            exp_v[i] = [spd * cos(yaw_a), spd * sin(yaw_a), spd_u]
        else:
            yaw_a = omega * ti
            exp_v[i] = [v_horiz * cos(yaw_a), v_horiz * sin(yaw_a), v_up]
        exp_o[i] = [0, 0, omega * ti]
    # Integrate velocity for position
    for i in range(1, N):
        exp_p[i] = exp_p[i-1] + exp_v[i-1] * dt

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "Helix (INS only)",
            "Circular motion + 1 m/s climb. Full 360 deg loop at 3 m/s, R=15m.")


def ins_high_speed_slalom():
    """High-speed slalom: alternating sharp turns.

    Vehicle at 10 m/s with alternating +/- 45 deg yaw segments,
    like driving through a slalom course. This stresses rapid
    yaw changes and high centripetal accelerations.
    """
    v = 10.0
    sr = 200
    dt = 1.0 / sr
    yaw_amplitude = pi / 4  # 45 deg
    turn_time = 0.5         # seconds per half-turn
    n_turns = 8

    omega = yaw_amplitude / turn_time  # rad/s per turn segment
    n_per_turn = int(turn_time * sr)
    n_ramp = 20
    N = n_ramp + n_turns * n_per_turn

    t = np.arange(N) * dt
    accel = np.zeros((N, 3))
    accel[:, 2] = G
    gyro = np.zeros((N, 3))

    # Ramp-up
    accel[:n_ramp, 0] = v / (n_ramp * dt)

    # Alternating turns
    for k in range(n_turns):
        start = n_ramp + k * n_per_turn
        end = start + n_per_turn
        sign = 1 if k % 2 == 0 else -1
        gyro[start:end, 2] = sign * omega
        accel[start:end, 1] = sign * v * omega  # centripetal

    res = _run_ins(accel, gyro, sr)

    # Analytical
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    exp_o = np.zeros((N, 3))
    yaw = 0.0
    px, py = 0.0, 0.0
    spd = 0.0

    for i in range(N):
        if i < n_ramp:
            spd += (v / (n_ramp * dt)) * dt
        else:
            spd = v

        # Get current gyro
        if i < N:
            yaw += gyro[i, 2] * dt

        vx = spd * cos(yaw)
        vy = spd * sin(yaw)
        px += vx * dt
        py += vy * dt

        exp_p[i] = [px, py, 0]
        exp_v[i] = [vx, vy, 0]
        exp_o[i] = [0, 0, yaw]

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "High-speed slalom (INS only)",
            f"10 m/s with {n_turns} alternating +/-45 deg turns at {omega:.1f} rad/s.")


def ins_pitched_spiral():
    """Climbing spiral: constant pitch + yaw, testing full 3D rotation.

    Body is pitched up 15 deg while turning. The rotation matrix must
    correctly decompose gravity into all three ENU axes.
    """
    pitch_target = pi / 12  # 15 deg
    v_body = 4.0            # body-frame forward speed
    R = 25.0
    sr = 200
    dt = 1.0 / sr

    omega = v_body * cos(pitch_target) / R  # yaw rate for horizontal circle
    T = 2 * pi / omega
    N = int(T * sr) + 1
    t = np.arange(N) * dt

    n_pitch_ramp = 50   # samples to ramp pitch
    n_speed_ramp = 20   # samples to ramp speed

    accel = np.zeros((N, 3))
    gyro = np.zeros((N, 3))

    # Build IMU signals analytically
    for i in range(N):
        ti = i * dt
        # Pitch ramps up then stays constant
        if i < n_pitch_ramp:
            pitch_i = pitch_target * (i + 1) / n_pitch_ramp
            gyro[i, 1] = pitch_target / (n_pitch_ramp * dt)
        else:
            pitch_i = pitch_target

        # Speed ramps up then stays constant
        if i < n_speed_ramp:
            spd = v_body * (i + 1) / n_speed_ramp
            accel[i, 0] = v_body / (n_speed_ramp * dt)
        else:
            spd = v_body

        # Yaw rate constant from the start
        gyro[i, 2] = omega

        # Centripetal acceleration (in body-Left direction)
        v_horiz = spd * cos(pitch_i)
        accel[i, 1] = v_horiz * omega

        # Gravity reaction in body frame:
        # At pitch angle, the body accelerometer reads:
        #   a_body = Rbn^T * [0, 0, +9.81] + body_dynamic_accel
        # Rbn^T * [0,0,G] for yaw-only-at-this-instant (body frame sees gravity):
        #   forward: G * sin(pitch)  (gravity component along body-x)
        #   left:    0
        #   up:      G * cos(pitch)  (gravity component along body-z)
        # But wait - FLU convention with our Ry:
        #   Ry^T @ [0,0,G] = [G*sin(pitch), 0, G*cos(pitch)]
        # This is the specific force from gravity that the accel measures.
        accel[i, 0] += G * sin(pitch_i)
        accel[i, 2] += G * cos(pitch_i)

    # Subtract the default gravity that _run_ins doesn't add
    # Actually, we built the full accel including gravity, so don't add G again
    # We need to NOT set accel[:,2] = G at the top since we compute it ourselves
    # Let's just rebuild cleanly:
    accel2 = np.zeros((N, 3))
    gyro2 = np.zeros((N, 3))

    for i in range(N):
        if i < n_pitch_ramp:
            pitch_i = pitch_target * (i + 1) / n_pitch_ramp
            gyro2[i, 1] = pitch_target / (n_pitch_ramp * dt)
        else:
            pitch_i = pitch_target

        if i < n_speed_ramp:
            spd = v_body * (i + 1) / n_speed_ramp
            accel2[i, 0] = v_body / (n_speed_ramp * dt)  # forward thrust
        else:
            spd = v_body

        gyro2[i, 2] = omega

        v_horiz = spd * cos(pitch_i)
        accel2[i, 1] = v_horiz * omega  # centripetal

        # Gravity in body frame (Rbn^T @ [0,0,G])
        # For FLU with our Ry convention:
        #   Rbn = Rz(yaw) @ Ry(pitch) @ Rx(0)
        #   Rbn^T @ [0,0,G] (only pitch matters for gravity projection):
        #     Ry(pitch)^T @ [0,0,G] = [G*sin(pitch), 0, G*cos(pitch)]
        accel2[i, 0] += G * sin(pitch_i)
        accel2[i, 2] += G * cos(pitch_i)

    res = _run_ins(accel2, gyro2, sr)

    # Analytical trajectory
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    exp_o = np.zeros((N, 3))
    yaw_a = 0.0
    px, py, pz = 0.0, 0.0, 0.0

    for i in range(N):
        if i < n_pitch_ramp:
            pitch_i = pitch_target * (i + 1) / n_pitch_ramp
        else:
            pitch_i = pitch_target

        if i < n_speed_ramp:
            spd = v_body * (i + 1) / n_speed_ramp
        else:
            spd = v_body

        yaw_a += omega * dt

        ve = spd * cos(pitch_i) * cos(yaw_a)
        vn = spd * cos(pitch_i) * sin(yaw_a)
        vu = spd * sin(pitch_i)

        px += ve * dt
        py += vn * dt
        pz += vu * dt

        exp_p[i] = [px, py, pz]
        exp_v[i] = [ve, vn, vu]
        exp_o[i] = [0, pitch_i, yaw_a]

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "Pitched spiral (INS only, 3D)",
            f"Climb at 15 deg pitch while circling. v={v_body} m/s, R={R}m.")


def ins_accel_decel_uturn():
    """Accelerate, U-turn, decelerate, stop.

    Tests:
    - Forward acceleration from standstill
    - 180 deg yaw turn
    - Deceleration (negative body-forward accel)
    - The vehicle should end near x=0 (returned to start region)
    """
    sr = 200
    dt = 1.0 / sr
    v_max = 8.0
    accel_rate = 4.0  # m/s^2

    t_accel = v_max / accel_rate  # 2s
    t_turn = 2.0  # seconds for 180 deg turn
    omega_turn = pi / t_turn
    t_cruise = 1.0
    t_decel = v_max / accel_rate  # 2s

    n_accel = int(t_accel * sr)
    n_cruise1 = int(t_cruise * sr)
    n_turn = int(t_turn * sr)
    n_cruise2 = int(t_cruise * sr)
    n_decel = int(t_decel * sr)

    N = n_accel + n_cruise1 + n_turn + n_cruise2 + n_decel
    t = np.arange(N) * dt

    accel = np.zeros((N, 3))
    accel[:, 2] = G
    gyro = np.zeros((N, 3))

    idx = 0
    # Phase 1: accelerate
    accel[idx:idx+n_accel, 0] = accel_rate
    idx += n_accel
    # Phase 2: cruise
    idx += n_cruise1
    # Phase 3: U-turn (180 deg)
    gyro[idx:idx+n_turn, 2] = omega_turn
    accel[idx:idx+n_turn, 1] = v_max * omega_turn  # centripetal
    idx += n_turn
    # Phase 4: cruise (heading reversed)
    idx += n_cruise2
    # Phase 5: decelerate
    accel[idx:idx+n_decel, 0] = -accel_rate
    idx += n_decel

    res = _run_ins(accel, gyro, sr)

    # Analytical
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    exp_o = np.zeros((N, 3))
    yaw = 0.0
    spd = 0.0
    px, py = 0.0, 0.0

    for i in range(N):
        # Determine body-forward acceleration
        a_fwd = accel[i, 0] if accel[i, 0] != 0 else 0
        # Don't count gravity component
        if abs(a_fwd) < 0.01:
            a_fwd = 0

        yaw += gyro[i, 2] * dt
        spd += a_fwd * dt
        spd = max(spd, 0)  # can't go negative speed

        vx = spd * cos(yaw)
        vy = spd * sin(yaw)
        px += vx * dt
        py += vy * dt

        exp_p[i] = [px, py, 0]
        exp_v[i] = [vx, vy, 0]
        exp_o[i] = [0, 0, yaw]

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "Accel-UTurn-Decel (INS only)",
            f"Accel to {v_max} m/s, 180 deg turn, decel to stop. Tests full cycle.")


def ins_long_straight():
    """Long straight drive at constant speed for 30 seconds.

    With perfect IMU, position error should stay < 1cm.
    This is the baseline: if this drifts, the equations are wrong.
    """
    v = 15.0  # m/s (54 km/h)
    sr = 200
    dt = 1.0 / sr
    T = 30.0  # seconds
    N = int(T * sr)
    t = np.arange(N) * dt

    n_ramp = 20
    accel = np.zeros((N, 3))
    accel[:, 2] = G
    accel[:n_ramp, 0] = v / (n_ramp * dt)
    gyro = np.zeros((N, 3))

    res = _run_ins(accel, gyro, sr)

    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    exp_o = np.zeros((N, 3))
    spd = 0.0
    px = 0.0
    for i in range(N):
        if i < n_ramp:
            spd += (v / (n_ramp * dt)) * dt
        else:
            spd = v
        px += spd * dt
        exp_p[i] = [px, 0, 0]
        exp_v[i] = [spd, 0, 0]

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "Long straight (INS only, 30s)",
            f"Constant {v} m/s straight East for 30s. Error should be < 1cm.")


# =====================================================================
#  GROUP B: GPS-AIDED SCENARIOS
# =====================================================================

def gps_racetrack():
    """Racetrack (oval): two straights + two semicircles with GPS.

    The filter must track position through curves and straights while
    GPS corrects drift. Tests that GPS updates properly anchor the
    trajectory AND that orientation tracks correctly.
    """
    v = 8.0
    R = 30.0
    omega = v / R
    sr = 100  # lower rate for GPS-aided (more realistic)
    dt = 1.0 / sr
    gps_rate = 1.0

    straight_len = 60.0  # meters
    t_straight = straight_len / v
    t_semicircle = pi / omega

    n_ramp = 10
    n_straight = int(t_straight * sr)
    n_semi = int(t_semicircle * sr)

    # Ramp + straight + semi + straight + semi
    N = n_ramp + 2 * n_straight + 2 * n_semi
    t = np.arange(N) * dt

    accel = np.zeros((N, 3))
    accel[:, 2] = G
    gyro = np.zeros((N, 3))

    # Ramp-up
    accel[:n_ramp, 0] = v / (n_ramp * dt)

    idx = n_ramp
    # Straight 1
    idx += n_straight
    # Semicircle 1 (left turn)
    gyro[idx:idx+n_semi, 2] = omega
    accel[idx:idx+n_semi, 1] = v * omega
    idx += n_semi
    # Straight 2
    idx += n_straight
    # Semicircle 2 (left turn)
    gyro[idx:idx+n_semi, 2] = omega
    accel[idx:idx+n_semi, 1] = v * omega

    # Generate numerically-exact truth from pure INS pass
    exp_p, exp_v, exp_o, lla_traj = _generate_truth(accel, gyro, sr)

    res = _run_gps_aided(accel, gyro, lla_traj, exp_v, sr=sr,
                         gps_rate=gps_rate, params=_gps_aided_params(Rpos=2.0))

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "Racetrack (GPS-aided)",
            f"Oval: 2x{straight_len}m straights + 2 semicircles R={R}m at {v} m/s. GPS at {gps_rate} Hz.")


def gps_figure8_long():
    """Long figure-8 with GPS at 1 Hz.

    Two full laps of a figure-8 (4 circles total). The filter must
    maintain orientation tracking over multiple yaw reversals with
    only 1 Hz GPS position corrections.
    """
    v = 6.0
    R = 25.0
    omega = v / R
    sr = 100
    dt = 1.0 / sr
    gps_rate = 1.0

    T_circle = 2 * pi / omega
    n_ramp = 10
    n_circle = int(T_circle * sr)
    n_laps = 2

    # 2 laps x (left circle + right circle)
    N = n_ramp + n_laps * 2 * n_circle
    t = np.arange(N) * dt

    accel = np.zeros((N, 3))
    accel[:, 2] = G
    gyro = np.zeros((N, 3))

    accel[:n_ramp, 0] = v / (n_ramp * dt)

    idx = n_ramp
    for lap in range(n_laps):
        # Left circle
        gyro[idx:idx+n_circle, 2] = omega
        accel[idx:idx+n_circle, 1] = v * omega
        idx += n_circle
        # Right circle
        gyro[idx:idx+n_circle, 2] = -omega
        accel[idx:idx+n_circle, 1] = -v * omega
        idx += n_circle

    # Generate numerically-exact truth from pure INS pass
    exp_p, exp_v, exp_o, lla_traj = _generate_truth(accel, gyro, sr)

    res = _run_gps_aided(accel, gyro, lla_traj, exp_v, sr=sr,
                         gps_rate=gps_rate, params=_gps_aided_params(Rpos=3.0))

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "Figure-8 x2 laps (GPS-aided)",
            f"2 laps of figure-8 at {v} m/s, R={R}m. GPS at {gps_rate} Hz.")


def gps_stop_and_go():
    """Stop-and-go driving: accel, cruise, stop, wait, repeat.

    Tests the filter's ability to handle velocity transitions with
    GPS corrections, and verifies the stationary segments don't drift.
    """
    sr = 100
    dt = 1.0 / sr
    gps_rate = 1.0
    v = 10.0
    a = 3.0  # m/s^2

    t_accel = v / a
    t_cruise = 3.0
    t_decel = v / a
    t_stop = 5.0

    n_accel = int(t_accel * sr)
    n_cruise = int(t_cruise * sr)
    n_decel = int(t_decel * sr)
    n_stop = int(t_stop * sr)
    n_segment = n_accel + n_cruise + n_decel + n_stop
    n_repeats = 3

    N = n_repeats * n_segment
    t = np.arange(N) * dt

    accel = np.zeros((N, 3))
    accel[:, 2] = G
    gyro = np.zeros((N, 3))

    for rep in range(n_repeats):
        base = rep * n_segment
        # Accelerate
        accel[base:base+n_accel, 0] = a
        # Cruise (no accel)
        # Decelerate
        accel[base+n_accel+n_cruise:base+n_accel+n_cruise+n_decel, 0] = -a
        # Stop (no accel)

    # Generate numerically-exact truth from pure INS pass
    exp_p, exp_v, exp_o, lla_traj = _generate_truth(accel, gyro, sr)

    res = _run_gps_aided(accel, gyro, lla_traj, exp_v, sr=sr,
                         gps_rate=gps_rate, params=_gps_aided_params(Rpos=2.0))

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "Stop-and-Go (GPS-aided)",
            f"3 cycles of accel to {v} m/s, cruise, decel, stop 5s. GPS at {gps_rate} Hz.")


# =====================================================================
#  GROUP C: GPS OUTAGE SCENARIOS
# =====================================================================

def outage_straight_60s():
    """60-second GPS outage during straight driving.

    Vehicle drives straight East at 15 m/s. GPS available for first 30s,
    then 60s outage, then GPS returns for 30s.
    This is a hard test: 60s of pure INS at high speed.
    """
    v = 15.0
    sr = 100
    dt = 1.0 / sr
    gps_rate = 1.0
    T = 120.0  # total time
    N = int(T * sr)
    t = np.arange(N) * dt

    n_ramp = 10
    accel = np.zeros((N, 3))
    accel[:, 2] = G
    accel[:n_ramp, 0] = v / (n_ramp * dt)
    gyro = np.zeros((N, 3))

    # Generate numerically-exact truth from pure INS pass
    exp_p, exp_v, exp_o, lla_traj = _generate_truth(accel, gyro, sr)

    outage = {'start': 30.0, 'duration': 60.0}
    res = _run_gps_aided(accel, gyro, lla_traj, exp_v, sr=sr,
                         gps_rate=gps_rate,
                         params=_gps_aided_params(Rpos=2.0),
                         outage_config=outage)

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "60s outage, straight (GPS-aided)",
            f"Straight East at {v} m/s. GPS for 30s, outage 60s, GPS returns 30s.",
            outage)


def outage_curve_30s():
    """30-second outage during a curve.

    Vehicle drives a large circle. GPS outage hits during the turn.
    This is harder than a straight outage because yaw drifts compound
    the position error.
    """
    v = 10.0
    R = 50.0
    omega = v / R
    sr = 100
    dt = 1.0 / sr
    gps_rate = 1.0

    T = 90.0  # total time (covers more than one lap)
    N = int(T * sr)
    t = np.arange(N) * dt

    n_ramp = 10
    accel = np.zeros((N, 3))
    accel[:, 2] = G
    accel[:n_ramp, 0] = v / (n_ramp * dt)
    gyro = np.zeros((N, 3))

    # Constant left turn
    gyro[:, 2] = omega
    accel[:, 1] = v * omega

    # Generate numerically-exact truth from pure INS pass
    exp_p, exp_v, exp_o, lla_traj = _generate_truth(accel, gyro, sr)

    outage = {'start': 20.0, 'duration': 30.0}
    res = _run_gps_aided(accel, gyro, lla_traj, exp_v, sr=sr,
                         gps_rate=gps_rate,
                         params=_gps_aided_params(Rpos=2.0),
                         outage_config=outage)

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "30s outage during curve (GPS-aided)",
            f"Constant left turn at {v} m/s, R={R}m. 30s outage from t=20s.",
            outage)


def outage_figure8_45s():
    """45-second outage spanning a yaw reversal in a figure-8.

    The hardest outage scenario: the outage covers the transition from
    one circle direction to the other. Yaw estimation must handle the
    reversal without GPS.
    """
    v = 6.0
    R = 20.0
    omega = v / R
    sr = 100
    dt = 1.0 / sr
    gps_rate = 1.0

    T_circle = 2 * pi / omega
    n_ramp = 10
    n_circle = int(T_circle * sr)

    # GPS for first circle, outage during transition + second circle start
    N = n_ramp + 2 * n_circle + int(10 * sr)  # extra 10s after
    t = np.arange(N) * dt

    accel = np.zeros((N, 3))
    accel[:, 2] = G
    gyro = np.zeros((N, 3))

    accel[:n_ramp, 0] = v / (n_ramp * dt)

    idx = n_ramp
    # Left circle
    gyro[idx:idx+n_circle, 2] = omega
    accel[idx:idx+n_circle, 1] = v * omega
    idx += n_circle
    # Right circle
    gyro[idx:idx+n_circle, 2] = -omega
    accel[idx:idx+n_circle, 1] = -v * omega

    # Generate numerically-exact truth from pure INS pass
    exp_p, exp_v, exp_o, lla_traj = _generate_truth(accel, gyro, sr)

    # Outage starts near end of first circle, covers the transition
    t_outage_start = (n_ramp + n_circle - int(5 * sr)) / sr
    outage = {'start': t_outage_start, 'duration': 45.0}

    res = _run_gps_aided(accel, gyro, lla_traj, exp_v, sr=sr,
                         gps_rate=gps_rate,
                         params=_gps_aided_params(Rpos=2.0),
                         outage_config=outage)

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "45s outage across figure-8 reversal",
            f"Figure-8 at {v} m/s, R={R}m. 45s outage covering yaw reversal.",
            outage)


def outage_reconvergence():
    """Test that the filter re-converges after a long outage.

    30s GPS, 40s outage, 50s GPS. Focus is on HOW FAST the filter
    snaps back to truth after GPS returns. Position error at the end
    should be close to GPS noise level.
    """
    v = 12.0
    sr = 100
    dt = 1.0 / sr
    gps_rate = 1.0
    T = 120.0
    N = int(T * sr)
    t = np.arange(N) * dt

    n_ramp = 10
    accel = np.zeros((N, 3))
    accel[:, 2] = G
    accel[:n_ramp, 0] = v / (n_ramp * dt)

    # Add a gentle curve for realism
    omega = 0.02  # very gentle turn
    gyro = np.zeros((N, 3))
    gyro[:, 2] = omega
    accel[:, 1] = v * omega

    # Generate numerically-exact truth from pure INS pass
    exp_p, exp_v, exp_o, lla_traj = _generate_truth(accel, gyro, sr)

    outage = {'start': 30.0, 'duration': 40.0}
    res = _run_gps_aided(accel, gyro, lla_traj, exp_v, sr=sr,
                         gps_rate=gps_rate,
                         params=_gps_aided_params(Rpos=2.0),
                         outage_config=outage)

    return (res, _make_expected(exp_p, exp_v, exp_o), t,
            "Re-convergence after 40s outage",
            f"Gentle curve at {v} m/s. 30s GPS, 40s outage, 50s GPS. "
            "End error should be small.",
            outage)


# =====================================================================
#  Plotting
# =====================================================================

def _enforce_min_span(ax, axis='both', min_span=None):
    if min_span is None:
        return
    if axis == 'both':
        axes_list = ['x', 'y']
    elif axis == 'all':
        axes_list = ['x', 'y', 'z']
    else:
        axes_list = [axis]
    for a in axes_list:
        if a == 'x':
            lo, hi = ax.get_xlim()
        elif a == 'y':
            lo, hi = ax.get_ylim()
        else:
            lo, hi = ax.get_zlim()
        span = hi - lo
        if span < min_span:
            mid = (lo + hi) / 2
            if a == 'x':
                ax.set_xlim(mid - min_span/2, mid + min_span/2)
            elif a == 'y':
                ax.set_ylim(mid - min_span/2, mid + min_span/2)
            else:
                ax.set_zlim(mid - min_span/2, mid + min_span/2)


def plot_scenario(res, exp, t, title, desc, outage=None):
    """16-panel diagnostic plot (4x4 grid).

    Row 0: E-N trajectory · Position vs time · Position error · Pos error ± 1σ
    Row 1: Velocity vs time · Velocity error · Vel error ± 1σ · Speed
    Row 2: Orientation vs time · Orientation error · Orient error ± 1σ · 3D trajectory
    Row 3: Pos jump/step (spike detect) · Orient jump/step · Bias estimates · Covariance diagonals

    GPS update instants are shown as thin green vertical lines.
    Spikes (>3× median step size) are flagged with red markers.
    """
    p = res['p']
    v = res['v']
    r = res['r']
    sp = res['std_pos']
    sv = res['std_vel']
    so = res['std_orient']
    ba = res['bias_acc']
    bg = res['bias_gyr']
    exp_p = exp['pos']
    exp_v = exp['vel']
    exp_o = exp['orient']

    fig = plt.figure(figsize=(26, 18), constrained_layout=True)
    fig.suptitle(f"{title}\n{desc}", fontsize=11, fontweight='bold')
    gs = fig.add_gridspec(4, 4)

    # Determine GPS update instants from sample rate
    N = len(t)
    sr = N / t[-1] if t[-1] > 0 else 100
    gps_interval = max(1, int(sr))  # assume 1 Hz GPS
    gps_indices = np.arange(0, N, gps_interval)

    # Outage shading + GPS update markers
    def shade(ax):
        if outage is not None:
            t_start = outage['start']
            t_end = t_start + outage['duration']
            ax.axvspan(t_start, t_end, alpha=0.12, color='red', label='GPS outage')
        # Thin green lines at GPS epochs (only the first 500 to avoid clutter)
        for gi in gps_indices[:500]:
            gt = t[gi] if gi < N else None
            if gt is not None:
                if outage is not None:
                    if outage['start'] <= gt <= outage['start'] + outage['duration']:
                        continue
                ax.axvline(gt, color='lime', alpha=0.18, lw=0.4)

    colors = ['r', 'g', 'b']

    # ── Row 0 ──
    # 1) E-N trajectory
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(exp_p[:, 0], exp_p[:, 1], 'g--', lw=2, label='Analytical')
    ax.plot(p[:, 0], p[:, 1], 'b-', lw=1, label='EKF')
    ax.plot(p[0, 0], p[0, 1], 'ko', ms=8, label='Start')
    ax.plot(p[-1, 0], p[-1, 1], 'r*', ms=12, label='End')
    if outage is not None:
        i_start = int(outage['start'] * sr)
        i_end = min(int((outage['start'] + outage['duration']) * sr), N - 1)
        ax.plot(p[i_start:i_end, 0], p[i_start:i_end, 1], 'r-', lw=2,
                alpha=0.6, label='During outage')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    ax.set_title('Trajectory (E-N)'); ax.legend(fontsize=6); ax.grid(True)
    _enforce_min_span(ax, 'both', min_span=1.0)
    ax.set_aspect('equal')

    # 2) Position vs time
    ax = fig.add_subplot(gs[0, 1])
    for j, (lbl, c) in enumerate(zip(['East', 'North', 'Up'], colors)):
        ax.plot(t, exp_p[:, j], c+'--', lw=1.5, label=f'{lbl} truth')
        ax.plot(t, p[:, j], c+'-', lw=0.8, label=f'{lbl} EKF')
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Position [m]')
    ax.set_title('Position vs Time'); ax.legend(fontsize=5); ax.grid(True)

    # 3) Position error
    ax = fig.add_subplot(gs[0, 2])
    p_err = p - exp_p
    p_err_norm = np.linalg.norm(p_err, axis=1)
    for j, (lbl, c) in enumerate(zip(['E', 'N', 'U'], colors)):
        ax.plot(t, p_err[:, j], c+'-', lw=0.8, label=lbl)
    ax.plot(t, p_err_norm, 'k-', lw=1.5, label='3D norm')
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error [m]')
    ax.set_title('Position Error'); ax.legend(fontsize=6); ax.grid(True)

    # 4) Position error ± 1σ
    ax = fig.add_subplot(gs[0, 3])
    for j, (lbl, c) in enumerate(zip(['E', 'N', 'U'], colors)):
        ax.fill_between(t, p_err[:, j] - sp[:, j], p_err[:, j] + sp[:, j],
                        color=c, alpha=0.15)
        ax.plot(t, p_err[:, j], c+'-', lw=0.8, label=lbl)
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error ± 1σ [m]')
    ax.set_title('Position Error ± σ'); ax.legend(fontsize=6); ax.grid(True)

    # ── Row 1 ──
    # 5) Velocity vs time
    ax = fig.add_subplot(gs[1, 0])
    for j, (lbl, c) in enumerate(zip(['v_E', 'v_N', 'v_U'], colors)):
        ax.plot(t, exp_v[:, j], c+'--', lw=1.5, label=f'{lbl} truth')
        ax.plot(t, v[:, j], c+'-', lw=0.8, label=f'{lbl} EKF')
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Velocity vs Time'); ax.legend(fontsize=5); ax.grid(True)

    # 6) Velocity error
    ax = fig.add_subplot(gs[1, 1])
    v_err = v - exp_v
    v_err_norm = np.linalg.norm(v_err, axis=1)
    for j, (lbl, c) in enumerate(zip(['v_E', 'v_N', 'v_U'], colors)):
        ax.plot(t, v_err[:, j], c+'-', lw=0.8, label=lbl)
    ax.plot(t, v_err_norm, 'k-', lw=1.5, label='3D norm')
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error [m/s]')
    ax.set_title('Velocity Error'); ax.legend(fontsize=6); ax.grid(True)

    # 7) Velocity error ± 1σ
    ax = fig.add_subplot(gs[1, 2])
    for j, (lbl, c) in enumerate(zip(['v_E', 'v_N', 'v_U'], colors)):
        ax.fill_between(t, v_err[:, j] - sv[:, j], v_err[:, j] + sv[:, j],
                        color=c, alpha=0.15)
        ax.plot(t, v_err[:, j], c+'-', lw=0.8, label=lbl)
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error ± 1σ [m/s]')
    ax.set_title('Velocity Error ± σ'); ax.legend(fontsize=6); ax.grid(True)

    # 8) Speed comparison
    ax = fig.add_subplot(gs[1, 3])
    speed_est = np.linalg.norm(v, axis=1)
    speed_exp = np.linalg.norm(exp_v, axis=1)
    ax.plot(t, speed_exp, 'g--', lw=1.5, label='Truth |v|')
    ax.plot(t, speed_est, 'b-', lw=1, label='EKF |v|')
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Speed [m/s]')
    ax.set_title('Speed'); ax.legend(); ax.grid(True)

    # ── Row 2 ──
    # 9) Orientation vs time
    ax = fig.add_subplot(gs[2, 0])
    for j, (lbl, c) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], colors)):
        ax.plot(t, np.degrees(exp_o[:, j]), c+'--', lw=1.5, label=f'{lbl} truth')
        ax.plot(t, np.degrees(r[:, j]), c+'-', lw=0.8, label=f'{lbl} EKF')
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Angle [deg]')
    ax.set_title('Orientation vs Time'); ax.legend(fontsize=5); ax.grid(True)

    # 10) Orientation error
    ax = fig.add_subplot(gs[2, 1])
    o_err = r - exp_o
    o_err = (o_err + pi) % (2 * pi) - pi
    for j, (lbl, c) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], colors)):
        ax.plot(t, np.degrees(o_err[:, j]), c+'-', lw=0.8, label=lbl)
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error [deg]')
    ax.set_title('Orientation Error'); ax.legend(fontsize=6); ax.grid(True)

    # 11) Orientation error ± 1σ
    ax = fig.add_subplot(gs[2, 2])
    for j, (lbl, c) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], colors)):
        ax.fill_between(t, np.degrees(o_err[:, j]) - np.degrees(so[:, j]),
                        np.degrees(o_err[:, j]) + np.degrees(so[:, j]),
                        color=c, alpha=0.15)
        ax.plot(t, np.degrees(o_err[:, j]), c+'-', lw=0.8, label=lbl)
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error ± 1σ [deg]')
    ax.set_title('Orient Error ± σ'); ax.legend(fontsize=6); ax.grid(True)

    # 12) 3D trajectory (with equal aspect ratio to avoid visual spikes)
    ax = fig.add_subplot(gs[2, 3], projection='3d')
    ax.plot(exp_p[:, 0], exp_p[:, 1], exp_p[:, 2], 'g--', lw=2, label='Truth')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-', lw=1, label='EKF')
    ax.set_xlabel('East'); ax.set_ylabel('North'); ax.set_zlabel('Up')
    ax.set_title('3D Trajectory'); ax.legend()
    # Force equal axis scaling so that mm-level Up noise doesn't get
    # stretched to look like 100 m spikes when E-N spans hundreds of metres.
    _all = np.vstack([p, exp_p])
    _mid = [(_all[:, i].max() + _all[:, i].min()) / 2 for i in range(3)]
    _half = max(_all[:, i].max() - _all[:, i].min() for i in range(3)) / 2 * 1.05
    ax.set_xlim(_mid[0] - _half, _mid[0] + _half)
    ax.set_ylim(_mid[1] - _half, _mid[1] + _half)
    ax.set_zlim(_mid[2] - _half, _mid[2] + _half)

    # ── Row 3: Spike detection & diagnostics ──
    # 13) Position jump per step (spike detector)
    ax = fig.add_subplot(gs[3, 0])
    dp = np.diff(p, axis=0)
    dp_norm = np.linalg.norm(dp, axis=1)
    dt_step = t[1] - t[0] if len(t) > 1 else 0.01
    ax.plot(t[1:], dp_norm / dt_step, 'k-', lw=0.5, alpha=0.6,
            label='|Δp|/Δt (approx speed)')
    med = np.median(dp_norm)
    spike_mask = dp_norm > 3 * max(med, 1e-10)
    if np.any(spike_mask):
        ax.plot(t[1:][spike_mask], dp_norm[spike_mask] / dt_step, 'rv', ms=3,
                label=f'Spikes (>3× median): {np.sum(spike_mask)}')
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Δpos/Δt [m/s]')
    ax.set_title('Position Jumps (spike detector)'); ax.legend(fontsize=6); ax.grid(True)

    # 14) Orientation jump per step (spike detector)
    ax = fig.add_subplot(gs[3, 1])
    dr = np.diff(r, axis=0)
    dr = (dr + pi) % (2 * pi) - pi
    for j, (lbl, c) in enumerate(zip(['ΔRoll', 'ΔPitch', 'ΔYaw'], colors)):
        ax.plot(t[1:], np.degrees(dr[:, j]) / dt_step, c+'-', lw=0.5, alpha=0.6,
                label=f'{lbl}/Δt [deg/s]')
    # Highlight pitch spikes
    dpitch_deg = np.abs(np.degrees(dr[:, 1]))
    med_p = np.median(dpitch_deg)
    spike_p = dpitch_deg > 3 * max(med_p, 1e-8)
    if np.any(spike_p):
        ax.plot(t[1:][spike_p], np.degrees(dr[spike_p, 1]) / dt_step, 'rv', ms=3,
                label=f'Pitch spikes: {np.sum(spike_p)}')
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Rate [deg/s]')
    ax.set_title('Orient Jumps (spike detector)'); ax.legend(fontsize=5); ax.grid(True)

    # 15) Bias estimates
    ax = fig.add_subplot(gs[3, 2])
    for j, (lbl, c) in enumerate(zip(['ba_x', 'ba_y', 'ba_z'], colors)):
        ax.plot(t, ba[:, j], c+'-', lw=0.8, label=lbl)
    for j, (lbl, c) in enumerate(zip(['bg_x', 'bg_y', 'bg_z'], ['m', 'c', 'y'])):
        ax.plot(t, bg[:, j], c+'--', lw=0.8, label=lbl)
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Bias')
    ax.set_title('Bias Estimates'); ax.legend(fontsize=5); ax.grid(True)

    # 16) Covariance diagonals
    ax = fig.add_subplot(gs[3, 3])
    ax.semilogy(t, np.linalg.norm(sp, axis=1), 'r-', lw=1, label='σ_pos (norm)')
    ax.semilogy(t, np.linalg.norm(sv, axis=1), 'g-', lw=1, label='σ_vel (norm)')
    ax.semilogy(t, np.degrees(np.linalg.norm(so, axis=1)), 'b-', lw=1,
                label='σ_orient (deg, norm)')
    shade(ax)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('σ (log scale)')
    ax.set_title('Covariance Evolution'); ax.legend(fontsize=6); ax.grid(True)

    return fig


# =====================================================================
#  Scenario registry
# =====================================================================

SCENARIOS = {
    # Group A: Pure INS
    'figure8':        ('ins', ins_figure8),
    'helix':          ('ins', ins_helix),
    'slalom':         ('ins', ins_high_speed_slalom),
    'spiral':         ('ins', ins_pitched_spiral),
    'uturn':          ('ins', ins_accel_decel_uturn),
    'long_straight':  ('ins', ins_long_straight),
    # Group B: GPS-aided
    'racetrack':      ('gps_aided', gps_racetrack),
    'gps_figure8':    ('gps_aided', gps_figure8_long),
    'stop_go':        ('gps_aided', gps_stop_and_go),
    # Group C: GPS outage
    'outage_straight': ('outage', outage_straight_60s),
    'outage_curve':    ('outage', outage_curve_30s),
    'outage_fig8':     ('outage', outage_figure8_45s),
    'reconvergence':   ('outage', outage_reconvergence),
}


def main():
    args = sys.argv[1:]

    if '--list' in args:
        print("Available scenarios:")
        for name, (group, _) in SCENARIOS.items():
            print(f"  [{group:10s}]  {name}")
        return

    # Filter by group
    if '--group' in args:
        idx = args.index('--group')
        if idx + 1 < len(args):
            group_filter = args[idx + 1]
            selected = [n for n, (g, _) in SCENARIOS.items() if g == group_filter]
            if not selected:
                print(f"Unknown group '{group_filter}'. Available: ins, gps_aided, outage")
                return
        else:
            print("--group requires a value: ins, gps_aided, outage")
            return
    elif not args:
        selected = list(SCENARIOS.keys())
    else:
        selected = [a for a in args if a in SCENARIOS]
        unknown = [a for a in args if a not in SCENARIOS and a != '--group']
        if unknown:
            print(f"Unknown scenarios: {unknown}")
            print(f"Available: {list(SCENARIOS.keys())}")
            return

    print(f"Running {len(selected)} scenario(s): {', '.join(selected)}")
    print("=" * 70)

    for name in selected:
        group, scenario_fn = SCENARIOS[name]

        print(f"\n{'='*70}")
        print(f"  [{group.upper()}]  {name}")
        print(f"{'='*70}")

        result = scenario_fn()

        # Outage scenarios return 6 values, others return 5
        if len(result) == 6:
            res, exp, t, title, desc, outage = result
        else:
            res, exp, t, title, desc = result
            outage = None

        p = res['p']
        exp_p = exp['pos']
        exp_v = exp['vel']
        exp_o = exp['orient']
        v = res['v']
        r = res['r']

        err = np.linalg.norm(p - exp_p, axis=1)
        v_err = np.linalg.norm(v - exp_v, axis=1)
        o_err = r - exp_o
        o_err = (o_err + pi) % (2 * pi) - pi

        print(f"  {desc}")
        print(f"  Duration: {t[-1]:.1f}s, Samples: {len(t)}")
        print(f"  Final position:  E={p[-1,0]:+.3f}  N={p[-1,1]:+.3f}  U={p[-1,2]:+.3f}")
        print(f"  Expected:        E={exp_p[-1,0]:+.3f}  N={exp_p[-1,1]:+.3f}  U={exp_p[-1,2]:+.3f}")
        print(f"  Pos error:  final={err[-1]:.4f} m  max={np.max(err):.4f} m  mean={np.mean(err):.4f} m")
        print(f"  Vel error:  final={v_err[-1]:.4f} m/s  max={np.max(v_err):.4f} m/s")
        o_err_deg = np.degrees(np.max(np.abs(o_err), axis=0))
        print(f"  Orient err: R={o_err_deg[0]:.3f} deg  P={o_err_deg[1]:.3f} deg  Y={o_err_deg[2]:.3f} deg")

        # Verdicts
        if group == 'ins':
            if np.max(err) < 0.1:
                print("  VERDICT: PASS (< 0.1m error)")
            elif np.max(err) < 1.0:
                print(f"  VERDICT: MARGINAL ({np.max(err):.3f}m error)")
            else:
                print(f"  VERDICT: FAIL ({np.max(err):.1f}m error)")

        if outage is not None:
            t_start = outage['start']
            t_end = t_start + outage['duration']
            sr = len(t) / t[-1]
            i_start = int(t_start * sr)
            i_end = min(int(t_end * sr), len(err) - 1)
            outage_err = err[i_start:i_end]
            post_outage_err = err[min(i_end + int(10*sr), len(err)-1):]
            print(f"  During outage:  max={np.max(outage_err):.2f}m  mean={np.mean(outage_err):.2f}m")
            if len(post_outage_err) > 0:
                print(f"  10s after GPS returns: mean={np.mean(post_outage_err):.2f}m")

        plot_scenario(res, exp, t, title, desc, outage)

    plt.show()


if __name__ == "__main__":
    main()
