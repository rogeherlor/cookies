# -*- coding: utf-8 -*-
"""
Visualize EKF test trajectories.

Run this script to see the estimated vs analytical trajectories for each
test scenario. Uses the same synthetic IMU inputs as the pytest tests.

Usage:
    python visualize_test_trajectories.py                  # show all
    python visualize_test_trajectories.py circle            # show one
    python visualize_test_trajectories.py east north circle # show several
    python visualize_test_trajectories.py --list            # list available scenarios
"""
import sys
import os
import numpy as np
from math import sin, cos, tan, sqrt, pi
from pathlib import Path

# Add EKF source to path
_ekf_src = str(Path(__file__).resolve().parent.parent / "scripts" / "positioning" / "python")
if _ekf_src not in sys.path:
    sys.path.insert(0, _ekf_src)

from data_loader import NavigationData
import ekf_core

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)

# ── Helpers (same as test file) ─────────────────────────────────────

LLA0 = np.array([40.4168, -3.7038, 650.0])
G = 9.81


def _enu_to_lla(e, n, u):
    import pymap3d as pm
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


def _zero_bias_params():
    return dict(
        Qpos=1e-20, Qvel=1e-20,
        QorientXY=1e-20, QorientZ=1e-20,
        Qacc=1e-20, QgyrXY=1e-20, QgyrZ=1e-20,
        Rpos=1.0,
        beta_acc=-1e-12, beta_gyr=-1e-12,
        P_pos_std=1e-6, P_vel_std=1e-6, P_orient_std=1e-6,
        P_acc_std=1e-12, P_gyr_std=1e-12,
    )


def _make_nav(N, sr, accel, gyro):
    return NavigationData(
        accel_flu=accel.astype(np.float64),
        gyro_flu=gyro.astype(np.float64),
        vel_enu=np.zeros((N, 3)),
        lla=np.tile(LLA0, (N, 1)),
        orient=np.zeros((N, 3)),
        gps_available=np.zeros(N, dtype=bool),
        sample_rate=float(sr),
        dataset_name="viz_test",
        gps_rate=1.0,
        lla0=LLA0.copy(),
    )


def _run(accel, gyro, sr=200, use_3d=True):
    N = accel.shape[0]
    nav = _make_nav(N, sr, accel, gyro)
    return ekf_core.run_ekf(nav, _zero_bias_params(), outage_config=None,
                            use_3d_rotation=use_3d)


# ── Expected-state helpers ──────────────────────────────────────────

def _make_expected(pos, vel=None, orient=None, dt=None):
    """Bundle expected arrays into a dict.  Compute velocity from position
    via finite differences if not supplied analytically."""
    N = pos.shape[0]
    if vel is None and dt is not None:
        vel = np.zeros_like(pos)
        vel[1:] = np.diff(pos, axis=0) / dt
    if orient is None:
        orient = np.zeros((N, 3))  # all zeros if not specified
    return dict(pos=pos, vel=vel, orient=orient)


# ── Scenario builders ──────────────────────────────────────────────
# Each returns: (result_dict, expected_dict, t, title, description)
#   expected_dict has keys: pos (Nx3), vel (Nx3), orient (Nx3 rad)

def scenario_straight_east():
    sr, dt, N = 200, 1/200, 1000
    t = np.arange(N) * dt
    accel = np.zeros((N, 3)); accel[:, 2] = G
    accel[t < 2.0, 0] = 1.0
    gyro = np.zeros((N, 3))
    res = _run(accel, gyro, sr)
    # Analytical position & velocity
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    for i, ti in enumerate(t):
        if ti <= 2.0:
            exp_p[i, 0] = 0.5 * ti**2
            exp_v[i, 0] = ti
        else:
            exp_p[i, 0] = 2.0 + 2.0 * (ti - 2.0)
            exp_v[i, 0] = 2.0
    exp_o = np.zeros((N, 3))  # yaw=0 throughout
    return res, _make_expected(exp_p, exp_v, exp_o), t, "Straight East",\
           "1 m/s² forward for 2s then coast. yaw=0 → Forward=East."


def scenario_straight_north():
    sr, dt, N = 200, 1/200, 1000
    t = np.arange(N) * dt
    accel = np.zeros((N, 3)); accel[:, 2] = G
    accel[t < 2.0, 1] = 1.0  # Left = North at yaw=0
    gyro = np.zeros((N, 3))
    res = _run(accel, gyro, sr)
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    for i, ti in enumerate(t):
        if ti <= 2.0:
            exp_p[i, 1] = 0.5 * ti**2
            exp_v[i, 1] = ti
        else:
            exp_p[i, 1] = 2.0 + 2.0 * (ti - 2.0)
            exp_v[i, 1] = 2.0
    exp_o = np.zeros((N, 3))
    return res, _make_expected(exp_p, exp_v, exp_o), t, "Straight North",\
           "1 m/s² body-Left for 2s. At yaw=0, Left=North."


def scenario_straight_up():
    sr, dt, N = 200, 1/200, 1000
    t = np.arange(N) * dt
    accel = np.zeros((N, 3)); accel[:, 2] = G
    accel[t < 2.0, 2] += 1.0
    gyro = np.zeros((N, 3))
    res = _run(accel, gyro, sr)
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    for i, ti in enumerate(t):
        if ti <= 2.0:
            exp_p[i, 2] = 0.5 * ti**2
            exp_v[i, 2] = ti
        else:
            exp_p[i, 2] = 2.0 + 2.0 * (ti - 2.0)
            exp_v[i, 2] = 2.0
    exp_o = np.zeros((N, 3))
    return res, _make_expected(exp_p, exp_v, exp_o), t, "Straight Up",\
           "1 m/s² body-Up for 2s. Gravity cancelled."


def scenario_yaw90_north():
    sr, dt = 200, 1/200
    N = 1000; n_spin = 100
    t = np.arange(N) * dt
    yaw_rate = (pi/2) / (n_spin * dt)
    gyro = np.zeros((N, 3)); gyro[:n_spin, 2] = yaw_rate
    accel = np.zeros((N, 3)); accel[:, 2] = G
    accel[n_spin:n_spin+400, 0] = 1.0
    res = _run(accel, gyro, sr)
    # Analytical
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    exp_o = np.zeros((N, 3))
    v_n = 0.0
    for i in range(N):
        if i < n_spin:
            exp_o[i, 2] = yaw_rate * i * dt
        else:
            exp_o[i, 2] = pi / 2
        if n_spin <= i < n_spin + 400:
            v_n += dt * 1.0
        exp_v[i, 1] = v_n
        if i > 0:
            exp_p[i, 1] = exp_p[i-1, 1] + dt * v_n
    return res, _make_expected(exp_p, exp_v, exp_o), t, "Yaw 90° then North",\
           "Spin yaw to π/2 (Forward=North) then 1 m/s² forward."


def scenario_yaw180_west():
    sr, dt = 200, 1/200
    N = 1000; n_spin = 100
    t = np.arange(N) * dt
    yaw_rate = pi / (n_spin * dt)
    gyro = np.zeros((N, 3)); gyro[:n_spin, 2] = yaw_rate
    accel = np.zeros((N, 3)); accel[:, 2] = G
    accel[n_spin:n_spin+400, 0] = 1.0
    res = _run(accel, gyro, sr)
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    exp_o = np.zeros((N, 3))
    v_e = 0.0
    for i in range(N):
        if i < n_spin:
            exp_o[i, 2] = yaw_rate * i * dt
        else:
            exp_o[i, 2] = pi
        if n_spin <= i < n_spin + 400:
            v_e -= dt * 1.0  # forward at yaw=π → −East
        exp_v[i, 0] = v_e
        if i > 0:
            exp_p[i, 0] = exp_p[i-1, 0] + dt * v_e
    return res, _make_expected(exp_p, exp_v, exp_o), t, "Yaw 180° then West",\
           "Spin yaw to π (Forward=−East) then 1 m/s² forward → moves West."


def scenario_circle():
    v, R, omega = 2.0, 5.0, 2.0/5.0
    T = 2 * pi * R / v
    sr, dt = 200, 1/200
    N = int(T * sr) + 1
    t = np.arange(N) * dt
    accel = np.zeros((N, 3)); accel[:, 2] = G; accel[:, 1] = v**2/R
    gyro = np.zeros((N, 3)); gyro[:, 2] = omega
    n_ramp = 10; accel[:n_ramp, 0] = v / (n_ramp * dt)
    res = _run(accel, gyro, sr)
    exp_p = np.zeros((N, 3))
    exp_p[:, 0] = R * np.sin(omega * t)
    exp_p[:, 1] = R * (1 - np.cos(omega * t))
    exp_v = np.zeros((N, 3))
    exp_v[:, 0] = v * np.cos(omega * t)
    exp_v[:, 1] = v * np.sin(omega * t)
    # During ramp-up, velocity builds from 0 to v
    for i in range(n_ramp):
        ramp_speed = (i + 1) * (v / n_ramp)
        yaw_i = omega * t[i]
        exp_v[i, 0] = ramp_speed * cos(yaw_i)
        exp_v[i, 1] = ramp_speed * sin(yaw_i)
    exp_o = np.zeros((N, 3))
    exp_o[:, 2] = omega * t   # yaw increases linearly
    return res, _make_expected(exp_p, exp_v, exp_o), t, "Full Circle",\
           f"v={v} m/s, R={R} m, ω={omega:.2f} rad/s. Should return to origin."


def scenario_deceleration():
    sr, dt, N = 200, 1/200, 800
    t = np.arange(N) * dt
    accel = np.zeros((N, 3)); accel[:, 2] = G
    accel[t < 2.0, 0] = 1.0; accel[t >= 2.0, 0] = -1.0
    gyro = np.zeros((N, 3))
    res = _run(accel, gyro, sr)
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    for i, ti in enumerate(t):
        if ti <= 2.0:
            exp_p[i, 0] = 0.5 * ti**2
            exp_v[i, 0] = ti
        else:
            tp = ti - 2.0
            exp_p[i, 0] = 2.0 + 2.0 * tp - 0.5 * tp**2
            exp_v[i, 0] = max(2.0 - tp, 0.0)  # stops at t=4s
    exp_o = np.zeros((N, 3))
    return res, _make_expected(exp_p, exp_v, exp_o), t, "Accelerate-Decelerate",\
           "1 m/s² East for 2s, then −1 m/s² for 2s. Should stop at 4 m."


def scenario_s_curve():
    sr, dt = 200, 1/200
    v, omega = 2.0, 0.4
    turn_time = (pi/2) / omega
    n_straight = int(1.0 * sr); n_turn = int(turn_time * sr)
    N = 3 * n_straight + 2 * n_turn
    t = np.arange(N) * dt
    accel = np.zeros((N, 3)); accel[:, 2] = G
    gyro = np.zeros((N, 3))
    n_ramp = 10; accel[:n_ramp, 0] = v / (n_ramp * dt)
    centripetal = v * omega
    s1 = n_straight; t1 = s1 + n_turn; s2 = t1 + n_straight; t2 = s2 + n_turn
    gyro[s1:t1, 2] = omega; accel[s1:t1, 1] = centripetal
    gyro[s2:t2, 2] = -omega; accel[s2:t2, 1] = -centripetal
    res = _run(accel, gyro, sr)
    # Analytical via numerical integration
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    exp_o = np.zeros((N, 3))
    yaw_a = 0.0
    for i in range(N):
        if s1 <= i < t1:
            yaw_a += omega * dt if i > 0 else 0.0
        elif s2 <= i < t2:
            yaw_a -= omega * dt if i > 0 else 0.0
        exp_o[i, 2] = yaw_a
        vx = v * cos(yaw_a); vy = v * sin(yaw_a)
        exp_v[i, 0] = vx; exp_v[i, 1] = vy
        if i > 0:
            exp_p[i, 0] = exp_p[i-1, 0] + vx * dt
            exp_p[i, 1] = exp_p[i-1, 1] + vy * dt
    # First few samples have ramp-up, set velocity correctly
    for i in range(n_ramp):
        ramp_v = (i + 1) * (v / n_ramp)
        exp_v[i, 0] = ramp_v
    return res, _make_expected(exp_p, exp_v, exp_o), t, "S-Curve",\
           "Straight → left 90° → straight → right 90° → straight."


def scenario_pitched_climb():
    sr, dt, N = 200, 1/200, 600
    t = np.arange(N) * dt
    n_spin = 100; pitch_target = pi/6
    pitch_rate = pitch_target / (n_spin * dt)
    gyro = np.zeros((N, 3)); gyro[:n_spin, 1] = pitch_rate
    accel = np.zeros((N, 3))
    for i in range(N):
        pitch_i = min(pitch_rate * i * dt, pitch_target) if i < n_spin else pitch_target
        thrust = 2.0 if i >= n_spin else 0.0
        Rbn_i = _Ry(pitch_i)
        grav_body = Rbn_i.T @ np.array([0, 0, G])
        accel[i] = np.array([thrust, 0, 0]) + grav_body
    res = _run(accel, gyro, sr)
    # Analytical: after spin-up, 2 m/s² at 30° pitch
    exp_p = np.zeros((N, 3))
    exp_v = np.zeros((N, 3))
    exp_o = np.zeros((N, 3))
    ve, vu = 0.0, 0.0
    for i in range(N):
        pitch_i = min(pitch_rate * i * dt, pitch_target) if i < n_spin else pitch_target
        exp_o[i, 1] = pitch_i
        if i >= n_spin:
            ve += dt * 2.0 * cos(pitch_target)
            vu += dt * 2.0 * sin(pitch_target)
        exp_v[i, 0] = ve; exp_v[i, 2] = vu
        if i > 0:
            exp_p[i, 0] = exp_p[i-1, 0] + ve * dt
            exp_p[i, 2] = exp_p[i-1, 2] + vu * dt
    return res, _make_expected(exp_p, exp_v, exp_o), t, "Pitched Climb (30°)",\
           "Pitch nose-up 30° then 2 m/s² forward → East + Up."


def scenario_axis_mapping():
    """Three short runs: thrust along each body axis at yaw=0."""
    sr, dt, N = 200, 1/200, 200
    t = np.arange(N) * dt
    results = []
    labels = ["Forward→East", "Left→North", "Up→Up"]
    for ax in range(3):
        accel = np.zeros((N, 3)); accel[:, 2] = G; accel[:, ax] += 1.0
        gyro = np.zeros((N, 3))
        results.append(_run(accel, gyro, sr))
    return results, t, labels


# ── Plotting ────────────────────────────────────────────────────────

def _enforce_min_span(ax, axis='both', min_span=None):
    """Ensure axis limits have at least `min_span` range, centred on current
    data.  Prevents tiny values (1e-5) from distorting the view.
    Supports 'x', 'y', 'z' (for 3D axes), or 'both' (x+y), or 'all' (x+y+z)."""
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
        elif a == 'z' and hasattr(ax, 'get_zlim'):
            lo, hi = ax.get_zlim()
        else:
            continue
        span = hi - lo
        if span < min_span:
            mid = (lo + hi) / 2
            new_lo, new_hi = mid - min_span / 2, mid + min_span / 2
            if a == 'x':
                ax.set_xlim(new_lo, new_hi)
            elif a == 'y':
                ax.set_ylim(new_lo, new_hi)
            elif a == 'z' and hasattr(ax, 'set_zlim'):
                ax.set_zlim(new_lo, new_hi)


def plot_trajectory_2d(res, exp, t, title, desc):
    """Plot estimated vs expected trajectory with 9 panels:
    E-N trajectory, position vs time, position error,
    velocity vs time, velocity error, speed,
    orientation vs time, orientation error, 3D trajectory."""
    p = res['p']
    v = res['v']
    r = res['r']
    exp_p = exp['pos']
    exp_v = exp['vel']
    exp_o = exp['orient']

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    fig.suptitle(f"{title}\n{desc}", fontsize=12, fontweight='bold')

    # Use GridSpec for 3 rows × 3 cols layout
    gs = fig.add_gridspec(3, 3)

    # ── Row 0 ──────────────────────────────────────────────────────

    # 1) E-N trajectory
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(exp_p[:, 0], exp_p[:, 1], 'g--', lw=2, label='Expected')
    ax.plot(p[:, 0], p[:, 1], 'b-', lw=1, label='EKF estimated')
    ax.plot(p[0, 0], p[0, 1], 'ko', ms=8, label='Start')
    ax.plot(p[-1, 0], p[-1, 1], 'r*', ms=12, label='End')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    ax.set_title('Trajectory (E-N)'); ax.legend(fontsize=7); ax.grid(True)
    # Enforce minimum visible range so near-perfect tracks aren't squished
    _enforce_min_span(ax, 'both', min_span=1.0)
    ax.set_aspect('equal')

    # 2) Position components vs time
    ax = fig.add_subplot(gs[0, 1])
    for i, (lbl, c) in enumerate(zip(['East', 'North', 'Up'], ['r', 'g', 'b'])):
        ax.plot(t, exp_p[:, i], c + '--', lw=1.5, label=f'{lbl} exp')
        ax.plot(t, p[:, i], c + '-', lw=1, label=f'{lbl} EKF')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Position [m]')
    ax.set_title('Position vs Time'); ax.legend(fontsize=6); ax.grid(True)
    _enforce_min_span(ax, 'y', min_span=1.0)

    # 3) Position error
    ax = fig.add_subplot(gs[0, 2])
    p_err = p - exp_p
    for i, (lbl, c) in enumerate(zip(['East', 'North', 'Up'], ['r', 'g', 'b'])):
        ax.plot(t, p_err[:, i], c + '-', lw=1, label=lbl)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error [m]')
    ax.set_title('Position Error (est − exp)'); ax.legend(); ax.grid(True)
    _enforce_min_span(ax, 'y', min_span=0.1)

    # ── Row 1 ──────────────────────────────────────────────────────

    # 4) Velocity vs time
    ax = fig.add_subplot(gs[1, 0])
    for i, (lbl, c) in enumerate(zip(['v_E', 'v_N', 'v_U'], ['r', 'g', 'b'])):
        ax.plot(t, exp_v[:, i], c + '--', lw=1.5, label=f'{lbl} exp')
        ax.plot(t, v[:, i], c + '-', lw=1, label=f'{lbl} EKF')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Velocity vs Time'); ax.legend(fontsize=6); ax.grid(True)
    _enforce_min_span(ax, 'y', min_span=0.5)

    # 5) Velocity error
    ax = fig.add_subplot(gs[1, 1])
    v_err = v - exp_v
    for i, (lbl, c) in enumerate(zip(['v_E', 'v_N', 'v_U'], ['r', 'g', 'b'])):
        ax.plot(t, v_err[:, i], c + '-', lw=1, label=lbl)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error [m/s]')
    ax.set_title('Velocity Error (est − exp)'); ax.legend(); ax.grid(True)
    _enforce_min_span(ax, 'y', min_span=0.1)

    # 6) Speed comparison
    ax = fig.add_subplot(gs[1, 2])
    speed_est = np.linalg.norm(v, axis=1)
    speed_exp = np.linalg.norm(exp_v, axis=1)
    ax.plot(t, speed_exp, 'g--', lw=1.5, label='Expected |v|')
    ax.plot(t, speed_est, 'b-', lw=1, label='EKF |v|')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Speed [m/s]')
    ax.set_title('Speed Magnitude'); ax.legend(); ax.grid(True)
    _enforce_min_span(ax, 'y', min_span=0.5)

    # ── Row 2 ──────────────────────────────────────────────────────

    # 7) Orientation vs time
    ax = fig.add_subplot(gs[2, 0])
    for i, (lbl, c) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], ['r', 'g', 'b'])):
        ax.plot(t, np.degrees(exp_o[:, i]), c + '--', lw=1.5, label=f'{lbl} exp')
        ax.plot(t, np.degrees(r[:, i]), c + '-', lw=1, label=f'{lbl} EKF')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Angle [°]')
    ax.set_title('Orientation vs Time'); ax.legend(fontsize=6); ax.grid(True)
    _enforce_min_span(ax, 'y', min_span=5.0)

    # 8) Orientation error
    ax = fig.add_subplot(gs[2, 1])
    o_err = r - exp_o
    # Wrap angle differences to [-π, π]
    o_err = (o_err + pi) % (2 * pi) - pi
    for i, (lbl, c) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], ['r', 'g', 'b'])):
        ax.plot(t, np.degrees(o_err[:, i]), c + '-', lw=1, label=lbl)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Error [°]')
    ax.set_title('Orientation Error (est − exp)'); ax.legend(); ax.grid(True)
    _enforce_min_span(ax, 'y', min_span=1.0)

    # 9) 3D trajectory
    ax = fig.add_subplot(gs[2, 2], projection='3d')
    ax.plot(exp_p[:, 0], exp_p[:, 1], exp_p[:, 2], 'g--', lw=2, label='Expected')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-', lw=1, label='EKF')
    ax.set_xlabel('East'); ax.set_ylabel('North'); ax.set_zlabel('Up')
    ax.set_title('3D Trajectory'); ax.legend()
    _enforce_min_span(ax, 'all', min_span=1.0)

    return fig


def plot_axis_mapping(results, t, labels):
    """Plot axis mapping test: 3 subplots, one per body axis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Axis Mapping at yaw=0: 1 m/s² along each body axis",
                 fontsize=12, fontweight='bold')

    enu_labels = ['East', 'North', 'Up']
    colors = ['r', 'g', 'b']

    for ax_idx, (res, label) in enumerate(zip(results, labels)):
        ax = axes[ax_idx]
        p = res['p']
        for i in range(3):
            ax.plot(t, p[:, i], colors[i] + '-', lw=1.5, label=enu_labels[i])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [m]')
        ax.set_title(f'Body {label}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    return fig


# ── Scenario registry ──────────────────────────────────────────────

SCENARIOS = {
    'east':      scenario_straight_east,
    'north':     scenario_straight_north,
    'up':        scenario_straight_up,
    'yaw90':     scenario_yaw90_north,
    'yaw180':    scenario_yaw180_west,
    'circle':    scenario_circle,
    'decel':     scenario_deceleration,
    's_curve':   scenario_s_curve,
    'pitch':     scenario_pitched_climb,
    'axes':      scenario_axis_mapping,
}


def main():
    args = sys.argv[1:]

    if '--list' in args:
        print("Available scenarios:")
        for name in SCENARIOS:
            print(f"  {name}")
        return

    if not args:
        selected = list(SCENARIOS.keys())
    else:
        selected = [a for a in args if a in SCENARIOS]
        unknown = [a for a in args if a not in SCENARIOS]
        if unknown:
            print(f"Unknown scenarios: {unknown}")
            print(f"Available: {list(SCENARIOS.keys())}")
            return

    print(f"Running {len(selected)} scenario(s): {', '.join(selected)}")

    for name in selected:
        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}")

        if name == 'axes':
            results, t, labels = scenario_axis_mapping()
            plot_axis_mapping(results, t, labels)
        else:
            res, exp, t, title, desc = SCENARIOS[name]()
            p = res['p']
            exp_p = exp['pos']
            err = np.linalg.norm(p - exp_p, axis=1)
            v = res['v']
            exp_v = exp['vel']
            r = res['r']
            exp_o = exp['orient']
            print(f"  {desc}")
            print(f"  Final position:  E={p[-1,0]:+.3f}  N={p[-1,1]:+.3f}  U={p[-1,2]:+.3f}")
            print(f"  Expected:        E={exp_p[-1,0]:+.3f}  N={exp_p[-1,1]:+.3f}  U={exp_p[-1,2]:+.3f}")
            print(f"  Position error:  {err[-1]:.4f} m  (max {np.max(err):.4f} m)")
            v_err = np.linalg.norm(v - exp_v, axis=1)
            print(f"  Velocity error:  {v_err[-1]:.4f} m/s  (max {np.max(v_err):.4f} m/s)")
            o_err = r - exp_o
            o_err = (o_err + pi) % (2 * pi) - pi
            o_err_deg = np.degrees(np.max(np.abs(o_err), axis=0))
            print(f"  Orient error:    R={o_err_deg[0]:.3f}°  P={o_err_deg[1]:.3f}°  Y={o_err_deg[2]:.3f}°")
            plot_trajectory_2d(res, exp, t, title, desc)

    plt.show()


if __name__ == "__main__":
    main()
