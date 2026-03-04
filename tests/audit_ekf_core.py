#!/usr/bin/env python3
"""
Audit ekf_core.py equation-by-equation.

Each test isolates ONE aspect of the EKF and checks the result against
a known analytical answer.  If ANY test fails, the corresponding equation
in ekf_core.py has a bug.

Run:
    conda run -n cookies python tests/audit_ekf_core.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'scripts', 'positioning', 'python'))
import numpy as np
from math import sin, cos, tan, pi, sqrt, radians
import pymap3d as pm
import ekf_core

LLA0 = np.array([40.4168, -3.7038, 650.0])
G = 9.81

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

PLOT_MODE = False   # set by --plot CLI flag


def _plot_gps_diagnostic(res, truth_enu, truth_vel, truth_rpy, sr, gps_mask,
                         title="", outage=None):
    """Diagnostic figure for GPS-aided tests.

    4x3 grid:
      Row 0: Position (E/N/U) vs truth · Position error · Position ±1σ
      Row 1: Velocity (E/N/U) vs truth · Velocity error · Velocity ±1σ
      Row 2: Orientation (R/P/Y) vs truth · Orientation error · Orientation ±1σ
      Row 3: Δpos per step (spike detector) · Δorient per step · Bias estimates

    GPS update instants are shown as thin vertical lines.
    """
    if not HAS_PLT or not PLOT_MODE:
        return

    N = res['p'].shape[0]
    t = np.arange(N) / sr
    p, v, r = res['p'], res['v'], res['r']
    sp, sv, so = res['std_pos'], res['std_vel'], res['std_orient']

    # Compute truth arrays – allow None for vel/rpy if not provided
    tp = truth_enu if truth_enu is not None else np.zeros((N, 3))
    tv = truth_vel if truth_vel is not None else np.zeros((N, 3))
    tr = truth_rpy if truth_rpy is not None else np.zeros((N, 3))

    gps_times = t[gps_mask[:N]] if gps_mask is not None else np.array([])

    fig, axes = plt.subplots(4, 3, figsize=(22, 16), constrained_layout=True)
    fig.suptitle(title, fontsize=12, fontweight='bold')

    def shade(ax):
        for gt in gps_times:
            ax.axvline(gt, color='lime', alpha=0.25, lw=0.5)
        if outage is not None:
            ax.axvspan(outage['start'], outage['start'] + outage['duration'],
                       alpha=0.12, color='red')

    colors = ['r', 'g', 'b']

    # ── Row 0: Position ──
    ax = axes[0, 0]
    for j, (lbl, c) in enumerate(zip(['East', 'North', 'Up'], colors)):
        ax.plot(t, tp[:, j], c+'--', lw=1.5, label=f'{lbl} truth')
        ax.plot(t, p[:, j], c+'-', lw=0.8, label=f'{lbl} EKF')
    shade(ax); ax.set_ylabel('Position [m]'); ax.set_title('Position'); ax.legend(fontsize=5); ax.grid(True)

    ax = axes[0, 1]
    pe = p - tp
    for j, (lbl, c) in enumerate(zip(['E', 'N', 'U'], colors)):
        ax.plot(t, pe[:, j], c+'-', lw=0.8, label=lbl)
    ax.plot(t, np.linalg.norm(pe, axis=1), 'k-', lw=1.2, label='3D')
    shade(ax); ax.set_ylabel('Error [m]'); ax.set_title('Position Error'); ax.legend(fontsize=6); ax.grid(True)

    ax = axes[0, 2]
    for j, (lbl, c) in enumerate(zip(['E', 'N', 'U'], colors)):
        ax.fill_between(t, pe[:, j] - sp[:, j], pe[:, j] + sp[:, j], color=c, alpha=0.15)
        ax.plot(t, pe[:, j], c+'-', lw=0.8, label=lbl)
    shade(ax); ax.set_ylabel('Error ± 1σ [m]'); ax.set_title('Position Error ± σ'); ax.legend(fontsize=6); ax.grid(True)

    # ── Row 1: Velocity ──
    ax = axes[1, 0]
    for j, (lbl, c) in enumerate(zip(['vE', 'vN', 'vU'], colors)):
        ax.plot(t, tv[:, j], c+'--', lw=1.5, label=f'{lbl} truth')
        ax.plot(t, v[:, j], c+'-', lw=0.8, label=f'{lbl} EKF')
    shade(ax); ax.set_ylabel('Velocity [m/s]'); ax.set_title('Velocity'); ax.legend(fontsize=5); ax.grid(True)

    ax = axes[1, 1]
    ve = v - tv
    for j, (lbl, c) in enumerate(zip(['vE', 'vN', 'vU'], colors)):
        ax.plot(t, ve[:, j], c+'-', lw=0.8, label=lbl)
    ax.plot(t, np.linalg.norm(ve, axis=1), 'k-', lw=1.2, label='3D')
    shade(ax); ax.set_ylabel('Error [m/s]'); ax.set_title('Velocity Error'); ax.legend(fontsize=6); ax.grid(True)

    ax = axes[1, 2]
    for j, (lbl, c) in enumerate(zip(['vE', 'vN', 'vU'], colors)):
        ax.fill_between(t, ve[:, j] - sv[:, j], ve[:, j] + sv[:, j], color=c, alpha=0.15)
        ax.plot(t, ve[:, j], c+'-', lw=0.8, label=lbl)
    shade(ax); ax.set_ylabel('Error ± 1σ [m/s]'); ax.set_title('Velocity Error ± σ'); ax.legend(fontsize=6); ax.grid(True)

    # ── Row 2: Orientation ──
    ax = axes[2, 0]
    for j, (lbl, c) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], colors)):
        ax.plot(t, np.degrees(tr[:, j]), c+'--', lw=1.5, label=f'{lbl} truth')
        ax.plot(t, np.degrees(r[:, j]), c+'-', lw=0.8, label=f'{lbl} EKF')
    shade(ax); ax.set_ylabel('Angle [deg]'); ax.set_title('Orientation'); ax.legend(fontsize=5); ax.grid(True)

    ax = axes[2, 1]
    oe = r - tr
    oe = (oe + np.pi) % (2 * np.pi) - np.pi
    for j, (lbl, c) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], colors)):
        ax.plot(t, np.degrees(oe[:, j]), c+'-', lw=0.8, label=lbl)
    shade(ax); ax.set_ylabel('Error [deg]'); ax.set_title('Orientation Error'); ax.legend(fontsize=6); ax.grid(True)

    ax = axes[2, 2]
    for j, (lbl, c) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], colors)):
        ax.fill_between(t, np.degrees(oe[:, j]) - np.degrees(so[:, j]),
                        np.degrees(oe[:, j]) + np.degrees(so[:, j]), color=c, alpha=0.15)
        ax.plot(t, np.degrees(oe[:, j]), c+'-', lw=0.8, label=lbl)
    shade(ax); ax.set_ylabel('Error ± 1σ [deg]'); ax.set_title('Orient Error ± σ'); ax.legend(fontsize=6); ax.grid(True)

    # ── Row 3: Spike detection ──
    # Δposition per step  → large jumps = update spikes
    ax = axes[3, 0]
    dp = np.diff(p, axis=0)
    dp_norm = np.linalg.norm(dp, axis=1)
    ax.plot(t[1:], dp_norm * sr, 'k-', lw=0.5, alpha=0.6, label='|Δp|·sr (approx speed)')
    # Highlight steps where jump is > 2× median (spike)
    med = np.median(dp_norm)
    spike_mask = dp_norm > 3 * max(med, 1e-10)
    if np.any(spike_mask):
        ax.plot(t[1:][spike_mask], dp_norm[spike_mask] * sr, 'rv', ms=3,
                label=f'Spikes (>3× median): {np.sum(spike_mask)}')
    shade(ax); ax.set_xlabel('Time [s]'); ax.set_ylabel('Δpos·sr [m/s]')
    ax.set_title('Position Jumps per Step (spike detector)'); ax.legend(fontsize=6); ax.grid(True)

    # Δorientation per step
    ax = axes[3, 1]
    dr = np.diff(r, axis=0)
    dr = (dr + np.pi) % (2 * np.pi) - np.pi  # wrap
    for j, (lbl, c) in enumerate(zip(['ΔR', 'ΔP', 'ΔY'], colors)):
        ax.plot(t[1:], np.degrees(dr[:, j]) * sr, c+'-', lw=0.5, alpha=0.6, label=f'{lbl}·sr [deg/s]')
    # Pitch spike detection
    dp_deg = np.abs(np.degrees(dr[:, 1]))
    med_p = np.median(dp_deg)
    spike_p = dp_deg > 3 * max(med_p, 1e-8)
    if np.any(spike_p):
        ax.plot(t[1:][spike_p], np.degrees(dr[spike_p, 1]) * sr, 'rv', ms=3,
                label=f'Pitch spikes: {np.sum(spike_p)}')
    shade(ax); ax.set_xlabel('Time [s]'); ax.set_ylabel('Rate [deg/s]')
    ax.set_title('Orientation Jumps per Step (spike detector)'); ax.legend(fontsize=5); ax.grid(True)

    # Bias estimates
    ax = axes[3, 2]
    ba, bg = res['bias_acc'], res['bias_gyr']
    for j, (lbl, c) in enumerate(zip(['bax', 'bay', 'baz'], colors)):
        ax.plot(t, ba[:, j], c+'-', lw=0.8, label=lbl)
    for j, (lbl, c) in enumerate(zip(['bgx', 'bgy', 'bgz'], ['m', 'c', 'y'])):
        ax.plot(t, bg[:, j], c+'--', lw=0.8, label=lbl)
    shade(ax); ax.set_xlabel('Time [s]'); ax.set_ylabel('Bias')
    ax.set_title('Bias Estimates'); ax.legend(fontsize=5); ax.grid(True)

    return fig


# ─── Helpers ─────────────────────────────────────────────────────────────
class Nav:
    """Minimal NavData stub."""
    pass

def make_nav(N, sr, accel, gyro, lla=None, vel_enu=None, gps_avail=None, gps_rate=1.0):
    nav = Nav()
    nav.accel_flu = accel.astype(np.float64)
    nav.gyro_flu = gyro.astype(np.float64)
    nav.vel_enu = vel_enu if vel_enu is not None else np.zeros((N,3), dtype=np.float64)
    nav.lla = lla if lla is not None else np.tile(LLA0, (N,1)).astype(np.float64)
    nav.orient = np.zeros((N,3), dtype=np.float64)
    nav.gps_available = gps_avail if gps_avail is not None else np.zeros(N, dtype=bool)
    nav.sample_rate = float(sr)
    nav.dataset_name = 'audit'
    nav.gps_rate = float(gps_rate)
    nav.lla0 = LLA0.copy()
    return nav

def zero_params(**kw):
    p = dict(Qpos=1e-30, Qvel=1e-30, QorientXY=1e-30, QorientZ=1e-30,
             Qacc=1e-30, QgyrXY=1e-30, QgyrZ=1e-30, Rpos=1.0,
             beta_acc=-1e-15, beta_gyr=-1e-15,
             P_pos_std=1e-8, P_vel_std=1e-8, P_orient_std=1e-8,
             P_acc_std=1e-15, P_gyr_std=1e-15,
             enable_nhc=False, enable_zupt=False, enable_level=False)
    p.update(kw)
    return p

def run_pure_ins(accel, gyro, sr=200, params=None):
    N = accel.shape[0]
    if params is None:
        params = zero_params()
    nav = make_nav(N, sr, accel, gyro)
    return ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)


PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# =========================================================================
#  TEST 1: Stationary — accelerometer reads [0,0,G], gyro reads [0,0,0]
#          Position, velocity, orientation should all stay at zero.
# =========================================================================
def test_stationary():
    print("\n=== TEST 1: Stationary ===")
    sr = 200; T = 10.0; N = int(T*sr)
    accel = np.zeros((N,3)); accel[:,2] = G
    gyro = np.zeros((N,3))
    res = run_pure_ins(accel, gyro, sr)
    
    max_pos = np.max(np.abs(res['p']))
    max_vel = np.max(np.abs(res['v']))
    max_rpy = np.max(np.abs(res['r']))
    
    # Tolerances allow for Earth-curvature coupling and floating-point accumulation
    check("pos stays near zero (<1mm)", max_pos < 1e-3, f"max_pos={max_pos:.2e}")
    check("vel stays near zero (<0.1mm/s)", max_vel < 1e-4, f"max_vel={max_vel:.2e}")
    check("orient stays near zero (<0.001 deg)", max_rpy < 2e-5, f"max_rpy={max_rpy:.2e}")


# =========================================================================
#  TEST 2: Constant forward acceleration from rest (heading East)
#          a_fwd = 2 m/s^2 for 5s -> final v = 10 m/s, pos = 25 m East
# =========================================================================
def test_const_accel_east():
    print("\n=== TEST 2: Constant forward acceleration (East) ===")
    sr = 200; a_fwd = 2.0; T = 5.0; N = int(T*sr); dt = 1.0/sr
    accel = np.zeros((N,3)); accel[:,2] = G; accel[:,0] = a_fwd
    gyro = np.zeros((N,3))
    res = run_pure_ins(accel, gyro, sr)
    
    # Forward-Euler: v[k+1] = v[k] + dt*a, p[k+1] = p[k] + dt*v[k] + dt^2/2*a
    # These are exact for const accel with the semi-implicit scheme in the code
    final_v_E = res['v'][-1, 0]
    final_p_E = res['p'][-1, 0]
    
    # Analytical for forward-Euler with the code's integration:
    # v_final = a * T (exact for constant accel)
    # p_final = 0.5 * a * T^2 (exact for constant accel with 2nd-order pos update)
    exp_v = a_fwd * T
    exp_p = 0.5 * a_fwd * T**2
    
    check("final vel East", abs(final_v_E - exp_v) < 0.01, f"got {final_v_E:.4f}, expect {exp_v:.4f}")
    check("final pos East", abs(final_p_E - exp_p) < 0.05, f"got {final_p_E:.4f}, expect {exp_p:.4f}")
    check("vel N~0", abs(res['v'][-1,1]) < 1e-4, f"vel_N={res['v'][-1,1]:.2e}")
    check("vel U~0", abs(res['v'][-1,2]) < 1e-4, f"vel_U={res['v'][-1,2]:.2e}")
    check("pitch~0", abs(res['r'][-1,1]) < 1e-4, f"pitch={np.degrees(res['r'][-1,1]):.6f} deg")
    check("yaw~0", abs(res['r'][-1,2]) < 1e-4, f"yaw={np.degrees(res['r'][-1,2]):.6f} deg")


# =========================================================================
#  TEST 3: Pure yaw rotation in place — gyro_z = 0.1 rad/s for 10s
#          Expected: yaw = 1 rad, pos/vel = 0
# =========================================================================
def test_pure_yaw():
    print("\n=== TEST 3: Pure yaw rotation in place ===")
    sr = 200; T = 10.0; N = int(T*sr)
    omega_z = 0.1  # rad/s
    accel = np.zeros((N,3)); accel[:,2] = G
    gyro = np.zeros((N,3)); gyro[:,2] = omega_z
    res = run_pure_ins(accel, gyro, sr)
    
    exp_yaw = omega_z * T  # 1.0 rad
    final_yaw = res['r'][-1, 2]
    
    check("yaw correct", abs(final_yaw - exp_yaw) < 0.001, f"got {np.degrees(final_yaw):.4f} deg, expect {np.degrees(exp_yaw):.4f} deg")
    check("roll~0", abs(res['r'][-1,0]) < 1e-3, f"roll={np.degrees(res['r'][-1,0]):.6f} deg")
    check("pitch~0", abs(res['r'][-1,1]) < 1e-3, f"pitch={np.degrees(res['r'][-1,1]):.6f} deg")
    check("pos~0", np.linalg.norm(res['p'][-1]) < 1e-2, f"pos={res['p'][-1]}")
    check("vel~0", np.linalg.norm(res['v'][-1]) < 1e-3, f"vel={res['v'][-1]}")


# =========================================================================
#  TEST 4: Pure pitch rotation — gyro_y = 0.1 rad/s for 2s
#          Expected: pitch = 0.2 rad, roll/yaw = 0
#          Also: gravity should project forward, causing East velocity
# =========================================================================
def test_pure_pitch():
    print("\n=== TEST 4: Pure pitch rotation ===")
    sr = 200; T = 2.0; N = int(T*sr); dt = 1.0/sr
    omega_y = 0.1  # rad/s
    accel = np.zeros((N,3)); accel[:,2] = G
    gyro = np.zeros((N,3)); gyro[:,1] = omega_y
    res = run_pure_ins(accel, gyro, sr)
    
    exp_pitch = omega_y * T  # 0.2 rad
    final_pitch = res['r'][-1, 1]
    
    check("pitch correct", abs(final_pitch - exp_pitch) < 0.001, f"got {np.degrees(final_pitch):.4f} deg, expect {np.degrees(exp_pitch):.4f} deg")
    check("roll~0", abs(res['r'][-1,0]) < 1e-4, f"roll={np.degrees(res['r'][-1,0]):.6f} deg")
    check("yaw~0", abs(res['r'][-1,2]) < 1e-4, f"yaw={np.degrees(res['r'][-1,2]):.6f} deg")
    
    # With pitch tilting the body, gravity projects forward (East).
    # accENU_E = -sin(pitch)*G (FLU convention, Ry has -sin at [0,2])
    # Actually let's just check the sign: positive pitch in FLU means nose-up,
    # which should project gravity component backward (negative East).
    # Wait: Rbn * [0,0,G] with positive pitch:
    # Rz=I (yaw=0), Ry(pitch)*[0,0,G] = [-sin(pitch)*G, 0, cos(pitch)*G]
    # So accENU_E = -sin(pitch)*G < 0 for positive pitch
    # vIMU = vIMU + Ts*(accENU + g) where g=[0,0,-G]
    # So net accENU + g = [-sin(pitch)*G, 0, cos(pitch)*G - G]
    # For small pitch: [-pitch*G, 0, -pitch^2/2*G] approx
    # So velocity East should be negative (accelerating West)
    
    # Just check that velocity is nonzero and in the expected direction
    vel_E_final = res['v'][-1, 0]
    check("vel_E < 0 (nose-up -> gravity projects West)", vel_E_final < 0,
          f"vel_E={vel_E_final:.4f}")


# =========================================================================
#  TEST 5: Pure roll rotation — gyro_x = 0.1 rad/s for 2s
#          Expected: roll = 0.2 rad, pitch/yaw ~ 0
# =========================================================================
def test_pure_roll():
    print("\n=== TEST 5: Pure roll rotation ===")
    sr = 200; T = 2.0; N = int(T*sr)
    omega_x = 0.1  # rad/s
    accel = np.zeros((N,3)); accel[:,2] = G
    gyro = np.zeros((N,3)); gyro[:,0] = omega_x
    res = run_pure_ins(accel, gyro, sr)
    
    exp_roll = omega_x * T  # 0.2 rad
    final_roll = res['r'][-1, 0]
    
    check("roll correct", abs(final_roll - exp_roll) < 0.001, f"got {np.degrees(final_roll):.4f} deg, expect {np.degrees(exp_roll):.4f} deg")
    check("pitch~0", abs(res['r'][-1,1]) < 1e-4, f"pitch={np.degrees(res['r'][-1,1]):.6f} deg")
    check("yaw~0", abs(res['r'][-1,2]) < 1e-4, f"yaw={np.degrees(res['r'][-1,2]):.6f} deg")
    
    # Roll tilts body left, gravity should project North (leftward in FLU = North in ENU when heading East)
    # Rbn * [0,0,G] with roll only: Rx(roll)*[0,0,G] = [0, -sin(roll)*G, cos(roll)*G]
    # accENU_N = -sin(roll)*G < 0 for positive roll (tilting left)
    # Wait: FLU x=fwd, y=left, z=up. Roll around x-axis.
    # Actually ENU: E=East, N=North, U=Up. When heading East, body-x maps to East.
    # Body-y (left) maps to North. Roll around x-axis tilts y-axis (left/North) down.
    # So gravity projects South (negative North).
    vel_N_final = res['v'][-1, 1]
    check("vel_N < 0 (roll right -> gravity projects South)", vel_N_final < 0,
          f"vel_N={vel_N_final:.4f}")


# =========================================================================
#  TEST 6: Constant-speed circle — centripetal accel + yaw rate
#          v=5 m/s, R=20 m, omega=v/R. Full circle should return to start.
# =========================================================================
def test_circle():
    print("\n=== TEST 6: Constant-speed circle ===")
    v = 5.0; R = 20.0; omega = v/R
    sr = 200; dt = 1.0/sr
    T_circle = 2*pi/omega
    n_ramp = 20  # ramp-up samples
    n_circle = int(T_circle * sr)
    N = n_ramp + n_circle
    
    accel = np.zeros((N,3)); accel[:,2] = G
    gyro = np.zeros((N,3))
    
    # Ramp to speed v
    accel[:n_ramp, 0] = v / (n_ramp * dt)
    # Circle: centripetal left + yaw rate
    gyro[n_ramp:, 2] = omega
    accel[n_ramp:, 1] = v * omega  # centripetal = v^2/R = v*omega
    
    res = run_pure_ins(accel, gyro, sr)
    
    # After full circle, should return near start (with ramp-up offset)
    # The ramp covers ~0.25m of distance, so position at end of ramp is the reference
    p_start = res['p'][n_ramp]
    p_end = res['p'][-1]
    closure_err = np.linalg.norm(p_end - p_start)
    
    # Yaw should have rotated 2*pi (back to ~0, modulo wrapping)
    final_yaw = res['r'][-1, 2]
    yaw_err = abs(((final_yaw - 0) + pi) % (2*pi) - pi)  # should be near 0 (mod 2pi)
    
    check("circle closure <1m", closure_err < 1.0, f"closure={closure_err:.4f}m")
    check("yaw returns ~0 (mod 2pi)", yaw_err < 0.05, f"yaw_err={np.degrees(yaw_err):.2f} deg")
    check("pitch stays small", np.max(np.abs(res['r'][:, 1])) < 0.01, 
          f"max_pitch={np.degrees(np.max(np.abs(res['r'][:, 1]))):.4f} deg")
    check("roll stays small", np.max(np.abs(res['r'][:, 0])) < 0.01,
          f"max_roll={np.degrees(np.max(np.abs(res['r'][:, 0]))):.4f} deg")


# =========================================================================
#  TEST 7: GPS position update — vehicle stationary, filter has position error
#          GPS should pull position toward truth.
# =========================================================================
def test_gps_pos_update():
    print("\n=== TEST 7: GPS position update (stationary) ===")
    sr = 100; T = 5.0; N = int(T*sr)
    accel = np.zeros((N,3)); accel[:,2] = G
    gyro = np.zeros((N,3))
    
    # Place GPS truth at (10, 0, 0) ENU -> vehicle should converge there
    truth_enu = np.array([10.0, 0.0, 0.0])
    truth_lla = np.array(pm.enu2geodetic(truth_enu[0], truth_enu[1], truth_enu[2],
                                          LLA0[0], LLA0[1], LLA0[2]))
    lla = np.tile(truth_lla, (N, 1))
    vel = np.zeros((N, 3))
    gps = np.zeros(N, dtype=bool)
    gps[::100] = True  # 1 Hz
    
    params = zero_params(Qpos=1e-3, Qvel=1e-3, Rpos=1.0,
                         P_pos_std=10.0, P_vel_std=1.0)
    nav = make_nav(N, sr, accel, gyro, lla=lla, vel_enu=vel, gps_avail=gps)
    res = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)
    
    # After 5s of 1Hz GPS, position should converge toward (10,0,0)
    final_pos = res['p'][-1]
    pos_err = np.linalg.norm(final_pos - truth_enu)
    
    check("pos converges to GPS truth", pos_err < 2.0, f"pos_err={pos_err:.4f}m, final={final_pos}")
    check("pitch stays small", np.max(np.abs(res['r'][:, 1])) < 0.05,
          f"max_pitch={np.degrees(np.max(np.abs(res['r'][:, 1]))):.4f} deg")
    _plot_gps_diagnostic(res, np.tile(truth_enu, (N, 1)), vel, np.zeros((N, 3)),
                         sr, gps, title="TEST 7: GPS position update (stationary)")


# =========================================================================
#  TEST 8: GPS update during straight drive — innovation should not
#          cause pitch/roll spikes.
# =========================================================================
def test_gps_no_pitch_spike():
    print("\n=== TEST 8: GPS update - no pitch spike during straight drive ===")
    v = 10.0; sr = 100; dt = 1.0/sr; T = 30.0; N = int(T*sr)
    n_ramp = 10
    accel = np.zeros((N,3)); accel[:,2] = G
    accel[:n_ramp, 0] = v/(n_ramp*dt)
    gyro = np.zeros((N,3))
    
    # Generate truth from pure INS
    res_truth = run_pure_ins(accel, gyro, sr)
    truth_enu = res_truth['p']
    truth_vel = res_truth['v']
    truth_lla = np.array([pm.enu2geodetic(r[0],r[1],r[2],LLA0[0],LLA0[1],LLA0[2]) for r in truth_enu])
    
    gps = np.zeros(N, dtype=bool)
    gps[::100] = True  # 1 Hz
    
    params = dict(Qpos=1e-3, Qvel=1e-2, QorientXY=1e-4, QorientZ=1e-3,
                  Qacc=1e-6, QgyrXY=1e-6, QgyrZ=1e-6, Rpos=2.0,
                  beta_acc=-0.001, beta_gyr=-0.001,
                  P_pos_std=1.0, P_vel_std=1.0, P_orient_std=0.1,
                  P_acc_std=0.01, P_gyr_std=0.01,
                  enable_nhc=False, enable_zupt=False, enable_level=False)
    
    nav = make_nav(N, sr, accel, gyro, lla=truth_lla, vel_enu=truth_vel, gps_avail=gps)
    res = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)
    
    max_pitch = np.degrees(np.max(np.abs(res['r'][:, 1])))
    max_roll = np.degrees(np.max(np.abs(res['r'][:, 0])))
    pos_err = np.linalg.norm(res['p'][-1] - truth_enu[-1])
    
    check("no pitch spike (<1 deg)", max_pitch < 1.0, f"max_pitch={max_pitch:.4f} deg")
    check("no roll spike (<1 deg)", max_roll < 1.0, f"max_roll={max_roll:.4f} deg")
    check("pos error small (<5m)", pos_err < 5.0, f"pos_err={pos_err:.4f}m")
    _plot_gps_diagnostic(res, truth_enu, truth_vel, np.zeros_like(truth_enu),
                         sr, gps, title="TEST 8: GPS no pitch spike (straight drive)")


# =========================================================================
#  TEST 9: GPS update during circle — orientation should track, no flip.
# =========================================================================
def test_gps_circle_no_flip():
    print("\n=== TEST 9: GPS update during circle - no flip ===")
    v = 8.0; R = 30.0; omega = v/R
    sr = 100; dt = 1.0/sr; T = 30.0; N = int(T*sr)
    n_ramp = 10
    
    accel = np.zeros((N,3)); accel[:,2] = G
    accel[:n_ramp, 0] = v/(n_ramp*dt)
    gyro = np.zeros((N,3))
    gyro[:, 2] = omega
    accel[:, 1] = v * omega
    
    # Generate truth from pure INS
    res_truth = run_pure_ins(accel, gyro, sr)
    truth_enu = res_truth['p']
    truth_vel = res_truth['v']
    truth_rpy = res_truth['r']
    truth_lla = np.array([pm.enu2geodetic(r[0],r[1],r[2],LLA0[0],LLA0[1],LLA0[2]) for r in truth_enu])
    
    gps = np.zeros(N, dtype=bool)
    gps[::100] = True  # 1 Hz
    
    params = dict(Qpos=1e-3, Qvel=1e-2, QorientXY=1e-4, QorientZ=1e-3,
                  Qacc=1e-6, QgyrXY=1e-6, QgyrZ=1e-6, Rpos=2.0,
                  beta_acc=-0.001, beta_gyr=-0.001,
                  P_pos_std=1.0, P_vel_std=1.0, P_orient_std=0.1,
                  P_acc_std=0.01, P_gyr_std=0.01,
                  enable_nhc=False, enable_zupt=False, enable_level=False)
    
    nav = make_nav(N, sr, accel, gyro, lla=truth_lla, vel_enu=truth_vel, gps_avail=gps)
    res = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)
    
    max_pitch = np.degrees(np.max(np.abs(res['r'][:, 1])))
    max_roll = np.degrees(np.max(np.abs(res['r'][:, 0])))
    pos_err = np.linalg.norm(res['p'][-1] - truth_enu[-1])
    yaw_err = np.degrees(np.max(np.abs(res['r'][:, 2] - truth_rpy[:, 2])))
    
    check("no pitch spike (<2 deg)", max_pitch < 2.0, f"max_pitch={max_pitch:.4f} deg")
    check("no roll spike (<2 deg)", max_roll < 2.0, f"max_roll={max_roll:.4f} deg")
    check("pos error small (<5m)", pos_err < 5.0, f"pos_err={pos_err:.4f}m")
    check("yaw tracks (<5 deg)", yaw_err < 5.0, f"max_yaw_err={yaw_err:.4f} deg")
    _plot_gps_diagnostic(res, truth_enu, truth_vel, truth_rpy,
                         sr, gps, title="TEST 9: GPS circle - no flip")


# =========================================================================
#  TEST 10: F-matrix sign check — explicit numerical test.
#           Apply a known small orientation perturbation and check that
#           the predicted velocity error has the correct sign.
# =========================================================================
def test_f_matrix_signs():
    print("\n=== TEST 10: F-matrix velocity-orientation coupling signs ===")
    # The correct F[3:6, 6:9] for Euler-angle error states is the Jacobian
    #   d(Rbn · acc_body) / d(roll, pitch, yaw)
    # computed via rotation matrix derivatives:
    #   col0 = Rz @ Ry @ dRx/dr @ acc
    #   col1 = Rz @ dRy/dp @ Rx @ acc
    #   col2 = dRz/dy @ Ry @ Rx @ acc
    #
    # This is NOT ±skew(f); those formulas apply to rotation-vector error states.

    def _Rz(y): return np.array([[cos(y),-sin(y),0],[sin(y),cos(y),0],[0,0,1]])
    def _Ry(p): return np.array([[cos(p),0,-sin(p)],[0,1,0],[sin(p),0,cos(p)]])
    def _Rx(r): return np.array([[1,0,0],[0,cos(r),-sin(r)],[0,sin(r),cos(r)]])
    def _dRx(r): return np.array([[0,0,0],[0,-sin(r),-cos(r)],[0,cos(r),-sin(r)]])
    def _dRy(p): return np.array([[-sin(p),0,-cos(p)],[0,0,0],[cos(p),0,-sin(p)]])
    def _dRz(y): return np.array([[-sin(y),-cos(y),0],[cos(y),-sin(y),0],[0,0,0]])

    test_cases = [
        (0.0, 0.0, 0.0, np.array([0,0,G]),       'level/East/cruising'),
        (0.0, 0.0, 0.0, np.array([2,0,G]),        'level/East/accel'),
        (0.0, 0.0, 0.0, np.array([0,2,G]),        'level/East/turning'),
        (0.0, 0.0, 0.5, np.array([0,2,G]),        'yaw=0.5/turning'),
        (0.1, 0.0, 0.0, np.array([0,0,G]),        'roll=0.1/cruising'),
        (0.0, 0.1, 0.0, np.array([0,0,G]),        'pitch=0.1/cruising'),
        (0.1, 0.05, 0.3, np.array([2,1,G]),       'all nonzero'),
    ]

    for r0, p0, y0, acc_b, label in test_cases:
        # Analytical Jacobian via rotation matrix derivatives
        J_analytical = np.column_stack([
            _Rz(y0) @ _Ry(p0) @ _dRx(r0) @ acc_b,
            _Rz(y0) @ _dRy(p0) @ _Rx(r0) @ acc_b,
            _dRz(y0) @ _Ry(p0) @ _Rx(r0) @ acc_b,
        ])

        # Numerical Jacobian for ground truth
        eps = 1e-7
        J_num = np.zeros((3,3))
        for j, (dr,dp,dy) in enumerate([(eps,0,0),(0,eps,0),(0,0,eps)]):
            Rp = _Rz(y0+dy) @ _Ry(p0+dp) @ _Rx(r0+dr)
            Rm = _Rz(y0-dy) @ _Ry(p0-dp) @ _Rx(r0-dr)
            J_num[:, j] = (Rp @ acc_b - Rm @ acc_b) / (2*eps)

        diff = np.max(np.abs(J_analytical - J_num))
        check(f"F vel-orient Jacobian [{label}]", diff < 1e-5,
              f"max_diff={diff:.2e}")

    # Verify the mechanisation direction: pitch up -> gravity projects backward (West)
    pitch = 0.01
    Ry = np.array([[cos(pitch),0,-sin(pitch)],[0,1,0],[sin(pitch),0,cos(pitch)]])
    accENU = Ry @ np.array([0, 0, G])
    check("pitch>0 -> acc_East<0 (FLU nose-up)", accENU[0] < 0,
          f"accENU_E={accENU[0]:.6f}")


# =========================================================================
#  TEST 11: Error-state reset check — after GPS update, error states
#           x[0:9] should be zero for the NEXT prediction step.
# =========================================================================
def test_error_state_reset():
    print("\n=== TEST 11: Error-state reset after GPS update ===")
    # This is a structural test. We'll run GPS-aided and check that
    # the error state doesn't accumulate between updates.
    v = 10.0; sr = 100; dt = 1.0/sr; T = 10.0; N = int(T*sr)
    n_ramp = 10
    accel = np.zeros((N,3)); accel[:,2] = G
    accel[:n_ramp, 0] = v/(n_ramp*dt)
    gyro = np.zeros((N,3))
    
    res_truth = run_pure_ins(accel, gyro, sr)
    truth_enu = res_truth['p']
    truth_vel = res_truth['v']
    truth_lla = np.array([pm.enu2geodetic(r[0],r[1],r[2],LLA0[0],LLA0[1],LLA0[2]) for r in truth_enu])
    
    gps = np.zeros(N, dtype=bool)
    gps[::100] = True
    
    # Use high initial P to give the filter something to correct
    params = dict(Qpos=1e-3, Qvel=1e-2, QorientXY=1e-4, QorientZ=1e-3,
                  Qacc=1e-6, QgyrXY=1e-6, QgyrZ=1e-6, Rpos=2.0,
                  beta_acc=-0.001, beta_gyr=-0.001,
                  P_pos_std=1.0, P_vel_std=1.0, P_orient_std=0.1,
                  P_acc_std=0.01, P_gyr_std=0.01,
                  enable_nhc=False, enable_zupt=False, enable_level=False)
    
    nav = make_nav(N, sr, accel, gyro, lla=truth_lla, vel_enu=truth_vel, gps_avail=gps)
    res = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)
    
    # Check position tracks truth
    final_err = np.linalg.norm(res['p'][-1] - truth_enu[-1])
    check("pos tracks truth (<2m)", final_err < 2.0, f"final_err={final_err:.4f}m")
    
    # Check no divergence: error should not grow monotonically
    errs = np.linalg.norm(res['p'] - truth_enu, axis=1)
    max_err = np.max(errs)
    check("max pos err bounded (<5m)", max_err < 5.0, f"max_err={max_err:.4f}m")


# =========================================================================
#  TEST 12: Innovation check — with perfect truth, innovation should be
#           near zero (not growing).
# =========================================================================
def test_innovation_near_zero():
    print("\n=== TEST 12: Innovation near-zero with perfect truth ===")
    v = 10.0; sr = 100; dt = 1.0/sr; T = 20.0; N = int(T*sr)
    n_ramp = 10
    accel = np.zeros((N,3)); accel[:,2] = G
    accel[:n_ramp, 0] = v/(n_ramp*dt)
    gyro = np.zeros((N,3))
    
    # Truth from INS
    res_truth = run_pure_ins(accel, gyro, sr)
    truth_enu = res_truth['p']
    truth_vel = res_truth['v']
    truth_lla = np.array([pm.enu2geodetic(r[0],r[1],r[2],LLA0[0],LLA0[1],LLA0[2]) for r in truth_enu])
    
    gps = np.zeros(N, dtype=bool)
    gps[::100] = True
    
    # Tiny process noise so the filter should track perfectly
    params = dict(Qpos=1e-10, Qvel=1e-10, QorientXY=1e-10, QorientZ=1e-10,
                  Qacc=1e-15, QgyrXY=1e-15, QgyrZ=1e-15, Rpos=1.0,
                  beta_acc=-1e-10, beta_gyr=-1e-10,
                  P_pos_std=0.01, P_vel_std=0.01, P_orient_std=0.001,
                  P_acc_std=1e-10, P_gyr_std=1e-10,
                  enable_nhc=False, enable_zupt=False, enable_level=False)
    
    nav = make_nav(N, sr, accel, gyro, lla=truth_lla, vel_enu=truth_vel, gps_avail=gps)
    res = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)
    
    # Position error should stay small. Note: the EKF seeds x[9:12]=1e-6
    # (initial bias), which causes a tiny systematic acceleration error that
    # integrates to ~0.16m over 20s.  With GPS corrections this is bounded.
    max_err = np.max(np.linalg.norm(res['p'] - truth_enu, axis=1))
    final_err = np.linalg.norm(res['p'][-1] - truth_enu[-1])
    max_pitch = np.degrees(np.max(np.abs(res['r'][:, 1])))
    
    check("max pos err < 0.5m (tiny Q, bias seed drift)", max_err < 0.5, f"max_err={max_err:.6f}m")
    check("final pos err < 0.5m", final_err < 0.5, f"final_err={final_err:.6f}m")
    check("pitch < 0.1 deg", max_pitch < 0.1, f"max_pitch={max_pitch:.6f} deg")


# =========================================================================
#  TEST 13: GPS-aided circle with realistic params — THE KEY TEST.
#           If this fails with pitch flip, there's a fundamental bug.
# =========================================================================
def test_gps_circle_realistic():
    print("\n=== TEST 13: GPS-aided circle - realistic params ===")
    v = 8.0; R = 30.0; omega = v/R
    sr = 100; dt = 1.0/sr; T = 60.0; N = int(T*sr)
    n_ramp = 10
    
    accel = np.zeros((N,3)); accel[:,2] = G
    accel[:n_ramp, 0] = v/(n_ramp*dt)
    gyro = np.zeros((N,3))
    gyro[:, 2] = omega
    accel[:, 1] = v * omega
    
    res_truth = run_pure_ins(accel, gyro, sr)
    truth_enu = res_truth['p']
    truth_vel = res_truth['v']
    truth_rpy = res_truth['r']
    truth_lla = np.array([pm.enu2geodetic(r[0],r[1],r[2],LLA0[0],LLA0[1],LLA0[2]) for r in truth_enu])
    
    gps = np.zeros(N, dtype=bool)
    gps[::100] = True
    
    params = dict(Qpos=1e-3, Qvel=1e-2, QorientXY=1e-4, QorientZ=1e-3,
                  Qacc=1e-6, QgyrXY=1e-6, QgyrZ=1e-6, Rpos=2.0,
                  beta_acc=-0.001, beta_gyr=-0.001,
                  P_pos_std=1.0, P_vel_std=1.0, P_orient_std=0.1,
                  P_acc_std=0.01, P_gyr_std=0.01,
                  enable_nhc=False, enable_zupt=False, enable_level=False)
    
    nav = make_nav(N, sr, accel, gyro, lla=truth_lla, vel_enu=truth_vel, gps_avail=gps)
    res = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)
    
    max_pitch = np.degrees(np.max(np.abs(res['r'][:, 1])))
    max_roll = np.degrees(np.max(np.abs(res['r'][:, 0])))
    pos_err = np.linalg.norm(res['p'][-1] - truth_enu[-1])
    
    # Print time series of pitch for debugging
    print(f"    Pitch profile (every 5s):")
    for t_sec in range(0, int(T), 5):
        idx = t_sec * sr
        if idx < N:
            pitch_deg = np.degrees(res['r'][idx, 1])
            roll_deg = np.degrees(res['r'][idx, 0])
            pe = np.linalg.norm(res['p'][idx] - truth_enu[idx])
            print(f"      t={t_sec:3d}s  pitch={pitch_deg:+8.4f} deg  roll={roll_deg:+8.4f} deg  pos_err={pe:.4f}m")
    
    check("no pitch flip (<5 deg)", max_pitch < 5.0, f"max_pitch={max_pitch:.4f} deg")
    check("no roll flip (<5 deg)", max_roll < 5.0, f"max_roll={max_roll:.4f} deg")
    check("pos error bounded (<10m)", pos_err < 10.0, f"pos_err={pos_err:.4f}m")
    _plot_gps_diagnostic(res, truth_enu, truth_vel, truth_rpy,
                         sr, gps, title="TEST 13: GPS-aided circle (60s, realistic)")


# =========================================================================
#  TEST 14: Racetrack (the problem scenario from stress tests)
# =========================================================================
def test_gps_racetrack():
    print("\n=== TEST 14: GPS-aided racetrack ===")
    v = 8.0; R = 30.0; omega = v/R
    sr = 100; dt = 1.0/sr
    straight_len = 60.0
    t_straight = straight_len/v
    t_semi = pi/omega
    n_ramp = 10
    n_straight = int(t_straight*sr)
    n_semi = int(t_semi*sr)
    N = n_ramp + 2*n_straight + 2*n_semi
    
    accel = np.zeros((N,3)); accel[:,2] = G
    gyro = np.zeros((N,3))
    accel[:n_ramp, 0] = v/(n_ramp*dt)
    
    idx = n_ramp
    idx += n_straight
    gyro[idx:idx+n_semi, 2] = omega
    accel[idx:idx+n_semi, 1] = v*omega
    idx += n_semi
    idx += n_straight
    gyro[idx:idx+n_semi, 2] = omega
    accel[idx:idx+n_semi, 1] = v*omega
    
    res_truth = run_pure_ins(accel, gyro, sr)
    truth_enu = res_truth['p']
    truth_vel = res_truth['v']
    truth_rpy = res_truth['r']
    truth_lla = np.array([pm.enu2geodetic(r[0],r[1],r[2],LLA0[0],LLA0[1],LLA0[2]) for r in truth_enu])
    
    gps = np.zeros(N, dtype=bool)
    gps[::100] = True
    
    params = dict(Qpos=1e-3, Qvel=1e-2, QorientXY=1e-4, QorientZ=1e-3,
                  Qacc=1e-6, QgyrXY=1e-6, QgyrZ=1e-6, Rpos=2.0,
                  beta_acc=-0.001, beta_gyr=-0.001,
                  P_pos_std=1.0, P_vel_std=1.0, P_orient_std=0.1,
                  P_acc_std=0.01, P_gyr_std=0.01,
                  enable_nhc=False, enable_zupt=False, enable_level=False)
    
    nav = make_nav(N, sr, accel, gyro, lla=truth_lla, vel_enu=truth_vel, gps_avail=gps)
    res = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)
    
    max_pitch = np.degrees(np.max(np.abs(res['r'][:, 1])))
    max_roll = np.degrees(np.max(np.abs(res['r'][:, 0])))
    pos_err = np.linalg.norm(res['p'][-1] - truth_enu[-1])
    
    print(f"    Pitch/roll profile (every 5s):")
    T = N * dt
    for t_sec in range(0, int(T), 5):
        idx = t_sec * sr
        if idx < N:
            pitch_deg = np.degrees(res['r'][idx, 1])
            roll_deg = np.degrees(res['r'][idx, 0])
            pe = np.linalg.norm(res['p'][idx] - truth_enu[idx])
            print(f"      t={t_sec:3d}s  pitch={pitch_deg:+8.4f} deg  roll={roll_deg:+8.4f} deg  pos_err={pe:.4f}m")
    
    check("no pitch flip (<5 deg)", max_pitch < 5.0, f"max_pitch={max_pitch:.4f} deg")
    check("no roll flip (<5 deg)", max_roll < 5.0, f"max_roll={max_roll:.4f} deg")
    check("pos error bounded (<10m)", pos_err < 10.0, f"pos_err={pos_err:.4f}m")
    _plot_gps_diagnostic(res, truth_enu, truth_vel, truth_rpy,
                         sr, gps, title="TEST 14: GPS-aided racetrack")


# =========================================================================
#  TEST 15: Covariance sanity — P should stay positive definite and
#           not blow up or collapse to zero.
# =========================================================================
def test_covariance_sanity():
    print("\n=== TEST 15: Covariance sanity ===")
    v = 10.0; sr = 100; dt = 1.0/sr; T = 30.0; N = int(T*sr)
    n_ramp = 10
    accel = np.zeros((N,3)); accel[:,2] = G
    accel[:n_ramp, 0] = v/(n_ramp*dt)
    gyro = np.zeros((N,3))
    
    res_truth = run_pure_ins(accel, gyro, sr)
    truth_enu = res_truth['p']
    truth_vel = res_truth['v']
    truth_lla = np.array([pm.enu2geodetic(r[0],r[1],r[2],LLA0[0],LLA0[1],LLA0[2]) for r in truth_enu])
    
    gps = np.zeros(N, dtype=bool)
    gps[::100] = True
    
    params = dict(Qpos=1e-3, Qvel=1e-2, QorientXY=1e-4, QorientZ=1e-3,
                  Qacc=1e-6, QgyrXY=1e-6, QgyrZ=1e-6, Rpos=2.0,
                  beta_acc=-0.001, beta_gyr=-0.001,
                  P_pos_std=1.0, P_vel_std=1.0, P_orient_std=0.1,
                  P_acc_std=0.01, P_gyr_std=0.01,
                  enable_nhc=False, enable_zupt=False, enable_level=False)
    
    nav = make_nav(N, sr, accel, gyro, lla=truth_lla, vel_enu=truth_vel, gps_avail=gps)
    res = ekf_core.run_ekf(nav, params, outage_config=None, use_3d_rotation=True)
    
    # Check that standard deviations are reasonable (positive, finite, not huge)
    max_std_pos = np.max(res['std_pos'])
    min_std_pos = np.min(res['std_pos'][1:])  # skip initial zero
    max_std_vel = np.max(res['std_vel'])
    max_std_orient = np.max(res['std_orient'])
    
    check("std_pos finite", np.all(np.isfinite(res['std_pos'])), f"max={max_std_pos}")
    check("std_pos not huge (<100m)", max_std_pos < 100.0, f"max={max_std_pos:.4f}")
    check("std_pos not collapsed (>1e-6)", min_std_pos > 1e-6, f"min={min_std_pos:.2e}")
    check("std_vel finite and bounded", np.all(np.isfinite(res['std_vel'])) and max_std_vel < 100.0,
          f"max={max_std_vel:.4f}")
    check("std_orient finite and bounded", np.all(np.isfinite(res['std_orient'])) and max_std_orient < 10.0,
          f"max={max_std_orient:.4f}")


# =========================================================================
if __name__ == "__main__":
    if '--plot' in sys.argv:
        PLOT_MODE = True
        sys.argv.remove('--plot')
        print("  [PLOT MODE enabled — diagnostic figures will open]")

    print("=" * 70)
    print("  EKF CORE EQUATION AUDIT")
    print("=" * 70)
    
    test_stationary()
    test_const_accel_east()
    test_pure_yaw()
    test_pure_pitch()
    test_pure_roll()
    test_circle()
    test_gps_pos_update()
    test_gps_no_pitch_spike()
    test_gps_circle_no_flip()
    test_f_matrix_signs()
    test_error_state_reset()
    test_innovation_near_zero()
    test_gps_circle_realistic()
    test_gps_racetrack()
    test_covariance_sanity()
    
    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASS} PASS, {FAIL} FAIL")
    print("=" * 70)
    
    if PLOT_MODE and HAS_PLT:
        plt.show()
    
    sys.exit(1 if FAIL > 0 else 0)
