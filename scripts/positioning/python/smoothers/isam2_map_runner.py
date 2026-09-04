# -*- coding: utf-8 -*-
"""
iSAM2 Track-Constrained Online Smoother
=======================================
Reference (inspiration):

    D. Wilbers, C. Merfels, and C. Stachniss, "Localization with sliding window
    factor graphs on third-party maps for automated driving," in 2019 International
    Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 5951–5957.
    https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/wilbers2019icra.pdf

Core idea
---------
A known track geometry (a surveyed / manually-labelled polyline — the railway
track centre-line, or a third-party HD-map) is used as a drift-bounding prior
during GNSS outages.  Unlike a raw position anchor, the track is incorporated as
an **anisotropic point-to-curve constraint**: at every 1 Hz node inside the
outage the *current estimated* position is projected onto the nearest point of
the track and the pose is pulled **perpendicularly onto the track** (tight
across-track and vertical, loose along-track).  The IMU decides how far *along*
the track the vehicle is; the map decides only that it is *on* the track.  This
is the position-domain analogue of a non-holonomic constraint.

Deviation from Wilbers et al.
-----------------------------
Wilbers keeps landmark variables in the graph and constrains those landmarks
with the map (eq. 4), so the pose is corrected only *indirectly* through
landmark-observation factors, and the data association is solved online with
ICP + temporal smoothing.  Here there are no landmark variables and no landmark
sensor: the map factor is applied **directly to the pose** as a lateral
constraint, and "association" is a purely geometric **online nearest-point
projection** of the estimated position onto the track polyline (gated by
arc-length continuity and heading to avoid wrong-branch snapping).  Crucially the
projection uses the *estimated* trajectory, never ground truth, so it is a
realizable online scheme rather than an offline oracle.

Anisotropy
----------
The constraint is realised by reusing ``gtsam.GPSFactor`` anchored at the
projected foot point ``p*`` with a **full covariance** oriented to the local
track frame::

    R_track = A · diag(sigma_lon^2, sigma_lat^2, sigma_up^2) · A^T,
    A = [ t_hat | n_hat | z_hat ]   (columns),

with sigma_lon large ("free" along-track) and sigma_lat / sigma_up tight.  The
huge along-track variance makes the anchor pull only in the perpendicular /
vertical directions, i.e. exactly a lateral point-to-curve constraint.

Differences from isam2_runner.py
--------------------------------
1. A TrackMap (polyline + KDTree + cumulative arc-length + per-segment tangents)
   is built from nav_data.lla before the loop.
2. During the outage a node is instantiated at every 1 Hz epoch (as in the base
   estimator's normal operation) so the track anchor has a node to attach to.
3. The outage factor is the anisotropic point-to-curve GPSFactor described above,
   inserted only when a valid projection is found.

All other aspects (coordinate frames, IMU preintegration, GPS factors, output
format) are identical to isam2_runner.py.

NOTE — numpy compatibility: install the conda-forge gtsam build (the pip wheel
gtsam==4.2 segfaults on numpy 2.x):  conda install -c conda-forge gtsam
"""

import sys
import numpy as np
import pymap3d as pm
from pathlib import Path
from math import sin, cos, atan2, asin
from scipy.spatial import KDTree

_SCRIPTS = Path(__file__).parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ── Lazy GTSAM import ─────────────────────────────────────────────────────────

def _import_gtsam():
    try:
        import gtsam
        from gtsam.symbol_shorthand import X, V, B
        return gtsam, X, V, B
    except ImportError as e:
        raise ImportError(
            "GTSAM is required for the iSAM2 Track smoother.\n"
            "Install with:  conda install -c conda-forge gtsam\n"
            f"Original error: {e}"
        ) from e


# ── Default parameters ────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # IMU noise (continuous-time, pre-integration)
    'acc_noise_sigma':   0.1,     # [m/s²/√s]  accelerometer white noise density
    'gyr_noise_sigma':   1e-3,    # [rad/s/√s] gyroscope white noise density
    'acc_bias_sigma':    1e-3,    # [m/s²/√Hz] accelerometer bias random-walk
    'gyr_bias_sigma':    1e-5,    # [rad/s/√Hz] gyroscope bias random-walk
    # GPS measurement noise
    'Rpos':              4.0,     # position std dev [m]; covariance = Rpos²·I₃
    # iSAM2 re-linearisation
    'isam2_relinearize_threshold': 0.1,
    'isam2_relinearize_skip':      1,
    # Initial state uncertainty (std dev)
    'P_pos_std':    1.0,     # [m]
    'P_vel_std':    0.3,     # [m/s]
    'P_orient_std': 0.1,     # [rad]
    'P_acc_std':    1e-2,    # [m/s²]  initial bias uncertainty
    'P_gyr_std':    1e-3,    # [rad/s]
    # Track constraint parameters (point-to-curve, applied during outage only)
    'track_spacing_m':        5.0,    # polyline resample spacing [m]
    'sigma_lat':              1.0,    # across-track std dev [m] (tight — HD-map accuracy)
    'sigma_up':               1.0,    # vertical std dev [m] (tight)
    'sigma_lon':           1000.0,    # along-track std dev [m] (loose — effectively free)
    'track_search_window_m':  150.0,  # arc-length half-window for online projection [m]
    'track_heading_gate':     0.6,    # max |tangent − heading| for a valid match [rad]
    'track_gate_m':           25.0,   # reject the anchor if the predicted position is
                                      # farther than this from the projected track point
                                      # (outlier / lost-lock guard — prevents divergence)
}


# ── Rotation utilities ────────────────────────────────────────────────────────

def _euler_to_Rbn(rpy):
    """ZYX Euler angles (roll, pitch, yaw) → body-to-nav rotation matrix."""
    r, p, y = rpy
    cr, sr = cos(r), sin(r)
    cp, sp = cos(p), sin(p)
    cy, sy = cos(y), sin(y)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr           ],
    ])


def _rbn_to_rpy(Rbn):
    """Rotation matrix (body-to-nav) → ZYX Euler angles [roll, pitch, yaw]."""
    pitch = asin(-Rbn[2, 0])
    if abs(cos(pitch)) > 1e-8:
        roll = atan2(Rbn[2, 1], Rbn[2, 2])
        yaw  = atan2(Rbn[1, 0], Rbn[0, 0])
    else:
        roll = 0.0
        yaw  = atan2(-Rbn[0, 1], Rbn[1, 1])
    return np.array([roll, pitch, yaw])


def _in_outage(i, sample_rate, outage_cfg):
    """Return True if sample i is inside the GPS outage window."""
    if outage_cfg is None:
        return False
    t = i / sample_rate
    return outage_cfg['start'] <= t < outage_cfg['start'] + outage_cfg['duration']


def _mat_to_rot3(gtsam, R):
    """Create gtsam.Rot3 from a 3×3 numpy rotation matrix (element-by-element)."""
    return gtsam.Rot3(
        R[0, 0], R[0, 1], R[0, 2],
        R[1, 0], R[1, 1], R[1, 2],
        R[2, 0], R[2, 1], R[2, 2],
    )


# ── Track map (surveyed polyline + online projection) ─────────────────────────

class TrackMap:
    """
    Known track geometry as an ENU polyline with an online point-to-curve query.

    Built once from a source trajectory (the surveyed / manually-labelled track;
    here simulated by resampling nav_data.lla by arc length).  Stores:

        verts   : (M, 3) ENU vertices of the polyline.
        seg_dir : (M-1, 2) unit tangents of each segment in the horizontal plane.
        seg_len : (M-1,)   segment lengths [m] (horizontal).
        s_vert  : (M,)     cumulative arc length at each vertex [m].

    A ``scipy.spatial.KDTree`` on the 2-D vertices gives an O(log M) coarse
    nearest-vertex lookup; ``project`` then refines to the exact perpendicular
    foot on the neighbouring segments, gated by arc-length continuity and heading.
    """

    def __init__(self, verts: np.ndarray):
        self.verts = verts                      # (M, 3)
        d = np.diff(verts[:, :2], axis=0)       # (M-1, 2) horizontal segment vectors
        seg_len = np.linalg.norm(d, axis=1)     # (M-1,)
        seg_len = np.maximum(seg_len, 1e-9)
        self.seg_len = seg_len
        self.seg_dir = d / seg_len[:, None]     # (M-1, 2) unit tangents
        self.s_vert = np.concatenate([[0.0], np.cumsum(seg_len)])  # (M,)
        self._kdtree = KDTree(verts[:, :2])

    @property
    def n_segments(self) -> int:
        return len(self.seg_len)

    def _foot_on_segment(self, j: int, q_xy: np.ndarray):
        """
        Perpendicular foot of q_xy on segment j (clamped to the segment).

        Returns (foot_xy, t_on_seg, along_len) where t_on_seg ∈ [0, 1] is the
        fractional position along the segment and along_len its metric offset from
        the segment start.
        """
        a = self.verts[j, :2]
        t = float(np.clip((q_xy - a) @ self.seg_dir[j], 0.0, self.seg_len[j]))
        foot_xy = a + t * self.seg_dir[j]
        return foot_xy, t / self.seg_len[j], t

    def project(self, pos_enu: np.ndarray, s_prev: float, heading: float,
                window_m: float, heading_gate: float):
        """
        Project ``pos_enu`` onto the track, online.

        Candidate segments are those whose start-vertex arc length lies within
        ``window_m`` of ``s_prev`` (continuity gate) and whose tangent agrees with
        ``heading`` to within ``heading_gate`` (branch gate).  Among candidates the
        one giving the smallest perpendicular distance is chosen.

        Args:
            pos_enu      : (3,) estimated ENU position.
            s_prev       : previous along-track arc length [m] (search centre).
            heading      : estimated heading atan2(E, N)-consistent angle [rad]
                           of the horizontal velocity/tangent; NaN disables the gate.
            window_m     : arc-length half-window [m].
            heading_gate : max tangent/heading mismatch [rad].

        Returns:
            (p_star, t_hat, n_hat, s_new) or None if nothing qualifies.
            p_star : (3,) foot point on the track (altitude interpolated).
            t_hat  : (2,) horizontal unit tangent at the foot.
            n_hat  : (2,) horizontal unit normal (left of tangent).
            s_new  : updated along-track arc length [m].
        """
        q_xy = pos_enu[:2]

        # Coarse candidates from arc-length window around s_prev.
        lo = np.searchsorted(self.s_vert, s_prev - window_m) - 1
        hi = np.searchsorted(self.s_vert, s_prev + window_m) + 1
        lo = max(lo, 0)
        hi = min(hi, self.n_segments)
        if hi <= lo:
            # Fall back to KDTree nearest vertex (e.g. first call / s_prev unknown).
            _, jv = self._kdtree.query(q_xy)
            lo = max(int(jv) - 3, 0)
            hi = min(int(jv) + 3, self.n_segments)

        best = None  # (dist, foot_xy, t_hat2, along_len, j)
        for j in range(lo, hi):
            t_hat2 = self.seg_dir[j]
            if np.isfinite(heading):
                seg_ang = atan2(t_hat2[0], t_hat2[1])  # atan2(E, N)
                dang = abs((seg_ang - heading + np.pi) % (2 * np.pi) - np.pi)
                if dang > heading_gate:
                    continue
            foot_xy, _, along = self._foot_on_segment(j, q_xy)
            dist = np.linalg.norm(q_xy - foot_xy)
            if best is None or dist < best[0]:
                best = (dist, foot_xy, t_hat2, along, j)

        if best is None:
            return None

        _, foot_xy, t_hat2, along, j = best
        s_new = self.s_vert[j] + along

        # Interpolate altitude along the chosen segment.
        frac = along / self.seg_len[j]
        alt = (1.0 - frac) * self.verts[j, 2] + frac * self.verts[j + 1, 2]
        p_star = np.array([foot_xy[0], foot_xy[1], alt])

        n_hat2 = np.array([-t_hat2[1], t_hat2[0]])   # left normal
        return p_star, t_hat2, n_hat2, s_new


def build_track_map(lla: np.ndarray, lla0: np.ndarray,
                    spacing_m: float = 5.0) -> TrackMap:
    """
    Build a TrackMap by resampling a source trajectory in ENU by arc length.

    In a real deployment the polyline would come from the surveyed railway track
    or a third-party HD-map; here it is simulated by resampling nav_data.lla.

    Args:
        lla       : (N, 3) geodetic positions [lat°, lon°, alt_m] — source track.
        lla0      : (3,)   ENU reference origin [lat0°, lon0°, alt0_m].
        spacing_m : target vertex spacing along the track [m].

    Returns:
        TrackMap with M ≤ N resampled vertices.
    """
    e, n, u = pm.geodetic2enu(
        lla[:, 0], lla[:, 1], lla[:, 2],
        lla0[0],   lla0[1],   lla0[2],
    )
    p_all = np.column_stack([e, n, u])   # (N, 3)

    verts = [p_all[0]]
    last = p_all[0]
    for i in range(1, len(p_all)):
        if np.linalg.norm(p_all[i, :2] - last[:2]) >= spacing_m:
            verts.append(p_all[i])
            last = p_all[i]
    if np.linalg.norm(p_all[-1, :2] - verts[-1][:2]) > 1e-6:
        verts.append(p_all[-1])
    verts = np.asarray(verts)
    if len(verts) < 2:                    # degenerate: pad so segments exist
        verts = np.vstack([p_all[0], p_all[-1]])
    return TrackMap(verts)


def _track_covariance(gtsam, t_hat2, n_hat2, s_lon, s_lat, s_up):
    """
    Full 3×3 GPS-factor covariance oriented to the local track frame:

        R = A · diag(s_lon², s_lat², s_up²) · Aᵀ,  A = [t̂ | n̂ | ẑ].

    The large along-track variance makes the anchor pull only perpendicularly and
    vertically — an anisotropic point-to-curve (lateral) constraint.
    """
    A = np.array([
        [t_hat2[0], n_hat2[0], 0.0],
        [t_hat2[1], n_hat2[1], 0.0],
        [0.0,       0.0,       1.0],
    ])
    D = np.diag([s_lon ** 2, s_lat ** 2, s_up ** 2])
    R = A @ D @ A.T
    return gtsam.noiseModel.Gaussian.Covariance(R)


# ── Public interface ──────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run the iSAM2 track-constrained online smoother and return navigation
    estimates.

    Identical to isam2_runner.run() except that, during a GNSS outage, each 1 Hz
    node receives an anisotropic point-to-curve constraint pulling its position
    perpendicularly onto a known track polyline (see module docstring).  Outside
    the outage the ordinary GPS factor is used unchanged.

    Args:
        nav_data       : NavigationData dataclass (data_loader.py).
        params         : Optional dict overriding DEFAULT_PARAMS.
        outage_config  : Optional {'start': t1_s, 'duration': d_s} for GPS
                         blackout.  Track constraints are applied in this window.
        use_3d_rotation: Accepted for interface compatibility; always ignored.

    Transparently dispatches to whichever environment actually has GTSAM —
    native in-process if importable here, otherwise bridged through the
    isolated /opt/conda-gtsam interpreter as a subprocess (see
    fgo_batch_runner.run()'s docstring for the full rationale; same pattern).

    Returns:
        dict with keys: p, v, r, bias_acc, bias_gyr,
                        std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr.
    """
    try:
        import gtsam  # noqa: F401
    except ImportError:
        from ._conda_gtsam_bridge import run_via_conda_subprocess
        return run_via_conda_subprocess("isam2_map", nav_data, params, outage_config, use_3d_rotation)
    return _run_native(nav_data, params, outage_config, use_3d_rotation)


def _run_native(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """The original, in-process GTSAM implementation — see run()'s docstring."""
    gtsam, X, V, B = _import_gtsam()

    p_cfg = dict(DEFAULT_PARAMS)
    if params:
        p_cfg.update(params)

    accel_flu = nav_data.accel_flu      # (N, 3) body-frame IMU [m/s²]
    gyro_flu  = nav_data.gyro_flu       # (N, 3) body-frame gyro [rad/s]
    orient    = nav_data.orient         # (N, 3) Euler [roll, pitch, yaw] [rad]
    vel_enu   = nav_data.vel_enu        # (N, 3) ENU velocity [m/s]
    lla       = nav_data.lla            # (N, 3) geodetic [lat, lon, alt]
    lla0      = nav_data.lla0
    N         = accel_flu.shape[0]
    Ts        = 1.0 / nav_data.sample_rate

    # GPS positions in ENU
    e, n, u = pm.geodetic2enu(
        lla[:, 0], lla[:, 1], lla[:, 2],
        lla0[0], lla0[1], lla0[2],
    )
    p_gps = np.column_stack([e, n, u])     # (N, 3) ENU positions from GPS

    # ── Track map (surveyed polyline; used for online projection in outage) ────
    track = build_track_map(lla, lla0, p_cfg['track_spacing_m'])
    s_track = 0.0          # running along-track arc-length estimate [m]

    # ── Output arrays ─────────────────────────────────────────────────────────
    p_out        = np.zeros((N, 3))
    v_out        = np.zeros((N, 3))
    r_out        = np.zeros((N, 3))
    b_acc_out    = np.zeros((N, 3))
    b_gyr_out    = np.zeros((N, 3))
    std_pos      = np.zeros((N, 3))
    std_vel      = np.zeros((N, 3))
    std_orient   = np.zeros((N, 3))
    std_b_acc    = np.zeros((N, 3))
    std_b_gyr    = np.zeros((N, 3))

    # ── iSAM2 setup ───────────────────────────────────────────────────────────
    isam2_p = gtsam.ISAM2Params()
    isam2_p.setRelinearizeThreshold(p_cfg['isam2_relinearize_threshold'])
    isam2_p.relinearizeSkip = int(p_cfg['isam2_relinearize_skip'])
    isam = gtsam.ISAM2(isam2_p)

    # ── IMU preintegration parameters (ENU Z-up frame) ────────────────────────
    pim_params = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
    pim_params.setAccelerometerCovariance(
        np.eye(3) * p_cfg['acc_noise_sigma'] ** 2)
    pim_params.setGyroscopeCovariance(
        np.eye(3) * p_cfg['gyr_noise_sigma'] ** 2)
    pim_params.setIntegrationCovariance(np.eye(3) * 1e-8)
    pim_params.setBiasAccCovariance(np.eye(3) * p_cfg['acc_bias_sigma'] ** 2)
    pim_params.setBiasOmegaCovariance(np.eye(3) * p_cfg['gyr_bias_sigma'] ** 2)
    pim_params.setBiasAccOmegaInit(np.eye(6) * 1e-8)

    bias_prev = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    pim = gtsam.PreintegratedCombinedMeasurements(pim_params, bias_prev)

    # ── Initial state ─────────────────────────────────────────────────────────
    Rbn0  = _euler_to_Rbn(orient[0])
    rot0  = _mat_to_rot3(gtsam, Rbn0)
    pose_prev = gtsam.Pose3(rot0, p_gps[0])
    vel_prev  = vel_enu[0].copy()

    p_out[0]  = p_gps[0]
    v_out[0]  = vel_prev
    r_out[0]  = orient[0].copy()

    # ── Insert initial priors into factor graph ────────────────────────────────
    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    values.insert(X(0), pose_prev)
    values.insert(V(0), vel_prev)
    values.insert(B(0), bias_prev)

    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        p_cfg['P_orient_std'], p_cfg['P_orient_std'], p_cfg['P_orient_std'],
        p_cfg['P_pos_std'],    p_cfg['P_pos_std'],    p_cfg['P_pos_std'],
    ]))
    vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        p_cfg['P_vel_std'], p_cfg['P_vel_std'], p_cfg['P_vel_std'],
    ]))
    bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        p_cfg['P_acc_std'], p_cfg['P_acc_std'], p_cfg['P_acc_std'],
        p_cfg['P_gyr_std'], p_cfg['P_gyr_std'], p_cfg['P_gyr_std'],
    ]))

    graph.push_back(gtsam.PriorFactorPose3(X(0), pose_prev, pose_noise))
    graph.push_back(gtsam.PriorFactorVector(V(0), vel_prev, vel_noise))
    graph.push_back(gtsam.PriorFactorConstantBias(B(0), bias_prev, bias_noise))

    isam.update(graph, values)
    result = isam.calculateEstimate()

    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # GPS noise model (normal operation)
    gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        p_cfg['Rpos'], p_cfg['Rpos'], p_cfg['Rpos'],
    ]))

    nav_state_prev = gtsam.NavState(pose_prev, vel_prev)
    k = 0   # variable key index

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i in range(N - 1):
        # Integrate one IMU sample
        pim.integrateMeasurement(accel_flu[i], gyro_flu[i], Ts)

        # Propagate NavState forward at IMU rate for output fill
        nav_prop = pim.predict(nav_state_prev, bias_prev)
        p_out[i + 1] = nav_prop.pose().translation()
        v_out[i + 1] = nav_prop.velocity()
        R_prop       = nav_prop.pose().rotation().matrix()
        r_out[i + 1] = _rbn_to_rpy(R_prop)
        std_pos[i + 1]    = std_pos[i]
        std_vel[i + 1]    = std_vel[i]
        std_orient[i + 1] = std_orient[i]
        std_b_acc[i + 1]  = std_b_acc[i]
        std_b_gyr[i + 1]  = std_b_gyr[i]
        b_acc_out[i + 1]  = b_acc_out[i]
        b_gyr_out[i + 1]  = b_gyr_out[i]

        # ── Three-way branch: normal GPS / outage+track / IMU-only ──────────
        is_gps_tick = nav_data.gps_available[i + 1]
        in_outage   = _in_outage(i + 1, nav_data.sample_rate, outage_config)
        do_update   = False       # whether to run iSAM2 update this step

        if is_gps_tick and not in_outage:
            # ── NORMAL GPS: identical to vanilla isam2_runner ────────────
            k += 1
            nav_pred = pim.predict(nav_state_prev, bias_prev)

            values.insert(X(k), nav_pred.pose())
            values.insert(V(k), nav_pred.velocity())
            values.insert(B(k), bias_prev)

            graph.push_back(gtsam.CombinedImuFactor(
                X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim,
            ))
            graph.push_back(gtsam.GPSFactor(X(k), p_gps[i + 1], gps_noise))
            do_update = True

        elif is_gps_tick and in_outage:
            # ── OUTAGE: IMU + anisotropic track constraint ───────────────
            # Instantiate a node at GPS rate even though GPS is absent, so the
            # track anchor has a node to attach to.  Project the *predicted*
            # position onto the track and pull it perpendicularly onto it.
            k += 1
            nav_pred = pim.predict(nav_state_prev, bias_prev)

            values.insert(X(k), nav_pred.pose())
            values.insert(V(k), nav_pred.velocity())
            values.insert(B(k), bias_prev)

            graph.push_back(gtsam.CombinedImuFactor(
                X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim,
            ))

            pos_pred = np.asarray(nav_pred.pose().translation())
            vel_pred = np.asarray(nav_pred.velocity())
            speed_h  = float(np.linalg.norm(vel_pred[:2]))
            heading  = atan2(vel_pred[0], vel_pred[1]) if speed_h > 0.1 else np.nan

            proj = track.project(
                pos_pred, s_track, heading,
                p_cfg['track_search_window_m'], p_cfg['track_heading_gate'],
            )
            if proj is not None:
                p_star, t_hat, n_hat, s_new = proj
                # Outlier / lost-lock guard: only anchor when the predicted
                # position is close to the projected track point.  During a long
                # outage the dead-reckoned prediction can drift beyond the track's
                # local search window and snap onto a wrong (e.g. parallel or
                # returning) segment; anchoring there injects a catastrophic pull.
                # Rejecting the association (and not advancing the arc-length lock)
                # bounds the error to the pure dead-reckoning level instead.
                if np.linalg.norm(pos_pred[:2] - p_star[:2]) <= p_cfg['track_gate_m']:
                    s_track = s_new
                    track_noise = _track_covariance(
                        gtsam, t_hat, n_hat,
                        p_cfg['sigma_lon'], p_cfg['sigma_lat'], p_cfg['sigma_up'],
                    )
                    graph.push_back(gtsam.GPSFactor(X(k), p_star, track_noise))
            do_update = True

        # ── Shared iSAM2 solve + state extraction ───────────────────────
        if do_update:
            isam.update(graph, values)
            isam.update()
            isam.update()
            result = isam.calculateEstimate()

            graph  = gtsam.NonlinearFactorGraph()
            values = gtsam.Values()

            pose_k = result.atPose3(X(k))
            vel_k  = result.atVector(V(k))
            bias_k = result.atConstantBias(B(k))

            pose_prev      = pose_k
            vel_prev       = vel_k
            bias_prev      = bias_k
            nav_state_prev = gtsam.NavState(pose_k, vel_k)

            pim.resetIntegrationAndSetBias(bias_k)

            # Keep the along-track estimate consistent with the corrected pose.
            if not in_outage:
                proj_c = track.project(
                    np.asarray(pose_k.translation()), s_track, np.nan,
                    p_cfg['track_search_window_m'], np.pi,
                )
                if proj_c is not None:
                    s_track = proj_c[3]

            # Overwrite propagated output with corrected (smoothed) values
            p_out[i + 1] = pose_k.translation()
            v_out[i + 1] = vel_k
            R_k           = pose_k.rotation().matrix()
            r_out[i + 1] = _rbn_to_rpy(R_k)
            b_a = bias_k.accelerometer()
            b_g = bias_k.gyroscope()
            b_acc_out[i + 1] = b_a
            b_gyr_out[i + 1] = b_g

            # Marginal covariance (extracted at GPS rate, ~1 Hz)
            try:
                cov_pose = isam.marginalCovariance(X(k))
                cov_vel  = isam.marginalCovariance(V(k))
                cov_bias = isam.marginalCovariance(B(k))
                std_pos[i + 1]    = np.sqrt(np.maximum(np.diag(cov_pose[3:6, 3:6]), 0))
                std_vel[i + 1]    = np.sqrt(np.maximum(np.diag(cov_vel),            0))
                std_orient[i + 1] = np.sqrt(np.maximum(np.diag(cov_pose[0:3, 0:3]), 0))
                std_b_acc[i + 1]  = np.sqrt(np.maximum(np.diag(cov_bias[0:3, 0:3]), 0))
                std_b_gyr[i + 1]  = np.sqrt(np.maximum(np.diag(cov_bias[3:6, 3:6]), 0))
            except RuntimeError:
                # Indeterminate marginal (GTSAM RuntimeError), legitimate on the
                # first GPS steps — std_* stays at held/zero there (reported
                # uncertainty only, trajectory unaffected). Narrowed from
                # `except Exception` so a real bug above is no longer masked.
                pass

    return {
        'p':           p_out,
        'v':           v_out,
        'r':           r_out,
        'bias_acc':    b_acc_out,
        'bias_gyr':    b_gyr_out,
        'std_pos':     std_pos,
        'std_vel':     std_vel,
        'std_orient':  std_orient,
        'std_bias_acc': std_b_acc,
        'std_bias_gyr': std_b_gyr,
    }
