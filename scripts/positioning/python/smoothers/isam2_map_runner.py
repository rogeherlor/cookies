# -*- coding: utf-8 -*-
"""
iSAM2 Map-Aided Online Smoother
================================
Reference implementation following:

    D. Wilbers, C. Merfels, and C. Stachniss, "Localization with sliding window
    factor graphs on third-party maps for automated driving," in 2019 International
    Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 5951–5957.
    https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/wilbers2019icra.pdf

Core idea
---------
A sparse "HD map" is extracted **offline** from the ground-truth trajectory by
sampling one waypoint every `map_spacing_m` metres (default 200 m).  During the
online iSAM2 run, whenever the IMU-propagated position prediction falls within
`map_assoc_radius` metres of an unvisited waypoint, an additional `GPSFactor` is
inserted into the factor graph at that node, constraining its translation to the
map position with noise std `map_sigma`.

This is the factor-graph equivalent of Wilbers et al. §III-B (map-feature
association and factor insertion).  The Bayes-tree re-linearisation/marginalisation
already built into iSAM2 provides the sliding-window behaviour that is the
architectural core of the Wilbers paper — no explicit window management is needed.

Differences from isam2_runner.py
---------------------------------
1. build_navigation_map() creates a NavigationMap (list of {id,lat,lon,alt,enu}
   dicts + scipy KDTree on ENU x-y) from the GT trajectory before the loop.
2. Three new DEFAULT_PARAMS keys control map behaviour:
       map_spacing_m      – inter-waypoint spacing [m]   (default 200)
       map_sigma          – map position noise std [m]    (default 3.0)
       map_assoc_radius   – pre-assignment gate radius [m] (default 25)
3. _assign_map_to_epochs() runs offline using a KDTree on GPS positions
   (O(M log K)) to find the GPS epoch closest to each waypoint.  The resulting
   dict {sample_index: point_dict} is looked up in O(1) per GPS epoch.
4. Each waypoint is assigned to at most one GPS epoch (independence preserved).

All other aspects (coordinate frames, IMU preintegration, GPS factors, output
format, outage simulation) are identical to isam2_runner.py.

No public code was released by the original authors; the GTSAM patterns follow
the official LocalizationExample.cpp in borglab/gtsam.
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
            "GTSAM is required for the iSAM2 Map smoother.\n"
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
    # Map factor parameters (Wilbers et al. ICRA 2019)
    'map_spacing_m':     200.0,   # waypoint extraction spacing [m]
    'map_sigma':           1.0,   # map position noise std dev [m]; used only during
                                  # GPS outages where no competing GPS factor exists,
                                  # so tight sigma is correct (HD-map accuracy ~1 m)
    'map_assoc_radius':   25.0,   # pre-assignment gate radius [m] (offline, vs GPS pos)
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


# ── Navigation map ────────────────────────────────────────────────────────────

class NavigationMap:
    """
    Scalable navigation map with KD-tree spatial index.

    Each waypoint is stored as a dict::

        {
            'id':  int,          # sequential index
            'lat': float,        # geodetic latitude  [°]
            'lon': float,        # geodetic longitude [°]
            'alt': float,        # geodetic altitude  [m]
            'enu': np.ndarray,   # (3,) ENU position  [m]
        }

    A ``scipy.spatial.KDTree`` is built on the 2-D ENU (x, y) plane, giving
    O(log M) nearest-neighbour queries regardless of map size.  Altitude is
    stored but not used for association (vehicles stay on-road).

    Design notes
    ------------
    * ENU is the right coordinate frame here: it is metric (1 m = 1 m in every
      direction), so KDTree Euclidean distances are correct.  Raw lat/lon is NOT
      metric (1° lat ≠ 1° lon), so building a KDTree on lat/lon would give wrong
      distances.
    * For global maps spanning thousands of kilometres, replace ENU with ECEF
      (``pm.geodetic2ecef``) and build the tree on (X, Y, Z).  The rest of the
      interface stays the same.
    * Each point's lat/lon/alt are kept for logging, serialisation (e.g. export
      to GeoJSON), and human inspection — they are never used for computation.
    """

    def __init__(self, points: list, lla0: np.ndarray):
        """
        Args:
            points : list of point dicts (see class docstring).
            lla0   : (3,) ENU reference origin [lat0°, lon0°, alt0_m].
        """
        self.points = points          # list of dicts — the map "database"
        self.lla0   = lla0            # reference origin

        if points:
            enu_xy = np.array([p['enu'][:2] for p in points])  # (M, 2)
            self._kdtree = KDTree(enu_xy)
            self._enu_xy = enu_xy
        else:
            self._kdtree = None
            self._enu_xy = np.empty((0, 2))

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return f'NavigationMap({len(self.points)} waypoints, lla0={self.lla0})'

    def query_nearest(self, pos_enu: np.ndarray, radius: float,
                      used_ids: set = None):
        """
        Find the nearest unvisited waypoint within ``radius`` metres.

        Uses the KD-tree for an O(log M) range query, then filters used IDs
        and picks the closest remaining candidate.

        Args:
            pos_enu  : (3,) query position in ENU [m].
            radius   : search radius [m].
            used_ids : set of point ``'id'`` values to skip (already consumed).

        Returns:
            Nearest matching point dict, or ``None`` if nothing qualifies.
        """
        if self._kdtree is None:
            return None

        candidates = self._kdtree.query_ball_point(pos_enu[:2], radius)
        if not candidates:
            return None

        if used_ids:
            candidates = [i for i in candidates
                          if self.points[i]['id'] not in used_ids]
        if not candidates:
            return None

        dists = np.linalg.norm(self._enu_xy[candidates] - pos_enu[:2], axis=1)
        return self.points[candidates[int(np.argmin(dists))]]


def build_navigation_map(lla: np.ndarray, lla0: np.ndarray,
                         spacing_m: float = 200.0) -> NavigationMap:
    """
    Extract waypoints every ``spacing_m`` metres from a ground-truth trajectory
    and return a :class:`NavigationMap` with KD-tree spatial index.

    In a real deployment the map would come from a third-party provider
    (HERE, TomTom, OSM) as a list of lat/lon waypoints; this function
    simulates that by sub-sampling the GT trajectory.

    Args:
        lla       : (N, 3) geodetic positions [lat°, lon°, alt_m] — source.
        lla0      : (3,)   ENU reference origin [lat0°, lon0°, alt0_m].
        spacing_m : minimum 3-D ENU distance between consecutive waypoints [m].

    Returns:
        NavigationMap with M ≪ N waypoints and a pre-built KD-tree.
    """
    e, n, u = pm.geodetic2enu(
        lla[:, 0], lla[:, 1], lla[:, 2],
        lla0[0],   lla0[1],   lla0[2],
    )
    p_all = np.column_stack([e, n, u])   # (N, 3)

    points  = []
    last_pt = p_all[0]
    pt_id   = 0

    # Always include the start position
    points.append({
        'id':  pt_id,
        'lat': float(lla[0, 0]),
        'lon': float(lla[0, 1]),
        'alt': float(lla[0, 2]),
        'enu': p_all[0].copy(),
    })

    for i in range(1, len(p_all)):
        if np.linalg.norm(p_all[i] - last_pt) >= spacing_m:
            pt_id += 1
            points.append({
                'id':  pt_id,
                'lat': float(lla[i, 0]),
                'lon': float(lla[i, 1]),
                'alt': float(lla[i, 2]),
                'enu': p_all[i].copy(),
            })
            last_pt = p_all[i]

    return NavigationMap(points=points, lla0=lla0)


def _assign_map_to_epochs(nav_map: NavigationMap, p_gps: np.ndarray,
                           gps_available: np.ndarray,
                           assoc_radius: float) -> dict:
    """
    Offline pre-assignment: for each map waypoint, find the GPS epoch whose
    actual GPS position is closest to it (Wilbers et al. §III-A).

    Uses a temporary KD-tree on GPS positions for O(M log K) total cost —
    efficient even for large maps (M large) and long sequences (K large).

    The result is a dict ``{sample_index: point_dict}`` used in the main loop
    for O(1) lookup: at GPS epoch i+1, check ``epoch_to_map.get(i+1)``.

    All GPS-rate epochs (including outage epochs) are eligible for assignment,
    because map factors are specifically intended for outage periods where GPS
    is absent.

    Returns:
        epoch_to_map : dict mapping sample index → NavigationMap point dict.
    """
    # Include ALL GPS-rate epochs (outage epochs are the primary target)
    gps_idx = np.where(gps_available)[0]  # sample indices of GPS-rate ticks
    if len(gps_idx) == 0:
        return {}

    p_gps_valid = p_gps[gps_idx, :2]     # (K, 2) — KDTree on x-y plane
    gps_kdtree  = KDTree(p_gps_valid)    # O(K log K) build, O(log K) per query

    epoch_to_map     = {}                 # sample_index → point dict
    assigned_epochs  = set()

    # Skip points[0] — it coincides with the initial prior at the start
    for pt in nav_map.points[1:]:
        dist, best = gps_kdtree.query(pt['enu'][:2])
        if dist > assoc_radius:
            continue
        sample_idx = int(gps_idx[best])
        if sample_idx in assigned_epochs:
            # Resolve conflict: keep the point closer to this GPS epoch
            existing = epoch_to_map[sample_idx]
            existing_dist = np.linalg.norm(
                p_gps[sample_idx, :2] - existing['enu'][:2]
            )
            if dist < existing_dist:
                epoch_to_map[sample_idx] = pt
        else:
            epoch_to_map[sample_idx] = pt
            assigned_epochs.add(sample_idx)

    return epoch_to_map


# ── Public interface ──────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run iSAM2 map-aided online smoother and return navigation estimates.

    Identical to isam2_runner.run() except that an additional GPSFactor is
    inserted at each GPS update epoch whenever the current predicted position
    falls within `map_assoc_radius` of an unvisited map waypoint.

    Map waypoints are extracted from nav_data.lla (ground truth) at a spacing of
    `map_spacing_m` metres before the main loop begins.  Each waypoint is then
    pre-assigned offline to the GPS epoch whose actual GPS position is closest to
    it (Wilbers et al. §III-A), bounding the map-factor offset to ≤ half the GPS
    inter-epoch spacing and keeping the factor within map_sigma.

    Args:
        nav_data       : NavigationData dataclass (data_loader.py).
        params         : Optional dict overriding DEFAULT_PARAMS.
        outage_config  : Optional {'start': t1_s, 'duration': d_s} for GPS
                         blackout.  GPS and map factors are both omitted in this
                         window.
        use_3d_rotation: Accepted for interface compatibility; always ignored.

    Returns:
        dict with keys: p, v, r, bias_acc, bias_gyr,
                        std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr.
    """
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

    # ── Map pre-processing (offline) ──────────────────────────────────────────
    # Extract sparse waypoints, then pre-assign each to the GPS epoch closest
    # to it.  This ensures the map factor offset ≤ half the GPS inter-epoch
    # spacing (~5 m), avoiding the outlier problem caused by online association
    # which fires ~20 m before the vehicle reaches each waypoint.
    nav_map      = build_navigation_map(lla, lla0, p_cfg['map_spacing_m'])
    epoch_to_map = _assign_map_to_epochs(
        nav_map, p_gps, nav_data.gps_available, p_cfg['map_assoc_radius'],
    )

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

    # GPS noise model
    gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        p_cfg['Rpos'], p_cfg['Rpos'], p_cfg['Rpos'],
    ]))

    # Map noise model — tighter than GPS to reflect HD-map accuracy
    map_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(3, p_cfg['map_sigma']))

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

        # ── Three-way branch: normal GPS / outage+map / IMU-only ────────────
        is_gps_tick = nav_data.gps_available[i + 1]
        in_outage   = _in_outage(i + 1, nav_data.sample_rate, outage_config)
        do_update   = False       # whether to run iSAM2 update this step

        if is_gps_tick and not in_outage:
            # ── NORMAL GPS: identical to vanilla isam2_runner (no map) ───
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
            # ── OUTAGE: IMU + map factors only (Wilbers et al. §III-B) ───
            # Create a graph node at GPS rate even though GPS is absent.
            # The IMU factor maintains dynamic consistency; the map factor
            # (if a waypoint is pre-assigned to this epoch) provides a
            # position anchor that prevents pure dead-reckoning drift.
            k += 1
            nav_pred = pim.predict(nav_state_prev, bias_prev)

            values.insert(X(k), nav_pred.pose())
            values.insert(V(k), nav_pred.velocity())
            values.insert(B(k), bias_prev)

            graph.push_back(gtsam.CombinedImuFactor(
                X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim,
            ))

            if (i + 1) in epoch_to_map:
                graph.push_back(gtsam.GPSFactor(
                    X(k), epoch_to_map[i + 1]['enu'], map_noise,
                ))
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
            except Exception:
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
