"""
Data Loader for Navigation Systems

This module provides standardized data loading from various sources:
- KITTI dataset (.p pickle files, ai-imu-dr format, 100 Hz)
- COOKIES dataset

All loaders return data in a standardized format for use with navigation algorithms.
"""

import os
import pickle
import numpy as np
import pymap3d as pm
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class NavigationData:
    """
    Standardized navigation data structure.
    
    All arrays are Nx3 numpy arrays where N is the number of samples.
    """
    accel_flu: np.ndarray      # Accelerometer data in FLU frame [forward, left, up] (m/s²)
    gyro_flu: np.ndarray       # Gyroscope data in FLU frame [roll rate, pitch rate, yaw rate] (rad/s)
    vel_enu: np.ndarray        # Velocity in ENU frame [east, north, up] (m/s)
    lla: np.ndarray            # Position in geodetic coordinates [lat, lon, alt] (deg, deg, m)
    orient: np.ndarray         # Orientation as Euler angles [roll, pitch, yaw] (rad)
    gps_available: np.ndarray  # Boolean array indicating which samples have actual GPS measurements (N,)

    # Metadata
    sample_rate: float         # Sampling rate in Hz
    dataset_name: str          # Name/identifier of the dataset
    gps_rate: float            # GPS update rate in Hz
    
    # GNSS derived observables (aligned to IMU timeline)
    # - gps_speed_mps: Speed over ground (m/s)
    # - gps_cog_rad: Course over ground in navigation frame (rad), convention: atan2(E, N)
    # These are always provided as arrays. When GNSS is not available at an epoch,
    # entries may be NaN.
    gps_speed_mps: np.ndarray = None
    gps_cog_rad: np.ndarray = None
    
    lla0: np.ndarray = None    # ENU origin coordinates [lat, lon, alt] (deg, deg, m)
    time: np.ndarray = None    # Time array in seconds from start (optional, for asynchronous data)
    
    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.lla)
    
    def validate(self):
        """Validate that all data arrays have consistent shapes."""
        n_samples = len(self.lla)
        
        assert self.accel_flu.shape == (n_samples, 3), f"accel_flu shape mismatch: {self.accel_flu.shape}"
        assert self.gyro_flu.shape == (n_samples, 3), f"gyro_flu shape mismatch: {self.gyro_flu.shape}"
        assert self.vel_enu.shape == (n_samples, 3), f"vel_enu shape mismatch: {self.vel_enu.shape}"
        assert self.lla.shape == (n_samples, 3), f"lla shape mismatch: {self.lla.shape}"
        assert self.orient.shape == (n_samples, 3), f"orient shape mismatch: {self.orient.shape}"
        assert self.gps_available.shape == (n_samples,), f"gps_available shape mismatch: {self.gps_available.shape}"
        assert self.gps_available.dtype == bool, f"gps_available must be boolean array"

        assert self.gps_speed_mps is not None, "gps_speed_mps must be provided (use NaN when unavailable)"
        assert self.gps_cog_rad is not None, "gps_cog_rad must be provided (use NaN when unavailable)"
        assert self.gps_speed_mps.shape == (n_samples,), f"gps_speed_mps shape mismatch: {self.gps_speed_mps.shape}"
        assert self.gps_cog_rad.shape == (n_samples,), f"gps_cog_rad shape mismatch: {self.gps_cog_rad.shape}"
        
        return True


def load_kitti_pickle(filepath: str, sample_rate: float = 100.0,
                      gps_rate: float = 1.0) -> NavigationData:
    """
    Load KITTI data from ai-imu-dr pickle (.p) format at 100 Hz.

    Source: https://github.com/mbrossar/ai-imu-dr

    Coordinate frames (no conversion needed):
        Body frame  : FLU (Forward, Left, Up)  — u[:, :3] = gyro, u[:, 3:6] = accel
        Nav frame   : ENU (East, North, Up)    — v_gt = [ve, vn, vu]
        Orientation : ZYX Euler [roll, pitch, yaw] in radians

    Pickle keys:
        t        (N,)   timestamps relative to sequence start [s]
        u        (N,6)  [gyro_flu(3), accel_flu(3)] float32
        ang_gt   (N,3)  [roll, pitch, yaw] float32
        p_gt     (N,3)  ENU position relative to first GPS fix [m] float32
        v_gt     (N,3)  [ve, vn, vu] velocity [m/s] float32
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"KITTI pickle not found: {filepath}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    def _to_numpy(x):
        return x.numpy() if hasattr(x, 'numpy') else np.array(x)

    t       = _to_numpy(data['t']).astype(np.float64)
    u       = _to_numpy(data['u']).astype(np.float64)
    ang_gt  = _to_numpy(data['ang_gt']).astype(np.float64)
    p_gt    = _to_numpy(data['p_gt']).astype(np.float64)
    v_gt    = _to_numpy(data['v_gt']).astype(np.float64)

    t -= t[0]  # ensure timestamps start at 0

    gyro_flu  = u[:, :3]   # [wx, wy, wz] in FLU body frame
    accel_flu = u[:, 3:6]  # [ax, ay, az] in FLU body frame
    vel_enu   = v_gt        # [East, North, Up]
    orient    = ang_gt      # [roll, pitch, yaw]

    # Back-convert relative ENU positions to approximate geodetic coordinates.
    # Using Karlsruhe, Germany as reference (KITTI recording area).
    # The round-trip enu2geodetic → geodetic2enu is exact for the same lla0,
    # so all filters receive consistent relative positions.
    lla0 = np.array([49.0, 8.4, 110.0])
    lat, lon, alt = pm.enu2geodetic(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                                    lla0[0], lla0[1], lla0[2])
    lla = np.column_stack([lat, lon, alt])

    N = len(t)
    gps_interval = max(1, int(round(sample_rate / gps_rate)))
    gps_available = np.zeros(N, dtype=bool)
    gps_available[::gps_interval] = True

    gps_indices = np.where(gps_available)[0]
    if len(gps_indices) > 1:
        imu_between = np.diff(gps_indices)
        print(f"IMU samples between GPS updates: mean={imu_between.mean():.1f}, "
              f"min={imu_between.min()}, max={imu_between.max()}")

    vE = vel_enu[:, 0]
    vN = vel_enu[:, 1]
    gps_speed_mps = np.sqrt(vE**2 + vN**2)
    gps_cog_rad   = np.arctan2(vE, vN)

    dataset_name = Path(filepath).stem
    print(f"KITTI pickle '{dataset_name}': {N} samples at {sample_rate} Hz, "
          f"GPS at {gps_rate} Hz ({gps_available.sum()} updates), "
          f"duration={t[-1]:.1f}s")

    nav_data = NavigationData(
        accel_flu=accel_flu,
        gyro_flu=gyro_flu,
        vel_enu=vel_enu,
        lla=lla,
        orient=orient,
        gps_available=gps_available,
        sample_rate=sample_rate,
        dataset_name=dataset_name,
        gps_rate=gps_rate,
        lla0=lla0,
        time=t,
        gps_speed_mps=gps_speed_mps,
        gps_cog_rad=gps_cog_rad,
    )
    nav_data.validate()
    return nav_data


def load_cookies_data(filepath: str, target_rate: float = 100.0) -> NavigationData:
    """
    Load COOKIES dataset from log file, downsampled to a uniform grid.

    Args:
        filepath: Path to the COOKIES .log file
        target_rate: Output sample rate in Hz (default: 100 Hz).
                     The raw ~200 Hz IMU is linearly interpolated onto this grid.

    Returns:
        NavigationData object with standardized format

    COOKIES Log Format (Inartrans_v2_imu firmware):
        - Timestamp: [YYYY-MM-DD HH:MM:SS.nnnnnnnnn]
        - IMU: IMU t=<ms>; A=<ax>,<ay>,<az>; G=<gx>,<gy>,<gz>
        - GPS: G[PN] V=<A|V>; Mod=<mode>; T=<ms>; D=<DDMMYY>; Lat=<lat>; Lon=<lon>;
               Alt=<alt>; Vel=<vel>; COG=<cog>; PDOP=<pdop>

    Scaling factors (from CHANGES_IMU_GNSS.md):
        - accel: raw × 1000 (milli-g), divide by 1000 and multiply by 9.81 for m/s²
        - gyro: raw × 100 (deg/s × 100), divide by 100 and convert to rad/s
        - lat/lon: NMEA DDMM.MMMM × 10000 (integer)
        - alt: meters × 100
        - vel: m/s × 100
        - COG: degrees × 100

    Note: Longitude should be negative for western hemisphere (Madrid is ~-3.7°W)
    """
    import re
    from datetime import datetime
    from scipy.interpolate import interp1d
    import pymap3d as pm

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"COOKIES file not found: {filepath}")

    # Lists to store parsed data
    timestamps = []
    accel_raw = []  # Raw accelerometer in body frame
    gyro_raw = []   # Raw gyroscope in body frame
    gps_data = []   # GPS measurements: [timestamp, lat, lon, alt, speed, cog_rad]

    # Regular expressions for parsing
    timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]'
    imu_pattern = r'IMU t=(\d+); A=(-?\d+),(-?\d+),(-?\d+); G=(-?\d+),(-?\d+),(-?\d+)'
    gps_pattern = r'G[PN] V=([AV]); Mod=(\d+); T=(\d+); D=(\d+); Lat=(\d+); Lon=(\d+); Alt=(-?\d+); Vel=(\d+); COG=(\d+); PDOP=(\d+)'

    # Track last GPS values to detect re-printed duplicates
    last_gps_values = None

    print(f"Parsing COOKIES log file: {filepath}")
    with open(filepath, 'r') as f:
        for line in f:
            # Extract timestamp
            ts_match = re.search(timestamp_pattern, line)
            if not ts_match:
                continue

            ts_str = ts_match.group(1)
            # Handle nanosecond precision (9 digits) - Python only supports microseconds (6 digits)
            if '.' in ts_str:
                date_part, frac_part = ts_str.rsplit('.', 1)
                frac_part = frac_part[:6]
                ts_str = f"{date_part}.{frac_part}"
            ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')

            # Check for IMU data
            imu_match = re.search(imu_pattern, line)
            if imu_match:
                groups = imu_match.groups()
                # groups[0] is device timestamp (ms), groups[1:4] accel, groups[4:7] gyro
                ax, ay, az = int(groups[1]), int(groups[2]), int(groups[3])
                gx, gy, gz = int(groups[4]), int(groups[5]), int(groups[6])

                timestamps.append(ts)

                # Accel: milli-g → g → m/s²
                accel_mg = np.array([ax, ay, az]) / 1000.0
                # Gyro: dps×100 → dps
                gyro_dps = np.array([gx, gy, gz]) / 100.0

                # Apply axis transformation from COOKIES frame to EKF FLU frame
                # COOKIES: x_cookie -> -x_ekf (forward)
                #          y_cookie -> -z_ekf (up)
                #          z_cookie -> -y_ekf (left)
                accel_flu = np.array([
                    -accel_mg[0] * 9.81,      # Forward = -x_cookie * g
                    -accel_mg[2] * 9.81,      # Left = -z_cookie * g
                    -accel_mg[1] * 9.81       # Up = -y_cookie * g
                ])

                gyro_flu = np.array([
                    -np.radians(gyro_dps[0]),  # Roll rate (around forward) = -x_cookie
                    -np.radians(gyro_dps[2]),  # Pitch rate (around left) = -z_cookie
                    -np.radians(gyro_dps[1])   # Yaw rate (around up) = -y_cookie
                ])

                accel_raw.append(accel_flu)
                gyro_raw.append(gyro_flu)

            # Check for GPS data (only valid fixes: V=A)
            gps_match = re.search(gps_pattern, line)
            if gps_match:
                groups = gps_match.groups()
                validity = groups[0]
                if validity != 'A':
                    continue

                lat_s = int(groups[4])
                lon_s = int(groups[5])
                alt_s = int(groups[6])
                vel_s = int(groups[7])
                cog_s = int(groups[8])

                # Only add GPS data if values have changed (dedup re-printed epochs)
                current_gps_values = (lat_s, lon_s, alt_s, vel_s, cog_s)
                if last_gps_values is None or current_gps_values != last_gps_values:
                    # Convert from NMEA format (DDMM.MMMM stored as integer without decimal)
                    # e.g., 40264900 = 4026.4900 = 40° 26.4900' = 40 + 26.4900/60 = 40.44150°
                    lat_deg = lat_s // 1000000
                    lat_min = (lat_s % 1000000) / 10000.0
                    lat = lat_deg + lat_min / 60.0

                    # Longitude: NMEA format, variable degree digits
                    if lon_s < 10000000:  # Single digit degrees (DMM.MMMM)
                        lon_deg = lon_s // 1000000
                        lon_min = (lon_s % 1000000) / 10000.0
                        lon = lon_deg + lon_min / 60.0
                    elif lon_s < 100000000:  # Double digit degrees (DDMM.MMMM)
                        lon_deg = lon_s // 1000000
                        lon_min = (lon_s % 1000000) / 10000.0
                        lon = lon_deg + lon_min / 60.0
                    else:  # Triple digit degrees (DDDMM.MMMM)
                        lon_deg = lon_s // 1000000
                        lon_min = (lon_s % 1000000) / 10000.0
                        lon = lon_deg + lon_min / 60.0

                    # Madrid is in western hemisphere - longitude should be negative
                    if lon > 0 and lon < 30:
                        lon = -lon

                    alt = alt_s / 100.0
                    vel = float(vel_s) / 100.0           # m/s × 100 → m/s
                    cog_rad = np.radians(cog_s / 100.0)  # deg × 100 → deg → rad

                    gps_data.append([ts, lat, lon, alt, vel, cog_rad])
                    last_gps_values = current_gps_values

    if len(timestamps) == 0:
        raise ValueError("No IMU data found in COOKIES log file")

    if len(gps_data) == 0:
        raise ValueError("No GPS data found in COOKIES log file")

    # Check if we have at least 10 valid GPS measurements to start
    min_gps_required = 10
    if len(gps_data) < min_gps_required:
        raise ValueError(f"Insufficient GPS data: found {len(gps_data)} updates, need at least {min_gps_required} to start")

    print(f"Parsed {len(timestamps)} IMU samples and {len(gps_data)} unique GPS updates")

    # Convert to numpy arrays
    accel_raw = np.array(accel_raw)
    gyro_raw = np.array(gyro_raw)

    # Process GPS data first to establish lla0 from 10th GPS measurement
    gps_array = np.array(gps_data, dtype=object)

    # Use 10th GPS measurement as the origin (lla0) for ENU conversion
    lla0 = [
        float(gps_array[9, 1]),  # lat from 10th GPS
        float(gps_array[9, 2]),  # lon from 10th GPS
        float(gps_array[9, 3])   # alt from 10th GPS
    ]
    print(f"Using GPS measurement #10 as ENU origin (lla0): lat={lla0[0]:.7f}°, lon={lla0[1]:.7f}°, alt={lla0[2]:.2f}m")

    # Find timestamp of 10th GPS measurement to trim IMU data
    gps_start_time = gps_array[9, 0]

    # Trim IMU data to start from 10th GPS measurement
    imu_start_idx = 0
    for idx, ts in enumerate(timestamps):
        if ts >= gps_start_time:
            imu_start_idx = idx
            break

    timestamps = timestamps[imu_start_idx:]
    accel_raw = accel_raw[imu_start_idx:]
    gyro_raw = gyro_raw[imu_start_idx:]

    print(f"Trimmed IMU data to start from 10th GPS measurement: {len(timestamps)} samples remaining")

    # Convert timestamps to seconds from start
    t0 = timestamps[0]
    time_imu = np.array([(t - t0).total_seconds() for t in timestamps])

    # Print timestamp analysis
    dt_actual = np.diff(time_imu)
    actual_mean_rate = 1.0 / np.mean(dt_actual)
    actual_median_rate = 1.0 / np.median(dt_actual)
    print(f"Timestamp analysis:")
    print(f"  Raw rate: mean={actual_mean_rate:.1f} Hz, median={actual_median_rate:.1f} Hz")
    print(f"  dt range: [{np.min(dt_actual)*1000:.2f}ms, {np.max(dt_actual)*1000:.2f}ms], jitter std={np.std(dt_actual)*1000:.3f}ms")

    # Downsample to uniform target_rate grid via linear interpolation
    # This guarantees every slot has a valid IMU value (no gaps)
    sample_rate = target_rate
    time_uniform = np.arange(time_imu[0], time_imu[-1], 1.0 / target_rate)

    print(f"Downsampling ~{actual_mean_rate:.0f} Hz → {target_rate:.0f} Hz uniform grid ({len(time_uniform)} samples)")

    accel_flu = np.array([interp1d(time_imu, accel_raw[:, i], kind='linear')(time_uniform)
                          for i in range(3)]).T
    gyro_flu = np.array([interp1d(time_imu, gyro_raw[:, i], kind='linear')(time_uniform)
                         for i in range(3)]).T

    # Process GPS data - use from 10th measurement onwards, relative to new t0
    time_gps = np.array([(t - t0).total_seconds() for t in gps_array[9:, 0]])
    lat_gps = gps_array[9:, 1].astype(float)
    lon_gps = gps_array[9:, 2].astype(float)
    alt_gps = gps_array[9:, 3].astype(float)
    speed_gps = gps_array[9:, 4].astype(float)
    cog_gps = gps_array[9:, 5].astype(float)

    # If COG wasn't logged, derive it from successive GNSS positions
    if not np.all(np.isfinite(cog_gps)):
        enu_gps = np.array([pm.geodetic2enu(lat, lon, alt, lla0[0], lla0[1], lla0[2])
                            for lat, lon, alt in zip(lat_gps, lon_gps, alt_gps)], dtype=float)
        d = np.diff(enu_gps[:, 0:2], axis=0)
        cog_derived = np.arctan2(d[:, 0], d[:, 1])
        if len(cog_derived) == 0:
            cog_derived = np.array([0.0])
        cog_gps_filled = np.empty_like(cog_gps, dtype=float)
        cog_gps_filled[:] = np.nan
        cog_gps_filled[0] = cog_derived[0]
        cog_gps_filled[1:] = cog_derived
        mask = np.isfinite(cog_gps)
        cog_gps = np.where(mask, cog_gps, cog_gps_filled)

    # Interpolate GPS to uniform grid timestamps
    if len(time_gps) > 1:
        f_lat = interp1d(time_gps, lat_gps, kind='linear', fill_value='extrapolate')
        f_lon = interp1d(time_gps, lon_gps, kind='linear', fill_value='extrapolate')
        f_alt = interp1d(time_gps, alt_gps, kind='linear', fill_value='extrapolate')
        f_speed = interp1d(time_gps, speed_gps, kind='linear', fill_value='extrapolate')

        lat_interp = f_lat(time_uniform)
        lon_interp = f_lon(time_uniform)
        alt_interp = f_alt(time_uniform)
        speed_interp = f_speed(time_uniform)

        # Circular interpolation for course: interpolate sin/cos then atan2
        f_cog_sin = interp1d(time_gps, np.sin(cog_gps), kind='linear', fill_value='extrapolate')
        f_cog_cos = interp1d(time_gps, np.cos(cog_gps), kind='linear', fill_value='extrapolate')
        cog_interp = np.arctan2(f_cog_sin(time_uniform), f_cog_cos(time_uniform))
    else:
        lat_interp = np.full(len(time_uniform), lat_gps[0])
        lon_interp = np.full(len(time_uniform), lon_gps[0])
        alt_interp = np.full(len(time_uniform), alt_gps[0])
        speed_interp = np.full(len(time_uniform), speed_gps[0])
        cog_interp = np.full(len(time_uniform), cog_gps[0])

    lla = np.column_stack([lat_interp, lon_interp, alt_interp])
    enu_positions = np.array([pm.geodetic2enu(lat, lon, alt, lla0[0], lla0[1], lla0[2])
                              for lat, lon, alt in zip(lat_interp, lon_interp, alt_interp)])

    # Compute velocity from position differences
    dt_uniform = 1.0 / target_rate
    vel_enu = np.zeros((len(time_uniform), 3))
    vel_enu[1:] = np.diff(enu_positions, axis=0) / dt_uniform
    vel_enu[0] = vel_enu[1]

    # Estimate orientation from accelerometer
    orient = np.zeros((len(time_uniform), 3))
    for i in range(len(time_uniform)):
        ax, ay, az = accel_flu[i]
        orient[i, 0] = np.arctan2(ay, az)
        orient[i, 1] = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        orient[i, 2] = 0.0

    # Create GPS availability mask
    gps_rate = len(gps_data) / time_gps[-1] if len(gps_data) > 1 else 1.0
    gps_available = np.zeros(len(time_uniform), dtype=bool)
    tolerance = 0.5 / target_rate  # half a sample period
    for gps_time in time_gps:
        closest_idx = np.argmin(np.abs(time_uniform - gps_time))
        if np.abs(time_uniform[closest_idx] - gps_time) < tolerance:
            gps_available[closest_idx] = True

    dataset_name = Path(filepath).name.split('.')[0]

    # Print GPS statistics
    gps_indices = np.where(gps_available)[0]
    if len(gps_indices) > 1:
        imu_between_gps = np.diff(gps_indices)
        print(f"IMU samples between GPS updates: mean={np.mean(imu_between_gps):.1f}, min={np.min(imu_between_gps)}, max={np.max(imu_between_gps)}")
    else:
        print("Warning: Not enough GPS updates to compute inter-GPS statistics")

    print(f"Loaded {len(time_uniform)} samples at {sample_rate:.0f} Hz with {np.sum(gps_available)} GPS updates ({gps_rate:.2f} Hz)")

    nav_data = NavigationData(
        accel_flu=accel_flu,
        gyro_flu=gyro_flu,
        vel_enu=vel_enu,
        lla=lla,
        orient=orient,
        gps_available=gps_available,
        sample_rate=sample_rate,
        dataset_name=dataset_name,
        gps_rate=gps_rate,
        lla0=np.array(lla0),
        time=time_uniform,
        gps_speed_mps=speed_interp,
        gps_cog_rad=cog_interp,
    )

    nav_data.validate()

    return nav_data



def get_data_loader(filepath: str, sample_rate: float = None) -> NavigationData:
    """
    Automatically detect file type and load with appropriate loader.

    Args:
        filepath: Path to the data file
        sample_rate: Sampling rate in Hz (auto-detected per format if None:
                     100 Hz for KITTI pickle, 10 Hz target for COOKIES)

    Returns:
        NavigationData object with standardized format
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if filepath.suffix == '.p':
        sr = sample_rate if sample_rate is not None else 100.0
        return load_kitti_pickle(str(filepath), sample_rate=sr)

    elif filepath.suffix == '.log':
        sr = sample_rate if sample_rate is not None else 100.0
        return load_cookies_data(str(filepath), target_rate=sr)

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


# KITTI odometry sequence number → raw drive name mapping.
# Seq 03 is absent: no raw OXTS data was available (discarded in the ai-imu-dr paper).
# Seqs 00, 02, 05 have 2-second data gaps but are still usable.
# Best test sequences (clean data): 01, 04, 06, 07, 08, 09, 10.
KITTI_SEQ_TO_DRIVE = {
    '00': '2011_10_03_drive_0027_extract',
    '01': '2011_10_03_drive_0042_extract',
    '02': '2011_10_03_drive_0034_extract',
    # '03': not available — no raw data
    '04': '2011_09_30_drive_0016_extract',
    '05': '2011_09_30_drive_0018_extract',
    '06': '2011_09_30_drive_0020_extract',
    '07': '2011_09_30_drive_0027_extract',
    '08': '2011_09_30_drive_0028_extract',
    '09': '2011_09_30_drive_0033_extract',
    '10': '2011_09_30_drive_0034_extract',
}


def get_kitti_dataset(dataset_id: str, base_dir: Optional[Path] = None,
                      sample_rate: float = 100.0) -> NavigationData:
    """
    Load a KITTI sequence by odometry sequence number or full drive name.

    Args:
        dataset_id: Two-digit sequence number ('00'–'10', seq 03 absent) OR
                    full drive name ('2011_10_03_drive_0027_extract')
        base_dir: Override for datasets/raw_kitti/ directory
        sample_rate: Sampling rate in Hz (default: 100 Hz)

    Returns:
        NavigationData object

    Example:
        data = get_kitti_dataset('00')   # seq 00 — drive 2011_10_03_drive_0027
        data = get_kitti_dataset('08')   # seq 08 — best long urban sequence
    """
    if base_dir is None:
        script_dir = Path(__file__).parent
        base_dir = script_dir / '../../../datasets/raw_kitti'

    # Accept both '00' and full drive name
    drive_name = KITTI_SEQ_TO_DRIVE.get(dataset_id, dataset_id)
    filepath = Path(base_dir) / f'{drive_name}.p'
    return load_kitti_pickle(str(filepath), sample_rate=sample_rate)


# Hand-curated list of clean cookies sequences for LOO training/evaluation.
# Each entry maps a short ID → (dataset_folder, log_file).
# - Only validated, gap-free recordings are listed here.
# - All sequences are loaded at 100 Hz (downsampled from ~200 Hz raw).
# - Add new entries here when additional collection sessions are validated.
COOKIES_CLEAN_SEQS = {
    'c01': ('castellana_260326_5', 'ttyUSB0_2026-03-26_17-13-57.183651572.log'),
    'c02': ('castellana_260326_5', 'ttyUSB0_2026-03-26_17-19-45.172268339.log'),
    'c03': ('castellana_260326_5', 'ttyUSB0_2026-03-26_17-30-32.981970598.log'),
    'c04': ('castellana_260326_5', 'ttyUSB0_2026-03-26_17-36-31.435293876.log'),
    'c05': ('castellana_260326_5', 'ttyUSB0_2026-03-26_17-41-14.804439366.log'),
}


def get_cookies_dataset(dataset_id: str, base_dir: Optional[Path] = None,
                       sample_rate: float = 100.0, log_file: str = None) -> NavigationData:
    """
    Convenience function to load COOKIES dataset by ID.

    Args:
        dataset_id: COOKIES dataset folder name (e.g., 'castellana_260326_5')
        base_dir: Base directory for datasets (default: auto-detect from script location)
        sample_rate: Target sampling rate in Hz (default: 100 Hz, downsampled from ~200 Hz raw)
        log_file: Specific log filename to load. If None, uses first available log file.

    Returns:
        NavigationData object

    Example:
        data = get_cookies_dataset('castellana_260326_5')
        data = get_cookies_dataset('castellana_260326_5', log_file='ttyUSB1_2026-03-26_17-30-32.987323239.log')

    Note:
        Each .log file in the dataset directory represents data from a different device.
        - ttyUSB0, ttyUSB1, etc. are different physical sensors/devices
        - You should select which device contains the relevant sensor data
    """
    if base_dir is None:
        script_dir = Path(__file__).parent
        base_dir = script_dir / '../../../datasets/raw_cookies'

    dataset_dir = base_dir / dataset_id

    if not dataset_dir.exists():
        raise FileNotFoundError(f"COOKIES dataset directory not found: {dataset_dir}")

    if log_file:
        filepath = dataset_dir / log_file
        if not filepath.exists():
            available_files = [f.name for f in sorted(dataset_dir.glob('*.log'))]
            raise FileNotFoundError(
                f"Log file '{log_file}' not found in {dataset_dir}.\n"
                f"Available files: {', '.join(available_files)}"
            )
        print(f"Loading COOKIES dataset '{dataset_id}' from: {log_file}")
    else:
        log_files = sorted(list(dataset_dir.glob('*.log')))

        if not log_files:
            raise FileNotFoundError(f"No .log files found in {dataset_dir}")

        filepath = log_files[0]
        device_name = filepath.name.split('_')[0]
        print(f"Loading COOKIES dataset '{dataset_id}' from: {filepath.name}")
        print(f"Note: Using device '{device_name}'. Specify log_file='<filename>' to select a different sensor.")

    return load_cookies_data(str(filepath), target_rate=sample_rate)


def get_cookies_dataset_by_id(seq_id: str, base_dir: Optional[Path] = None,
                               sample_rate: float = 100.0) -> NavigationData:
    """
    Load a clean cookies sequence by its short ID (e.g. 'c01').

    IDs are defined in COOKIES_CLEAN_SEQS. Always loads at 100 Hz.

    Example:
        data = get_cookies_dataset_by_id('c03')
    """
    if seq_id not in COOKIES_CLEAN_SEQS:
        raise KeyError(
            f"Unknown cookies sequence '{seq_id}'. "
            f"Available: {sorted(COOKIES_CLEAN_SEQS)}"
        )
    folder, log_file = COOKIES_CLEAN_SEQS[seq_id]
    return get_cookies_dataset(folder, base_dir, sample_rate, log_file=log_file)
