"""
Data Loader for Navigation Systems

This module provides standardized data loading from various sources:
- KITTI dataset (.mat files)
- COOKIES dataset
- Other custom formats

All loaders return data in a standardized format for use with navigation algorithms.
"""

import os
import numpy as np
from scipy.io import loadmat
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


def load_kitti_mat(filepath: str, sample_rate: float = 10.0) -> NavigationData:
    """
    Load KITTI dataset from .mat file.
    
    Args:
        filepath: Path to the .mat file
        sample_rate: Sampling rate in Hz (default: 10 Hz)
    
    Returns:
        NavigationData object with standardized format
    
    Expected .mat structure:
        - accel_flu: Nx3 array (accelerometer in FLU frame)
        - gyro_flu: Nx3 array (gyroscope in FLU frame)
        - vel_enu: Nx3 array (velocity in ENU frame)
        - lla: Nx3 array (latitude, longitude, altitude)
        - rpy: Nx3 array (roll, pitch, yaw)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"KITTI file not found: {filepath}")
    
    # Load MATLAB file
    data = loadmat(filepath)
    
    # Extract data arrays
    accel_flu = np.array(data['accel_flu'], dtype=np.float64)
    gyro_flu = np.array(data['gyro_flu'], dtype=np.float64)
    vel_enu = np.array(data['vel_enu'], dtype=np.float64)
    lla = np.array(data['lla'], dtype=np.float64)
    orient = np.array(data['rpy'], dtype=np.float64)
    
    # Get dataset name from filename
    dataset_name = Path(filepath).stem
    
    # Create GPS availability mask (1 Hz updates)
    # Assume GPS arrives at 1 Hz, so mark every (sample_rate) samples
    n_samples = len(lla)
    gps_rate = 1.0  # Hz
    gps_interval = int(sample_rate / gps_rate)
    gps_available = np.zeros(n_samples, dtype=bool)
    gps_available[::gps_interval] = True  # Mark GPS updates at 1 Hz
    
    # Compute IMU samples between GPS updates statistics
    gps_indices = np.where(gps_available)[0]
    if len(gps_indices) > 1:
        imu_between_gps = np.diff(gps_indices)
        mean_imu_between = np.mean(imu_between_gps)
        min_imu_between = np.min(imu_between_gps)
        max_imu_between = np.max(imu_between_gps)
        print(f"IMU samples between GPS updates: mean={mean_imu_between:.1f}, min={min_imu_between}, max={max_imu_between}")
    
    print(f"KITTI data: {n_samples} samples at {sample_rate} Hz, GPS updates at {gps_rate} Hz ({np.sum(gps_available)} updates)")

    # Derive GNSS speed/course from provided ENU velocity (used as GNSS velocity observation)
    vE = vel_enu[:, 0]
    vN = vel_enu[:, 1]
    gps_speed_mps = np.sqrt(vE**2 + vN**2)
    gps_cog_rad = np.arctan2(vE, vN)  # atan2(E, N)
    
    # Create NavigationData object
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
        lla0=lla[0].copy(),  # ENU origin = first GPS position
        gps_speed_mps=gps_speed_mps,
        gps_cog_rad=gps_cog_rad,
    )
    
    # Validate data consistency
    nav_data.validate()
    
    return nav_data


def load_cookies_data(filepath: str, sample_rate: float = None, resample: bool = False, target_rate: float = 10.0) -> NavigationData:
    """
    Load COOKIES dataset from log file.
    
    Args:
        filepath: Path to the COOKIES .log file
        sample_rate: Auto-detected from data if None
        resample: Whether to resample to uniform rate (default: True)
        target_rate: Target sample rate in Hz if resampling (default: 10 Hz)
    
    Returns:
        NavigationData object with standardized format
    
    COOKIES Log Format:
        - Timestamp: [YYYY-MM-DD HH:MM:SS.nnnnnnnnn]
        - IMU: A = ax, ay, az;G = gx, gy, gz;Ts1 = dt; Sum = cumsum;
        - GPS: KUpd; lat_s: lat_scaled, lon_s: lon_scaled, alt_s: alt_scaled; vel_s: vel_scaled;
        
    Scaling factors (from hardware):
        - lat/lon: GPS outputs NMEA format (DDMM.MMMM) as integer without decimal point
          e.g., 40264184 = 4026.4184 = 40° 26.4184' = 40 + 26.4184/60 = 40.440306°
          Format: First 2-3 digits are degrees, remaining 6 digits are minutes × 10000
        - alt: divide by 100 to get meters
        - vel: divide by 100 to get m/s
        - accel: raw values in milli-g, divide by 1000 and multiply by 9.81 for m/s²
        - gyro: raw values in milli-dps, divide by 1000 and convert to rad/s
        
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
    gps_data = []   # GPS measurements: [timestamp, lat, lon, alt, speed_raw, cog_rad_or_nan]
    
    # Regular expressions for parsing
    timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]'
    imu_pattern = r'A = (-?\d+), (-?\d+), (-?\d+);G = (-?\d+), (-?\d+), (-?\d+)'
    # Firmware prints: "KUpd; lat_s: %ld; lon_s: %ld; alt_s: %ld;  vel_s: %ld; cog_s: %ld"
    # Some older logs may omit cog_s, so make it optional.
    gps_pattern = r'KUpd; lat_s:\s*(\d+)[,;]\s*lon_s:\s*(\d+)[,;]\s*alt_s:\s*(\d+)[,;]\s*vel_s:\s*(\d+)(?:[,;]\s*cog_s:\s*(\d+))?'
    
    # Track last GPS values to detect actual updates (GPS updates at 1Hz, but KUpd printed at ~8Hz)
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
            # Split timestamp at decimal point and truncate fractional seconds to 6 digits
            if '.' in ts_str:
                date_part, frac_part = ts_str.rsplit('.', 1)
                frac_part = frac_part[:6]  # Keep only first 6 digits (microseconds)
                ts_str = f"{date_part}.{frac_part}"
            ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
            
            # Check for IMU data
            imu_match = re.search(imu_pattern, line)
            if imu_match:
                ax, ay, az = map(int, imu_match.groups()[0:3])
                gx, gy, gz = map(int, imu_match.groups()[3:6])
                
                timestamps.append(ts)
                
                # Convert from milli-g to m/s² and milli-dps to rad/s
                accel_mg = np.array([ax, ay, az]) / 1000.0  # milli-g to g
                gyro_mdps = np.array([gx, gy, gz]) / 1000.0  # milli-dps to dps
                
                # Apply axis transformation from COOKIES frame to EKF FLU frame
                # COOKIES: x_cookie -> -x_ekf (forward)
                #          y_cookie -> -z_ekf (up)  
                #          z_cookie -> -y_ekf (left)
                # Result: [Forward, Left, Up] in EKF frame
                accel_flu = np.array([
                    -accel_mg[0] * 9.81,      # Forward = -x_cookie * g
                    -accel_mg[2] * 9.81,      # Left = -z_cookie * g
                    -accel_mg[1] * 9.81       # Up = -y_cookie * g
                ])
                
                gyro_flu = np.array([
                    -np.radians(gyro_mdps[0]),  # Roll rate (around forward) = -x_cookie
                    -np.radians(gyro_mdps[2]),  # Pitch rate (around left) = -z_cookie
                    -np.radians(gyro_mdps[1])   # Yaw rate (around up) = -y_cookie
                ])
                
                accel_raw.append(accel_flu)
                gyro_raw.append(gyro_flu)
            
            # Check for GPS data
            gps_match = re.search(gps_pattern, line)
            if gps_match:
                groups = gps_match.groups()
                lat_s, lon_s, alt_s, vel_s = map(int, groups[0:4])
                cog_s_raw = groups[4]
                cog_s_int = int(cog_s_raw) if cog_s_raw is not None else None
                
                # Only add GPS data if values have changed (actual 1Hz update, not just re-printed value)
                current_gps_values = (lat_s, lon_s, alt_s, vel_s, cog_s_int)
                if last_gps_values is None or current_gps_values != last_gps_values:
                    # Convert from NMEA format (DDMM.MMMM stored as integer without decimal)
                    # e.g., 40264184 = 4026.4184 = 40° 26.4184' = 40 + 26.4184/60 = 40.440306°
                    
                    # Latitude: format DDMM.MMMM (first 2 digits = degrees, rest = minutes)
                    lat_deg = lat_s // 1000000  # 40264184 // 1000000 = 40
                    lat_min = (lat_s % 1000000) / 10000.0  # (40264184 % 1000000) / 10000 = 26.4184
                    lat = lat_deg + lat_min / 60.0  # 40 + 26.4184/60 = 40.440306°
                    
                    # Longitude: format depends on value range
                    # For single-digit degrees (< 10°): DMM.MMMM = 7 digits (e.g., 3415075 = 3° 41.5075')
                    # For double-digit degrees (10-99°): DDMM.MMMM = 8 digits  
                    # For triple-digit degrees (100-180°): DDDMM.MMMM = 9 digits
                    # Madrid example: 3415075 = 3° 41.5075' = 3 + 41.5075/60 = 3.69179°
                    if lon_s < 10000000:  # Single digit degrees (DMM.MMMM format)
                        lon_deg = lon_s // 1000000  # 3415075 // 1000000 = 3
                        lon_min = (lon_s % 1000000) / 10000.0  # 415075 / 10000 = 41.5075
                        lon = lon_deg + lon_min / 60.0  # 3 + 41.5075/60 = 3.69179°
                    elif lon_s < 100000000:  # Double digit degrees (DDMM.MMMM format)
                        lon_deg = lon_s // 1000000
                        lon_min = (lon_s % 1000000) / 10000.0
                        lon = lon_deg + lon_min / 60.0
                    else:  # Triple digit degrees (DDDMM.MMMM format)
                        lon_deg = lon_s // 1000000
                        lon_min = (lon_s % 1000000) / 10000.0
                        lon = lon_deg + lon_min / 60.0
                    
                    # Madrid is in western hemisphere - longitude should be negative
                    # The C code doesn't log the hemisphere sign, so we apply it based on known location
                    if lon > 0 and lon < 30:  # Western Europe/Mediterranean longitude range
                        lon = -lon
                    
                    alt = alt_s / 100.0
                    # Firmware scaling (see `simplicity/Inartrans_v1_imu/flex-callbacks.c`):
                    #   vel_GNSS_u = knots * 100 / 1.94384    -> (m/s * 100)
                    #   log prints vel_s = vel_GNSS_u * 100   -> (m/s * 10000)
                    vel = float(vel_s) / 10000.0

                    def _decode_cog_to_rad(cog_s: Optional[int]) -> float:
                        """
                        Decode course-over-ground from firmware integer.

                        Firmware scaling (current decision):
                          cog_GNSS_u = radians * 1
                          log prints cog_s = cog_GNSS_u * 10000 -> (rad * 10000)
                        """
                        if cog_s is None:
                            return float('nan')
                        return float(cog_s) / 10000.0

                    cog_rad = _decode_cog_to_rad(cog_s_int) if cog_s_int is not None else float('nan')
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
    
    print(f"Parsed {len(timestamps)} IMU samples and {len(gps_data)} unique GPS updates (~{len(gps_data)/(len(timestamps)/sample_rate if sample_rate else 1):.1f} Hz)")
    
    # Convert to numpy arrays
    accel_raw = np.array(accel_raw)
    gyro_raw = np.array(gyro_raw)
    
    # Process GPS data first to establish lla0 from 10th GPS measurement
    gps_array = np.array(gps_data, dtype=object)
    
    # Use 10th GPS measurement as the origin (lla0) for ENU conversion
    # This ensures we have a stable GPS fix before starting
    lla0 = [
        float(gps_array[9, 1]),  # lat from 10th GPS
        float(gps_array[9, 2]),  # lon from 10th GPS
        float(gps_array[9, 3])   # alt from 10th GPS
    ]
    print(f"Using GPS measurement #10 as ENU origin (lla0): lat={lla0[0]:.7f}°, lon={lla0[1]:.7f}°, alt={lla0[2]:.2f}m")
    
    # Find timestamp of 10th GPS measurement to trim IMU data
    gps_start_time = gps_array[9, 0]  # datetime object
    
    # Trim IMU data to start from 10th GPS measurement
    imu_start_idx = 0
    for idx, ts in enumerate(timestamps):
        if ts >= gps_start_time:
            imu_start_idx = idx
            break
    
    # Trim arrays
    timestamps = timestamps[imu_start_idx:]
    accel_raw = accel_raw[imu_start_idx:]
    gyro_raw = gyro_raw[imu_start_idx:]
    
    print(f"Trimmed IMU data to start from 10th GPS measurement: {len(timestamps)} samples remaining")
    
    # Convert timestamps to seconds from start (now starting from 10th GPS)
    t0 = timestamps[0]
    time_imu = np.array([(t - t0).total_seconds() for t in timestamps])
    
    # Determine sample rate from actual timestamps
    if sample_rate is None:
        # Calculate actual mean sample rate from timestamp differences
        dt_actual = np.diff(time_imu)
        actual_mean_rate = 1.0 / np.mean(dt_actual)
        actual_median_rate = 1.0 / np.median(dt_actual)
        nominal_rate = 200.0  # Ts1 = 5ms (what the sensor uses internally)
        
        print(f"Timestamp analysis:")
        print(f"  Nominal rate (Ts1=5ms): {nominal_rate:.2f} Hz (used by sensor for integration)")
        print(f"  Actual mean rate: {actual_mean_rate:.2f} Hz (mean dt={np.mean(dt_actual)*1000:.2f}ms)")
        print(f"  Actual median rate: {actual_median_rate:.2f} Hz (median dt={np.median(dt_actual)*1000:.2f}ms)")
        print(f"  dt range: [{np.min(dt_actual)*1000:.2f}ms, {np.max(dt_actual)*1000:.2f}ms]")
        print(f"  Jitter std: {np.std(dt_actual)*1000:.3f}ms")
        
        # Use nominal rate for EKF (matches sensor's internal integration)
        # The timestamp jitter is due to logging/OS overhead, not actual sensor timing
        sample_rate = nominal_rate
    
    # Process GPS data - use from 10th measurement onwards, relative to new t0
    time_gps = np.array([(t - t0).total_seconds() for t in gps_array[9:, 0]])
    lat_gps = gps_array[9:, 1].astype(float)
    lon_gps = gps_array[9:, 2].astype(float)
    alt_gps = gps_array[9:, 3].astype(float)
    speed_gps = gps_array[9:, 4].astype(float)
    cog_gps = gps_array[9:, 5].astype(float)  # may contain NaN if not logged

    # If COG wasn't logged, derive it from successive GNSS positions (course over ground).
    # This keeps the interface consistent: we always provide speed + course when GNSS is available.
    if not np.all(np.isfinite(cog_gps)):
        enu_gps = np.array([pm.geodetic2enu(lat, lon, alt, lla0[0], lla0[1], lla0[2])
                            for lat, lon, alt in zip(lat_gps, lon_gps, alt_gps)], dtype=float)
        d = np.diff(enu_gps[:, 0:2], axis=0)
        dt = np.diff(time_gps)
        dt = np.maximum(dt, 1e-3)
        # Heading from displacement; pad last with previous
        cog_derived = np.arctan2(d[:, 0], d[:, 1])
        if len(cog_derived) == 0:
            cog_derived = np.array([0.0])
        cog_gps_filled = np.empty_like(cog_gps, dtype=float)
        cog_gps_filled[:] = np.nan
        cog_gps_filled[0] = cog_derived[0]
        cog_gps_filled[1:] = cog_derived
        # Prefer logged values where finite
        mask = np.isfinite(cog_gps)
        cog_gps = np.where(mask, cog_gps, cog_gps_filled)
    
    # Interpolate GPS to IMU timestamps
    if len(time_gps) > 1:
        f_lat = interp1d(time_gps, lat_gps, kind='linear', fill_value='extrapolate')
        f_lon = interp1d(time_gps, lon_gps, kind='linear', fill_value='extrapolate')
        f_alt = interp1d(time_gps, alt_gps, kind='linear', fill_value='extrapolate')
        f_speed = interp1d(time_gps, speed_gps, kind='linear', fill_value='extrapolate')
        
        lat_interp = f_lat(time_imu)
        lon_interp = f_lon(time_imu)
        alt_interp = f_alt(time_imu)
        speed_interp = f_speed(time_imu)

        # Circular interpolation for course: interpolate sin/cos then atan2
        cog_sin = np.sin(cog_gps)
        cog_cos = np.cos(cog_gps)
        f_cog_sin = interp1d(time_gps, cog_sin, kind='linear', fill_value='extrapolate')
        f_cog_cos = interp1d(time_gps, cog_cos, kind='linear', fill_value='extrapolate')
        cog_interp = np.arctan2(f_cog_sin(time_imu), f_cog_cos(time_imu))
    else:
        # If only one GPS sample, use it for all
        lat_interp = np.full(len(time_imu), lat_gps[0])
        lon_interp = np.full(len(time_imu), lon_gps[0])
        alt_interp = np.full(len(time_imu), alt_gps[0])
        speed_interp = np.full(len(time_imu), speed_gps[0])
        cog_interp = np.full(len(time_imu), cog_gps[0])
    
    lla = np.column_stack([lat_interp, lon_interp, alt_interp])
    enu_positions = np.array([pm.geodetic2enu(lat, lon, alt, lla0[0], lla0[1], lla0[2]) 
                              for lat, lon, alt in zip(lat_interp, lon_interp, alt_interp)])
    
    # Compute velocity from position differences (kept for backwards compatibility).
    # Note: you also have GNSS-derived observables `gps_speed_mps` / `gps_cog_rad`
    # which can be fused as measurements in the EKF.
    dt_imu = np.diff(time_imu)
    dt_imu = np.append(dt_imu, dt_imu[-1])  # Pad last value
    vel_enu = np.zeros((len(time_imu), 3))
    vel_enu[1:] = np.diff(enu_positions, axis=0) / dt_imu[1:, np.newaxis]
    vel_enu[0] = vel_enu[1]  # Copy first velocity
    
    # Estimate orientation from accelerometer (initial attitude)
    # Assuming stationary or low dynamics at start
    # Roll and pitch from gravity, yaw from initial heading (assume 0)
    orient = np.zeros((len(time_imu), 3))
    for i in range(len(time_imu)):
        ax, ay, az = accel_raw[i]
        # Roll (rotation about x-axis)
        orient[i, 0] = np.arctan2(ay, az)
        # Pitch (rotation about y-axis)  
        orient[i, 1] = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        # Yaw - would need magnetometer or GPS heading, set to 0 for now
        orient[i, 2] = 0.0
    
    # Create GPS availability mask based on actual unique GPS updates
    gps_rate = len(gps_data) / time_gps[-1] if len(gps_data) > 1 else 1.0
    
    # Optionally resample to uniform rate
    if resample and target_rate != sample_rate:
        print(f"Resampling to uniform {target_rate:.2f} Hz grid")
        
        # Create uniform time grid
        time_uniform = np.arange(time_imu[0], time_imu[-1], 1.0/target_rate)
        
        # Interpolate all data to uniform grid
        accel_flu = np.array([interp1d(time_imu, accel_raw[:, i], kind='linear')(time_uniform) 
                              for i in range(3)]).T
        gyro_flu = np.array([interp1d(time_imu, gyro_raw[:, i], kind='linear')(time_uniform) 
                             for i in range(3)]).T
        vel_enu_interp = np.array([interp1d(time_imu, vel_enu[:, i], kind='linear')(time_uniform) 
                                   for i in range(3)]).T
        lla_interp = np.array([interp1d(time_imu, lla[:, i], kind='linear')(time_uniform) 
                               for i in range(3)]).T
        orient_interp = np.array([interp1d(time_imu, orient[:, i], kind='linear')(time_uniform) 
                                  for i in range(3)]).T
        
        # Mark GPS availability at resampled timestamps
        # GPS is available when resampled timestamp is close to an actual GPS timestamp
        gps_available = np.zeros(len(time_uniform), dtype=bool)
        tolerance = 0.05  # 50ms tolerance for GPS matching
        for gps_time in time_gps:
            closest_idx = np.argmin(np.abs(time_uniform - gps_time))
            if np.abs(time_uniform[closest_idx] - gps_time) < tolerance:
                gps_available[closest_idx] = True
        
        sample_rate = target_rate
        time_array = time_uniform
    else:
        # Keep original asynchronous timestamps
        accel_flu = accel_raw
        gyro_flu = gyro_raw
        vel_enu_interp = vel_enu
        lla_interp = lla
        orient_interp = orient
        time_array = time_imu
        
        # Mark GPS availability at original IMU timestamps
        gps_available = np.zeros(len(time_imu), dtype=bool)
        tolerance = 0.05  # 50ms tolerance for GPS matching
        for gps_time in time_gps:
            closest_idx = np.argmin(np.abs(time_imu - gps_time))
            if np.abs(time_imu[closest_idx] - gps_time) < tolerance:
                gps_available[closest_idx] = True
    
    # Get dataset name from filename
    dataset_name = Path(filepath).parent.name
    
    # Compute IMU samples between GPS updates statistics
    gps_indices = np.where(gps_available)[0]
    if len(gps_indices) > 1:
        imu_between_gps = np.diff(gps_indices)  # Number of IMU samples between consecutive GPS updates
        mean_imu_between = np.mean(imu_between_gps)
        min_imu_between = np.min(imu_between_gps)
        max_imu_between = np.max(imu_between_gps)
        print(f"IMU samples between GPS updates: mean={mean_imu_between:.1f}, min={min_imu_between}, max={max_imu_between}")
    else:
        print("Warning: Not enough GPS updates to compute inter-GPS statistics")
    
    print(f"Loaded {len(lla_interp)} samples at nominal {sample_rate:.2f} Hz with {np.sum(gps_available)} GPS updates ({gps_rate:.2f} Hz)")
    
    # Create NavigationData object
    nav_data = NavigationData(
        accel_flu=accel_flu,
        gyro_flu=gyro_flu,
        vel_enu=vel_enu_interp,
        lla=lla_interp,
        orient=orient_interp,
        gps_available=gps_available,
        sample_rate=sample_rate,
        dataset_name=dataset_name,
        gps_rate=gps_rate,
        lla0=np.array(lla0),
        time=time_array,
        gps_speed_mps=speed_interp,
        gps_cog_rad=cog_interp,
    )
    
    nav_data.validate()
    
    return nav_data



def get_data_loader(filepath: str, sample_rate: float = 10.0) -> NavigationData:
    """
    Automatically detect file type and load with appropriate loader.
    
    Args:
        filepath: Path to the data file
        sample_rate: Sampling rate in Hz
    
    Returns:
        NavigationData object with standardized format
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Detect file type by extension
    if filepath.suffix == '.mat':
        # Check if it's in KITTI directory
        if 'kitti' in str(filepath).lower():
            return load_kitti_mat(str(filepath), sample_rate)
        else:
            # Try KITTI format by default for .mat files
            return load_kitti_mat(str(filepath), sample_rate)
    
    elif filepath.suffix == '.log':
        # COOKIES log file
        return load_cookies_data(str(filepath), sample_rate=None, resample=True, target_rate=sample_rate)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def get_kitti_dataset(dataset_id: str, base_dir: Optional[Path] = None, 
                     sample_rate: float = 10.0) -> NavigationData:
    """
    Convenience function to load KITTI dataset by ID.
    
    Args:
        dataset_id: KITTI dataset identifier (e.g., '10_03_0034')
        base_dir: Base directory for datasets (default: auto-detect from script location)
        sample_rate: Sampling rate in Hz
    
    Returns:
        NavigationData object
    
    Example:
        data = get_kitti_dataset('10_03_0034')
    """
    if base_dir is None:
        # Auto-detect base directory relative to this script
        script_dir = Path(__file__).parent
        base_dir = script_dir / '../../../datasets/raw_kitti'
    
    filepath = base_dir / f'{dataset_id}.mat'
    
    return load_kitti_mat(str(filepath), sample_rate)


def get_cookies_dataset(dataset_id: str, base_dir: Optional[Path] = None, 
                       sample_rate: float = 10.0, log_file: str = None) -> NavigationData:
    """
    Convenience function to load COOKIES dataset by ID.
    
    Args:
        dataset_id: COOKIES dataset folder name (e.g., 'castellana_270226_5')
        base_dir: Base directory for datasets (default: auto-detect from script location)
        sample_rate: Target sampling rate in Hz for resampling (default: 10 Hz)
        log_file: Specific log filename to load (e.g., 'ttyUSB0_2026-02-27_11-45-04.542636036.log'). 
                 If None, uses first available log file.
    
    Returns:
        NavigationData object
    
    Example:
        data = get_cookies_dataset('castellana_270226_5')  # Use first device
        data = get_cookies_dataset('castellana_270226_5', log_file='ttyUSB0_2026-02-27_11-45-04.542636036.log')  # Specific file
    
    Note:
        Each .log file in the dataset directory represents data from a different device.
        - ttyUSB0, ttyUSB1, etc. are different physical sensors/devices
        - You should select which device contains the relevant sensor data
    """
    if base_dir is None:
        # Auto-detect base directory relative to this script
        script_dir = Path(__file__).parent
        base_dir = script_dir / '../../../datasets/raw_cookies'
    
    dataset_dir = base_dir / dataset_id
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"COOKIES dataset directory not found: {dataset_dir}")
    
    # Select specific log file or use first available
    if log_file:
        filepath = dataset_dir / log_file
        if not filepath.exists():
            # Show available files
            available_files = [f.name for f in sorted(dataset_dir.glob('*.log'))]
            raise FileNotFoundError(
                f"Log file '{log_file}' not found in {dataset_dir}.\n"
                f"Available files: {', '.join(available_files)}"
            )
        print(f"Loading COOKIES dataset '{dataset_id}' from: {log_file}")
    else:
        # Find log files in directory
        log_files = sorted(list(dataset_dir.glob('*.log')))
        
        if not log_files:
            raise FileNotFoundError(f"No .log files found in {dataset_dir}")
        
        # Use first log file
        filepath = log_files[0]
        device_name = filepath.name.split('_')[0]
        print(f"Loading COOKIES dataset '{dataset_id}' from: {filepath.name}")
        print(f"Note: Using device '{device_name}'. Specify log_file='<filename>' to select a different sensor.")
    
    # Resample to target_rate when one is explicitly requested (default None = keep 200 Hz raw)
    do_resample = sample_rate is not None
    return load_cookies_data(str(filepath), sample_rate=None, resample=do_resample, target_rate=sample_rate or 200.0)
