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
    
    # Metadata
    sample_rate: float         # Sampling rate in Hz
    dataset_name: str          # Name/identifier of the dataset
    
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
    
    # Create NavigationData object
    nav_data = NavigationData(
        accel_flu=accel_flu,
        gyro_flu=gyro_flu,
        vel_enu=vel_enu,
        lla=lla,
        orient=orient,
        sample_rate=sample_rate,
        dataset_name=dataset_name
    )
    
    # Validate data consistency
    nav_data.validate()
    
    return nav_data


def load_cookies_data(filepath: str, sample_rate: float = 10.0) -> NavigationData:
    """
    Load COOKIES dataset from custom format.
    
    Args:
        filepath: Path to the COOKIES data file
        sample_rate: Sampling rate in Hz
    
    Returns:
        NavigationData object with standardized format
    
    TODO: Implement based on COOKIES data format
    """
    raise NotImplementedError("COOKIES data loader not yet implemented")



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
