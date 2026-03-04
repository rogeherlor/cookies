"""
Shared fixtures and helpers for EKF tests.

Adds the EKF source directory to sys.path so that ekf_core, data_loader, etc.
can be imported directly.
"""
import sys
from pathlib import Path

# Add the EKF Python source directory to the import path
_ekf_src = str(Path(__file__).resolve().parent.parent / "scripts" / "positioning" / "python")
if _ekf_src not in sys.path:
    sys.path.insert(0, _ekf_src)
