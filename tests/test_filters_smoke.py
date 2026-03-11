"""
Smoke tests for all 7 INS/GNSS filters.

Verifies the universal filter contract:
  - run() accepts (nav_data, params, outage_config, use_3d_rotation)
  - Returns a dict with all 10 required keys
  - All arrays have shape (N, 3)
  - No NaN or Inf anywhere
  - std_* arrays are non-negative
  - 2D rotation mode and outage_config are accepted without error
"""
import importlib
import numpy as np
import pytest

from conftest import make_nav_data

# ---------------------------------------------------------------------------
# All filter names
# ---------------------------------------------------------------------------
ALL_FILTERS = [
    "ekf_vanilla",
    "ekf_enhanced",
    "eskf_vanilla",
    "eskf_enhanced",
    "iekf_vanilla",
    "iekf_enhanced",
    "imu_only",
]

REQUIRED_OUTPUT_KEYS = {
    "p", "v", "r",
    "bias_acc", "bias_gyr",
    "std_pos", "std_vel", "std_orient",
    "std_bias_acc", "std_bias_gyr",
}

STD_KEYS = ["std_pos", "std_vel", "std_orient", "std_bias_acc", "std_bias_gyr"]


@pytest.fixture(params=ALL_FILTERS)
def filter_module(request):
    return importlib.import_module(f"filters.{request.param}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFilterInterface:

    def test_runs_without_error(self, filter_module, stationary_nav_data, tight_params):
        result = filter_module.run(
            stationary_nav_data,
            params=tight_params,
            outage_config=None,
            use_3d_rotation=True,
        )
        assert result is not None

    def test_output_has_all_required_keys(self, filter_module, stationary_nav_data, tight_params):
        result = filter_module.run(stationary_nav_data, params=tight_params)
        missing = REQUIRED_OUTPUT_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_output_shapes_are_N_by_3(self, filter_module, stationary_nav_data, tight_params):
        N = len(stationary_nav_data.accel_flu)
        result = filter_module.run(stationary_nav_data, params=tight_params)
        for key in REQUIRED_OUTPUT_KEYS:
            arr = result[key]
            assert arr.shape == (N, 3), \
                f"filter={filter_module.__name__}, key={key}: shape {arr.shape} != ({N}, 3)"

    def test_no_nan_in_outputs(self, filter_module, stationary_nav_data, tight_params):
        result = filter_module.run(stationary_nav_data, params=tight_params)
        for key in REQUIRED_OUTPUT_KEYS:
            assert not np.any(np.isnan(result[key])), \
                f"NaN found in {key} for {filter_module.__name__}"

    def test_no_inf_in_outputs(self, filter_module, stationary_nav_data, tight_params):
        result = filter_module.run(stationary_nav_data, params=tight_params)
        for key in REQUIRED_OUTPUT_KEYS:
            assert not np.any(np.isinf(result[key])), \
                f"Inf found in {key} for {filter_module.__name__}"

    def test_std_arrays_nonnegative(self, filter_module, stationary_nav_data, tight_params):
        """
        imu_only returns zeros (not negative), so >= 0 holds for all 7 filters.
        Kalman filters should have positive uncertainty (>= 0 is the weaker check).
        """
        result = filter_module.run(stationary_nav_data, params=tight_params)
        for key in STD_KEYS:
            assert np.all(result[key] >= 0.0), \
                f"Negative value in {key} for {filter_module.__name__}"

    def test_2d_rotation_mode_accepted(self, filter_module, stationary_nav_data, tight_params):
        result = filter_module.run(
            stationary_nav_data,
            params=tight_params,
            use_3d_rotation=False,
        )
        N = len(stationary_nav_data.accel_flu)
        assert result["p"].shape == (N, 3)

    def test_outage_config_accepted(self, filter_module, stationary_nav_data, tight_params):
        outage = {"start": 0.2, "duration": 0.5}
        result = filter_module.run(
            stationary_nav_data,
            params=tight_params,
            outage_config=outage,
        )
        assert result is not None

    def test_none_params_uses_defaults(self, filter_module, stationary_nav_data):
        """Passing params=None should not crash (filter uses built-in defaults)."""
        result = filter_module.run(stationary_nav_data, params=None)
        assert result is not None
        assert result["p"].shape[1] == 3


# ---------------------------------------------------------------------------
# Kalman-filter-specific: covariance must be positive after initialization
# ---------------------------------------------------------------------------

KALMAN_FILTERS = [f for f in ALL_FILTERS if f != "imu_only"]


@pytest.fixture(params=KALMAN_FILTERS)
def kalman_module(request):
    return importlib.import_module(f"filters.{request.param}")


class TestKalmanCovariancePositive:

    def test_std_pos_positive_after_first_step(self, kalman_module, stationary_nav_data, tight_params):
        """Kalman filters should have strictly positive position uncertainty."""
        result = kalman_module.run(stationary_nav_data, params=tight_params)
        assert np.all(result["std_pos"][1:] > 0.0), \
            f"std_pos has non-positive values for {kalman_module.__name__}"

    def test_std_vel_positive_after_first_step(self, kalman_module, stationary_nav_data, tight_params):
        result = kalman_module.run(stationary_nav_data, params=tight_params)
        assert np.all(result["std_vel"][1:] > 0.0), \
            f"std_vel has non-positive values for {kalman_module.__name__}"
