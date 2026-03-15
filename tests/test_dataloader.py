"""
Tests for data_loader.py.

Covers:
  - NavigationData.validate() contract (shapes, dtypes, required fields)
  - GPS mask generation (1 Hz from 100 Hz → every 100th sample)
  - FLU frame convention (stationary accel = [0,0,+9.81])
  - GRAVITY constant = [0,0,-9.81] in ENU
  - lla0 set to Karlsruhe reference
  - load_kitti_pickle (requires KITTI .p file, skipped otherwise)
"""
import numpy as np
import pytest
from conftest import make_nav_data, LLA0_DEFAULT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_nav(N=50):
    return make_nav_data(N=N, sample_rate=10.0)


# ---------------------------------------------------------------------------
# NavigationData validation contract
# ---------------------------------------------------------------------------

class TestNavigationDataValidation:

    def test_validate_passes_on_consistent_data(self):
        nd = _base_nav(50)
        nd.validate()   # must not raise

    def test_validate_rejects_mismatched_accel_shape(self):
        nd = _base_nav(50)
        nd.accel_flu = np.zeros((49, 3))   # one row short
        with pytest.raises(AssertionError):
            nd.validate()

    def test_validate_rejects_mismatched_gyro_shape(self):
        nd = _base_nav(50)
        nd.gyro_flu = np.zeros((50, 2))    # wrong number of columns
        with pytest.raises(AssertionError):
            nd.validate()

    def test_validate_rejects_non_bool_gps_available(self):
        nd = _base_nav(50)
        nd.gps_available = nd.gps_available.astype(int)   # int, not bool
        with pytest.raises(AssertionError):
            nd.validate()

    def test_validate_requires_gps_speed_mps(self):
        nd = _base_nav(50)
        nd.gps_speed_mps = None
        with pytest.raises(AssertionError):
            nd.validate()

    def test_validate_requires_gps_cog_rad(self):
        nd = _base_nav(50)
        nd.gps_cog_rad = None
        with pytest.raises(AssertionError):
            nd.validate()

    def test_validate_rejects_wrong_gps_speed_shape(self):
        nd = _base_nav(50)
        nd.gps_speed_mps = np.zeros(49)    # one short
        with pytest.raises(AssertionError):
            nd.validate()


# ---------------------------------------------------------------------------
# GPS mask generation
# ---------------------------------------------------------------------------

class TestGpsMaskGeneration:

    def test_kitti_1hz_from_100hz(self):
        """At 100 Hz, 1 Hz GPS → True every 100 samples, False in between."""
        N = 1000
        gps_avail = np.zeros(N, dtype=bool)
        gps_avail[::100] = True         # pattern used by load_kitti_pickle

        assert gps_avail[0]   == True
        assert gps_avail[100] == True
        assert gps_avail[50]  == False
        assert gps_avail[99]  == False
        assert np.sum(gps_avail) == 10  # 10 GPS updates in 1000 samples

    def test_gps_mask_dtype_is_bool(self):
        nd = _base_nav(50)
        assert nd.gps_available.dtype == bool

    def test_no_gps_by_default(self):
        nd = make_nav_data(N=50, sample_rate=10.0)
        assert not np.any(nd.gps_available), "Default nav has no GPS"

    def test_gps_available_supplied_correctly(self):
        N = 100
        gps = np.zeros(N, dtype=bool)
        gps[::10] = True
        nd = make_nav_data(N=N, sample_rate=10.0, gps_available=gps)
        assert np.sum(nd.gps_available) == 10


# ---------------------------------------------------------------------------
# FLU frame convention
# ---------------------------------------------------------------------------

class TestFluFrameConvention:

    def test_stationary_accel_positive_z(self):
        """
        A sensor at rest in FLU frame measures the reaction to gravity.
        Gravity in ENU = [0, 0, -9.81].
        Specific force = coord_accel - g = [0,0,0] - [0,0,-9.81] = [0,0,+9.81].
        """
        nd = make_nav_data(N=10, accel_flu=np.tile([0.0, 0.0, 9.81], (10, 1)))
        assert nd.accel_flu[0, 2] > 0, "FLU Up axis must be positive for stationary sensor"
        np.testing.assert_allclose(nd.accel_flu[0], [0.0, 0.0, 9.81], atol=1e-9)

    def test_stationary_accel_zero_horizontal(self):
        """Stationary sensor has no horizontal specific force."""
        nd = make_nav_data(N=10, accel_flu=np.tile([0.0, 0.0, 9.81], (10, 1)))
        np.testing.assert_allclose(nd.accel_flu[0, :2], [0.0, 0.0], atol=1e-9)

    def test_gravity_constant_is_negative_enu_z(self):
        """GRAVITY constant in all filter modules must be [0, 0, -9.81]."""
        from filters import ekf_vanilla
        np.testing.assert_allclose(ekf_vanilla.GRAVITY, [0.0, 0.0, -9.81], atol=1e-12)

    def test_specific_force_formula(self):
        """
        Verify: specific_force = coord_accel - gravity.
        Stationary → coord_accel = 0, gravity = [0,0,-9.81] in ENU,
        so specific_force = [0,0,+9.81].
        """
        g_enu = np.array([0.0, 0.0, -9.81])
        coord_accel = np.zeros(3)
        specific_force = coord_accel - g_enu
        np.testing.assert_allclose(specific_force, [0.0, 0.0, 9.81], atol=1e-12)


# ---------------------------------------------------------------------------
# LLA0 convention
# ---------------------------------------------------------------------------

class TestLla0Convention:

    def test_lla0_equals_first_lla(self):
        """
        make_nav_data must set lla0 = lla[0] so that
        geodetic2enu(lla[0], lla0) returns [0, 0, 0] (ENU origin at start).
        """
        nd = make_nav_data(N=50, sample_rate=10.0)
        np.testing.assert_allclose(nd.lla0, nd.lla[0], atol=1e-7)

    def test_lla0_matches_default_reference(self):
        """Without custom lla0, the reference is the Munich test point."""
        nd = make_nav_data(N=20)
        np.testing.assert_allclose(nd.lla0, LLA0_DEFAULT, atol=1e-7)

    def test_custom_lla0_is_respected(self):
        custom = np.array([40.4168, -3.7038, 650.0])   # Madrid
        nd = make_nav_data(N=20, lla0=custom)
        np.testing.assert_allclose(nd.lla0, custom, atol=1e-12)


# ---------------------------------------------------------------------------
# load_kitti_pickle (skipped if KITTI .p file absent)
# ---------------------------------------------------------------------------

@pytest.mark.requires_data
class TestLoadKittiPickle:

    def test_returns_navigation_data(self, kitti_data):
        kitti_data.validate()   # must not raise

    def test_sample_rate_is_100hz(self, kitti_data):
        assert kitti_data.sample_rate == 100.0

    def test_gps_rate_is_1hz(self, kitti_data):
        assert kitti_data.gps_rate == 1.0

    def test_kitti_gps_update_count(self, kitti_data):
        """45700 samples at 100 Hz, 1 Hz GPS → ~457 GPS updates."""
        n_gps = np.sum(kitti_data.gps_available)
        assert 400 < n_gps < 500, f"Expected ~457 GPS updates, got {n_gps}"

    def test_kitti_accel_shape(self, kitti_data):
        N = len(kitti_data.accel_flu)
        assert kitti_data.accel_flu.shape == (N, 3)

    def test_kitti_gps_available_dtype(self, kitti_data):
        assert kitti_data.gps_available.dtype == bool

    def test_kitti_lla0_set(self, kitti_data):
        assert kitti_data.lla0 is not None
        assert kitti_data.lla0.shape == (3,)
