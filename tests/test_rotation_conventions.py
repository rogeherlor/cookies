"""
Tests for FLU/ENU rotation conventions.

Convention summary (tested here):
  Body frame : FLU  — x = Forward, y = Left,  z = Up
  Nav  frame : ENU  — x = East,    y = North,  z = Up

  Rotation matrix R_bn: body → navigation
  Built as R_bn = Rz(yaw) @ Ry(pitch) @ Rx(roll)   (ZYX Euler)

  yaw = 0     → Forward ≡ East      (R_bn @ [1,0,0] = [1,0,0])
  yaw = π/2   → Forward ≡ North     (R_bn @ [1,0,0] = [0,1,0])
  pitch > 0   → nose tilts up       (R_bn @ [1,0,0] has positive ENU z)

  Gravity in ENU: g_enu = [0, 0, -9.81]
  Stationary specific force in body: [0, 0, +9.81]
    (sensor reads reaction to gravity, not gravity itself)

Tests use internal helper functions imported directly from the filter modules:
  ekf_vanilla  : _euler_to_Rbn, _skew, GRAVITY
  eskf_vanilla : _qfrom_euler, _qto_Rbn, _qto_rpy, _qnorm
"""
import numpy as np
import pytest
from math import pi

from filters import ekf_vanilla, eskf_vanilla


# ---------------------------------------------------------------------------
# 1. Euler-angle R_bn (ekf_vanilla)
# ---------------------------------------------------------------------------

class TestEulerRbn:

    def test_identity_at_zero_angles(self):
        R = ekf_vanilla._euler_to_Rbn([0.0, 0.0, 0.0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_yaw0_forward_maps_to_east(self):
        """At yaw=0, FLU Forward (body x) → ENU East (nav x)."""
        R = ekf_vanilla._euler_to_Rbn([0.0, 0.0, 0.0])
        nav = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(nav, [1.0, 0.0, 0.0], atol=1e-12)

    def test_yaw0_left_maps_to_north(self):
        """At yaw=0, FLU Left (body y) → ENU North (nav y)."""
        R = ekf_vanilla._euler_to_Rbn([0.0, 0.0, 0.0])
        nav = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(nav, [0.0, 1.0, 0.0], atol=1e-12)

    def test_yaw0_up_maps_to_up(self):
        """At yaw=0, FLU Up (body z) → ENU Up (nav z)."""
        R = ekf_vanilla._euler_to_Rbn([0.0, 0.0, 0.0])
        nav = R @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(nav, [0.0, 0.0, 1.0], atol=1e-12)

    def test_yaw90_forward_maps_to_north(self):
        """At yaw=π/2, FLU Forward → ENU North."""
        R = ekf_vanilla._euler_to_Rbn([0.0, 0.0, pi / 2])
        nav = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(nav, [0.0, 1.0, 0.0], atol=1e-12)

    def test_yaw90_left_maps_to_west(self):
        """At yaw=π/2, FLU Left (body y) → ENU West (nav -x)."""
        R = ekf_vanilla._euler_to_Rbn([0.0, 0.0, pi / 2])
        nav = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(nav, [-1.0, 0.0, 0.0], atol=1e-12)

    def test_yaw180_forward_maps_to_west(self):
        """At yaw=π, FLU Forward → ENU West."""
        R = ekf_vanilla._euler_to_Rbn([0.0, 0.0, pi])
        nav = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(nav, [-1.0, 0.0, 0.0], atol=1e-12)

    def test_positive_pitch_tilts_nose_down(self):
        """
        FLU + ZYX Euler convention: positive pitch rotates around body-y (Left).
        Right-hand rule around +y (Left): Forward (+x) rotates toward -z (Down).
        R_bn @ [1,0,0] at pitch=+θ = [cosθ, 0, -sinθ] → z < 0 (nose DOWN in ENU).
        """
        R = ekf_vanilla._euler_to_Rbn([0.0, pi / 6, 0.0])
        nav = R @ np.array([1.0, 0.0, 0.0])
        assert nav[2] < 0, \
            f"Positive pitch in FLU/ENU tilts nose DOWN (z<0). Got z={nav[2]:.4f}"

    def test_negative_pitch_tilts_nose_up(self):
        """Negative pitch tilts Forward axis above horizontal (positive ENU z)."""
        R = ekf_vanilla._euler_to_Rbn([0.0, -pi / 6, 0.0])
        nav = R @ np.array([1.0, 0.0, 0.0])
        assert nav[2] > 0, \
            f"Negative pitch in FLU/ENU tilts nose UP (z>0). Got z={nav[2]:.4f}"

    @pytest.mark.parametrize("rpy", [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, 0.0, pi / 3],
        [0.3, -0.2, 1.5],
        [-pi / 6, pi / 8, -pi / 4],
    ])
    def test_rbn_is_orthogonal(self, rpy):
        R = ekf_vanilla._euler_to_Rbn(rpy)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10,
                                   err_msg=f"R @ R.T != I for rpy={rpy}")
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10,
                                   err_msg=f"det(R) != 1 for rpy={rpy}")

    def test_gravity_cancellation_stationary(self):
        """
        Rbn^T @ g_enu gives gravity in body frame.
        At identity rotation: R^T @ [0,0,-9.81] = [0,0,-9.81].
        Sensor reads specific force = a - g = 0 - (-9.81) = +9.81 along Up.
        """
        R = ekf_vanilla._euler_to_Rbn([0.0, 0.0, 0.0])
        g_enu = ekf_vanilla.GRAVITY                       # [0, 0, -9.81]
        g_body = R.T @ g_enu
        np.testing.assert_allclose(g_body, [0.0, 0.0, -9.81], atol=1e-12)
        # Specific force (what IMU reads) = 0 - g_body = [0, 0, +9.81]
        specific_force = np.zeros(3) - g_body
        np.testing.assert_allclose(specific_force, [0.0, 0.0, 9.81], atol=1e-12)

    def test_gravity_constant_enu(self):
        """GRAVITY must be [0, 0, -9.81] in ENU (negative z)."""
        np.testing.assert_allclose(ekf_vanilla.GRAVITY, [0.0, 0.0, -9.81], atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Quaternion R_bn (eskf_vanilla)
# ---------------------------------------------------------------------------

class TestQuaternionRbn:

    def test_identity_quaternion_gives_identity_Rbn(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = eskf_vanilla._qto_Rbn(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("rpy", [
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.0, -0.2, 0.0],
        [0.0, 0.0, pi / 3],
        [0.3, -0.1, 1.2],
        [-0.15, 0.08, -0.9],
    ])
    def test_euler_quaternion_Rbn_consistency(self, rpy):
        """
        Euler-based and quaternion-based Rbn must be identical for all angles.
        This is the cross-filter consistency check between EKF and ESKF/IEKF.
        """
        R_euler = ekf_vanilla._euler_to_Rbn(rpy)
        q       = eskf_vanilla._qfrom_euler(*rpy)
        R_quat  = eskf_vanilla._qto_Rbn(q)
        np.testing.assert_allclose(R_euler, R_quat, atol=1e-10,
                                   err_msg=f"Euler/Quat Rbn mismatch at rpy={rpy}")

    def test_yaw90_forward_to_north_quaternion(self):
        """Quaternion path: yaw=π/2 → Forward → North, same as Euler path."""
        q   = eskf_vanilla._qfrom_euler(0.0, 0.0, pi / 2)
        R   = eskf_vanilla._qto_Rbn(q)
        nav = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(nav, [0.0, 1.0, 0.0], atol=1e-12)

    @pytest.mark.parametrize("rpy", [
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, 0.0, 1.0],
        [0.1, -0.2, 0.8],
    ])
    def test_qto_rpy_roundtrip(self, rpy):
        """Euler → quaternion → Euler must roundtrip."""
        rpy_in = np.array(rpy)
        q      = eskf_vanilla._qfrom_euler(*rpy_in)
        rpy_out = eskf_vanilla._qto_rpy(q)
        np.testing.assert_allclose(rpy_in, rpy_out, atol=1e-10,
                                   err_msg=f"RPY roundtrip failed for {rpy}")

    def test_qnorm_gives_unit_quaternion(self):
        q = np.array([2.0, 1.0, 0.5, 0.3])
        q_normed = eskf_vanilla._qnorm(q)
        np.testing.assert_allclose(np.linalg.norm(q_normed), 1.0, atol=1e-12)

    def test_qfrom_euler_returns_unit_quaternion(self):
        q = eskf_vanilla._qfrom_euler(0.3, -0.1, 1.2)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-12)

    @pytest.mark.parametrize("rpy", [
        [0.0, 0.0, 0.0],
        [0.3, -0.2, 1.5],
    ])
    def test_qto_Rbn_is_orthogonal(self, rpy):
        q = eskf_vanilla._qfrom_euler(*rpy)
        R = eskf_vanilla._qto_Rbn(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Skew-symmetric matrix (ekf_vanilla)
# ---------------------------------------------------------------------------

class TestSkewSymmetric:

    def test_antisymmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        S = ekf_vanilla._skew(v)
        np.testing.assert_allclose(S + S.T, np.zeros((3, 3)), atol=1e-12)

    def test_diagonal_is_zero(self):
        v = np.array([4.0, 5.0, 6.0])
        S = ekf_vanilla._skew(v)
        np.testing.assert_allclose(np.diag(S), np.zeros(3), atol=1e-12)

    @pytest.mark.parametrize("v,u", [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ([0.1, -0.3, 0.7], [-0.5, 0.2, 1.1]),
    ])
    def test_cross_product_equivalence(self, v, u):
        """skew(v) @ u must equal v × u."""
        v, u = np.array(v), np.array(u)
        np.testing.assert_allclose(
            ekf_vanilla._skew(v) @ u,
            np.cross(v, u),
            atol=1e-12,
        )

    def test_shape_is_3x3(self):
        S = ekf_vanilla._skew(np.array([1.0, 2.0, 3.0]))
        assert S.shape == (3, 3)

    def test_zero_vector_gives_zero_matrix(self):
        S = ekf_vanilla._skew(np.zeros(3))
        np.testing.assert_allclose(S, np.zeros((3, 3)), atol=1e-12)
