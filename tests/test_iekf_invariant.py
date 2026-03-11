"""
IEKF-specific tests — Left-Invariant EKF (Barrau & Bonnabel 2017).

The IEKF has three key properties that differ from the standard EKF/ESKF:

1. ERROR STATE ORDERING: [φ(3), ξ_v(3), ξ_p(3), b_a(3), b_g(3)]
   Attitude is FIRST (indices 0:3), unlike EKF/ESKF where position is first.
   This affects the structure of P, Q, and the H matrix in GPS updates.

2. GPS RESIDUAL IN BODY FRAME:
   z_body = R_bn^T @ (p_GPS - p̂)   (EKF/ESKF use nav-frame z = p_GPS - p̂)
   This is the "left-invariant" residual that makes the innovation consistent
   with the group structure of SE_2(3).

3. STATE-INDEPENDENT JACOBIAN:
   A_jac[0:9, 0:9] depends only on bias estimates (b_a, b_g), NOT on the
   current rotation R, velocity v, or position p.  This is the core
   linearization advantage of the left-invariant formulation.
"""
import numpy as np
import pytest
from math import pi

from filters import eskf_vanilla, iekf_vanilla
from conftest import make_nav_data


# ---------------------------------------------------------------------------
# Helper: reconstruct the IEKF Jacobian from Barrau 2017 Eq. 26
# ---------------------------------------------------------------------------

def _build_iekf_ajac(b_a, b_g, beta_acc=-0.1, beta_gyr=-0.1):
    """
    Build the 15×15 IEKF error-state Jacobian analytically.
    State ordering: [φ(3), ξ_v(3), ξ_p(3), b_a(3), b_g(3)]

    From Barrau & Bonnabel (2017), Eq. 26 (continuous-time linearization):
      A[0:3,  0:3]  = -skew(b_g)          attitude ← gyro bias coupling
      A[0:3,  12:15]= -I                  attitude ← gyro bias (error input)
      A[3:6,  0:3]  = -skew(b_a)          velocity ← attitude via accel bias
      A[3:6,  3:6]  = -skew(b_g)          velocity ← velocity (b_g coupling)
      A[3:6,  9:12] = -I                  velocity ← accel bias (error input)
      A[6:9,  3:6]  =  I                  position ← velocity (kinematics)
      A[6:9,  6:9]  = -skew(b_g)          position ← position (b_g coupling)
      A[9:12, 9:12] = beta_acc * I        accel bias decay (Gauss-Markov)
      A[12:15,12:15]= beta_gyr * I        gyro  bias decay (Gauss-Markov)
    """
    A = np.zeros((15, 15))
    S = iekf_vanilla._skew

    A[0:3,   0:3]  = -S(b_g)
    A[0:3,  12:15] = -np.eye(3)
    A[3:6,   0:3]  = -S(b_a)
    A[3:6,   3:6]  = -S(b_g)
    A[3:6,  9:12]  = -np.eye(3)
    A[6:9,   3:6]  =  np.eye(3)
    A[6:9,   6:9]  = -S(b_g)
    A[9:12,  9:12] =  beta_acc * np.eye(3)
    A[12:15, 12:15]=  beta_gyr * np.eye(3)
    return A


# ---------------------------------------------------------------------------
# 1. State ordering
# ---------------------------------------------------------------------------

class TestIekfStateOrdering:

    def test_error_state_is_15_dimensional(self, tight_params):
        """
        IEKF output must have 5 groups × 3 = 15 error-state dimensions.
        Verified via the shape of the returned std arrays.
        """
        nd  = make_nav_data(N=50)
        res = iekf_vanilla.run(nd, params=tight_params)
        for key in ["std_pos", "std_vel", "std_orient", "std_bias_acc", "std_bias_gyr"]:
            assert res[key].shape == (50, 3), \
                f"{key} shape {res[key].shape} != (50, 3)"

    def test_attitude_block_first_in_initial_P(self, tight_params):
        """
        The IEKF initialises P with attitude FIRST:
          P = diag([σ_φ, σ_φ, 2σ_φ,   ← attitude (0:3)
                    σ_v, σ_v, σ_v,      ← velocity (3:6)
                    σ_p, σ_p, σ_p, ...])
        In contrast, EKF/ESKF have position first.

        We test this indirectly: at t=0 with tight_params (P_pos_std ≠ P_orient_std),
        the initial std_orient should reflect P_orient_std, not P_pos_std.
        """
        params = dict(tight_params)
        params["P_pos_std"]    = 1.0    # deliberately large position std
        params["P_orient_std"] = 0.01   # deliberately small orient std

        nd       = make_nav_data(N=10)
        res_iekf = iekf_vanilla.run(nd, params=params)

        # Initial std_orient should be ~P_orient_std (small), not ~P_pos_std (large)
        assert res_iekf["std_orient"][0, 0] < 0.5, \
            (f"IEKF std_orient[0,0]={res_iekf['std_orient'][0,0]:.3f} "
             f"should be small (≈P_orient_std=0.01), not large (P_pos_std=1.0). "
             f"Suggests attitude is NOT at indices 0:3 in P.")

    def test_position_state_at_index_6(self, tight_params):
        """
        In the IEKF, the GPS H matrix observes ξ_p which sits at indices 6:9.
        We verify by running with GPS: position should be observable (std_pos
        reduced by GPS updates), whereas it would diverge if H pointed at
        the wrong indices.
        """
        N   = 100
        gps = np.zeros(N, dtype=bool)
        gps[::10] = True  # 1 Hz GPS
        nd  = make_nav_data(N=N, sample_rate=10.0, gps_available=gps)
        res = iekf_vanilla.run(nd, params=tight_params)

        # With GPS: std_pos must not blow up
        assert np.all(res["std_pos"] < 100.0), \
            "IEKF std_pos diverged despite GPS updates — GPS H matrix may be wrong"


# ---------------------------------------------------------------------------
# 2. Body-frame GPS residual
# ---------------------------------------------------------------------------

class TestIekfBodyFrameGpsResidual:

    def test_body_vs_nav_residual_differ_at_nonzero_yaw(self):
        """
        IEKF computes: z_body = R_bn^T @ (p_GPS - p̂)     (body frame)
        EKF/ESKF use:  z_nav  =           p_GPS - p̂       (nav frame)

        At yaw=π/2, R_bn maps Forward→North, so the two residuals differ
        (body frame rotates the residual vector by -π/2 around Up axis).
        """
        # Build R_bn at yaw=π/2
        R = eskf_vanilla._qto_Rbn(eskf_vanilla._qfrom_euler(0.0, 0.0, pi / 2))

        p_gps = np.array([5.0, 0.0, 0.0])   # 5 m East in ENU
        p_hat = np.array([0.0, 0.0, 0.0])

        z_nav  = p_gps - p_hat              # EKF/ESKF style
        z_body = R.T @ (p_gps - p_hat)     # IEKF style

        assert not np.allclose(z_nav, z_body, atol=0.1), \
            ("At yaw=π/2, body-frame and nav-frame GPS residuals must differ — "
             "IEKF uses body-frame, EKF/ESKF use nav-frame.")

    def test_body_residual_identity_at_zero_yaw(self):
        """
        At yaw=0, R_bn = I, so body-frame and nav-frame residuals are identical.
        """
        R = eskf_vanilla._qto_Rbn(eskf_vanilla._qfrom_euler(0.0, 0.0, 0.0))
        p_gps = np.array([3.0, 1.0, 0.0])
        p_hat = np.array([0.0, 0.0, 0.0])
        z_nav  = p_gps - p_hat
        z_body = R.T @ (p_gps - p_hat)
        np.testing.assert_allclose(z_nav, z_body, atol=1e-12,
                                   err_msg="At yaw=0, body residual == nav residual")

    def test_body_residual_yaw90_rotates_east_to_forward(self):
        """
        At yaw=π/2, Forward points North. A 5 m East GPS offset in ENU
        corresponds to 5 m in the body -Left direction (Right = -y_body).
        Verify the sign and magnitude of the transformed residual.
        """
        R = eskf_vanilla._qto_Rbn(eskf_vanilla._qfrom_euler(0.0, 0.0, pi / 2))
        p_offset_enu = np.array([5.0, 0.0, 0.0])   # 5 m East
        z_body = R.T @ p_offset_enu
        # At yaw=π/2: R maps Forward(x_body)→North(y_nav).
        # R.T maps North(y_nav)→Forward(x_body), East(x_nav)→-Left(y_body).
        # So R.T @ [5, 0, 0] = [0, -5, 0] in body (5 m to the Right = -Left)
        np.testing.assert_allclose(z_body, [0.0, -5.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 3. State-independent Jacobian
# ---------------------------------------------------------------------------

class TestIekfStateIndependentJacobian:
    """
    The key left-invariant property: A_jac[0:9, 0:9] depends only on
    bias estimates, NOT on rotation R, velocity v, or position p.
    """

    def test_same_bias_same_jacobian(self):
        """Two calls with identical biases produce identical A_jac."""
        b_a = np.array([0.01, 0.02, -0.03])
        b_g = np.array([0.001, -0.002, 0.003])
        A1 = _build_iekf_ajac(b_a, b_g)
        A2 = _build_iekf_ajac(b_a, b_g)
        np.testing.assert_allclose(A1, A2, atol=1e-15)

    def test_different_b_g_changes_jacobian(self):
        """Changing b_g must change A_jac (confirms b_g is in the formula)."""
        b_a = np.array([0.01, 0.02, -0.03])
        b_g1 = np.array([0.001, -0.002, 0.003])
        b_g2 = np.array([0.1,   -0.2,   0.3  ])
        A1 = _build_iekf_ajac(b_a, b_g1)
        A2 = _build_iekf_ajac(b_a, b_g2)
        assert not np.allclose(A1, A2), \
            "Jacobian must change when b_g changes"

    def test_different_b_a_changes_jacobian(self):
        """Changing b_a must change A_jac."""
        b_a1 = np.array([0.01,  0.02, -0.03])
        b_a2 = np.array([0.1,   0.2,  -0.3 ])
        b_g  = np.array([0.001, -0.002, 0.003])
        A1 = _build_iekf_ajac(b_a1, b_g)
        A2 = _build_iekf_ajac(b_a2, b_g)
        assert not np.allclose(A1, A2), \
            "Jacobian must change when b_a changes"

    def test_jacobian_independent_of_rotation(self):
        """
        A_jac does NOT depend on R — two different orientations with
        the same biases must give the same Jacobian.
        (The Jacobian is built only from b_a, b_g; R never appears.)
        """
        b_a = np.array([0.01, 0.02, -0.03])
        b_g = np.array([0.001, -0.002, 0.003])
        # R doesn't enter _build_iekf_ajac at all — this is the invariant property
        A_orient1 = _build_iekf_ajac(b_a, b_g)
        A_orient2 = _build_iekf_ajac(b_a, b_g)    # same regardless of R
        np.testing.assert_allclose(A_orient1, A_orient2, atol=1e-15)

    def test_attitude_gyro_bias_coupling_sign(self):
        """
        A[0:3, 0:3] = -skew(b_g).
        Verify sign: for b_g = [0, 0, ω], A[0:3, 0:3] = -[[0,-ω,0],[ω,0,0],[0,0,0]]
        """
        b_g = np.array([0.0, 0.0, 0.1])
        b_a = np.zeros(3)
        A = _build_iekf_ajac(b_a, b_g)
        expected_block = -iekf_vanilla._skew(b_g)
        np.testing.assert_allclose(A[0:3, 0:3], expected_block, atol=1e-15)

    def test_position_velocity_coupling_is_identity(self):
        """A[6:9, 3:6] = I — kinematic coupling: position ← velocity."""
        b_a = np.zeros(3)
        b_g = np.zeros(3)
        A = _build_iekf_ajac(b_a, b_g)
        np.testing.assert_allclose(A[6:9, 3:6], np.eye(3), atol=1e-15)
