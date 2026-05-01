"""Shared utilities for all hands-on MPC practice steps.

Provides:
  - Robot / model constants
  - Centroidal dynamics construction
  - Euler discretization
  - Friction cone builder
  - Open-loop rollout for verification
  - Standing equilibrium helper
  - Toy 1D point-mass environment for early steps

Run any step from single_leg_stand/:
    uv run python hands_on_mpc/step01_mpc_concept.py
"""

import numpy as np
from scipy import sparse
import osqp

# ── Physical constants ───────────────────────────────────────────────────────
TOTAL_MASS  = 35.0          # kg  (Unitree G1 approximate)
GRAVITY_MAG = 9.81          # m/s²
G_VEC       = np.array([0.0, 0.0, -GRAVITY_MAG])

# ── MPC dimensions (match config.py) ────────────────────────────────────────
NX        = 9     # state: [com_pos(3), com_vel(3), ang_mom(3)]
NU        = 3     # control: ground reaction force [Fx, Fy, Fz]
N_HORIZON = 10    # prediction horizon (steps)
T_S       = 0.05  # MPC sampling period (seconds)

# ── Cost weights (match config.py Q, R, QN) ─────────────────────────────────
Q  = np.diag([100., 100., 100.,   # CoM position tracking
               1.,   1.,   1.,   # CoM velocity
               1.,   1.,   1.])  # angular momentum
R  = np.diag([0.001, 0.001, 0.001])  # input (GRF) regularization
QN = Q.copy()                         # terminal cost = stage cost

# ── Standing equilibrium ─────────────────────────────────────────────────────
COM_HEIGHT     = 0.76       # m  (nominal standing CoM height)
SUPPORT_POS_XY = np.array([0.0, -0.09])  # right foot position in XY

# ── Friction ─────────────────────────────────────────────────────────────────
MU          = 0.6     # friction coefficient
FRICTION_MU = MU      # alias used in step files
F_Z_MAX     = 500.0   # maximum normal force (N)


# ─────────────────────────────────────────────────────────────────────────────
# Core dynamics functions
# ─────────────────────────────────────────────────────────────────────────────

def make_centroidal_dynamics(mass: float, r_foot: np.ndarray) -> tuple:
    """Build continuous-time centroidal dynamics matrices A_c, B_c, d_c.

    State:  x = [c(3), c_dot(3), L(3)]
    Input:  u = [fx, fy, fz]   (contact force at r_foot)

    Continuous EOM:
      c_ddot   = (1/m) * u + g
      L_dot    = (r_foot - c) × u  ≈  r_perp × u  (linearized at equilibrium)

    Linearized around equilibrium (c_eq, zero velocity, L=0, u_eq=[0,0,mg]):
      A_c = [[0  I  0],
             [0  0  0],
             [0  0  0]]
      B_c = [[0         ],
             [I/m       ],
             [skew(r_eq)]]   where r_eq = r_foot - c_eq (equilibrium lever arm)
      d_c = [0, g, 0]        (affine gravity term)

    Returns: (A_c, B_c, d_c)  each as numpy arrays
    """
    A_c = np.zeros((NX, NX))
    A_c[0:3, 3:6] = np.eye(3)   # c_dot enters c dynamics

    B_c = np.zeros((NX, NU))
    B_c[3:6, :] = np.eye(3) / mass        # force → CoM acceleration

    # Lever arm cross product: skew(r) * f = r × f
    # Use equilibrium lever arm (constant for linearized model)
    r_eq = r_foot  # simplification: lever arm ≈ support foot position - CoM
    skew_r = _skew(r_eq)
    B_c[6:9, :] = skew_r                  # force → angular momentum rate

    d_c = np.zeros(NX)
    d_c[3:6] = np.array([0.0, 0.0, -GRAVITY_MAG])  # gravity acts on CoM

    return A_c, B_c, d_c


def discretize_euler(A_c: np.ndarray, B_c: np.ndarray,
                     d_c: np.ndarray, dt: float) -> tuple:
    """First-order Euler discretization.

    x_{k+1} = A_d x_k + B_d u_k + d_d

    A_d = I + A_c * dt
    B_d = B_c * dt
    d_d = d_c * dt
    """
    I   = np.eye(A_c.shape[0])
    A_d = I + A_c * dt
    B_d = B_c * dt
    d_d = d_c * dt
    return A_d, B_d, d_d


def standing_equilibrium(com_height: float = COM_HEIGHT,
                         support_xy: np.ndarray = SUPPORT_POS_XY) -> tuple:
    """Compute the equilibrium state and control for static standing.

    At equilibrium:
      - CoM is directly above support foot (or offset for balance)
      - CoM velocity = 0
      - Angular momentum = 0
      - GRF = [0, 0, mg]  (supporting full weight)

    Returns:
      x_eq (NX,), u_eq (NU,)
    """
    c_eq   = np.array([support_xy[0], support_xy[1], com_height])
    x_eq   = np.zeros(NX)
    x_eq[:3] = c_eq
    u_eq   = np.array([0.0, 0.0, TOTAL_MASS * GRAVITY_MAG])
    return x_eq, u_eq


def build_friction_cone(mu: float = MU, fz_max: float = F_Z_MAX) -> tuple:
    """Build linear friction cone constraint in OSQP format: l ≤ A_fcon @ u ≤ u.

    Constraints (pyramid approximation):
        -mu*fz ≤ fx ≤  mu*fz
        -mu*fz ≤ fy ≤  mu*fz
        0 ≤ fz ≤ fz_max

    Rearranged as 6 rows in l ≤ A@u ≤ u (OSQP convention):
      Row 0:  fx - mu*fz   → l=-inf, u=0
      Row 1: -fx - mu*fz   → l=-inf, u=0
      Row 2:  fy - mu*fz   → l=-inf, u=0
      Row 3: -fy - mu*fz   → l=-inf, u=0
      Row 4:  fz            → l=0,    u=fz_max

    Returns:
        A_fcon (5, 3), l_fcon (5,), u_fcon (5,)
    """
    INF = np.inf
    A = np.array([
        [ 1,  0, -mu],   # fx - mu*fz
        [-1,  0, -mu],   # -fx - mu*fz
        [ 0,  1, -mu],   # fy - mu*fz
        [ 0, -1, -mu],   # -fy - mu*fz
        [ 0,  0,  1 ],   # fz
    ], dtype=float)
    l = np.array([-INF, -INF, -INF, -INF, 0.0])
    u = np.array([0.0,   0.0,  0.0,  0.0, fz_max])
    return A, l, u


def rollout(A_d: np.ndarray, B_d: np.ndarray,
            d_d: np.ndarray, x0: np.ndarray,
            u_traj: np.ndarray) -> np.ndarray:
    """Simulate system forward for N steps given control trajectory.

    Args:
        A_d, B_d, d_d: discrete dynamics matrices
        x0: initial state (NX,)
        u_traj: (N, NU) control sequence

    Returns:
        x_traj: (N+1, NX) state trajectory  (x_traj[0] = x0)
    """
    N  = u_traj.shape[0]
    nx = x0.shape[0]
    x_traj = np.zeros((N + 1, nx))
    x_traj[0] = x0.copy()
    for k in range(N):
        x_traj[k + 1] = A_d @ x_traj[k] + B_d @ u_traj[k] + d_d
    return x_traj


# ─────────────────────────────────────────────────────────────────────────────
# Toy 1D point-mass environment (for step01 / bridging)
# ─────────────────────────────────────────────────────────────────────────────

class PointMass1D:
    """Simple 1D point-mass: x_ddot = u/m - g  (vertical axis).

    State: [z, z_dot]   Control: [Fz]
    Discrete (Euler): x_{k+1} = A*x + B*u + d
    """

    def __init__(self, mass: float = 1.0, dt: float = 0.05):
        self.mass = mass
        self.dt   = dt
        self.g    = GRAVITY_MAG

        # Continuous: A = [[0,1],[0,0]],  B = [[0],[1/m]],  d = [[0],[-g]]
        self.A_d = np.array([[1.0, dt], [0.0, 1.0]])
        self.B_d = np.array([[0.0], [dt / mass]])
        self.d_d = np.array([0.0, -self.g * dt])

        self.state = np.array([1.0, 0.0])   # [height=1m, vel=0]

    def reset(self, z: float = 1.0, z_dot: float = 0.0):
        self.state = np.array([z, z_dot])

    def step(self, u: float) -> np.ndarray:
        self.state = self.A_d @ self.state + self.B_d @ np.array([u]) + self.d_d
        return self.state.copy()

    @property
    def equilibrium_u(self) -> float:
        """Force needed to hover: u* = m*g."""
        return self.mass * self.g


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix for cross product: skew(v) @ w = v × w."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ],
    ])
