"""Step 04 — Discretization: continuous → discrete.

The centroidal dynamics are continuous-time:
    ẋ = A_c * x + B_c * u + d_c

But computers work in discrete steps. We need:
    x_{k+1} = A_d * x_k + B_d * u_k + d_d

This step covers:
  1. First-order Euler (what mpc.py uses): A_d = I + A_c*dt
  2. Zero-Order Hold (ZOH) — more accurate for large dt
  3. Accuracy comparison between the two
  4. Effect of choosing different dt on prediction quality

Run from single_leg_stand/:
    uv run python hands_on_mpc/step04_discretization.py
"""

import numpy as np
from scipy.linalg import expm

from _mpc_utils import (
    NX, NU, TOTAL_MASS, GRAVITY_MAG, COM_HEIGHT,
    SUPPORT_POS_XY, T_S, N_HORIZON,
    make_centroidal_dynamics, discretize_euler, standing_equilibrium, rollout,
)

# Build continuous dynamics
x_eq, u_eq = standing_equilibrium()
c_eq   = x_eq[:3]
r_foot = np.array([SUPPORT_POS_XY[0], SUPPORT_POS_XY[1], 0.0])
r_lever = r_foot - c_eq
A_c, B_c, d_c = make_centroidal_dynamics(TOTAL_MASS, r_lever)

print("=" * 60)
print("STEP 04 — Discretization: continuous → discrete time")
print("=" * 60)

# ── 1. Euler discretization ──────────────────────────────────────────────────
print("\n[Method 1: First-order Euler (used in mpc.py)]")
print("  A_d = I + A_c * dt")
print("  B_d = B_c * dt")
print("  d_d = d_c * dt")

dt = T_S   # 0.05 s
A_d_euler, B_d_euler, d_d_euler = discretize_euler(A_c, B_c, d_c, dt)

print(f"\n  dt = {dt} s  (MPC sampling period T_S from config.py)")
print(f"\n  A_d (Euler), shape {A_d_euler.shape}:")
print(f"  Diagonal: {A_d_euler.diagonal().round(4)}")
print(f"  Off-diagonal block A_d[0:3,3:6] (pos ← vel):\n  {A_d_euler[0:3, 3:6].round(4)}")
print(f"\n  B_d (Euler), rows 3-5 (CoM accel block):\n  {B_d_euler[3:6].round(6)}")
print(f"\n  d_d (Euler): {d_d_euler.round(6)}")
print(f"  d_d[3:6] = g*dt = {GRAVITY_MAG}*{dt} = {GRAVITY_MAG*dt:.4f}  ✓")

# ── 2. ZOH discretization (more accurate) ────────────────────────────────────
print("\n[Method 2: Zero-Order Hold (ZOH) — exact for piecewise constant input]")
print("  Uses matrix exponential: A_d_zoh = expm(A_c * dt)")
print("  More accurate than Euler for large dt.")

nx, nu = A_c.shape[0], B_c.shape[1]
M_aug = np.zeros((nx + nu, nx + nu))
M_aug[:nx, :nx] = A_c
M_aug[:nx, nx:] = B_c
expm_aug = expm(M_aug * dt)
A_d_zoh = expm_aug[:nx, :nx]
B_d_zoh = expm_aug[:nx, nx:]
d_d_zoh = d_c * dt   # affine term same for ZOH (gravity is constant)

print(f"\n  A_d (ZOH) diagonal: {A_d_zoh.diagonal().round(6)}")
print(f"  A_d (Euler) diagonal: {A_d_euler.diagonal().round(6)}")
print(f"  Max diff: {np.max(np.abs(A_d_zoh - A_d_euler)):.2e}")

# ── 3. Accuracy comparison ───────────────────────────────────────────────────
print("\n[Accuracy comparison: Euler vs ZOH over 5 steps from perturbed state]")
x_pert = x_eq.copy()
x_pert[0] += 0.05   # push CoM 5cm forward
x_pert[3]  = 0.1    # give it forward velocity

N_test = 5
u_const = np.tile(u_eq, (N_test, 1))

x_euler = rollout(A_d_euler, B_d_euler, d_d_euler, x_pert, u_const)
x_zoh   = rollout(A_d_zoh,   B_d_zoh,   d_d_zoh,   x_pert, u_const)

print(f"\n  {'step':>5}  {'Euler c_x':>10}  {'ZOH c_x':>9}  {'diff (mm)':>10}")
for k in range(N_test + 1):
    diff_mm = abs(x_euler[k, 0] - x_zoh[k, 0]) * 1000
    print(f"  {k:5d}  {x_euler[k,0]:10.5f}  {x_zoh[k,0]:9.5f}  {diff_mm:10.3f}")
print("  → For small dt=0.05s Euler is close to ZOH. Larger dt = more error.")

# ── 4. Effect of dt on discretization error ──────────────────────────────────
print("\n[Effect of dt on discretization error]")
print("  Comparing Euler vs ZOH over 1 second for different dt values:")

x_init = x_pert.copy()
print(f"\n  {'dt (s)':>8}  {'Euler c_x':>10}  {'ZOH c_x':>9}  {'error (mm)':>11}")
for dt_test in [0.001, 0.01, 0.05, 0.1, 0.2]:
    N_t = int(1.0 / dt_test)
    A_e, B_e, d_e = discretize_euler(A_c, B_c, d_c, dt_test)

    M_t = np.zeros((nx + nu, nx + nu))
    M_t[:nx, :nx] = A_c
    M_t[:nx, nx:] = B_c
    ep = expm(M_t * dt_test)
    A_z = ep[:nx, :nx]
    B_z = ep[:nx, nx:]
    d_z = d_c * dt_test

    u_t = np.tile(u_eq, (N_t, 1))
    xE = rollout(A_e, B_e, d_e, x_init, u_t)
    xZ = rollout(A_z, B_z, d_z, x_init, u_t)
    err_mm = abs(xE[-1, 0] - xZ[-1, 0]) * 1000
    print(f"  {dt_test:8.3f}  {xE[-1,0]:10.5f}  {xZ[-1,0]:9.5f}  {err_mm:11.4f}")
print("  → Large dt amplifies Euler error. mpc.py uses dt=0.05s (acceptable)")

# ── 5. Why T_S=0.05s is chosen ───────────────────────────────────────────────
print("\n[Why T_S = 0.05s (50ms) in config.py]")
print(f"  MPC horizon: N={N_HORIZON} steps × T_S={T_S}s = {N_HORIZON*T_S:.1f}s lookahead")
print(f"  MPC runs at 1/T_S = {1/T_S:.0f} Hz  (every {int(T_S/0.001)} sim steps of 1ms)")
print("  Fast enough for balance,  slow enough to be solvable in real time.")
print()
print("  Note: the SIMULATION runs at 1kHz (dt_sim=0.001s)")
print("        the MPC     runs at 20Hz  (T_S=0.05s)")
print("        the control applies MPC output for 50 sim steps between MPC solves")

# ── 6. Sanity check: A_d should have unit or near-unit eigenvalues ─────────
print("\n[Stability check: eigenvalues of A_d]")
eigs = np.linalg.eigvals(A_d_euler)
print(f"  Eigenvalues of A_d (Euler): {np.sort(np.abs(eigs)).round(4)}")
print("  All ≤ 1.0 → marginally stable (no exponential growth from A alone)")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters for single-leg balance:")
print("   mpc.py.set_dynamics(A_d, B_d, d_d) is called every MPC step.")
print("   If T_S is too large, Euler error makes the predicted trajectory wrong,")
print("   and the planned GRF won't keep the CoM where expected.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. Verify A_d @ x_eq + B_d @ u_eq + d_d ≈ x_eq (fixed point):
# x_next = A_d_euler @ x_eq + B_d_euler @ u_eq + d_d_euler
# print("Fixed point error:", np.linalg.norm(x_next - x_eq))
# Expected: small (equilibrium is preserved by discretization)

# 2. What's the maximum eigenvalue of A_d for dt=0.2s?
# A_big, _, _ = discretize_euler(A_c, B_c, d_c, 0.2)
# print("Max |eig(A_d)| at dt=0.2s:", np.max(np.abs(np.linalg.eigvals(A_big))))
# Expected: > 1.0 → unstable open-loop discrete system

# 3. Compare Euler prediction with a hand-computed 1-step:
# x0 = x_eq.copy(); x0[3] = 0.2  # give forward velocity
# x_euler_manual = x0 + (A_c @ x0 + B_c @ u_eq + d_c) * T_S
# x_euler_matrix = A_d_euler @ x0 + B_d_euler @ u_eq + d_d_euler
# print("Manual:", x_euler_manual[:3].round(5))
# print("Matrix:", x_euler_matrix[:3].round(5))
# Expected: identical (Euler discretization IS just x + f(x)*dt)
""")
