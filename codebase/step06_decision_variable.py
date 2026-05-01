"""Step 06 — Decision variable: the QP optimization variable z.

The MPC solves a Quadratic Program (QP):
    min  0.5 * zᵀ P z + qᵀ z
    s.t. A_eq z = b_eq           (dynamics + initial state)
         l ≤ A_ineq z ≤ u        (friction cone / force bounds)

But what is z?  It stacks all predicted states and controls:

    z = [ x_0, x_1, ..., x_N,  u_0, u_1, ..., u_{N-1} ]
         |←─── (N+1)*NX ────→|  |←─── N*NU ───→|
         |                                        |
         state part                          control part

This step covers:
  1. How z is structured (slicing into states and controls)
  2. How to index into z for any step k
  3. Building z from a trajectory
  4. Extracting a trajectory from z

Run from single_leg_stand/:
    uv run python hands_on_mpc/step06_decision_variable.py
"""

import numpy as np

from _mpc_utils import (
    NX, NU, N_HORIZON,
    standing_equilibrium, make_centroidal_dynamics,
    discretize_euler, rollout,
    TOTAL_MASS, T_S, COM_HEIGHT, SUPPORT_POS_XY,
)

x_eq, u_eq = standing_equilibrium()

print("=" * 60)
print("STEP 06 — Decision variable z: structure and indexing")
print("=" * 60)

# ── 1. Size of z ─────────────────────────────────────────────────────────────
N = N_HORIZON
NZ = (N + 1) * NX + N * NU

print(f"\n[Size of z]")
print(f"  NX = {NX}  (state dimension)")
print(f"  NU = {NU}  (control dimension)")
print(f"  N  = {N}   (horizon steps)")
print(f"  NZ = (N+1)*NX + N*NU = ({N}+1)*{NX} + {N}*{NU} = {NZ}")
print(f"\n  Layout:")
print(f"  z[0      : {(N+1)*NX}]  → {N+1} state vectors x_0...x_{N}   ({(N+1)*NX} elements)")
print(f"  z[{(N+1)*NX} : {NZ}]   → {N} control vectors u_0...u_{N-1}  ({N*NU} elements)")

# ── 2. Index functions ────────────────────────────────────────────────────────
print("\n[Indexing into z]")
print("  State  at step k: z[k*NX       : (k+1)*NX]")
print("  Control at step k: z[(N+1)*NX + k*NU : (N+1)*NX + (k+1)*NU]")
print()
print("  Helper functions:")

def x_slice(k):
    """Indices in z for state at step k."""
    return slice(k * NX, (k + 1) * NX)

def u_slice(k):
    """Indices in z for control at step k."""
    base = (N + 1) * NX
    return slice(base + k * NU, base + (k + 1) * NU)

for k in [0, 1, N // 2, N - 1, N]:
    sx = x_slice(k)
    print(f"    x_{k}: z[{sx.start}:{sx.stop}]", end="")
    if k < N:
        su = u_slice(k)
        print(f"   u_{k}: z[{su.start}:{su.stop}]")
    else:
        print("   (no u_N — N controls, N+1 states)")

# ── 3. Build z from a known trajectory ───────────────────────────────────────
print("\n[Building z from a known trajectory]")
print("  Given states X[N+1, NX] and controls U[N, NU], pack into z:")

# Simulate a simple trajectory from perturbed equilibrium
r_foot  = np.array([SUPPORT_POS_XY[0], SUPPORT_POS_XY[1], 0.0])
r_lever = r_foot - x_eq[:3]
A_c, B_c, d_c = make_centroidal_dynamics(TOTAL_MASS, r_lever)
A_d, B_d, d_d = discretize_euler(A_c, B_c, d_c, T_S)

x0 = x_eq.copy(); x0[0] += 0.03  # 3cm CoM offset
U  = np.tile(u_eq, (N, 1))        # constant equilibrium forces
X  = rollout(A_d, B_d, d_d, x0, U)

# Pack z
z = np.zeros(NZ)
for k in range(N + 1):
    z[x_slice(k)] = X[k]
for k in range(N):
    z[u_slice(k)] = U[k]

print(f"  z.shape = {z.shape}")
print(f"  z[x_0] = z[{x_slice(0).start}:{x_slice(0).stop}] = {z[x_slice(0)].round(4)}")
print(f"  z[u_0] = z[{u_slice(0).start}:{u_slice(0).stop}] = {z[u_slice(0)].round(2)}")
print(f"  z[x_N] = z[{x_slice(N).start}:{x_slice(N).stop}] = {z[x_slice(N)].round(4)}")

# ── 4. Recover trajectory from z ─────────────────────────────────────────────
print("\n[Recovering trajectory from z]")
X_rec = np.array([z[x_slice(k)] for k in range(N + 1)])
U_rec = np.array([z[u_slice(k)] for k in range(N)])

print(f"  X_rec matches original X: {np.allclose(X_rec, X)}")
print(f"  U_rec matches original U: {np.allclose(U_rec, U)}")

# ── 5. mpc.py connection ─────────────────────────────────────────────────────
print("\n[How mpc.py uses z]")
print("  In mpc.py (CentroidalMPC class):")
print()
print("  NZ = (N+1)*NX + N*NU")
print("  z is the OSQP decision variable; after solving:")
print("    result = self._solver.solve()")
print("    z_opt  = result.x")
print("    x_0_opt = z_opt[0:NX]       ← predicted next state")
print("    u_0_opt = z_opt[(N+1)*NX : (N+1)*NX + NU]  ← first control")
print()
print("  The robot applies only u_0 (first control step).")
print("  At next time step, the whole QP is solved again (receding horizon).")

# ── 6. Small N=2 example: hand-checkable ─────────────────────────────────────
print("\n[Small N=2 example — hand checkable]")
N2  = 2
NZ2 = (N2 + 1) * NX + N2 * NU
print(f"  N=2: NZ = ({N2}+1)*{NX} + {N2}*{NU} = {NZ2}")
print()
print(f"  z = [ x_0({NX}), x_1({NX}), x_2({NX}), u_0({NU}), u_1({NU}) ]")
print(f"       0       {NX}      {2*NX}      {3*NX}     {3*NX+NU}    {NZ2}")
print()
x0_2 = x_eq.copy(); x0_2[0] += 0.02
U2   = np.array([u_eq, u_eq])
X2   = rollout(A_d, B_d, d_d, x0_2, U2)

z2 = np.zeros(NZ2)
for k2 in range(N2 + 1):
    z2[k2 * NX:(k2 + 1) * NX] = X2[k2]
for k2 in range(N2):
    z2[(N2 + 1) * NX + k2 * NU:(N2 + 1) * NX + (k2 + 1) * NU] = U2[k2]

print(f"  z2[:6]  (x_0 first 6 elements): {z2[:6].round(4)}")
print(f"  z2[{3*NX}:{3*NX+NU}] (u_0):  {z2[3*NX:3*NX+NU].round(2)}")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters for single-leg balance:")
print("   When debugging mpc.py, you need to slice z_opt correctly.")
print("   Wrong slicing → you extract u from the state part → garbage control.")
print("   mpc.py does: u_opt = result.x[(N+1)*NX : (N+1)*NX + NU]")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. Verify dynamics consistency in z:
# For each k, check x_{k+1} == A_d @ x_k + B_d @ u_k + d_d
# for k in range(N):
#     xk     = z[x_slice(k)]
#     uk     = z[u_slice(k)]
#     xk1_pred = A_d @ xk + B_d @ uk + d_d
#     xk1_z    = z[x_slice(k+1)]
#     print(f"k={k}: dynamics error = {np.linalg.norm(xk1_pred - xk1_z):.2e}")
# Expected: all ~0 (z was built from a consistent rollout)

# 2. What happens if you use wrong slice for u?
# Accidentally read x data as u:
# wrong_u = z[0:NU]   # reads beginning of x_0 as control
# print("Wrong u:", wrong_u, "vs correct u_0:", z[u_slice(0)])
# Expected: very different values

# 3. For a larger N=20 horizon, count NZ and identify z slices:
# N_big = 20
# NZ_big = (N_big+1)*NX + N_big*NU
# print(f"N=20: NZ={NZ_big}")
# print(f"u_last starts at: {(N_big+1)*NX + (N_big-1)*NU}")
""")
