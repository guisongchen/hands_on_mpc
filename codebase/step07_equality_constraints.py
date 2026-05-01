"""Step 07 — Equality constraints: dynamics + initial state.

The QP equality constraints enforce:
  1. Initial state:  x_0 = x_current   (pin the first state to reality)
  2. Dynamics:       x_{k+1} = A_d*x_k + B_d*u_k + d_d  for k=0..N-1

In QP form:  A_eq * z = b_eq

This step covers:
  1. How to build A_eq and b_eq block-by-block
  2. Verify the constraint is satisfied by a consistent trajectory
  3. Understand why these rows uniquely define the prediction

Run from single_leg_stand/:
    uv run python hands_on_mpc/step07_equality_constraints.py
"""

import numpy as np
from _mpc_utils import (
    NX, NU, N_HORIZON,
    standing_equilibrium, make_centroidal_dynamics,
    discretize_euler, rollout,
    TOTAL_MASS, T_S, COM_HEIGHT, SUPPORT_POS_XY,
)

x_eq, u_eq = standing_equilibrium()
r_foot  = np.array([SUPPORT_POS_XY[0], SUPPORT_POS_XY[1], 0.0])
r_lever = r_foot - x_eq[:3]
A_c, B_c, d_c = make_centroidal_dynamics(TOTAL_MASS, r_lever)
A_d, B_d, d_d = discretize_euler(A_c, B_c, d_c, T_S)

N  = N_HORIZON
NZ = (N + 1) * NX + N * NU

print("=" * 60)
print("STEP 07 — Equality constraints: A_eq * z = b_eq")
print("=" * 60)

# ── 1. Structure overview ─────────────────────────────────────────────────────
print("\n[Constraint structure]")
n_eq = (N + 1) * NX     # 1 block for x0 + N blocks for dynamics
print(f"  A_eq shape: ({n_eq}, {NZ})")
print(f"  b_eq shape: ({n_eq},)")
print()
print("  Rows 0 to NX-1:      pin x_0 = x_current")
print("  Rows NX to 2*NX-1:   enforce x_1 = A_d*x_0 + B_d*u_0 + d_d")
print("  Rows 2*NX to 3*NX-1: enforce x_2 = A_d*x_1 + B_d*u_1 + d_d")
print("  ...   (N dynamics blocks total)")

# ── 2. Index helpers ──────────────────────────────────────────────────────────
def x_slice(k):
    return slice(k * NX, (k + 1) * NX)

def u_slice(k):
    base = (N + 1) * NX
    return slice(base + k * NU, base + (k + 1) * NU)

# ── 3. Build A_eq and b_eq ────────────────────────────────────────────────────
print("\n[Building A_eq and b_eq]")

A_eq = np.zeros((n_eq, NZ))
b_eq = np.zeros(n_eq)

# Block 0: x_0 = x_current  →  I * x_0 = x_current
x_current = x_eq.copy()
x_current[0] += 0.03  # simulate a 3cm offset

row = 0
A_eq[row:row + NX, x_slice(0)] = np.eye(NX)
b_eq[row:row + NX]             = x_current
row += NX

# Blocks 1..N: x_{k+1} = A_d*x_k + B_d*u_k + d_d
#   Rearranged: -A_d*x_k  -B_d*u_k  +I*x_{k+1} = d_d
for k in range(N):
    A_eq[row:row + NX, x_slice(k)]     = -A_d
    A_eq[row:row + NX, u_slice(k)]     = -B_d
    A_eq[row:row + NX, x_slice(k + 1)] =  np.eye(NX)
    b_eq[row:row + NX]                 =  d_d
    row += NX

print(f"  A_eq shape: {A_eq.shape}")
print(f"  Non-zeros in A_eq: {np.count_nonzero(A_eq)} of {A_eq.size}")
print(f"  Sparsity: {100*(1-np.count_nonzero(A_eq)/A_eq.size):.1f}% zeros  (very sparse!)")

# ── 4. Verify: a consistent z should satisfy A_eq * z = b_eq ─────────────────
print("\n[Verification: consistent z satisfies A_eq * z = b_eq]")

# Build z from a consistent rollout starting at x_current
U = np.tile(u_eq, (N, 1))
X = rollout(A_d, B_d, d_d, x_current, U)

z = np.zeros(NZ)
for k in range(N + 1):
    z[x_slice(k)] = X[k]
for k in range(N):
    z[u_slice(k)] = U[k]

residual = A_eq @ z - b_eq
print(f"  Max residual |A_eq*z - b_eq|: {np.max(np.abs(residual)):.2e}")
print(f"  {'✓ Constraints satisfied' if np.max(np.abs(residual)) < 1e-8 else '✗ CONSTRAINTS VIOLATED'}")

# ── 5. Violate dynamics and check ────────────────────────────────────────────
print("\n[What happens with an inconsistent z?]")
z_bad = z.copy()
z_bad[x_slice(1)] += 0.1  # perturb x_1 to violate dynamics

residual_bad = A_eq @ z_bad - b_eq
print(f"  Perturbed x_1 by 0.1 m:")
print(f"  Max residual: {np.max(np.abs(residual_bad)):.4f}  ← non-zero = constraint violated")
print("  QP would reject this z_bad — solver enforces feasibility.")

# ── 6. Block structure visualization ─────────────────────────────────────────
print("\n[Block structure of A_eq (small N=3 example)]")
N3  = 3
NZ3 = (N3 + 1) * NX + N3 * NU
n3  = (N3 + 1) * NX

A_small = np.zeros((n3, NZ3))

# Initial state
A_small[0:NX, 0:NX] = np.eye(NX)
# Dynamics
for k3 in range(N3):
    r = (k3 + 1) * NX
    A_small[r:r+NX, k3*NX:(k3+1)*NX]         = -A_d
    A_small[r:r+NX, (N3+1)*NX + k3*NU:(N3+1)*NX + (k3+1)*NU] = -B_d
    A_small[r:r+NX, (k3+1)*NX:(k3+2)*NX]     =  np.eye(NX)

# Show non-zero pattern as ASCII
print("  Non-zero pattern (each cell = one NX×NX or NX×NU block):")
state_cols = N3 + 1
ctrl_cols  = N3
all_cols   = state_cols + ctrl_cols
header  = "       " + "".join(f"x{k3:2d} " for k3 in range(state_cols))
header += "".join(f"u{k3:2d} " for k3 in range(ctrl_cols))
print("  " + header)
block_rows = N3 + 1
for br in range(block_rows):
    row_lbl = f"  {'x0=':4s}" if br == 0 else f"  {'xd'+str(br-1)+':':4s}"
    row_str = row_lbl
    for bc in range(state_cols):
        block = A_small[br*NX:(br+1)*NX, bc*NX:(bc+1)*NX]
        row_str += " ██ " if np.any(block != 0) else "    "
    ctrl_start = (N3 + 1) * NX
    for bc in range(ctrl_cols):
        block = A_small[br*NX:(br+1)*NX, ctrl_start+bc*NU:ctrl_start+(bc+1)*NU]
        row_str += " ██ " if np.any(block != 0) else "    "
    print(row_str)

# ── 7. mpc.py connection ─────────────────────────────────────────────────────
print("\n[How mpc.py builds equality constraints]")
print("  CentroidalMPC.build_equality_constraints(A_d, B_d, d_d, x_curr):")
print("  → builds A_eq (sparse via scipy.sparse) and b_eq")
print("  → passed to OSQP via: solver.setup(P, q, A_ineq, l, u)")
print("  (OSQP uses a combined constraint matrix, not separate eq/ineq)")
print("  See mpc.py for the actual implementation.")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters for single-leg balance:")
print("   The equality constraints force the QP to find a z that is physically")
print("   consistent. Without them, the optimizer could 'cheat' by setting")
print("   u=0 while claiming x stays at reference — impossible for a real robot.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. Check rank of A_eq (should be full row rank):
# rank = np.linalg.matrix_rank(A_eq)
# print(f"rank(A_eq) = {rank} / {n_eq}  (full = {rank == n_eq})")
# Expected: full row rank (each constraint is independent)

# 2. What if x_current is exactly x_eq?
# b_eq_new = b_eq.copy(); b_eq_new[:NX] = x_eq
# z_eq_rollout = np.zeros(NZ)
# X_eq = rollout(A_d, B_d, d_d, x_eq, np.tile(u_eq, (N,1)))
# for k in range(N+1): z_eq_rollout[x_slice(k)] = X_eq[k]
# for k in range(N):   z_eq_rollout[u_slice(k)] = u_eq
# resid = A_eq @ z_eq_rollout - b_eq_new
# print("Equilibrium residual:", np.max(np.abs(resid)))
# Expected: ~0

# 3. Inspect the sparsity of A_eq using a heatmap (requires matplotlib):
# import matplotlib.pyplot as plt
# plt.figure(); plt.spy(A_eq, markersize=1); plt.title("A_eq sparsity")
# plt.show()
""")
