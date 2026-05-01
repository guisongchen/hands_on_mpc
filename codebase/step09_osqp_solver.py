"""Step 09 — OSQP solver: setup, solve, and debug.

OSQP (Operator Splitting QP) solves:
    min  0.5 * zᵀ P z + qᵀ z
    s.t. l ≤ A z ≤ u

OSQP uses a COMBINED constraint matrix A that stacks:
  [ A_eq  ]       [ b_eq  ]       [ b_eq  ]
  [       ]  l =  [       ]  u =  [       ]
  [ A_ineq]       [ l_ineq]       [ u_ineq]

This step covers:
  1. Building P, q, A, l, u for a small MPC problem
  2. Calling osqp.setup() and osqp.solve()
  3. Reading the result: status, objective, z_opt
  4. Common failure modes and debug checklist

Run from single_leg_stand/:
    uv run python hands_on_mpc/step09_osqp_solver.py
"""

import numpy as np
import scipy.sparse as sp
import osqp

from _mpc_utils import (
    NX, NU, N_HORIZON, TOTAL_MASS, GRAVITY_MAG, T_S,
    FRICTION_MU, F_Z_MAX,
    Q, R, QN,
    standing_equilibrium, make_centroidal_dynamics,
    discretize_euler, build_friction_cone,
)

x_eq, u_eq = standing_equilibrium()
from _mpc_utils import COM_HEIGHT, SUPPORT_POS_XY
r_foot  = np.array([SUPPORT_POS_XY[0], SUPPORT_POS_XY[1], 0.0])
r_lever = r_foot - x_eq[:3]
A_c, B_c, d_c = make_centroidal_dynamics(TOTAL_MASS, r_lever)
A_d, B_d, d_d = discretize_euler(A_c, B_c, d_c, T_S)

N  = N_HORIZON
NZ = (N + 1) * NX + N * NU

print("=" * 60)
print("STEP 09 — OSQP solver: setup, solve, debug")
print("=" * 60)

# ── Index helpers ─────────────────────────────────────────────────────────────
def x_slice(k):
    return slice(k * NX, (k + 1) * NX)
def u_slice(k):
    base = (N + 1) * NX
    return slice(base + k * NU, base + (k + 1) * NU)

# ── 1. Build cost matrices P and q ────────────────────────────────────────────
print("\n[Cost matrices: P and q]")
print("  P is block-diagonal: Q for states, R for controls")

P_dense = np.zeros((NZ, NZ))
for k in range(N):
    P_dense[x_slice(k), x_slice(k)] = Q
    P_dense[u_slice(k), u_slice(k)] = R
P_dense[x_slice(N), x_slice(N)] = QN

P = sp.csc_matrix(P_dense)
print(f"  P shape: {P.shape},  non-zeros: {P.nnz}")

# Linear term q: -P * z_ref (track x_ref, control u_eq)
# For constant reference x_eq:
x_ref = np.tile(x_eq, N + 1)
u_ref = np.tile(u_eq, N)
z_ref = np.concatenate([x_ref, u_ref])
q = -P_dense @ z_ref
print(f"  q shape: {q.shape}")

# ── 2. Build constraint matrix A, l, u ────────────────────────────────────────
print("\n[Constraint matrix A, l, u]")

# Equality: A_eq * z = b_eq
n_eq = (N + 1) * NX
x_current = x_eq.copy()
x_current[0] += 0.05  # 5cm offset

A_eq = np.zeros((n_eq, NZ))
b_eq = np.zeros(n_eq)
A_eq[0:NX, x_slice(0)] = np.eye(NX)
b_eq[0:NX] = x_current
for k in range(N):
    row = (k + 1) * NX
    A_eq[row:row+NX, x_slice(k)]     = -A_d
    A_eq[row:row+NX, u_slice(k)]     = -B_d
    A_eq[row:row+NX, x_slice(k+1)]   =  np.eye(NX)
    b_eq[row:row+NX]                 =  d_d

# Inequality: friction cone per step
A_fcon, l_fcon, u_fcon = build_friction_cone(FRICTION_MU, F_Z_MAX)
n_per = len(l_fcon)
A_ineq = np.zeros((N * n_per, NZ))
l_ineq = np.zeros(N * n_per)
u_ineq = np.zeros(N * n_per)
for k in range(N):
    row = k * n_per
    col = (N + 1) * NX + k * NU
    A_ineq[row:row+n_per, col:col+NU] = A_fcon
    l_ineq[row:row+n_per] = l_fcon
    u_ineq[row:row+n_per] = u_fcon

# Stack equality and inequality
A_all = np.vstack([A_eq, A_ineq])
l_all = np.concatenate([b_eq, l_ineq])   # eq: l == u
u_all = np.concatenate([b_eq, u_ineq])

A_sp = sp.csc_matrix(A_all)
print(f"  A_eq rows: {n_eq}  A_ineq rows: {N*n_per}  Total: {n_eq + N*n_per}")
print(f"  A_sp shape: {A_sp.shape},  non-zeros: {A_sp.nnz}")

# ── 3. Setup OSQP ─────────────────────────────────────────────────────────────
print("\n[OSQP setup]")
solver = osqp.OSQP()
solver.setup(
    P, q,
    A_sp, l_all, u_all,
    warm_starting=True,
    verbose=False,
    eps_abs=1e-5,
    eps_rel=1e-5,
    max_iter=4000,
)
print("  solver.setup() done.")

# ── 4. Solve ──────────────────────────────────────────────────────────────────
print("\n[Solve and inspect result]")
result = solver.solve()

print(f"  Status:    {result.info.status}")
print(f"  Obj value: {result.info.obj_val:.4f}")
print(f"  Iterations:{result.info.iter}")
print(f"  Solve time:{result.info.solve_time*1000:.2f} ms")

# ── 5. Extract trajectory ─────────────────────────────────────────────────────
print("\n[Extracting optimal trajectory from z_opt]")
z_opt = result.x
x0_opt = z_opt[x_slice(0)]
u0_opt = z_opt[u_slice(0)]
xN_opt = z_opt[x_slice(N)]

print(f"  x_0 (predicted start): {x0_opt[:3].round(5)}")
print(f"  x_current (measured):  {x_current[:3].round(5)}")
print(f"  Diff:                  {np.abs(x0_opt[:3]-x_current[:3]).round(6)}")
print(f"  → x_0 must match x_current (equality constraint), diff should be ~0")

print(f"\n  u_0 (first GRF command): {u0_opt.round(2)} N")
print(f"  u_eq (equilibrium):      {u_eq.round(2)} N")
print(f"  u_0 - u_eq:              {(u0_opt - u_eq).round(2)} N")

print(f"\n  x_N (predicted final state CoM): {xN_opt[:3].round(4)} m")
print(f"  x_ref CoM:                       {x_eq[:3].round(4)} m")

# ── 6. Check friction cone on u_0 ────────────────────────────────────────────
print("\n[Verify u_0 satisfies friction cone]")
Au0 = A_fcon @ u0_opt
ok = np.all(Au0 >= l_fcon - 1e-6) and np.all(Au0 <= u_fcon + 1e-6)
print(f"  A_fcon @ u_0 in cone: {'✓' if ok else '✗'}")
print(f"  F_z = {u0_opt[2]:.2f} N,  |Fx|/Fz = {abs(u0_opt[0])/u0_opt[2]:.3f} (≤ μ={FRICTION_MU}?  {abs(u0_opt[0])/u0_opt[2]<=FRICTION_MU})")

# ── 7. Failure modes demo ─────────────────────────────────────────────────────
print("\n[Common failure modes]")

# Mode 1: Infeasible — impossible initial state
print("\n  Mode 1: Infeasible (CoM below ground)")
x_bad = x_eq.copy()
x_bad[2] = -0.1  # CoM underground

A_eq2 = A_eq.copy()
b_eq2 = b_eq.copy()
A_eq2[0:NX, x_slice(0)] = np.eye(NX)
b_eq2[0:NX] = x_bad
l_all2 = np.concatenate([b_eq2, l_ineq])
u_all2 = np.concatenate([b_eq2, u_ineq])

s2 = osqp.OSQP()
s2.setup(P, q, sp.csc_matrix(np.vstack([A_eq2, A_ineq])),
         l_all2, u_all2, verbose=False, max_iter=200)
r2 = s2.solve()
print(f"  Status with underground CoM: '{r2.info.status}'")

# Mode 2: Wrong dimensions
print("\n  Mode 2: Dimension mismatch (what the error looks like)")
print("  If q has wrong size, OSQP raises: ValueError or assert error")
print("  Always check: q.shape == (NZ,), P.shape == (NZ,NZ)")
print(f"  Here: NZ={NZ}, q.shape={q.shape}, P.shape={P.shape}")

# ── 8. Debug checklist ───────────────────────────────────────────────────────
print("\n[MPC debug checklist]")
print("  1. Status not 'solved'?")
print("     → Check A_eq, b_eq: is b_eq consistent with A_d/B_d?")
print("     → Check friction cone bounds: F_z_max too small?")
print("     → Increase max_iter or relax eps_abs/eps_rel")
print()
print("  2. u_0 violates friction cone?")
print("     → Check A_ineq column mapping (wrong ctrl_base offset?)")
print()
print("  3. x_0 ≠ x_current?")
print("     → Check equality block rows 0:NX: must be I at x_slice(0)")
print()
print("  4. Cost is huge?")
print("     → Check z_ref construction — is u_ref set to u_eq (not 0)?")
print("     → Check QN: terminal weight not too large causing solver issues")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters for single-leg balance:")
print("   mpc.py.solve() calls the OSQP solver every control step.")
print("   If the solver returns non-'solved' status, mpc.py falls back to u_eq.")
print("   Understanding OSQP debug output lets you diagnose balance failures.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. Warm-starting test (update state and re-solve):
# x_current2 = x_current.copy(); x_current2[0] += 0.01
# b_eq_new = b_eq.copy(); b_eq_new[:NX] = x_current2
# l_new = np.concatenate([b_eq_new, l_ineq])
# u_new = np.concatenate([b_eq_new, u_ineq])
# solver.update(l=l_new, u=u_new)
# result2 = solver.solve()
# print(f"Warm-start iters: {result2.info.iter} vs cold: {result.info.iter}")
# Expected: warm-start is faster (fewer iterations)

# 2. Watch the full predicted CoM trajectory:
# for k in range(N+1):
#     xk = z_opt[x_slice(k)]
#     print(f"  x_{k:2d} CoM: {xk[:3].round(4)}")
# Expected: CoM converges from x_current toward x_eq over horizon

# 3. Try very tight friction cone (μ=0.2) and see if solver still finds solution:
# A_tight, l_tight, u_tight = build_friction_cone(0.2, F_Z_MAX)
# A_ineq_tight = ... (rebuild with μ=0.2)
# ... re-setup and solve
# Expected: may fail if tight cone prevents recovery
""")
