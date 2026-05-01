"""Step 10 — Receding horizon: the full closed-loop MPC loop.

So far we've built:
  - Dynamics model (A_d, B_d, d_d)
  - Cost matrices (Q, R, QN)
  - Decision variable z
  - Equality constraints (dynamics + initial state)
  - Inequality constraints (friction cone)
  - OSQP solver

Now we combine everything into a closed-loop MPC simulation:
  for each step t:
    1. Observe current state x_t
    2. Solve QP → get optimal z
    3. Extract u_0 (first control)
    4. Apply u_0 to the system → get x_{t+1}
    5. Repeat

This is the "receding horizon" principle: plan N steps, execute 1.

Run from single_leg_stand/:
    uv run python hands_on_mpc/step10_receding_horizon.py
"""

import numpy as np
import scipy.sparse as sp
import osqp

from _mpc_utils import (
    NX, NU, N_HORIZON, TOTAL_MASS, GRAVITY_MAG, T_S,
    FRICTION_MU, F_Z_MAX,
    Q, R, QN,
    standing_equilibrium, make_centroidal_dynamics,
    discretize_euler, build_friction_cone, rollout,
    COM_HEIGHT, SUPPORT_POS_XY,
)

x_eq, u_eq = standing_equilibrium()
r_foot  = np.array([SUPPORT_POS_XY[0], SUPPORT_POS_XY[1], 0.0])
r_lever = r_foot - x_eq[:3]
A_c, B_c, d_c = make_centroidal_dynamics(TOTAL_MASS, r_lever)
A_d, B_d, d_d = discretize_euler(A_c, B_c, d_c, T_S)

N  = N_HORIZON
NZ = (N + 1) * NX + N * NU

A_fcon, l_fcon, u_fcon = build_friction_cone(FRICTION_MU, F_Z_MAX)
n_per = len(l_fcon)

print("=" * 60)
print("STEP 10 — Receding horizon: closed-loop MPC simulation")
print("=" * 60)

# ── Index helpers ─────────────────────────────────────────────────────────────
def x_slice(k):
    return slice(k * NX, (k + 1) * NX)
def u_slice(k):
    return slice((N + 1) * NX + k * NU, (N + 1) * NX + (k + 1) * NU)

# ── 1. Build static parts of QP (don't change between steps) ────────────────
print("\n[Building static QP components]")

# Cost: P (same every step)
P_dense = np.zeros((NZ, NZ))
for k in range(N):
    P_dense[x_slice(k), x_slice(k)] = Q
    P_dense[u_slice(k), u_slice(k)] = R
P_dense[x_slice(N), x_slice(N)] = QN
P_sp = sp.csc_matrix(P_dense)

# Linear term: track x_eq, u_eq
z_ref  = np.concatenate([np.tile(x_eq, N + 1), np.tile(u_eq, N)])
q_cost = -P_dense @ z_ref

# Friction cone (block-diagonal over horizon)
A_ineq_dense = np.zeros((N * n_per, NZ))
l_ineq = np.zeros(N * n_per)
u_ineq = np.zeros(N * n_per)
for k in range(N):
    row = k * n_per
    col = (N + 1) * NX + k * NU
    A_ineq_dense[row:row+n_per, col:col+NU] = A_fcon
    l_ineq[row:row+n_per] = l_fcon
    u_ineq[row:row+n_per] = u_fcon

print(f"  P: {P_sp.shape}, nnz={P_sp.nnz}")
print(f"  Friction cone A_ineq: {A_ineq_dense.shape}")

# ── 2. Closed-loop simulation function ───────────────────────────────────────
def build_equality(x_curr):
    """Build A_eq and b_eq for current state x_curr."""
    n_eq  = (N + 1) * NX
    A_eq  = np.zeros((n_eq, NZ))
    b_eq  = np.zeros(n_eq)
    A_eq[0:NX, x_slice(0)] = np.eye(NX)
    b_eq[0:NX] = x_curr
    for k in range(N):
        r = (k + 1) * NX
        A_eq[r:r+NX, x_slice(k)]     = -A_d
        A_eq[r:r+NX, u_slice(k)]     = -B_d
        A_eq[r:r+NX, x_slice(k + 1)] =  np.eye(NX)
        b_eq[r:r+NX]                 =  d_d
    return A_eq, b_eq

# ── 3. Simulate 30 steps from perturbed state ─────────────────────────────────
print("\n[Closed-loop simulation: 30 steps from perturbed initial state]")
print("  Initial perturbation: +5cm CoM forward, +0.1 m/s velocity")

x_sim = x_eq.copy()
x_sim[0] += 0.05
x_sim[3]  = 0.1

SIM_STEPS = 30
history_x = [x_sim.copy()]
history_u = []
status_list = []

# Initialize solver with first step
A_eq0, b_eq0 = build_equality(x_sim)
A_all = np.vstack([A_eq0, A_ineq_dense])
l_all = np.concatenate([b_eq0, l_ineq])
u_all = np.concatenate([b_eq0, u_ineq])

solver = osqp.OSQP()
solver.setup(P_sp, q_cost, sp.csc_matrix(A_all), l_all, u_all,
             warm_starting=True, verbose=False, eps_abs=1e-5, eps_rel=1e-5)

for t in range(SIM_STEPS):
    # Solve QP
    result = solver.solve()
    status = result.info.status
    status_list.append(status)

    if status == "solved":
        u_t = result.x[u_slice(0)]
    else:
        u_t = u_eq.copy()  # fallback to equilibrium

    # Apply control: x_{t+1} = A_d * x_t + B_d * u_t + d_d
    x_next = A_d @ history_x[-1] + B_d @ u_t + d_d
    history_x.append(x_next)
    history_u.append(u_t)

    # Update equality constraints for next state (warm start)
    A_eq_t, b_eq_t = build_equality(x_next)
    l_new = np.concatenate([b_eq_t, l_ineq])
    u_new = np.concatenate([b_eq_t, u_ineq])
    solver.update(l=l_new, u=u_new)

# ── 4. Report results ─────────────────────────────────────────────────────────
print(f"\n  {'t':>4}  {'CoM x (m)':>10}  {'CoM z (m)':>10}  {'Fz (N)':>8}  {'status':>12}")
for t in range(SIM_STEPS + 1):
    x_t = history_x[t]
    u_str  = f"{history_u[t][2]:8.1f}" if t < SIM_STEPS else "       -"
    s_str  = status_list[t]              if t < SIM_STEPS else "       -"
    print(f"  {t:4d}  {x_t[0]:10.5f}  {x_t[2]:10.5f}  {u_str}  {s_str:>12}")

# ── 5. Convergence summary ────────────────────────────────────────────────────
x_final = history_x[-1]
pos_err  = np.linalg.norm(x_final[:3] - x_eq[:3])
vel_err  = np.linalg.norm(x_final[3:6])
print(f"\n  Final position error: {pos_err*100:.2f} cm  (from x_eq)")
print(f"  Final velocity norm:  {vel_err:.4f} m/s")
print(f"  Solver failures:      {sum(1 for s in status_list if s != 'solved')}/{SIM_STEPS}")

if pos_err < 0.005:
    print("  ✓ Robot converged to equilibrium within 5mm")
else:
    print("  ✗ Robot did not fully converge — consider longer horizon or larger Q")

# ── 6. Receding horizon principle ────────────────────────────────────────────
print("\n[Receding horizon principle]")
print("  Each step: plan N=10 steps ahead, execute only 1 step.")
print("  Then: shift window forward, re-observe state, re-plan.")
print()
print("  Why not execute all N planned steps?")
print("    → Model is never perfect (disturbances, noise)")
print("    → Re-measuring x every step corrects accumulated errors")
print("    → Called 'Model Predictive Control' precisely for this reason")
print()
print(f"  Our simulation: N={N}, T_S={T_S}s → look {N*T_S:.2f}s ahead each solve")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters for single-leg balance:")
print("   stand_unified.py calls mpc.solve(x_current) at every control step.")
print("   The MPC output (u_0) is passed to WBC → joint torques → motors.")
print("   If MPC fails, the robot falls. Understanding this loop helps debug.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. Open-loop vs closed-loop comparison:
# Apply u_eq for all 30 steps WITHOUT re-solving:
# x_ol = x_sim.copy()
# hist_ol = [x_ol.copy()]
# for t in range(SIM_STEPS):
#     x_ol = A_d @ x_ol + B_d @ u_eq + d_d
#     hist_ol.append(x_ol.copy())
# print("Open-loop final CoM x:", hist_ol[-1][0])
# print("Closed-loop final CoM x:", history_x[-1][0])
# Expected: open-loop drifts, closed-loop converges

# 2. Try a larger perturbation (10cm offset):
# x_large = x_eq.copy(); x_large[0] += 0.1
# ... re-run simulation
# Does the MPC still converge? Does it take more steps?

# 3. Reduce N to 3 and compare convergence:
# Smaller horizon = shorter lookahead = slower or no convergence for big perturbations
""")
