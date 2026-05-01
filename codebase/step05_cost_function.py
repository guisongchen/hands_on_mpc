"""Step 05 — Cost function: what does the MPC minimize?

MPC minimizes a sum of quadratic costs over the prediction horizon:

    J = Σ_{k=0}^{N-1} [ (x_k - x_ref_k)ᵀ Q (x_k - x_ref_k)
                        + (u_k - u_eq)ᵀ R (u_k - u_eq) ]
        + (x_N - x_ref_N)ᵀ QN (x_N - x_ref_N)

This step covers:
  1. Role of Q, R, QN matrices
  2. Physical meaning of each weight
  3. How weights shape behavior (speed vs. effort trade-off)
  4. Unit scaling issues (why raw SI units can be poorly conditioned)

Run from single_leg_stand/:
    uv run python hands_on_mpc/step05_cost_function.py
"""

import numpy as np

from _mpc_utils import (
    NX, NU, N_HORIZON, TOTAL_MASS, GRAVITY_MAG, T_S,
    Q, R, QN,
    standing_equilibrium, make_centroidal_dynamics,
    discretize_euler, rollout,
    COM_HEIGHT, SUPPORT_POS_XY,
)

x_eq, u_eq = standing_equilibrium()

print("=" * 60)
print("STEP 05 — Cost function: Q, R, QN")
print("=" * 60)

# ── 1. State and control dimensions ─────────────────────────────────────────
print("\n[State and control layout (reminder)]")
state_labels  = ["c_x", "c_y", "c_z", "v_x", "v_y", "v_z", "L_x", "L_y", "L_z"]
control_labels = ["F_x", "F_y", "F_z"]
print("  x = [" + ", ".join(state_labels) + "]")
print("  u = [" + ", ".join(control_labels) + "]")
print(f"  NX={NX}, NU={NU}, N={N_HORIZON}")

# ── 2. Q matrix ──────────────────────────────────────────────────────────────
print("\n[Q matrix: state tracking weights]")
print("  Q penalizes deviation from the reference trajectory x_ref.")
print("  Q is diagonal (each state weighted independently).")
print()
print(f"  Q diagonal: {np.diag(Q).round(2)}")
print()
print("  Interpretation:")
for i, (lbl, qi) in enumerate(zip(state_labels, np.diag(Q))):
    unit = "m" if i < 3 else ("m/s" if i < 6 else "kg·m²/s")
    print(f"    Q[{i},{i}] = {qi:8.1f}  → {lbl:5s} ({unit})")

print()
print("  Note: Q[6:9] (angular momentum L) are smaller because:")
print("  L has larger magnitude (kg·m²/s) and we care most about CoM position.")

# ── 3. R matrix ──────────────────────────────────────────────────────────────
print("\n[R matrix: control effort weights]")
print("  R penalizes deviation of u from equilibrium u_eq.")
print("  Larger R → smoother forces, but slower response.")
print()
print(f"  R diagonal: {np.diag(R).round(6)}")
print()
print("  Interpretation:")
for i, (lbl, ri) in enumerate(zip(control_labels, np.diag(R))):
    print(f"    R[{i},{i}] = {ri:.6f}  → {lbl}  (N units)")
print()
print("  R is small → MPC is willing to use large forces.")
print("  This makes sense for a robot that needs to react quickly.")

# ── 4. Terminal cost QN ───────────────────────────────────────────────────────
print("\n[QN matrix: terminal cost at step N]")
print("  QN penalizes final state error more than intermediate steps.")
print("  This ensures the MPC drives toward the goal at end of horizon.")
print()
print(f"  QN diagonal: {np.diag(QN).round(2)}")
print(f"  QN/Q ratio:  {(np.diag(QN)/np.diag(Q)).round(2)}")
print()
print("  QN is larger than Q → robot must be close to x_ref at horizon end.")

# ── 5. Quadratic cost function ───────────────────────────────────────────────
print("\n[Computing the cost for a trajectory]")

# Build dynamics
r_foot  = np.array([SUPPORT_POS_XY[0], SUPPORT_POS_XY[1], 0.0])
r_lever = r_foot - x_eq[:3]
from _mpc_utils import make_centroidal_dynamics
A_c, B_c, d_c = make_centroidal_dynamics(TOTAL_MASS, r_lever)
A_d, B_d, d_d = discretize_euler(A_c, B_c, d_c, T_S)

# Reference: stay at equilibrium
x_ref = np.tile(x_eq, (N_HORIZON + 1, 1))

# Scenario A: apply exactly u_eq (optimal strategy from equilibrium)
u_eq_traj = np.tile(u_eq, (N_HORIZON, 1))
X_A = rollout(A_d, B_d, d_d, x_eq, u_eq_traj)

# Scenario B: perturbed start, still apply u_eq
x_pert = x_eq.copy()
x_pert[0] += 0.05   # 5cm CoM offset
x_pert[3]  = 0.1    # forward velocity
X_B = rollout(A_d, B_d, d_d, x_pert, u_eq_traj)

def compute_cost(X_traj, u_traj, x_ref_arr, u_eq_v):
    cost = 0.0
    for k in range(N_HORIZON):
        dx = X_traj[k] - x_ref_arr[k]
        du = u_traj[k] - u_eq_v
        cost += dx @ Q @ dx + du @ R @ du
    dx_N = X_traj[N_HORIZON] - x_ref_arr[N_HORIZON]
    cost += dx_N @ QN @ dx_N
    return cost

cost_A = compute_cost(X_A, u_eq_traj, x_ref, u_eq)
cost_B = compute_cost(X_B, u_eq_traj, x_ref, u_eq)

print(f"  Scenario A: start at x_eq, apply u_eq → J = {cost_A:.4f}")
print(f"  Scenario B: start perturbed,  apply u_eq → J = {cost_B:.4f}")
print(f"  Cost ratio B/A: {cost_B/max(cost_A,1e-12):.1f}×  (perturbed is costlier)")

# ── 6. Weight trade-off demo ─────────────────────────────────────────────────
print("\n[Weight trade-off: tracking vs effort]")
print("  If you increase R (penalize forces harder), the MPC uses less force.")
print("  If you increase Q (penalize state error harder), it tracks faster.")
print()
print("  Key ratios (what MPC actually sees):")
print(f"    Q_pos / R_F  ≈ {np.diag(Q)[0] / np.diag(R)[0]:.0f}  "
      f"→ MPC cares {np.diag(Q)[0]/np.diag(R)[0]:.0f}× more about pos than force")
print(f"    Q_vel / R_F  ≈ {np.diag(Q)[3] / np.diag(R)[0]:.0f}  "
      f"→ velocity tracking weight")

# ── 7. Unit scaling issue ────────────────────────────────────────────────────
print("\n[Unit scaling — why weights must account for different units]")
print("  State x mixes: positions (m), velocities (m/s), ang. mom. (kg·m²/s)")
print("  A 1 mm CoM error ≈ 0.001 m; a 1 kg·m²/s L error ≈ 1 in angular units.")
print()
print("  If Q was identity, L errors would dominate (they're in larger units).")
print("  Solution: normalize — Q[i] ∝ 1 / (acceptable_error_i)²")
print()
acceptable = {
    'c': 0.01,      # 1 cm position error OK
    'v': 0.1,       # 10 cm/s velocity error OK
    'L': 1.0,       # 1 kg·m²/s angular momentum error OK
    'F': 10.0,      # 10 N force variation OK
}
print("  Heuristic normalization Q[i] = 1/σ_i²:")
for name, sigma in acceptable.items():
    print(f"    σ_{name} = {sigma:5.3f} → Q = {1/sigma**2:8.1f}")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters for single-leg balance:")
print("   mpc.py: self._Q, self._R, self._QN fill the cost matrices in the QP.")
print("   If Q_pos is too small → MPC allows CoM to drift → robot falls.")
print("   If R is too large → MPC applies tiny forces → too slow to recover.")
print("   Tuning Q/R is one of the most important MPC design decisions.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. Increase Q_pos by 10× and compare cost for perturbed state:
# Q2 = Q.copy(); Q2[:3,:3] *= 10
# cost_B2 = 0.0
# for k in range(N_HORIZON):
#     dx = X_B[k] - x_ref[k]; du = u_eq_traj[k] - u_eq
#     cost_B2 += dx @ Q2 @ dx + du @ R @ du
# dx_N = X_B[N_HORIZON] - x_ref[N_HORIZON]
# cost_B2 += dx_N @ (QN * 1) @ dx_N   # keep QN same for simplicity
# print("Cost with 10x Q_pos:", cost_B2)
# Expected: cost increases, MPC will prioritize position recovery even more.

# 2. What if R = 0 (no effort penalty)?
# R0 = np.zeros_like(R)
# cost_noR = compute_cost(X_B, u_eq_traj, x_ref, u_eq)
# → Cost drops by the R penalty portion. Solver can use arbitrarily large forces.

# 3. Verify the minimum cost trajectory:
# At x_eq, u_eq: cost should equal zero (reference is equilibrium)
# print("Cost at perfect equilibrium:", compute_cost(X_A, u_eq_traj, x_ref, u_eq))
# Expected: ~0 (numerical noise only)
""")
