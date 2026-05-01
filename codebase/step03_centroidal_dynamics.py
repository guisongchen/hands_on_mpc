"""Step 03 — Centroidal dynamics: the model of the system.

The MPC needs a model: given state x and control u, what is x_next?

CENTROIDAL DYNAMICS MODEL:
  The centroidal dynamics of a robot simplify to Newton-Euler at the CoM:

    m * c_ddot = Σ f_i + m*g          (Newton: force = mass × acceleration)
    L_dot      = Σ (r_i - c) × f_i   (Euler: torque = angular momentum rate)

  For single-support (one contact force f at foot position r_foot):
    c_ddot = (1/m) * f + g
    L_dot  = (r_foot - c) × f   ← lever arm × force

  Linearized around equilibrium (c_eq, L=0, u_eq=[0,0,mg]):
    δx_dot = A_c * δx + B_c * u + d_c

  Where:
    A_c[0:3, 3:6] = I   (velocity integrates position)
    B_c[3:6, :]   = I/m (force → CoM acceleration)
    B_c[6:9, :]   = skew(r_foot - c_eq)  (torque from force × lever arm)
    d_c[3:6]      = [0, 0, -g]  (gravity)

NOTE: This is a SIMPLIFICATION. The full dynamics depend on all body
      positions, but for control purposes a centroidal model is sufficient.

Run from single_leg_stand/:
    uv run python hands_on_mpc/step03_centroidal_dynamics.py
"""

import numpy as np

from _mpc_utils import (
    NX, NU, TOTAL_MASS, GRAVITY_MAG, G_VEC,
    COM_HEIGHT, SUPPORT_POS_XY,
    make_centroidal_dynamics, standing_equilibrium, rollout,
    discretize_euler, _skew,
)

print("=" * 60)
print("STEP 03 — Centroidal dynamics model")
print("=" * 60)

# ── 1. Newton-Euler equations ────────────────────────────────────────────────
print("\n[Newton-Euler at the CoM]")
print("  m * c_ddot = Σ f_i + m * g     ← Newton (linear)")
print("  L_dot      = Σ (r_i - c) × f_i ← Euler  (angular)")
print()
print(f"  m = {TOTAL_MASS} kg  (robot mass)")
print(f"  g = {G_VEC}  m/s²")
print()
print("  Simplification: 1 contact force at foot position r_foot")
print("  → c_ddot = (1/m)*f + g")
print("  → L_dot  = (r_foot - c) × f")

# ── 2. Build A_c, B_c, d_c ───────────────────────────────────────────────────
print("\n[Continuous-time matrices: A_c, B_c, d_c]")

# Lever arm = r_foot - c_eq  (at equilibrium)
x_eq, u_eq = standing_equilibrium()
c_eq = x_eq[:3]
r_foot = np.array([SUPPORT_POS_XY[0], SUPPORT_POS_XY[1], 0.0])  # foot at ground
r_lever = r_foot - c_eq  # lever arm at equilibrium

print(f"\n  Equilibrium CoM: c_eq = {c_eq}")
print(f"  Support foot pos: r_foot = {r_foot}")
print(f"  Lever arm r_lever = r_foot - c_eq = {r_lever.round(4)}")

A_c, B_c, d_c = make_centroidal_dynamics(TOTAL_MASS, r_lever)

print(f"\n  A_c ({A_c.shape}):")
print("  [Rows 0-2: position kinematics]")
print("  [Rows 3-5: CoM acceleration (zero A_c part, force via B_c)]")
print("  [Rows 6-8: angular momentum (zero A_c part)]")
for i, row in enumerate(A_c):
    nz = np.nonzero(row)[0]
    labels = {0:'cx',1:'cy',2:'cz',3:'vx',4:'vy',5:'vz',6:'Lx',7:'Ly',8:'Lz'}
    parts = " + ".join(f"{row[j]:.2f}*{labels[j]}" for j in nz) if len(nz) else "0"
    lhs = {0:'ċx',1:'ċy',2:'ċz',3:'v̇x',4:'v̇y',5:'v̇z',6:'L̇x',7:'L̇y',8:'L̇z'}[i]
    print(f"  {lhs} = {parts}")

print(f"\n  B_c ({B_c.shape}):")
print(f"  Rows 3-5 (CoM accel): I/m =\n  {B_c[3:6].round(5)}")
print(f"  Rows 6-8 (L_dot):     skew(r_lever) =\n  {B_c[6:9].round(4)}")
print("  (skew matrix encodes cross product: r × f)")

print(f"\n  d_c = {d_c}  (gravity affine term: d_c[3:6] = g)")

# ── 3. Physical interpretation of B_c ────────────────────────────────────────
print("\n[Physical check: B_c encodes gravity + lever arm]")

# Apply u = u_eq = [0, 0, mg] at equilibrium → should give zero acceleration
x_dot_at_eq = A_c @ x_eq + B_c @ u_eq + d_c
print(f"  At equilibrium: A_c*x_eq + B_c*u_eq + d_c = {x_dot_at_eq.round(6)}")
print("  → Should be ~zero (equilibrium: no acceleration)")
print(f"  Max |x_dot_at_eq|: {np.max(np.abs(x_dot_at_eq)):.2e}  ({'✓ correct' if np.max(np.abs(x_dot_at_eq)) < 1e-6 else '✗ check model'})")

# ── 4. Lever arm cross product demo ──────────────────────────────────────────
print("\n[Lever arm cross product: why L_dot = r_lever × f]")
print("  skew(r) @ f  =  r × f  (cross product as matrix multiplication)")
print()
f_test = np.array([0.0, 0.0, TOTAL_MASS * GRAVITY_MAG])  # vertical force
L_dot_cross  = np.cross(r_lever, f_test)
L_dot_matrix = _skew(r_lever) @ f_test
print(f"  r_lever = {r_lever.round(4)}")
print(f"  f_test  = {f_test}  (full weight upward)")
print(f"  r × f (np.cross):    {L_dot_cross.round(4)}")
print(f"  skew(r) @ f (matrix):{L_dot_matrix.round(4)}")
print(f"  Match: {np.allclose(L_dot_cross, L_dot_matrix)}")
print()
print("  L_dot[1] (pitch) ≠ 0 means vertical force at foot lateral offset")
print("  creates a pitching moment — the robot would tilt forward/backward.")

# ── 5. Why gravity enters as d_c, not A_c ────────────────────────────────────
print("\n[Why gravity is an affine term d_c, not part of A_c]")
print("  A_c is the state-derivative matrix: x_dot = A_c*x + ...")
print("  Gravity doesn't depend on state x (it's constant).")
print("  → Gravity enters as d_c: x_dot = A_c*x + B_c*u + d_c")
print()
print("  At equilibrium, B_c*u_eq cancels d_c exactly:")
print(f"  B_c*u_eq = {(B_c@u_eq).round(4)}")
print(f"  d_c      = {d_c.round(4)}")
print(f"  B_c*u_eq + d_c = {(B_c@u_eq + d_c).round(6)}  (≈ zero)")

# ── 6. What the model ignores ────────────────────────────────────────────────
print("\n[Modeling simplifications — what this model IGNORES]")
print("  ✗ Individual joint positions/velocities")
print("  ✗ Variable inertia (assumes centroidal inertia is constant)")
print("  ✗ Foot rotation / CoP within support polygon")
print("  ✗ Full kinematics chain (pelvis, knees, ankles)")
print("  ✗ Multiple contact points / foot corners")
print("  ✓ Captures: CoM trajectory, net GRF, first-order angular momentum")
print("  ✓ Good enough for: planning CoM trajectory, deciding contact force")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters for single-leg balance:")
print("   mpc.py.set_dynamics(A_d, B_d, d_d) feeds these matrices to the QP.")
print("   If the lever arm r_lever is wrong (foot moved), B_c changes and the")
print("   MPC will compute wrong angular momentum corrections.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. What if the foot is directly under the CoM (r_lever = 0)?
# A_c2, B_c2, d_c2 = make_centroidal_dynamics(TOTAL_MASS, np.zeros(3))
# print("B_c rows 6-8 (angular momentum):", B_c2[6:9])
# Expected: all zeros — no torque from force at zero lever arm

# 2. Simulate 3 steps manually: apply u_eq (hover force) from x_eq
# A_d, B_d, d_d = discretize_euler(A_c, B_c, d_c, T_S=0.05)
# x = x_eq.copy()
# for k in range(3):
#     x_next = A_d @ x + B_d @ u_eq + d_d
#     print(f"Step {k+1}: x = {x_next[:3].round(4)} (should stay at c_eq)")
#     x = x_next
# Expected: CoM stays at c_eq (equilibrium is stable under u_eq)

# 3. Apply a horizontal disturb force and see L_dot change:
# f_disturb = np.array([50.0, 0.0, TOTAL_MASS*GRAVITY_MAG])  # +50N forward
# L_dot_disturb = _skew(r_lever) @ f_disturb
# print("L_dot from forward push:", L_dot_disturb.round(4))
# Expected: L_dot[1] (pitch) changes significantly
""")
