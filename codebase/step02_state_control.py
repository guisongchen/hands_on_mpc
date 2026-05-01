"""Step 02 — State and control vectors: physical meaning.

Before building any matrices, you must know exactly what every number
in the state and control vectors represents.

For the centroidal MPC:
  State x (9D):    [c_x, c_y, c_z,          ← CoM position (m)
                    c_dot_x, c_dot_y, c_dot_z, ← CoM velocity (m/s)
                    L_x, L_y, L_z]            ← centroidal angular momentum (kg·m²/s)

  Control u (3D):  [f_x, f_y, f_z]           ← ground reaction force (N)

These are the CENTROIDAL quantities: they live at the whole-body CoM,
not at individual joints. This is a deliberate simplification —
the MPC only cares about CoM motion, not individual joint angles.

Run from single_leg_stand/:
    uv run python hands_on_mpc/step02_state_control.py
"""

import numpy as np

from _mpc_utils import (
    NX, NU, N_HORIZON, T_S,
    TOTAL_MASS, GRAVITY_MAG, COM_HEIGHT, SUPPORT_POS_XY,
    standing_equilibrium,
)

print("=" * 60)
print("STEP 02 — State and control vectors")
print("=" * 60)

# ── 1. State vector x ─────────────────────────────────────────────────────────
print("\n[State vector x — shape (9,)]")
print("  Indices 0:3  → CoM position  c   [m]")
print("  Indices 3:6  → CoM velocity  c_dot [m/s]")
print("  Indices 6:9  → centroidal angular momentum L [kg·m²/s]")
print()
print("  ⚠  These are NOT joint angles or foot positions.")
print("      They are whole-body aggregate quantities computed from all body masses.")
print()
print(f"  NX = {NX}  (defined in config.py / _mpc_utils.py)")

# ── 2. Control vector u ──────────────────────────────────────────────────────
print("\n[Control vector u — shape (3,)]")
print("  Indices 0:3  → Ground Reaction Force (GRF) [N]")
print("     u[0] = Fx  (lateral,   tangential)")
print("     u[1] = Fy  (forward,   tangential)")
print("     u[2] = Fz  (vertical,  normal force, must be ≥ 0!)")
print()
print(f"  NU = {NU}  (3D force, no moment — simplified model)")
print()
print("  ⚠  This is the TOTAL GRF at the support foot, not a joint torque.")
print("      The WBC (whole-body controller) later converts GRF → joint torques.")

# ── 3. Physical bounds and typical values ─────────────────────────────────────
print("\n[Typical values during single-leg standing]")
x_eq, u_eq = standing_equilibrium()

print(f"\n  Equilibrium state x_eq:")
print(f"    CoM position  = {x_eq[:3]}  [m]")
print(f"    CoM velocity  = {x_eq[3:6]}  [m/s]  (zero at rest)")
print(f"    Angular mom.  = {x_eq[6:9]}  [kg·m²/s]  (zero at rest)")

print(f"\n  Equilibrium control u_eq:")
print(f"    GRF = {u_eq}  [N]")
print(f"    = [0, 0, m*g] = [0, 0, {TOTAL_MASS}×{GRAVITY_MAG}] = {TOTAL_MASS*GRAVITY_MAG:.1f} N")
print(f"    → At rest, support foot carries full robot weight upward.")

# ── 4. Coordinate system ─────────────────────────────────────────────────────
print("\n[Coordinate system convention]")
print("  x → forward (robot faces +x)")
print("  y → lateral (left is +y)")
print("  z → up (gravity is -z)")
print()
print("  CoM at equilibrium:")
print(f"    c_x ≈ 0.0 m  (no forward lean at rest)")
print(f"    c_y ≈ {SUPPORT_POS_XY[1]:.2f} m  (over support foot lateral position)")
print(f"    c_z ≈ {COM_HEIGHT:.2f} m  (CoM height)")

# ── 5. Interpreting L (angular momentum) ─────────────────────────────────────
print("\n[Angular momentum L — why it's in the state]")
print("  L = Σ_i  m_i * (r_i - c) × v_i  +  I_i * ω_i")
print("    = the 'spin' of the robot about its own CoM")
print()
print("  L = 0  → no spinning around CoM  (desired for standing)")
print("  L_x ≠ 0 → robot is rolling")
print("  L_y ≠ 0 → robot is pitching")
print("  L_z ≠ 0 → robot is yawing")
print()
print("  The MPC penalizes L ≠ 0, encouraging the robot to not spin over time.")

# ── 6. How x relates to MuJoCo state ─────────────────────────────────────────
print("\n[Connection to MuJoCo state (from step 3 of MuJoCo learning)]")
print("  MuJoCo qpos (30D) → centroidal x (9D) by:")
print("    c     = data.subtree_com[root_id]              (step 05 of MuJoCo)")
print("    c_dot = data.subtree_linvel[root_id]           (step 11 of MuJoCo)")
print("    L     = data.subtree_angmom[root_id]           (step 11 of MuJoCo)")
print()
print("  This massive reduction (30D → 9D) is the centroidal abstraction.")
print("  It loses joint-level detail but makes the MPC tractable.")

# ── 7. Unit magnitudes ───────────────────────────────────────────────────────
print("\n[Expected magnitudes — important for cost weight tuning]")
print(f"  CoM position error during shift: ~0.1 m")
print(f"  CoM velocity:                    ~0.1-0.5 m/s  (slow balance)")
print(f"  Angular momentum L:              ~0.1-1.0 kg·m²/s  (small for balance)")
print(f"  GRF Fz:                          ~{TOTAL_MASS*GRAVITY_MAG:.0f} N  (robot weight)")
print(f"  GRF Fx, Fy:                      ~0-50 N  (friction forces)")
print()
print("  ⚠  Units differ widely! This is WHY Q and R have very different values.")
print("      A 0.01m CoM error is just as important as a 10 N/m² cost difference.")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters for single-leg balance:")
print("   The state estimator (state_estimator.py) reads MuJoCo data and packs")
print("   it into exactly this x vector. mpc.py then optimizes over x, outputting")
print("   GRF u. If you confuse what's in x or u, the controller will fight itself.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. What is c_y at equilibrium for the LEFT foot support?
#    (swap support foot to +0.09 m, right is -0.09 m)
# x_left, u_left = standing_equilibrium(support_xy=np.array([0.0, 0.09]))
# print("Left support c_eq:", x_left[:3])
# Expected: c_y ≈ +0.09 m

# 2. If the robot leans 5° (0.087 rad) in roll, estimate L_x:
#    L_x ≈ I_xx * omega_x  (rough estimate)
#    moment of inertia ~10 kg·m²,  lean rate ~0.1 rad/s → L_x ~1 kg·m²/s
# I_xx = 10.0   # approximate whole-body roll inertia [kg·m²]
# omega_x = 0.1  # lean rate [rad/s]
# L_x_estimate = I_xx * omega_x
# print(f"Estimated L_x = {L_x_estimate} kg·m²/s")

# 3. What does the decision variable z look like in terms of x?
# N = 10
# nz = (N+1)*NX + N*NU
# print(f"Decision variable size: {nz}")
# print(f"  x_0 at indices 0:{NX}")
# print(f"  x_1 at indices {NX}:{2*NX}")
# ...
# print(f"  u_0 at indices {(N+1)*NX}:{(N+1)*NX + NU}")
""")
