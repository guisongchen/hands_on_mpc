"""Step 08 — Friction cone: limiting the ground reaction force.

A foot on the ground can only push, not pull.  
The friction cone encodes two physical constraints:
  1. Normal force must be upward:  F_z ≥ 0
  2. Friction force within limits: |F_x|, |F_y| ≤ μ * F_z

Combined:  μ * F_z ≥ |F_x|  and  μ * F_z ≥ |F_y|  and  F_z ≥ 0

Pyramid approximation (4 sides):
  -μ*F_z ≤ F_x ≤ μ*F_z
  -μ*F_z ≤ F_y ≤ μ*F_z
  F_z ≥ 0  (and optionally F_z ≤ F_z_max)

This step covers:
  1. Physical meaning of friction cone
  2. Building the inequality constraint matrix A_fcon
  3. Stacking friction cone over the full horizon
  4. Checking whether a force violates the cone

Run from single_leg_stand/:
    uv run python hands_on_mpc/step08_friction_cone.py
"""

import numpy as np
from _mpc_utils import (
    NX, NU, N_HORIZON, FRICTION_MU, F_Z_MAX, TOTAL_MASS, GRAVITY_MAG,
    build_friction_cone, standing_equilibrium,
)

x_eq, u_eq = standing_equilibrium()

print("=" * 60)
print("STEP 08 — Friction cone: ground reaction force limits")
print("=" * 60)

# ── 1. Physical intuition ────────────────────────────────────────────────────
print("\n[Physical friction cone]")
print("  Imagine the foot as a point contact on the floor.")
print("  The floor can push up (normal force) and grip sideways (friction).")
print()
print("  Coulomb friction: |F_tangential| ≤ μ * F_normal")
print(f"  μ = {FRICTION_MU}  (friction coefficient, typical for rubber on floor)")
print()
print("  Constraints per force:")
print("    F_z ≥ 0         (floor can only push, not pull)")
print("    F_x ≥ -μ * F_z  (no sliding forward)")
print("    F_x ≤  μ * F_z  (no sliding backward)")
print("    F_y ≥ -μ * F_z  (no sliding left)")
print("    F_y ≤  μ * F_z  (no sliding right)")
print("    F_z ≤ F_z_max   (optional structural limit)")
print(f"\n  F_z_max = {F_Z_MAX} N  (max normal force = ~{F_Z_MAX/(TOTAL_MASS*GRAVITY_MAG):.1f}× body weight)")

# ── 2. Build single friction cone matrix ────────────────────────────────────
print("\n[Friction cone for a single time step]")
print("  A_fcon shape: (6, NU)  — 6 rows (constraints) × 3 cols (Fx,Fy,Fz)")

A_fcon, l_fcon, u_fcon = build_friction_cone(FRICTION_MU, F_Z_MAX)

print(f"\n  A_fcon:\n{A_fcon}")
print(f"\n  l_fcon (lower bounds): {l_fcon}")
print(f"  u_fcon (upper bounds): {u_fcon}")
print()
print("  Constraint format:  l_fcon ≤ A_fcon @ u ≤ u_fcon")

# ── 3. Check some forces ─────────────────────────────────────────────────────
print("\n[Checking forces against the cone]")

def check_force(label, f):
    Au = A_fcon @ f
    ok = np.all(Au >= l_fcon - 1e-9) and np.all(Au <= u_fcon + 1e-9)
    status = "✓ in cone" if ok else "✗ VIOLATES CONE"
    print(f"  {label:40s}: {status}")
    if not ok:
        for i, (ai, li, ui, aui) in enumerate(zip(A_fcon, l_fcon, u_fcon, Au)):
            if not (li - 1e-9 <= aui <= ui + 1e-9):
                print(f"    → row {i} violated: {li:.1f} ≤ {aui:.2f} ≤ {ui:.1f}")

m_g = TOTAL_MASS * GRAVITY_MAG
check_force(f"u_eq = [0, 0, {m_g:.0f}N] (equilibrium)", u_eq)
check_force("F = [0, 0, 200] (light vertical)",  np.array([0.0, 0.0, 200.0]))
check_force("F = [0, 0, -50] (pulling up!)",     np.array([0.0, 0.0, -50.0]))
check_force("F = [300, 0, 200] (too much x)",    np.array([300.0, 0.0, 200.0]))
check_force("F = [60, 0, 200] (x = μ*Fz OK)",   np.array([60.0, 0.0, 200.0]))
check_force("F = [61, 0, 200] (x > μ*Fz, slip)", np.array([61.0, 0.0, 200.0]))
check_force(f"F = [0, 0, {F_Z_MAX+1:.0f}] (exceeds F_z_max)", np.array([0.0, 0.0, F_Z_MAX+1]))

# ── 4. Full horizon stacking ──────────────────────────────────────────────────
print("\n[Stacking friction cone over N horizon steps]")
print("  Each u_k must satisfy the friction cone independently.")
print("  Stack N copies of A_fcon into a block-diagonal matrix.")
print()

N      = N_HORIZON
NZ     = (N + 1) * NX + N * NU
n_fcon = len(l_fcon)  # rows per step (5 for pyramid cone)

# Build block-diagonal inequality constraint matrix
A_ineq = np.zeros((N * n_fcon, NZ))
l_ineq = np.zeros(N * n_fcon)
u_ineq = np.zeros(N * n_fcon)

ctrl_base = (N + 1) * NX
for k in range(N):
    row = k * n_fcon
    col = ctrl_base + k * NU
    A_ineq[row:row + n_fcon, col:col + NU] = A_fcon
    l_ineq[row:row + n_fcon] = l_fcon
    u_ineq[row:row + n_fcon] = u_fcon

print(f"  A_ineq shape: {A_ineq.shape}  ({N} steps × {n_fcon} constraints × {NU} controls)")
print(f"  Non-zeros: {np.count_nonzero(A_ineq)} / {A_ineq.size}  ({100*np.count_nonzero(A_ineq)/A_ineq.size:.1f}% dense)")

# Check equilibrium control satisfies all steps
U_const = np.tile(u_eq, N)  # flat vector
Au_full = A_ineq @ np.concatenate([np.zeros((N + 1) * NX), U_const])
ok_all = np.all(Au_full >= l_ineq - 1e-9) and np.all(Au_full <= u_ineq + 1e-9)
print(f"\n  Equilibrium forces for all {N} steps: {'✓ all in cone' if ok_all else '✗ violation'}")

# ── 5. Visualize cone cross-section ──────────────────────────────────────────
print("\n[Cone cross-section at F_z = 200 N]")
Fz   = 200.0
fmax = FRICTION_MU * Fz
print(f"  At F_z = {Fz} N, friction limit = μ*F_z = {fmax} N")
print(f"  Allowable F_x range: [{-fmax:.1f}, {fmax:.1f}] N")
print(f"  Allowable F_y range: [{-fmax:.1f}, {fmax:.1f}] N")
print()
print("  ASCII cross-section (Fx vs Fy at Fz=200N):")
size = 7
for row_i in range(size):
    line = "  "
    for col_j in range(size):
        Fx = (col_j - size//2) / (size//2) * fmax * 1.2
        Fy = (size//2 - row_i) / (size//2) * fmax * 1.2
        f = np.array([Fx, Fy, Fz])
        Au = A_fcon @ f
        in_cone = np.all(Au >= l_fcon - 1e-9) and np.all(Au <= u_fcon + 1e-9)
        line += "██" if in_cone else "  "
    print(line)
print(f"  (square footprint = pyramid cone; actual cone is circular)")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters for single-leg balance:")
print("   mpc.py.build_friction_cone() creates A_fcon and stacks it into the QP.")
print("   If μ is too low, the solver is overly restricted and can't plan recovery.")
print("   If F_z_max is too low, the solver may be infeasible when the robot lands.")
print("   Friction cone violations in simulation = foot slipping in reality.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. What is the maximum lateral force allowed at equilibrium (F_z = m*g)?
# Fz_eq = TOTAL_MASS * GRAVITY_MAG
# Fx_max = FRICTION_MU * Fz_eq
# print(f"Max Fx at equilibrium: {Fx_max:.1f} N = {Fx_max/Fz_eq*100:.1f}% of body weight")

# 2. Try μ = 0.3 (icy floor) vs μ = 1.0 (grippy rubber):
# for mu in [0.3, 0.6, 1.0]:
#     A, l, u = build_friction_cone(mu, F_Z_MAX)
#     f_lateral_max = mu * (TOTAL_MASS * GRAVITY_MAG)
#     print(f"  μ={mu}: max lateral force at eq = {f_lateral_max:.1f} N")
# → lower μ means tighter cone = harder to maintain balance with lean

# 3. Is u_eq inside a tight cone (μ=0.3)?
# A_tight, l_tight, u_tight = build_friction_cone(0.3, F_Z_MAX)
# Au = A_tight @ u_eq
# in_cone = np.all(Au >= l_tight) and np.all(Au <= u_tight)
# print("u_eq in tight cone (μ=0.3):", in_cone)
# Expected: True — u_eq is [0, 0, mg], purely vertical, no friction needed
""")
