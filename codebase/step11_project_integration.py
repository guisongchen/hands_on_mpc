"""Step 11 — Connecting to the real project: mpc.py integration.

All previous steps taught centroidal MPC from scratch.
Now we map every concept onto the actual CentroidalMPC class in mpc.py.

This step covers:
  1. CentroidalMPC class structure and call flow
  2. How state_estimator.py provides x_current
  3. How mpc.py → wbc_v2.py → joint torques chain works
  4. What happens at each major step in stand_unified.py

Run from single_leg_stand/:
    uv run python hands_on_mpc/step11_project_integration.py
"""

import sys
import os
import numpy as np

# Make parent (single_leg_stand/) importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("=" * 60)
print("STEP 11 — Connecting to the real project: mpc.py")
print("=" * 60)

# ── 1. Import and inspect the real CentroidalMPC ─────────────────────────────
print("\n[Importing and inspecting CentroidalMPC]")

try:
    from mpc import CentroidalMPC
    from config import SingleLegConfig as Cfg
    mpc = CentroidalMPC(Cfg)
    print("  CentroidalMPC loaded successfully")
except Exception as e:
    print(f"  Import failed: {e}")
    print("  → Run this script from single_leg_stand/ with 'uv run python'")
    sys.exit(1)

# ── 2. Inspect config values ──────────────────────────────────────────────────
print("\n[Config values used by MPC]")
fields = [
    ("N_HORIZON",   "prediction horizon steps"),
    ("T_S",         "sampling period (s)"),
    ("TOTAL_MASS",  "robot total mass (kg)"),
    ("MU",          "friction coefficient"),
    ("Q_DIAG",      "state cost diagonal"),
    ("R_DIAG",      "control cost diagonal"),
    ("QN_DIAG",     "terminal cost diagonal"),
    ("F_Z_MAX",     "max normal force (N)"),
]
for attr, desc in fields:
    val = getattr(Cfg, attr, "NOT FOUND")
    print(f"  Cfg.{attr:<15} = {val!r:<30}  ({desc})")

# ── 3. mpc.py method walkthrough ─────────────────────────────────────────────
print("\n[CentroidalMPC method walkthrough]")
print("""
  CentroidalMPC.__init__(cfg)
    → Reads N, T_S, mass, friction, Q/R/QN from config
    → Calls _build_cost_matrices()   ← step05 (cost function)
    → Calls _build_friction_cone()   ← step08 (friction cone)
    → Calls _setup_osqp()            ← step09 (OSQP setup)

  CentroidalMPC.solve(x_current, r_foot, x_ref)
    → Calls _update_dynamics(r_foot) ← step03/04 (centroidal dynamics + discretize)
    → Calls _build_equality(x_current) ← step07 (equality constraints)
    → Calls _update_qp()             ← updates P, q, A, l, u in OSQP
    → Calls solver.solve()           ← step09 (OSQP solve)
    → Extracts u_0 from z_opt        ← step06 (decision variable z)
    → Returns u_0 (GRF command)
""")

# ── 4. Trace a real solve ─────────────────────────────────────────────────────
print("[Tracing a real CentroidalMPC.solve() call]")

# Build a plausible state: close to equilibrium
try:
    from state_estimator import StateEstimator
    from robot_model import RobotModel
    import mujoco
    xml_path = "/home/ccc/projects/unitree_ros/robots/g1_description/g1_23dof.xml"
    if os.path.exists(xml_path):
        model = mujoco.MjModel.from_xml_path(xml_path)
        data  = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        robot = RobotModel(model, data)
        estimator = StateEstimator(robot, Cfg)
        x_current = estimator.get_state()
        r_foot    = robot.get_body_pos(Cfg.FOOT_BODY_NAME)
        print(f"  x_current (from MuJoCo T-pose): {x_current[:6].round(4)}")
        print(f"  r_foot:                          {r_foot.round(4)}")
        u_opt = mpc.solve(x_current, r_foot)
        print(f"  u_0 (GRF):  {u_opt.round(2)} N")
        print(f"  u_eq:       Fz ≈ {Cfg.TOTAL_MASS * 9.81:.1f} N")
        print(f"  Fz ratio:   {u_opt[2] / (Cfg.TOTAL_MASS * 9.81):.3f}  (should be near 1.0 from rest)")
    else:
        raise FileNotFoundError(xml_path)
except Exception as e:
    print(f"  MuJoCo trace skipped ({e})")
    print("  → Using synthetic state instead:")
    from _mpc_utils import standing_equilibrium, COM_HEIGHT, SUPPORT_POS_XY
    x_syn, u_eq_syn = standing_equilibrium()
    x_syn[0] += 0.02  # small offset
    r_foot_syn = np.array([SUPPORT_POS_XY[0], SUPPORT_POS_XY[1], 0.0])
    try:
        u_opt = mpc.solve(x_syn, r_foot_syn)
        print(f"  x_current (synthetic): {x_syn[:6].round(4)}")
        print(f"  u_0 (GRF): {u_opt.round(2)} N")
    except Exception as e2:
        print(f"  solve() also failed: {e2}")

# ── 5. State estimator role ───────────────────────────────────────────────────
print("\n[StateEstimator: how x_current is built from MuJoCo]")
print("""
  state_estimator.py.get_state() returns x = [c, c_dot, L]:

    c      = data.subtree_com[pelvis_id]     ← CoM position  (step11 mujoco)
    c_dot  = data.subtree_linvel[pelvis_id]  ← CoM velocity  (step11 mujoco)
    L      = data.subtree_angmom[pelvis_id]  ← angular mom.  (step11 mujoco)

  data.subtree_linvel/angmom require mj_subtreeVel() to be called first!
  (robot_model.py calls mj_subtreeVel in update())
""")

# ── 6. WBC chain ─────────────────────────────────────────────────────────────
print("[WBC chain: MPC output → joint torques]")
print("""
  mpc.solve()  → u_0 = [Fx, Fy, Fz]  (desired GRF)
       ↓
  wbc_v2.solve(u_0, x_current)
    → WBC QP: find joint torques τ that realize the GRF
    → Minimizes: |J^T τ - f_desired|² + |τ|²
    → Constraints: torque limits, joint limits
       ↓
  data.ctrl[:] = τ   (MuJoCo actuator input)
       ↓
  mj_step() → physics evolves → new state
""")

# ── 7. Stand_unified.py loop ─────────────────────────────────────────────────
print("[stand_unified.py control loop (simplified)]")
print("""
  while True:
    robot.update()                          # mj_step × 1
    if t % MPC_INTERVAL == 0:
      x_curr = state_estimator.get_state()  # [c, c_dot, L]
      r_foot = robot.get_body_pos(...)      # foot position
      u_grf  = mpc.solve(x_curr, r_foot)   # centroidal MPC
      tau    = wbc.solve(u_grf, x_curr)    # whole-body control
      data.ctrl[:] = tau                    # apply torques
""")

# ── 8. Common debugging entry points ─────────────────────────────────────────
print("[Debugging entry points in the real project]")
print("  Problem: robot falls → check in order:")
print()
print("  1. StateEstimator.get_state()  → is CoM/velocity/L_ang correct?")
print("     → Print x_current every step, compare to manual xpos[pelvis_id]")
print()
print("  2. CentroidalMPC.solve()       → is QP solved? status == 'solved'?")
print("     → Add: print(result.info.status) in mpc.py")
print()
print("  3. mpc output u_0              → does Fz ≈ mg? friction cone OK?")
print("     → Print u_0, check |u_0[0]|/u_0[2] ≤ μ")
print()
print("  4. WBC output tau              → are torques within limits?")
print("     → Print tau, compare to model.actuator_forcerange")
print()
print("  5. data.ctrl                   → is ctrl actually being set?")
print("     → Print data.ctrl before mj_step()")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters:")
print("   The full pipeline is: MuJoCo → StateEstimator → MPC → WBC → MuJoCo.")
print("   Knowing each interface helps you insert print statements at the right")
print("   place to debug instability without guessing.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. Print every solve status in stand_unified.py:
# In stand_unified.py, after mpc.solve(), add:
#   print(f"MPC status: {mpc._last_status}, u={u_grf}")
# Run: uv run python stand_unified.py  and observe the first 10 solves.

# 2. Inspect the OSQP result inside mpc.py:
# In CentroidalMPC.solve(), add:
#   print(f"  iter={result.info.iter}, obj={result.info.obj_val:.3f}")
# Expected: few iterations near equilibrium, more during perturbation recovery.

# 3. Check how mpc.py handles a failed solve (non-'solved' status):
# Look for the fallback in mpc.py:
# Likely returns u_eq or the previous solution.
""")
