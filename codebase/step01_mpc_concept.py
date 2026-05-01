"""Step 01 — What is MPC? Predict → Optimize → Apply.

Model Predictive Control answers one question every control loop:
"Given where I am now, what is the best action over the next N steps?"

The key idea: RECEDING HORIZON
  1. Predict the future N steps (using a model of the system)
  2. Optimize a cost function over those N steps to find best controls
  3. Apply ONLY the first control u_0
  4. Re-solve at the next timestep (with updated state)

Why not just solve once and apply everything?
  → Disturbances and model errors accumulate. Re-solving every step
     gives feedback (like closing the loop), so errors are corrected.

This step uses a TOY 1D point-mass (mass hovering against gravity)
to make the predict-optimize-apply cycle concrete and hand-checkable
before we deal with the full 9D centroidal robot model.

Run from single_leg_stand/:
    uv run python hands_on_mpc/step01_mpc_concept.py
"""

import numpy as np
import osqp
from scipy import sparse

from _mpc_utils import PointMass1D, rollout

print("=" * 60)
print("STEP 01 — MPC concept: predict → optimize → apply")
print("=" * 60)

env = PointMass1D(mass=1.0, dt=0.05)
N   = 3     # tiny horizon so you can read the numbers

# ── 1. The system model ──────────────────────────────────────────────────────
print("\n[The plant: 1D point-mass, state = [z, z_dot], control = [Fz]]")
print(f"  A_d = {env.A_d}  (identity + velocity integration)")
print(f"  B_d = {env.B_d.flatten()}  (force → acceleration)")
print(f"  d_d = {env.d_d}  (gravity affects velocity each step)")
print(f"  Equilibrium force u* = {env.equilibrium_u:.2f} N  (= mg, hover force)")

# ── 2. Open-loop prediction ──────────────────────────────────────────────────
print("\n[Open-loop prediction over N=3 steps]")
print("  Question: if we apply u=0 (no force) from z=1.0m, where do we end up?")

env.reset(z=1.0, z_dot=0.0)
x0 = env.state.copy()
u_zero = np.zeros((N, 1))
x_pred_zero = rollout(env.A_d, env.B_d, env.d_d, x0, u_zero)
print(f"\n  {'step':>5}  {'z (m)':>8}  {'z_dot (m/s)':>12}")
for k in range(N + 1):
    print(f"  {k:5d}  {x_pred_zero[k,0]:8.4f}  {x_pred_zero[k,1]:12.4f}")
print("  → Robot falls (z decreases). We need to optimize the forces.")

# ── 3. What "optimize" means: a tiny QP by hand ───────────────────────────────
print("\n[Optimization: find u_0,...,u_{N-1} that keep z close to 1.0m]")
print("  Cost: sum_k (z_k - 1.0)^2 + 0.001*(u_k - u*)^2  (minimize deviation + input)")
print("  This is a Quadratic Program (QP). Let's solve it with OSQP.")

# Build the tiny QP manually (N=3, nx=2, nu=1)
nx, nu = 2, 1
nz  = (N + 1) * nx + N * nu   # decision variable: [x0,x1,x2,x3, u0,u1,u2]
x_ref = np.array([1.0, 0.0])   # reference: hover at z=1.0m
u_ref = np.array([env.equilibrium_u])

# Cost matrix P (block diagonal Q, Q, Q, QN, R, R, R)
Q1  = np.diag([10.0, 0.1])
QN1 = np.diag([10.0, 0.1])
R1  = np.diag([0.001])
P_blocks = [Q1] * N + [QN1] + [R1] * N
P = sparse.block_diag(P_blocks, format="csc")

# Linear cost q = -P * z_ref (for quadratic tracking)
z_ref_full = np.zeros(nz)
for k in range(N):
    z_ref_full[k*nx:(k+1)*nx] = x_ref
z_ref_full[N*nx:(N+1)*nx] = x_ref
for k in range(N):
    z_ref_full[(N+1)*nx + k*nu : (N+1)*nx + (k+1)*nu] = u_ref
q_vec = -P @ z_ref_full

# Constraints: initial state + dynamics
n_init = nx
n_dyn  = N * nx
n_constr = n_init + n_dyn
A_con = np.zeros((n_constr, nz))

# Initial state: x0 = x0_hat
A_con[:nx, :nx] = np.eye(nx)

# Dynamics: x_{k+1} = A_d*x_k + B_d*u_k + d_d
for k in range(N):
    row = n_init + k * nx
    cx  = k * nx
    cxn = (k + 1) * nx
    cu  = (N + 1) * nx + k * nu
    A_con[row:row+nx, cxn:cxn+nx] = np.eye(nx)
    A_con[row:row+nx, cx:cx+nx]   = -env.A_d
    A_con[row:row+nx, cu:cu+nu]   = -env.B_d

l_con = np.zeros(n_constr)
u_con = np.zeros(n_constr)
l_con[:nx] = x0
u_con[:nx] = x0
for k in range(N):
    l_con[n_init + k*nx : n_init + (k+1)*nx] = env.d_d
    u_con[n_init + k*nx : n_init + (k+1)*nx] = env.d_d

solver = osqp.OSQP()
solver.setup(P=P, q=q_vec, A=sparse.csc_matrix(A_con),
             l=l_con, u=u_con, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
result = solver.solve()

print(f"\n  OSQP status: {result.info.status}  (should be 'solved')")
z_opt = result.x
x_opt = z_opt[:(N+1)*nx].reshape(N+1, nx)
u_opt = z_opt[(N+1)*nx:].reshape(N, nu)

print(f"\n  Optimal forces: {u_opt.flatten().round(3)} N")
print(f"  (Equilibrium hover force = {env.equilibrium_u:.2f} N)")
print(f"\n  Predicted trajectory with optimal forces:")
print(f"  {'step':>5}  {'z (m)':>8}  {'z_dot':>8}  {'u (N)':>8}")
for k in range(N):
    print(f"  {k:5d}  {x_opt[k,0]:8.4f}  {x_opt[k,1]:8.4f}  {u_opt[k,0]:8.3f}")
print(f"  {N:5d}  {x_opt[N,0]:8.4f}  {x_opt[N,1]:8.4f}  {'---':>8}")

# ── 4. Apply u_0 only — receding horizon ──────────────────────────────────────
print("\n[Receding horizon: apply u_0, discard u_1..u_{N-1}, re-solve next step]")
print("  This is the fundamental MPC trick: feedback via re-solving.")

env.reset(z=1.0, z_dot=0.0)
print(f"\n  {'t (s)':>6}  {'z (m)':>8}  {'z_dot':>8}  {'u_applied (N)':>14}")
for step in range(12):
    x_now = env.state.copy()
    l_con[:nx] = x_now
    u_con[:nx] = x_now
    solver.setup(P=P, q=q_vec, A=sparse.csc_matrix(A_con),
                 l=l_con, u=u_con, verbose=False)
    res = solver.solve()
    u0 = res.x[(N+1)*nx] if res.info.status_val == 1 else env.equilibrium_u
    env.step(u0)
    t = (step + 1) * env.dt
    print(f"  {t:6.2f}  {env.state[0]:8.4f}  {env.state[1]:8.4f}  {u0:14.3f}")
print("  → z stays near 1.0 m even though each step re-solves from scratch")

# ── 5. Open-loop vs closed-loop ───────────────────────────────────────────────
print("\n[Open-loop vs closed-loop under disturbance]")
print("  Disturbance: sudden z_dot = -0.5 m/s at t=0 (downward push)")

# Open-loop: plan once, apply all controls
env.reset(z=1.0, z_dot=-0.5)
x_pert = env.state.copy()
l_con[:nx] = x_pert; u_con[:nx] = x_pert
solver.setup(P=P, q=q_vec, A=sparse.csc_matrix(A_con), l=l_con, u=u_con, verbose=False)
res_ol = solver.solve()
u_plan = res_ol.x[(N+1)*nx:].reshape(N, 1)

env.reset(z=1.0, z_dot=-0.5)
z_openloop  = [env.state[0]]
z_closedloop= [env.state[0]]
env2 = PointMass1D(mass=1.0, dt=0.05)
env2.reset(z=1.0, z_dot=-0.5)

for k in range(N):
    env.step(float(u_plan[k, 0]))   # open-loop: pre-planned
    z_openloop.append(env.state[0])

    # Closed-loop: re-solve each step
    x_cl = env2.state.copy()
    l_con[:nx] = x_cl; u_con[:nx] = x_cl
    solver.setup(P=P, q=q_vec, A=sparse.csc_matrix(A_con), l=l_con, u=u_con, verbose=False)
    res_cl = solver.solve()
    u_cl0 = res_cl.x[(N+1)*nx] if res_cl.info.status_val == 1 else env2.equilibrium_u
    env2.step(u_cl0)
    z_closedloop.append(env2.state[0])

print(f"  {'step':>5}  {'z open-loop':>12}  {'z closed-loop':>14}")
for k in range(N + 1):
    print(f"  {k:5d}  {z_openloop[k]:12.4f}  {z_closedloop[k]:14.4f}")
print("  → Closed-loop stays closer to 1.0m than open-loop after disturbance")

# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("\n💡 Why this matters for single-leg balance:")
print("   mpc.py runs this exact loop every 50ms: given current CoM state,")
print("   solve for best contact force over N=10 steps, apply only the first.")
print("   Without receding horizon, a single unexpected push would cause a fall.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. What happens with horizon N=1 vs N=10?
#    Shorter horizon = more myopic decisions.
#    Try N=1 and repeat the closed-loop simulation above.
#    Expected: N=1 may oscillate more; N=10 is smoother.

# 2. Increase the disturbance: z_dot = -2.0 m/s
#    Does closed-loop still recover? Does open-loop fail completely?
#    Expected: closed-loop recovers; open-loop diverges.

# 3. What if the model is wrong? Use A_d with 1.5x wrong mass in the optimizer
#    but correct mass in the environment.
#    Expected: closed-loop still converges (feedback corrects model error);
#              open-loop diverges (no correction mechanism).
""")
