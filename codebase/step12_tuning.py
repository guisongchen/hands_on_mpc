"""Step 12 — Tuning the MPC: horizon, Q/R weights, and debugging.

MPC performance depends heavily on its parameters.
This step teaches you how each parameter affects balance quality,
and how to diagnose common issues systematically.

Parameters to tune:
  N         — horizon length (steps)
  T_S       — sampling period (s)
  Q         — state tracking weights
  R         — control effort weights
  QN        — terminal cost weights
  μ (MU)    — friction coefficient

Run from single_leg_stand/:
    uv run python hands_on_mpc/step12_tuning.py
"""

import numpy as np
import scipy.sparse as sp
import osqp

from _mpc_utils import (
    NX, NU, TOTAL_MASS, GRAVITY_MAG, T_S, N_HORIZON,
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

print("=" * 60)
print("STEP 12 — MPC tuning: horizon, Q/R/QN, diagnostics")
print("=" * 60)

# ── Closed-loop simulation helper ─────────────────────────────────────────────
def simulate_mpc(A_d, B_d, d_d, Q_m, R_m, QN_m, N_h, mu, fz_max,
                 x0, T_sim=30):
    """Run a closed-loop centroidal MPC simulation.

    Returns (history_x, solve_statuses).
    """
    NZ_h = (N_h + 1) * NX + N_h * NU

    def xs(k): return slice(k * NX, (k + 1) * NX)
    def us(k): return slice((N_h + 1) * NX + k * NU, (N_h + 1) * NX + (k + 1) * NU)

    # Cost
    P_d = np.zeros((NZ_h, NZ_h))
    for k in range(N_h):
        P_d[xs(k), xs(k)] = Q_m
        P_d[us(k), us(k)] = R_m
    P_d[xs(N_h), xs(N_h)] = QN_m
    P_sp = sp.csc_matrix(P_d)
    z_ref = np.concatenate([np.tile(x_eq, N_h + 1), np.tile(u_eq, N_h)])
    q_v   = -P_d @ z_ref

    # Friction cone
    A_fc, l_fc, u_fc = build_friction_cone(mu, fz_max)
    nf = len(l_fc)
    A_in = np.zeros((N_h * nf, NZ_h))
    l_in = np.zeros(N_h * nf)
    u_in = np.zeros(N_h * nf)
    for k in range(N_h):
        A_in[k*nf:(k+1)*nf, (N_h+1)*NX + k*NU:(N_h+1)*NX+(k+1)*NU] = A_fc
        l_in[k*nf:(k+1)*nf] = l_fc
        u_in[k*nf:(k+1)*nf] = u_fc

    def build_eq(xc):
        n_eq = (N_h + 1) * NX
        A_e  = np.zeros((n_eq, NZ_h))
        b_e  = np.zeros(n_eq)
        A_e[0:NX, xs(0)] = np.eye(NX)
        b_e[0:NX]        = xc
        for k in range(N_h):
            r = (k + 1) * NX
            A_e[r:r+NX, xs(k)]     = -A_d
            A_e[r:r+NX, us(k)]     = -B_d
            A_e[r:r+NX, xs(k+1)]   =  np.eye(NX)
            b_e[r:r+NX]            =  d_d
        return A_e, b_e

    A_e0, b_e0 = build_eq(x0)
    A_all = np.vstack([A_e0, A_in])
    l_all = np.concatenate([b_e0, l_in])
    u_all = np.concatenate([b_e0, u_in])
    sv = osqp.OSQP()
    sv.setup(P_sp, q_v, sp.csc_matrix(A_all), l_all, u_all,
             warm_starting=True, verbose=False, eps_abs=1e-5, eps_rel=1e-5)

    hist = [x0.copy()]
    statuses = []
    x_cur = x0.copy()
    for _ in range(T_sim):
        res = sv.solve()
        statuses.append(res.info.status)
        u_t = res.x[us(0)] if res.info.status == "solved" else u_eq.copy()
        x_cur = A_d @ x_cur + B_d @ u_t + d_d
        hist.append(x_cur.copy())
        _, b_new = build_eq(x_cur)
        l_new = np.concatenate([b_new, l_in])
        u_new = np.concatenate([b_new, u_in])
        sv.update(l=l_new, u=u_new)

    return np.array(hist), statuses


# Initial perturbation (same for all experiments)
x_pert = x_eq.copy()
x_pert[0] += 0.05  # 5cm forward CoM offset


# ── 1. Effect of horizon length N ────────────────────────────────────────────
print("\n[Effect of horizon length N]")
print("  Shorter N → less lookahead → slower convergence (or divergence).")
print("  Longer N  → more planning → better response, but slower solve.")
print()
print(f"  {'N':>4}  {'conv.(cm)':>10}  {'failures':>9}  {'note'}")

A_d0, B_d0, d_d0 = discretize_euler(A_c, B_c, d_c, T_S)
for N_test in [2, 5, N_HORIZON, 20]:
    hist, stats = simulate_mpc(A_d0, B_d0, d_d0, Q, R, QN, N_test,
                                FRICTION_MU, F_Z_MAX, x_pert, T_sim=30)
    final_err_cm = abs(hist[-1, 0] - x_eq[0]) * 100
    fails = sum(1 for s in stats if s != "solved")
    note = "too short" if N_test <= 3 else ("good" if N_test <= 15 else "long")
    print(f"  {N_test:4d}  {final_err_cm:10.2f}  {fails:9d}  {note}")


# ── 2. Effect of Q (state weight) ────────────────────────────────────────────
print("\n[Effect of Q — state tracking weight]")
print("  Large Q → aggressive tracking (fast convergence, large forces).")
print("  Small Q → lazy tracking (slow, smooth forces).")
print()
print(f"  {'Q_scale':>8}  {'conv.(cm)':>10}  {'max_Fz(N)':>10}")

hist_u = []  # store trajectories for force check
for q_scale in [0.01, 0.1, 1.0, 10.0]:
    Q_test = Q * q_scale
    hist, stats = simulate_mpc(A_d0, B_d0, d_d0, Q_test, R, QN, N_HORIZON,
                                FRICTION_MU, F_Z_MAX, x_pert, T_sim=30)
    err_cm = abs(hist[-1, 0] - x_eq[0]) * 100
    # Approximate max Fz: can't easily extract from pure rollout, show convergence
    print(f"  {q_scale:8.2f}  {err_cm:10.2f}  (forces not tracked in this simplified demo)")


# ── 3. Effect of R (control weight) ──────────────────────────────────────────
print("\n[Effect of R — control effort weight]")
print("  Large R → conservative forces → slow recovery.")
print("  Small R → aggressive forces → fast recovery but potentially noisy.")
print()
print(f"  {'R_scale':>8}  {'conv.(cm)':>10}")

for r_scale in [0.001, 0.01, 0.1, 1.0, 10.0]:
    R_test = R * r_scale
    hist, stats = simulate_mpc(A_d0, B_d0, d_d0, Q, R_test, QN, N_HORIZON,
                                FRICTION_MU, F_Z_MAX, x_pert, T_sim=30)
    err_cm = abs(hist[-1, 0] - x_eq[0]) * 100
    print(f"  {r_scale:8.3f}  {err_cm:10.2f}")


# ── 4. Effect of T_S (sampling period) ───────────────────────────────────────
print("\n[Effect of T_S — sampling period]")
print("  Small T_S → finer steps, more accurate → but N*T_S = shorter lookahead")
print("  Large T_S → longer lookahead per step → but discretization error grows")
print()
print(f"  {'T_S (s)':>8}  {'lookahead(s)':>13}  {'conv.(cm)':>10}")

for ts in [0.01, 0.02, 0.05, 0.1, 0.2]:
    A_dt, B_dt, d_dt = discretize_euler(A_c, B_c, d_c, ts)
    lookahead = ts * N_HORIZON
    # Need to simulate same real time (1.5s) → adjust T_sim steps
    T_sim_real = max(1, int(1.5 / ts))
    hist, stats = simulate_mpc(A_dt, B_dt, d_dt, Q, R, QN, N_HORIZON,
                                FRICTION_MU, F_Z_MAX, x_pert, T_sim=T_sim_real)
    err_cm = abs(hist[-1, 0] - x_eq[0]) * 100
    print(f"  {ts:8.3f}  {lookahead:13.2f}  {err_cm:10.2f}")


# ── 5. Tuning heuristics ─────────────────────────────────────────────────────
print("\n[Tuning heuristics for centroidal MPC in balance]")
print("""
  Start with:
    N   = 10          → 0.5s lookahead at T_S=0.05s
    T_S = 0.05s       → 20Hz MPC rate (fast enough for walking)
    Q_pos = 1e3       → 1mm CoM error costs 1 unit
    Q_vel = 1e2       → 10cm/s velocity error costs 1 unit
    Q_L   = 1e0       → 1 kg·m²/s angular momentum error costs 1 unit
    R     = 1e-4      → very small → MPC is willing to use large forces
    QN    = 10×Q      → strongly penalize terminal state error

  If robot falls forward:
    → Increase Q_pos (respond faster to CoM error)
    → Decrease R     (allow stronger corrective forces)
    → Increase N     (plan further ahead)

  If robot oscillates:
    → Decrease Q_pos or increase R
    → Check T_S — if too small, Euler error accumulates

  If solver fails (not 'solved'):
    → Check F_z_max — too small?
    → Check friction cone — μ too small for the required forces?
    → Increase max_iter in OSQP settings
""")


# ── 6. Diagnosis summary ─────────────────────────────────────────────────────
print("[Diagnosis flowchart]")
print("""
  Robot falls:
    1. Print status every step → is MPC solving successfully?
    2. Print u_0 → is Fz reasonable (≈ mg)? Is Fxy within μ*Fz?
    3. Print x_current → is StateEstimator giving sensible values?
    4. Print x_predicted vs x_actual → does model match reality?

  If model doesn't match:
    → Is gravity term d_c included? (x_dot = A_c*x + B_c*u + d_c)
    → Is lever arm r_lever correct? (r_foot - c_eq, updated each step)
    → Is T_S too large? (use ZOH discretization instead of Euler)

  If model matches but robot still falls:
    → Increase Q_pos or N (MPC knows but can't react fast enough)
    → Check WBC: is it actually tracking the desired GRF?
""")


# ── 💡 WHY THIS MATTERS FOR BALANCE ─────────────────────────────────────────
print("💡 Why this matters:")
print("   There is no single 'right' set of MPC parameters.")
print("   You must iterate: tune → simulate → observe → re-tune.")
print("   Understanding how each parameter affects behavior saves hours of guessing.")

# ── 🔬 EXPERIMENTS ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬 EXPERIMENTS")
print("─" * 60)
print("""
# 1. Find the minimum N that still achieves convergence to within 1cm:
# for N_test in range(2, 15):
#     hist, _ = simulate_mpc(..., N_h=N_test, T_sim=40)
#     err = abs(hist[-1, 0] - x_eq[0]) * 100
#     print(f"N={N_test}: err={err:.2f}cm")

# 2. What Q_scale causes divergence (CoM error grows)?
# for q_scale in [0.001, 0.0001]:
#     hist, _ = simulate_mpc(..., Q_test=Q*q_scale, T_sim=50)
#     print(f"Q_scale={q_scale}: final CoM={hist[-1,0]:.3f} m (x_eq={x_eq[0]:.3f})")

# 3. Simulate with a moving reference (track a forward step):
# x_ref_moving = x_eq.copy(); x_ref_moving[0] += 0.2  # step 20cm forward
# Modify simulate_mpc to accept x_ref instead of x_eq...
""")
