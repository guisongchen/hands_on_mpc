# MPC Learning Plan — Single-Leg Balance Robot

> **Goal:** Learn Model Predictive Control (MPC) from first principles,
> grounded in the centroidal MPC used in this project (`mpc.py`).
> After completing this plan you will be able to read, debug, and tune
> the MPC in `stand_unified.py`.

---

## Background

The robot uses a **centroidal MPC** to decide the ground reaction force (GRF)
that keeps the center of mass (CoM) balanced over the support foot.
Every 50 ms the MPC solves a quadratic program (QP) and passes the optimal
GRF to the whole-body controller (WBC), which converts it to joint torques.

**Stack:** `StateEstimator` → `CentroidalMPC` → `WBC` → `mj_step()`

---

## Steps and Practice Files

| Step | Topic | Practice file |
|------|-------|---------------|
| 01 | MPC concept: predict → optimize → apply | `hands_on_mpc/step01_mpc_concept.py` |
| 02 | State and control vectors | `hands_on_mpc/step02_state_control.py` |
| 03 | Centroidal dynamics: A_c, B_c, d_c | `hands_on_mpc/step03_centroidal_dynamics.py` |
| 04 | Discretization: Euler vs ZOH | `hands_on_mpc/step04_discretization.py` |
| 05 | Cost function: Q, R, QN | `hands_on_mpc/step05_cost_function.py` |
| 06 | Decision variable z | `hands_on_mpc/step06_decision_variable.py` |
| 07 | Equality constraints | `hands_on_mpc/step07_equality_constraints.py` |
| 08 | Friction cone (inequality constraints) | `hands_on_mpc/step08_friction_cone.py` |
| 09 | OSQP solver: setup, solve, debug | `hands_on_mpc/step09_osqp_solver.py` |
| 10 | Receding horizon: closed-loop MPC | `hands_on_mpc/step10_receding_horizon.py` |
| 11 | Project integration: mpc.py → WBC | `hands_on_mpc/step11_project_integration.py` |
| 12 | Tuning: N, Q/R, T_S, diagnostics | `hands_on_mpc/step12_tuning.py` |

Run any step from `single_leg_stand/`:
```bash
uv run python hands_on_mpc/step01_mpc_concept.py
```

---

## Step Details

### Step 01 — MPC concept: predict → optimize → apply

**Toy 1D example** (no robot, just math):
- A mass-spring system controlled by a force
- Open-loop: apply a fixed force — system drifts
- Closed-loop MPC: solve QP at each step, apply only the first optimal control
- Key takeaway: MPC combines prediction + optimization + feedback

**Key concepts:** horizon N, prediction model, cost function, receding horizon

---

### Step 02 — State and control vectors

**Physical meaning of x and u:**
```
x = [c_x, c_y, c_z,   v_x, v_y, v_z,   L_x, L_y, L_z]
     ←─── CoM pos ──→  ←── CoM vel ──→  ←── angular mom ──→

u = [F_x, F_y, F_z]   (ground reaction force at foot)
```
- `c` = CoM position (meters)
- `v` = CoM velocity (m/s)
- `L` = centroidal angular momentum (kg·m²/s)
- `F` = GRF from foot to robot (Newtons)

At equilibrium: `u_eq = [0, 0, m*g]` — foot pushes up with full body weight.

---

### Step 03 — Centroidal dynamics

**Newton-Euler at the CoM:**
```
m * c̈ = Σ f_i + m*g
L̇     = Σ (r_i - c) × f_i
```

Linearized continuous-time system:
```
ẋ = A_c * x + B_c * u + d_c
```
- `A_c[0:3, 3:6] = I`  — velocity integrates position
- `B_c[3:6, :]   = I/m` — force → CoM acceleration
- `B_c[6:9, :]   = skew(r_lever)` — torque from force × lever arm
- `d_c[3:6]      = [0, 0, -g]` — gravity affine term

**Lever arm:** `r_lever = r_foot - c_eq` — must be updated when foot moves.

---

### Step 04 — Discretization

Convert continuous `ẋ = A_c x + B_c u + d_c` to discrete `x_{k+1} = A_d x_k + B_d u_k + d_d`:

| Method | Formula | Used in project |
|--------|---------|-----------------|
| Euler | `A_d = I + A_c*dt`, `B_d = B_c*dt` | ✓ mpc.py |
| ZOH   | `A_d = expm(A_c*dt)` | more accurate |

`T_S = 0.05 s` (20 Hz) → 10-step horizon = 0.5 s lookahead.

---

### Step 05 — Cost function

```
J = Σ_{k=0}^{N-1} [ (x_k - x_ref)ᵀ Q (x_k - x_ref) + (u_k - u_eq)ᵀ R (u_k - u_eq) ]
  + (x_N - x_ref)ᵀ QN (x_N - x_ref)
```

| Matrix | Purpose | Typical values |
|--------|---------|----------------|
| `Q`    | State tracking | `diag([1e3, 1e3, 1e3, 1e2, 1e2, 1e2, 1, 1, 1])` |
| `R`    | Control effort | `diag([1e-4, 1e-4, 1e-4])` |
| `QN`   | Terminal state | `10 × Q` |

**Unit scaling:** Q weights must account for units. Position errors (m) are
small; angular momentum errors (kg·m²/s) are larger. Use `Q[i] ∝ 1/σ_i²`.

---

### Step 06 — Decision variable z

The QP optimization variable stacks all states and controls:
```
z = [ x_0, x_1, ..., x_N,   u_0, u_1, ..., u_{N-1} ]
     |←─── (N+1)*NX ────→|  |←─── N*NU ───→|
```

Total size: `NZ = (N+1)*NX + N*NU = 11×9 + 10×3 = 129`

Index helpers:
```python
x_k = z[k*NX : (k+1)*NX]
u_k = z[(N+1)*NX + k*NU : (N+1)*NX + (k+1)*NU]
```

---

### Step 07 — Equality constraints

Force z to satisfy dynamics + initial state:
```
A_eq * z = b_eq
```

Block structure:
- Row block 0:   `I * x_0 = x_current`
- Row block k+1: `-A_d * x_k - B_d * u_k + I * x_{k+1} = d_d`

Shape: `((N+1)*NX, NZ)` — very sparse (most entries zero).

---

### Step 08 — Friction cone

A foot on the ground can only push, not pull.

**Pyramid approximation (6 linear constraints per step):**
```
F_z ≥ 0
-μ F_z ≤ F_x ≤ μ F_z
-μ F_z ≤ F_y ≤ μ F_z
F_z ≤ F_z_max
```

`μ = 0.6` (rubber on floor), `F_z_max = 500 N`.

Stacked over N steps → block-diagonal `A_ineq` of shape `(6N, NZ)`.

---

### Step 09 — OSQP solver

OSQP solves the combined QP:
```
min  0.5 zᵀ P z + qᵀ z
s.t. l ≤ A z ≤ u
```
where A stacks `[A_eq; A_ineq]` with matching `l` and `u`.

**Key calls:**
```python
solver = osqp.OSQP()
solver.setup(P, q, A, l, u, warm_starting=True, verbose=False)
result = solver.solve()
z_opt  = result.x                    # optimal decision variable
status = result.info.status          # 'solved', 'infeasible', etc.
u_opt  = z_opt[(N+1)*NX : (N+1)*NX + NU]  # first control
```

**Common failure modes:**
- `infeasible` → F_z_max too small, or initial state physically impossible
- `max_iter_reached` → increase max_iter or loosen eps_abs/eps_rel
- garbage u → check column mapping in A_ineq

---

### Step 10 — Receding horizon

Closed-loop MPC loop:
```
for t in range(T_sim):
    result = solver.solve()           # plan N steps
    u_t    = z_opt[u_slice(0)]        # extract first control
    x_next = A_d @ x_t + B_d @ u_t + d_d  # apply to system
    update equality constraints with x_next
    solver.update(l=..., u=...)       # warm-start for next step
```

Key insight: only 1 step is executed per solve, then the window slides.
This feedback corrects model errors and disturbances.

---

### Step 11 — Project integration

Full pipeline in `stand_unified.py`:
```
mj_step()  →  StateEstimator.get_state()  →  CentroidalMPC.solve()
           →  WBC.solve()  →  data.ctrl[:]  →  mj_step()
```

**StateEstimator** reads from MuJoCo:
- `c`   = `data.subtree_com[pelvis_id]`
- `ṡ`   = `data.subtree_linvel[pelvis_id]`  (requires `mj_subtreeVel()`)
- `L`   = `data.subtree_angmom[pelvis_id]`

**Debugging entry points:**
1. Print `x_current` — is CoM/velocity sensible?
2. Print `result.info.status` — is MPC solving?
3. Print `u_0` — is Fz ≈ mg? Is |Fx|/Fz ≤ μ?
4. Print `tau` — are torques within limits?

---

### Step 12 — Tuning

| Parameter | Effect | Starting value |
|-----------|--------|----------------|
| `N`       | Larger → better planning, slower solve | 10 |
| `T_S`     | Smaller → more accurate, less lookahead | 0.05 s |
| `Q_pos`   | Larger → tighter CoM tracking | 1e3 |
| `Q_vel`   | Larger → penalize CoM drift velocity | 1e2 |
| `R`       | Smaller → allow larger forces | 1e-4 |
| `QN`      | Larger → must converge at horizon end | 10×Q |
| `μ`       | Match real surface friction | 0.6 |

**Tuning workflow:**
1. Run simulation with default values
2. Observe convergence, oscillation, or divergence
3. Adjust one parameter at a time
4. Re-run and compare

---

## Learning Order

```
01 → 02 → 03 → 04      (understand the system and model)
             ↓
          05 → 06 → 07 → 08  (build the QP)
                           ↓
                        09 → 10  (solve and close the loop)
                                  ↓
                               11 → 12  (integrate and tune)
```

## Key Equations Summary

```
Continuous dynamics:   ẋ = A_c x + B_c u + d_c
Discrete dynamics:     x_{k+1} = A_d x_k + B_d u_k + d_d
Cost:                  J = Σ (x-x_ref)ᵀQ(x-x_ref) + (u-u_eq)ᵀR(u-u_eq)
QP:                    min 0.5 zᵀPz + qᵀz  s.t. l ≤ Az ≤ u
Friction:              -μFz ≤ Fx ≤ μFz,  -μFz ≤ Fy ≤ μFz,  0 ≤ Fz ≤ Fz_max
```
