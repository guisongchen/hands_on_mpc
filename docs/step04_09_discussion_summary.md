# Steps 04–09 Discussion Summary

This document covers concepts discussed across `step04_discretization.py`, `step05_cost_function.py` (with fixes), `step08_friction_cone.py`, `step09_osqp_solver.py`, and supporting utilities in `_mpc_utils.py`.

---

## 1. Centroidal Dynamics & State-Space Form

### Physical Equations

For a single rigid body (robot) with a contact foot:

- **CoM translation:** $\ddot{\mathbf{c}} = \frac{1}{m}\mathbf{u} + \mathbf{g}$
- **Angular momentum:** $\dot{\mathbf{L}} = (\mathbf{r}_{\text{foot}} - \mathbf{c}) \times \mathbf{u}$

State: $\mathbf{x} = [\mathbf{c}, \dot{\mathbf{c}}, \mathbf{L}]^\top \in \mathbb{R}^9$, Control: $\mathbf{u} = [f_x, f_y, f_z]^\top \in \mathbb{R}^3$

### Linearization (`make_centroidal_dynamics`)

The function builds $A_c, B_c, \mathbf{d}_c$ — the **continuous-time, linearized** state-space matrices.

$$\dot{\mathbf{x}} = A_c \mathbf{x} + B_c \mathbf{u} + \mathbf{d}_c$$

| Block | Equation | Matrix entry |
|-------|----------|--------------|
| $\dot{\mathbf{c}}$ | $\dot{\mathbf{c}} = \dot{\mathbf{c}}$ | $A_c[0:3,\,3:6] = I$ |
| $\ddot{\mathbf{c}}$ | $\ddot{\mathbf{c}} = \mathbf{u}/m + \mathbf{g}$ | $B_c[3:6,\,:] = I/m$, $\mathbf{d}_c[3:6] = \mathbf{g}$ |
| $\dot{\mathbf{L}}$ | $\dot{\mathbf{L}} = \mathbf{r}_{\text{eq}} \times \mathbf{u}$ | $B_c[6:9,\,:] = \text{skew}(\mathbf{r}_{\text{eq}})$ |

**Key point:** `make_centroidal_dynamics` does **only linearization**. It approximates the nonlinear lever arm $(\mathbf{r}_{\text{foot}} - \mathbf{c})$ as a constant $\mathbf{r}_{\text{eq}}$ evaluated at equilibrium.

---

## 2. Discretization

### Euler (`discretize_euler`)

Assumes derivative is constant over $\Delta t$:

$$A_d = I + A_c \Delta t, \quad B_d = B_c \Delta t, \quad \mathbf{d}_d = \mathbf{d}_c \Delta t$$

This is a first-order approximation. It is simple and fast, but less accurate for large $\Delta t$ or stiff dynamics.

### Zero-Order Hold (ZOH)

Assumes input $\mathbf{u}$ is **piecewise constant** over $\Delta t$, then computes the **exact** solution of the linear ODE using the matrix exponential:

$$A_d = e^{A_c \Delta t}, \quad B_d = \int_0^{\Delta t} e^{A_c \tau} B_c \, d\tau$$

**Implementation trick:** Augment $[A_c, B_c; 0, 0]$ and take $\exp(\cdot \cdot \Delta t)$. The top-left block is $A_d$, top-right is $B_d$.

**When it matters:** For the centroidal model, $A_c^2 \approx 0$, so ZOH $\approx$ Euler. The difference becomes significant for systems with springs, rotations, or large $\Delta t$.

---

## 3. Eigenvalue Sanity Check

For the centroidal model, $A_d$ should have eigenvalues on the **unit circle** ($|\lambda| = 1$).

**Why:** The unforced dynamics ($\mathbf{u}=0$) are pure integrators — momentum is conserved and position drifts linearly. There is no natural damping or instability. Eigenvalues $>1$ would mean artificial blow-up; $<1$ would mean artificial damping (both are discretization bugs).

---

## 4. Cost Function Design ($Q$, $R$, $Q_N$)

### Roles

| Matrix | Penalizes | Effect |
|--------|-----------|--------|
| $Q$ | State deviation during horizon | Tracking accuracy |
| $Q_N$ | Terminal state deviation | Stability / goal-seeking at horizon end |
| $R$ | Control effort deviation | Smoothness / fuel efficiency |

### Why Not Just $Q \gg R$?

Making $Q$ huge and $R$ tiny produces an **aggressive** controller that demands instantaneous tracking. Problems:
- **Actuator saturation:** Real motors have force limits; the QP plans for fantasy forces.
- **Overshoot & oscillation:** Like a PID with infinite gain.
- **Noise amplification:** Tiny state-estimate noise becomes violent force spikes.
- **Model-mismatch instability:** High gain magnifies errors in $A_d, B_d$.

**Resolution:** Use **hard constraints** (friction cone, torque limits) to enforce physical limits, and tune $Q/R$ to set the desired *trade-off* within the feasible region.

### Unit Scaling

State variables have different units (meters, m/s, kg$\cdot$m$^2$/s). A naive identity $Q$ would let angular momentum errors dominate. Heuristic:

$$Q_i \propto \frac{1}{\sigma_i^2}, \quad R_j \propto \frac{1}{\sigma_j^2}$$

where $\sigma$ is the acceptable error for each variable. **Important:** state weights go in $Q$; control weights go in $R$.

### Cost Function Expansion in QP

The tracking cost $\frac{1}{2}(z - z_{\text{ref}})^\top P (z - z_{\text{ref}})$ expands to:

$$\frac{1}{2}z^\top P z + \underbrace{(-P z_{\text{ref}})^\top}_{q^\top} z + \text{const}$$

- $P$ is passed **unchanged** to the solver.
- $q = -P z_{\text{ref}}$ shifts the minimum to $z_{\text{ref}}$.
- The **negative sign** is essential: without it, the optimum would sit at $z = -z_{\text{ref}}$.

---

## 5. Friction Cone Constraints

### Physical Meaning

A foot on the ground can only **push**, not pull:

$$f_z \geq 0, \qquad |f_x|, |f_y| \leq \mu f_z$$

### Pyramid Approximation

Linearized as 4 inequality facets + normal force bounds:

```
 fx - μ·fz ≤ 0
-fx - μ·fz ≤ 0
 fy - μ·fz ≤ 0
-fy - μ·fz ≤ 0
 0 ≤ fz ≤ fz_max
```

### Implementation in OSQP

One-sided inequalities use $-\infty$ as the inactive lower bound:

$$-\infty \leq f_x - \mu f_z \leq 0$$

This is safe — OSQP treats $\pm\infty$ as absent bounds, avoiding numerical issues.

### What About Angular Momentum?

The friction cone **directly constrains only force**. Angular momentum $\mathbf{L}$ is bounded **indirectly** via $\dot{\mathbf{L}} = \mathbf{r} \times \mathbf{u}$. Since $\mathbf{u}$ is capped, the torque is capped, so $\mathbf{L}$ can only change at a finite rate. If you need explicit limits on $\mathbf{L}$ itself, add separate state inequality constraints.

---

## 6. Building the QP for OSQP

### Decision Variable

$$z = [\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_N, \mathbf{u}_0, \dots, \mathbf{u}_{N-1}]^\top$$

### Equality Constraints (Dynamics)

Stacked uniformly for all $k = 0, \dots, N-1$:

$$\mathbf{x}_{k+1} - A_d \mathbf{x}_k - B_d \mathbf{u}_k = \mathbf{d}_d$$

In the constraint matrix, $-A_d$ and $-B_d$ appear because terms are moved to the left-hand side.

### Inequality Constraints (Friction Cone)

Stacked over the horizon: each $\mathbf{u}_k$ must satisfy the same friction cone.

### Constraint Order

The order of stacking equalities vs. inequalities **does not matter** to the solver, as long as rows in $A$, $l$, and $u$ remain aligned. Any permutation yields the same optimal $z^*$.

---

## 7. Post-Solve Sanity Checks

Even though the friction cone is baked into the QP, the code checks $\mathbf{u}_0$ after solving:

```python
ok = np.all(A_fcon @ u0 >= l_fcon - 1e-6)
```

**Why:**
- **Numerical tolerance:** Solvers satisfy constraints up to `eps_abs`/`eps_rel`.
- **Bug detection:** Catches silent matrix-building errors.
- **Solver failure:** Flags intermediate iterates from non-converged solves.
- **Safety:** Independent confirmation of critical physics.

---

## 8. Solver Failure & Fallback

If OSQP returns a non-`"solved"` status (e.g., `max_iter_reached`, infeasible), the application code falls back to equilibrium control:

```python
u0 = res.x[u_index] if res.info.status_val == 1 else u_eq
```

**Why $u_{\text{eq}}$:** It is the **safest failure mode**. It cancels gravity ($u_{\text{eq}} = mg$) so the robot doesn't collapse, without requiring a valid optimization result. Applying a failed solver's raw output could command dangerous or physically impossible forces.

**Important:** The fallback is **application-layer logic**, not built into OSQP. The solver knows nothing about robots or safety.

---

## 9. Warm-Starting

In receding-horizon MPC, consecutive QPs are very similar. The previous solution shifted by one step is an excellent initial guess.

**OSQP API:**

```python
solver.setup(..., warm_start=True)
result = solver.solve()
# ... next timestep ...
solver.warm_start(x=result.x, y=result.y)
result2 = solver.solve()  # fewer iterations
```

**Benefits:**
- Reduces iterations from 20–50 (cold) to 2–5 (warm).
- Essential for real-time control at 50–1000 Hz.
- Works with `solver.update()` for efficient bound changes between steps.

**Explicit shifting:** Often you `np.roll` the previous trajectory forward by one step and copy the last control to fill the end, then pass that as the warm-start seed.

---

*Document generated from discussion on Steps 04–09.*
