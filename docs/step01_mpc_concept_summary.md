# Step 01 MPC Concept — Discussion Summary

This document summarizes the key concepts discussed around `codebase/step01_mpc_concept.py`, which implements a toy 1D point-mass MPC controller to illustrate the predict-optimize-apply cycle.

---

## 1. System Model (Discrete-Time Point Mass)

The plant is a 1D point mass with dynamics:

$$\ddot{z} = \frac{u}{m} - g$$

State: $x = [z, \dot{z}]^\top$, Control: $u = [F_z]$

**Discrete-time matrices (Euler integration):**

| Matrix | Value | Physical Meaning |
|--------|-------|------------------|
| $A_d$ | $\begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}$ | State transition. Position drifts by $v \cdot \Delta t$; velocity persists. |
| $B_d$ | $\begin{bmatrix} 0 \\ \Delta t / m \end{bmatrix}$ | Control effect. Force $u$ changes velocity by $(u/m) \cdot \Delta t$ per step. |
| $d_d$ | $\begin{bmatrix} 0 \\ -g \Delta t \end{bmatrix}$ | Gravity drift. Velocity decreases by $g \Delta t$ each step regardless of input. |

---

## 2. Cost Matrices

The MPC objective penalizes deviations from a reference trajectory. The cost matrices are block-diagonal:

```python
P_blocks = [Q] * N + [QN] + [R] * N
```

| Matrix | Role | In This Code |
|--------|------|--------------|
| $Q$ | Stage state cost (penalizes $x_k$ deviation during horizon) | `diag([10.0, 0.1])` — heavily penalize position error, lightly penalize velocity |
| $Q_N$ | Terminal state cost (penalizes $x_N$ at horizon end) | Same as $Q$. Helps with stability. |
| $R$ | Control cost (penalizes large inputs) | `diag([0.001])` — cheap to use thrust; controller prioritizes tracking over efficiency |

The relative magnitudes encode the controller's preferences: $Q \gg R$ means "get to the target accurately, even if it costs force."

---

## 3. Decision Variable Structure

The QP decision vector stacks all states and controls over the horizon:

$$z = \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_N \\ u_0 \\ u_1 \\ \vdots \\ u_{N-1} \end{bmatrix} \in \mathbb{R}^{(N+1)n_x + N n_u}$$

**Why include $x_0$?** Even though $x_0$ is fixed by the initial state measurement, including it in $z$ allows all $N$ dynamics constraints to use the **same uniform structure**:

$$x_{k+1} - A_d x_k - B_d u_k = d_d$$

Without $x_0$ in $z$, the first dynamics equation would need a special case.

---

## 4. QP Formulation

The MPC problem is cast as a standard QP:

$$\min_z \quad \frac{1}{2} z^\top P z + q^\top z \qquad \text{s.t.} \qquad l \leq A_{\text{con}} z \leq u$$

Since $l = u$ everywhere, this is effectively an **equality-constrained QP**.

### Cost Terms

- **Quadratic:** $P = \text{blkdiag}(Q, \dots, Q, Q_N, R, \dots, R)$
- **Linear:** $q = -P z_{\text{ref}}$

The linear term comes from expanding the tracking objective:

$$\frac{1}{2}(z - z_{\text{ref}})^\top P (z - z_{\text{ref}}) = \underbrace{\frac{1}{2} z^\top P z}_{\text{quadratic}} + \underbrace{(-P z_{\text{ref}})^\top z}_{\text{linear}} + \underbrace{\text{constant}}_{\text{dropped}}$$

Constant terms do not affect the optimal solution, so they are omitted.

### Constraint Matrix $A_{\text{con}}$

$$A_{\text{con}} = \left[\begin{array}{cccccc|ccccc} I & 0 & \cdots & 0 & 0 & 0 & 0 & 0 & \cdots & 0 \\ \hline -A_d & I & \cdots & 0 & 0 & 0 & -B_d & 0 & \cdots & 0 \\ 0 & -A_d & \ddots & 0 & 0 & 0 & 0 & -B_d & \cdots & 0 \\ \vdots & \vdots & \ddots & \ddots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & -A_d & I & 0 & 0 & 0 & \cdots & -B_d \end{array}\right]$$

- **Row 1** ($n_x$ rows): Initial state constraint $x_0 = x_{\text{meas}}$
- **Rows 2 to $N+1$** ($N \cdot n_x$ rows): Dynamics constraints

### Bounds $l = u$

$$l = u = \begin{bmatrix} x_{0,\text{meas}} \\ d_d \\ d_d \\ \vdots \\ d_d \end{bmatrix}$$

**Why $d_d$ and not 0?** Because the dynamics are affine: $x_{k+1} = A_d x_k + B_d u_k + d_d$. Rearranging gives $x_{k+1} - A_d x_k - B_d u_k = d_d$. If there were no gravity ($d_d = 0$), the bounds would be zero.

---

## 5. Receding Horizon Principle

After solving the QP, the controller obtains an optimal sequence $(u_0^*, u_1^*, \dots, u_{N-1}^*)$. However, it **only applies $u_0^*$** and discards the rest. At the next timestep, it:

1. Re-measures the current state $x_{\text{meas}}$
2. Updates the initial constraint $l[:n_x] = u[:n_x] = x_{\text{meas}}$
3. Re-solves the QP from scratch
4. Applies the new $u_0^*$

This is the **fundamental MPC trick**: feedback via re-solving.

### Why It Works

The optimizer is **stateless across timesteps**. It does not remember the previous plan. It only asks: *"Given where I am right now, what's the best action?"* This means:

- **Disturbances are corrected automatically.** A push off course becomes the new initial state for the next QP.
- **Model errors are tolerated.** Even with imperfect $A_d, B_d$, re-solving from the true state prevents drift accumulation.
- **Reference changes are handled for free.** A moving target just changes $z_{\text{ref}}$; no extra logic needed.

### The Price

Re-solving a QP every timestep is computationally expensive compared to evaluating a fixed feedback law (e.g., $u = -Kx$). Mitigation strategies include:

- **Short horizons** ($N = 10$–$50$)
- **Warm-starting** (seed the solver with the previous solution)
- **Fast QP solvers** (OSQP, qpOASES)
- **Explicit MPC** (pre-compute control laws offline for small problems)

---

## 6. Open-Loop vs. Closed-Loop Experiment

The code compares two strategies under a sudden disturbance ($\dot{z} = -0.5$ m/s push at $t=0$):

| Strategy | Behavior |
|----------|----------|
| **Open-loop** | Plan once at $t=0$, then blindly apply $u_0, u_1, u_2$ without looking at the state again. Errors accumulate; the mass drifts from the target. |
| **Closed-loop (MPC)** | Re-solve at every step using the **fresh current state** as the initial condition. The controller sees the drift and corrects. |

**Key insight:** The closed-loop side is exactly the receding horizon principle in action. The open-loop side shows what happens without feedback. The experiment demonstrates that **the act of re-optimizing from the true state is what provides robustness**, not the initial plan itself.

---

## 7. Implementation Notes

### Why For-Loops for Building $z_{\text{ref}}$?

The code uses explicit loops to fill $z_{\text{ref}}$ for clarity. In production, this can be vectorized:

```python
z_ref_full = np.concatenate([
    np.tile(x_ref, N + 1),
    np.tile(u_ref, N)
])
```

For $N=3$ and $n_x=2$, the loop overhead is negligible; readability is preferred in a tutorial.

### Why $x_0$'s Reference Value Is Irrelevant

In the code, $x_0$'s reference is set to $x_{\text{ref}}$ like the other states. However, because $x_0$ is **pinned by the equality constraint** $x_0 = x_{\text{meas}}$, the optimizer cannot change it. Any cost associated with $x_0$ is just a constant offset and does not affect the optimal control sequence. Setting it to $x_{\text{ref}}$, $0$, or any other value yields the same $u_0^*, \dots, u_{N-1}^*$.

---

*Document generated from discussion on `step01_mpc_concept.py`.*
