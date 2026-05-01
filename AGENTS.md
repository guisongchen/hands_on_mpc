# AGENTS.md — hands_on_mpc

Self-contained MPC learning tutorial (12 step-by-step Python scripts).

## Run steps

All steps run from the parent `single_leg_stand/` directory:
```bash
uv run python hands_on_mpc/step01_mpc_concept.py
```

Step 11 additionally imports `mpc`, `config`, `state_estimator` from the parent project — it will fall back to a synthetic demo if those aren't available.

## Structure

- `codebase/_mpc_utils.py` — shared utilities (dynamics, discretization, friction cone, 1D toy env)
- `codebase/step01_*.py` through `codebase/step12_*.py` — sequential lessons
- `codebase/mpc_learning_plan.md` — study guide with equations and parameter tables
- `docs/` — empty (no generated docs or artifacts)

## Dependencies

numpy, scipy, osqp. No project-level config (pyproject.toml, requirements.txt, etc.) exists in this repo — deps are managed by the parent project.

## What's missing

No tests, no linter, no formatter, no CI, no build step. Each script is standalone and prints its output to stdout. To verify changes, run the script and check printed diagnostics.
