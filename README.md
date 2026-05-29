# Agentic-VLA

`Agentic-VLA` is a research codebase for long-horizon embodied manipulation. It studies how a strong frozen VLA policy can be improved at inference time with lightweight process-control modules, rather than by replacing the action model itself.

The current system keeps `pi0.5 / pi05_libero` as the low-level action expert and augments it with an agentic execution layer built around transition repair, structured scene priors and memory, critic-based auditing, and planner-assisted control.

## Framework Overview

- **Low-level executor**: `pi0.5 / OpenPI`
- **High-level planner**: `Qwen3-VL-8B-Instruct`
- **Planner mode**: `on-demand`
- **Control target**: `10Hz`
- **Planner output**: structured JSON

The current agentic stack includes:

- `failure taxonomy`: `stall / collision / misgrasp / slip`
- `subgoal + FSM + verifier`
- `cooldown / cache / max_plans_per_episode`
- `Transition / Critic / GraphRAG-Memory`
- an explicit `Router + Expert + Verifier` skeleton for `Agent-level MoE v1`

## Experimental Results

All numbers below come from real interactive rollouts in the official `LIBERO` simulator.

| Setting | Suite | Success Rate | Episodes | Notes |
|---|---|---:|---:|---|
| `pi05_libero` baseline | `libero_spatial` | `99.0%` | `198/200` | Strong short-horizon baseline |
| `pi05_libero` baseline | `libero_object` | `98.0%` | `196/200` |  |
| `pi05_libero` baseline | `libero_goal` | `98.0%` | `196/200` |  |
| `pi05_libero` baseline | `libero_10` | `90.0%` | `180/200` | `Task 8 = 55%` |
| `Full-Agentic-VLA-Refined` | `libero_10` | `92.5%` | `185/200` | `Task 8: 55% -> 75%` |

In the current `libero_10` study, the most consistent gains come from transition-aware process control. `GraphRAG + Memory` helps on part of the difficult cases, while the current `Critic` behaves more like an auditing component than a fully mature recovery controller.

## Repository Structure

- `scripts/run_agentic_vla_libero.py`: main LIBERO rollout entrypoint
- `openpi/scripts/serve_policy.py`: policy server
- `openpi/src/openpi/policies/agentic_policy.py`: server-side agentic planner protocol
- `agentic_vla/`: earlier prototype package and reusable agentic modules
- `AGENT_LEVEL_MOE_PLAN.md`: current `Router + Expert + Verifier` design notes
- `EXPERIMENT_LOG.md`: chronological experiment log
- `RESULTS_EVIDENCE_GUIDE.md`: result directory map and evidence boundary

## Getting Started

### 1. Prepare the OpenPI / LIBERO environment

Please refer to:

- `openpi/README.md`
- `openpi/examples/libero/README.md`

### 2. Start the policy server

```bash
PYTHONPATH=/path/to/openpi/src python openpi/scripts/serve_policy.py --env LIBERO --port 8000
```

To enable the agentic planner:

```bash
PYTHONPATH=/path/to/openpi/src python openpi/scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  --agentic \
  --planner-model /path/to/Qwen3-VL-8B-Instruct
```

### 3. Run LIBERO evaluation

```bash
python scripts/run_agentic_vla_libero.py \
  --task-suite libero_10 \
  --trials 20 \
  --host 127.0.0.1 \
  --port 8000 \
  --transition \
  --graph-rag \
  --critic \
  --agentic-planner
```

## Current Focus

This repository is currently centered on:

- `LIBERO`, especially `libero_10`
- strong `pi0.5 / pi05_libero` baselines
- inference-time agentic augmentation for long-horizon execution
- deployment-aware design under constrained compute budgets

## Notes

- The repository keeps both the main evaluation pipeline and a historical prototype package for method development.
- We do not track local weights, result dumps, or runtime caches by default.
- For reproducible evaluation, prefer the real rollout pipeline under `scripts/` and `openpi/`.
