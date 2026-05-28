# Refined LIBERO-10 Experiment Plan (2026-05-18)

## Goal

Upgrade the current `refined Full` evidence from targeted follow-up into a unified, journal-ready `libero_10` result line that is directly comparable to `pi05_libero` under the same evaluation protocol.

## Current Evidence

- Main audited result:
  - `results/ablation_FULL_tuned_libero10_20260513_225055/summary.json`
  - `91/100 = 91.0%`
- Strong baseline:
  - `results/ablation_B0_pi05_libero_10_20260512/summary.json`
  - `180/200 = 90.0%`
- Refined weak-task follow-up already completed:
  - `results/ablation_FULL_tuned_v2_task8_20260515_160827/summary.json`
  - `results/ablation_FULL_tuned_v2_task9_20260515_162215/summary.json`
  - `results/ablation_FULL_tuned_v2_task6_20260515_162215/summary.json`
- Refined non-weak tasks already completed:
  - `results/ablation_FULL_refined_libero10_skip689_20260517_132924/summary.json`

## Phase 1: Same-Protocol Weak-Task Confirmation

Purpose: confirm that refined `Full` remains stronger on `Task 6/8/9` when the trial count is increased from `10` to `20`.

Run:

- suite: `libero_10`
- tasks: `8 9 6`
- method: `Full-Agentic-VLA-Refined`
- flags: `--transition --graph-rag --critic`
- trials per task: `20`

Success criterion:

- `Task 8` stays above old main `40%` and ideally above baseline `55%`
- `Task 9` stays at or above old main `90%`
- `Task 6` stays at or above old main `80%`
- No obvious collapse in mechanism statistics or rollout quality

Stop rule:

- If `Task 8` falls back near the old main result, stop before full-suite expansion and re-audit prompt/control interaction.

## Phase 2: Unified Refined Full on LIBERO-10

Purpose: produce a single auditable `refined Full` main result line for the paper.

Run:

- suite: `libero_10`
- method: `Full-Agentic-VLA-Refined`
- flags: `--transition --graph-rag --critic`
- trials per task: `20`

Execution strategy:

- Preferred: start a fresh full-suite run in one directory.
- Fast sanity option: use Phase 1 weak-task confirmation plus existing `skip689` evidence only as an internal gating signal, not as the final paper number.

Success criterion:

- Overall `libero_10` score is at least competitive with `pi05_libero`
- Weak tasks improve without introducing broad regressions on previously stable tasks

## Phase 3: Optional No-Regression Check

Only run if Phase 2 is positive and wall-clock budget remains.

- small-scale `libero_object` or `libero_goal` check
- goal: show that refined prompts/control do not obviously damage short-horizon suites

## Priority

- Must run now:
  - Phase 1
- Run next if Phase 1 is stable:
  - Phase 2
- Optional:
  - Phase 3

## Command Notes

- Policy server must be running first:
  - `conda run -n openpi python /home/admin1/ct/Agentic-VLA/openpi/scripts/serve_policy.py --env LIBERO`
- Phase 1 can be launched with:
  - `STAMP=<stamp> TRIALS=20 TASK_IDS="8 9 6" VARIANTS="full" bash /home/admin1/ct/Agentic-VLA/scripts/run_low_success_ablations.sh`
- Phase 2 should use a fresh result directory to avoid mixing with prior `10`-trial runs.

## Output Convention

- Phase 1:
  - `results/ablation_FULL_tuned_v2_task8_<stamp>/`
  - `results/ablation_FULL_tuned_v2_task9_<stamp>/`
  - `results/ablation_FULL_tuned_v2_task6_<stamp>/`
- Phase 2:
  - `results/ablation_FULL_refined_libero10_<stamp>/`

## Reporting

After each phase, record:

- result directory
- summary JSON path
- task success rates
- transition / critic statistics
- whether the next phase should proceed
