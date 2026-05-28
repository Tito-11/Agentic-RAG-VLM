"""Real LIBERO evaluation entrypoint for Agentic-VLA ablation experiments.

Supported ablation modes (controlled via CLI flags):
  A1. Baseline:            pi0_base (no flags)
  A2. + Transition Agent:  --transition
  A3. + Graph RAG + Memory: --graph-rag
  A4. + Critic/Retry:       --critic

Combinations:
  --transition --graph-rag   (A2+A3)
  --transition --critic       (A2+A4)
  --transition --graph-rag --critic  (Full Agentic-VLA)

Legacy (appendix supplement only):
  --vision-prompt
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Any

import imageio
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolve_openpi_root() -> pathlib.Path:
    candidates = []
    if env_root := os.environ.get("AGENTIC_VLA_OPENPI_ROOT"):
        candidates.append(pathlib.Path(env_root).expanduser())
    candidates.extend(
        (
            pathlib.Path("/home/admin1/ct/openpi-official"),
            PROJECT_ROOT / "openpi",
        )
    )
    for candidate in candidates:
        if (candidate / "src" / "openpi").exists():
            return candidate.resolve()
    raise RuntimeError(
        "Unable to locate a runnable openpi repository. "
        "Set AGENTIC_VLA_OPENPI_ROOT to the official openpi checkout."
    )


OPENPI_ROOT = _resolve_openpi_root()
OPENPI_SRC = OPENPI_ROOT / "src"
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages" / "openpi-client" / "src"
LIBERO_SRC = OPENPI_ROOT / "third_party" / "libero"

for extra_path in (PROJECT_ROOT, OPENPI_SRC, OPENPI_CLIENT_SRC, LIBERO_SRC):
    sys.path.insert(0, str(extra_path))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AgenticVLARealLIBERO")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

VISION_PROMPT_COLORS = [
    np.array([0, 255, 0], dtype=np.float32),
    np.array([255, 80, 0], dtype=np.float32),
    np.array([0, 120, 255], dtype=np.float32),
    np.array([255, 255, 0], dtype=np.float32),
    np.array([255, 0, 255], dtype=np.float32),
    np.array([0, 255, 255], dtype=np.float32),
    np.array([255, 165, 0], dtype=np.float32),
    np.array([138, 43, 226], dtype=np.float32),
]

# ===== Transition Agent Constants (A2) =====
TRANSITION_STALL_THRESHOLD = 0.0015
TRANSITION_STALL_WINDOW = 12
TRANSITION_MIN_CONTROL_STEPS = 60
TRANSITION_PROMPT = "slightly lift and stabilize the gripper while maintaining the current grasp"
TRANSITION_CHUNK_STEPS = 6
MAX_TRANSITIONS_PER_EPISODE = 1

# ===== Critic Agent Constants (A4) =====
CRITIC_CHECK_INTERVAL = 60
CRITIC_MIN_CONTROL_STEPS = 120
CRITIC_MAX_RETRIES = 1
CRITIC_RECOVERY_PROMPT = "slightly lift and stabilize the gripper while maintaining the current grasp"
CRITIC_RECOVERY_STEPS = 6

DEFAULT_VLA_EXPERT = "default_vla_expert"
TRANSITION_EXPERT = "transition_expert"
RECOVERY_EXPERT = "recovery_expert"
REGRASP_EXPERT = "regrasp_expert"
_ALLOWED_EXPERTS = {
    DEFAULT_VLA_EXPERT,
    TRANSITION_EXPERT,
    RECOVERY_EXPERT,
    REGRASP_EXPERT,
}

# ===== Graph RAG + Memory Constants (A3) =====
SCENE_PRIORS = {
    "moka pot": {
        "role": "manipulated_object",
        "priority": 3,
        "z_offset": 0.02,
        "force": "medium_grip",
        "yaw_hint": "approach_from_side",
        "stability_hint": "keep_upright_while_transporting",
    },
    "mug": {
        "role": "manipulated_object",
        "priority": 3,
        "z_offset": 0.012,
        "force": "medium_grip",
        "yaw_hint": "approach_from_side",
        "stability_hint": "keep_upright_while_transporting",
    },
    "red mug": {
        "role": "manipulated_object",
        "priority": 4,
        "z_offset": 0.01,
        "force": "medium_grip",
        "yaw_hint": "approach_from_top",
        "stability_hint": "keep_upright_while_transporting",
    },
    "plate": {
        "role": "manipulated_object",
        "priority": 2,
        "z_offset": 0.005,
        "force": "light_grip",
        "yaw_hint": "approach_from_side",
    },
    "pan": {
        "role": "manipulated_object",
        "priority": 2,
        "z_offset": 0.015,
        "force": "medium_grip",
        "yaw_hint": "approach_from_side",
    },
    "kettle": {
        "role": "manipulated_object",
        "priority": 2,
        "z_offset": 0.02,
        "force": "firm_grip",
        "yaw_hint": "approach_from_side",
        "stability_hint": "keep_upright_while_transporting",
    },
    "bowl": {
        "role": "manipulated_object",
        "priority": 2,
        "z_offset": 0.01,
        "force": "light_grip",
        "yaw_hint": "approach_from_top",
    },
    "chef knife": {
        "role": "manipulated_object",
        "priority": 2,
        "z_offset": 0.005,
        "force": "careful_grip",
        "yaw_hint": "approach_carefully",
    },
    "wooden cabinet": {
        "role": "target_region",
        "priority": 1,
        "target_hint": "align_with_the_handle_or_opening_before_contact",
        "release_hint": "avoid_premature_release",
    },
    "stove": {
        "role": "target_region",
        "priority": 1,
        "target_hint": "center_over_the_burner_before_release",
        "release_hint": "release_only_after_the_object_is_settled",
    },
    "microwave": {
        "role": "target_container",
        "priority": 1,
        "target_hint": "align_with_the_opening_and_insert_deeply_before_release",
        "release_hint": "close_the_door_only_after_secure_placement",
    },
}

TASK_CONTROL_PROFILES = {
    "default": {
        "transition_min_steps": TRANSITION_MIN_CONTROL_STEPS,
        "transition_stall_window": TRANSITION_STALL_WINDOW,
        "transition_stall_threshold": TRANSITION_STALL_THRESHOLD,
        "critic_min_steps": CRITIC_MIN_CONTROL_STEPS,
    },
    "stove": {
        "transition_min_steps": 140,
        "transition_stall_window": 16,
        "transition_stall_threshold": 0.0012,
        "critic_min_steps": 180,
    },
    "microwave": {
        "transition_min_steps": 140,
        "transition_stall_window": 16,
        "transition_stall_threshold": 0.0012,
        "critic_min_steps": 180,
    },
}


@dataclasses.dataclass
class RouterDecision:
    expert: str = DEFAULT_VLA_EXPERT
    trigger: str = ""
    reason: str = ""
    request_plan: bool = False
    subgoal: str = "execute"
    prompt_override: str | None = None
    client_hints: dict[str, Any] | None = None


@dataclasses.dataclass
class ExpertState:
    current_expert: str = DEFAULT_VLA_EXPERT
    current_subgoal: str = "execute"
    current_prompt: str = ""
    steps_in_expert: int = 0
    last_trigger: str = ""
    last_reason: str = ""
    agentic_action_scale: float = 1.0
    agentic_apply_steps: int = 0
    planner_cooldown_steps: int = 0
    subgoal_success_window: int = 0

    def reset_for_episode(self, default_prompt: str) -> None:
        self.current_expert = DEFAULT_VLA_EXPERT
        self.current_subgoal = "execute"
        self.current_prompt = default_prompt
        self.steps_in_expert = 0
        self.last_trigger = ""
        self.last_reason = ""
        self.agentic_action_scale = 1.0
        self.agentic_apply_steps = 0
        self.planner_cooldown_steps = 0
        self.subgoal_success_window = 0


@dataclasses.dataclass
class VerifierDecision:
    status: str = "continue"
    reason: str = ""


def _build_empty_expert_stats() -> dict[str, collections.Counter]:
    return {
        "switch_counts": collections.Counter(),
        "success_counts": collections.Counter(),
        "escalation_counts": collections.Counter(),
        "replan_counts": collections.Counter(),
    }


def _serialize_expert_stats(expert_stats: dict[str, collections.Counter]) -> dict[str, dict[str, int]]:
    return {key: dict(value) for key, value in expert_stats.items()}


def _clone_expert_stats(expert_stats: dict[str, collections.Counter]) -> dict[str, collections.Counter]:
    return {key: collections.Counter(value) for key, value in expert_stats.items()}


def _diff_expert_stats(
    after: dict[str, collections.Counter],
    before: dict[str, collections.Counter],
) -> dict[str, collections.Counter]:
    diff = _build_empty_expert_stats()
    for key in diff:
        diff[key].update(after.get(key, {}))
        diff[key].subtract(before.get(key, {}))
        diff[key] = collections.Counter({name: int(count) for name, count in diff[key].items() if int(count) > 0})
    return diff


def _materialize_expert_stats(serialized: dict[str, dict[str, int]] | None) -> dict[str, collections.Counter]:
    stats = _build_empty_expert_stats()
    if not isinstance(serialized, dict):
        return stats
    for key, values in serialized.items():
        if key in stats and isinstance(values, dict):
            stats[key].update({name: int(count) for name, count in values.items()})
    return stats


def _merge_expert_stats(
    base: dict[str, collections.Counter],
    delta: dict[str, collections.Counter],
) -> dict[str, collections.Counter]:
    merged = _clone_expert_stats(base)
    for key in merged:
        merged[key].update(delta.get(key, {}))
    return merged


def _record_expert_switch(expert_stats: dict[str, collections.Counter], expert_name: str) -> None:
    if expert_name:
        expert_stats["switch_counts"][expert_name] += 1


def _route_expert(
    *,
    failure_event: str = "",
    reason: str = "",
    expert_state: ExpertState | None = None,
    critic_stuck: bool = False,
    request_plan: bool = False,
) -> RouterDecision:
    failure_event = str(failure_event or "").strip().lower()
    current_expert = expert_state.current_expert if expert_state is not None else DEFAULT_VLA_EXPERT
    if failure_event == "state_gap":
        return RouterDecision(
            expert=TRANSITION_EXPERT,
            trigger="state_gap",
            reason=reason or "transition agent detected state gap",
            request_plan=request_plan,
            subgoal="recover",
        )
    if failure_event in {"stall", "collision"} or critic_stuck:
        return RouterDecision(
            expert=RECOVERY_EXPERT,
            trigger=failure_event or "critic",
            reason=reason,
            request_plan=request_plan,
            subgoal="recover",
        )
    if failure_event in {"misgrasp", "slip"}:
        return RouterDecision(
            expert=REGRASP_EXPERT,
            trigger=failure_event,
            reason=reason,
            request_plan=request_plan,
            subgoal="regrasp",
        )
    return RouterDecision(
        expert=current_expert,
        trigger=failure_event,
        reason=reason,
        request_plan=request_plan,
        subgoal=expert_state.current_subgoal if expert_state is not None else "execute",
    )


def _apply_router_decision(
    expert_state: ExpertState,
    decision: RouterDecision,
    expert_stats: dict[str, collections.Counter],
    *,
    default_prompt: str,
) -> None:
    next_expert = decision.expert if decision.expert in _ALLOWED_EXPERTS else expert_state.current_expert
    if next_expert != expert_state.current_expert:
        _record_expert_switch(expert_stats, next_expert)
        expert_state.current_expert = next_expert
        expert_state.steps_in_expert = 0
    expert_state.current_subgoal = decision.subgoal or expert_state.current_subgoal
    expert_state.last_trigger = decision.trigger
    expert_state.last_reason = decision.reason
    if decision.prompt_override and decision.prompt_override.strip():
        expert_state.current_prompt = decision.prompt_override.strip()
    elif next_expert == DEFAULT_VLA_EXPERT and not expert_state.current_prompt:
        expert_state.current_prompt = default_prompt
    if next_expert == DEFAULT_VLA_EXPERT and expert_state.current_subgoal not in {"execute", ""}:
        expert_state.current_subgoal = "execute"
        expert_state.current_prompt = default_prompt


def _apply_agentic_response(
    agentic_resp: dict[str, Any],
    *,
    expert_state: ExpertState,
    planner_action_counts: collections.Counter,
    expert_stats: dict[str, collections.Counter],
    default_prompt: str,
) -> None:
    action_name = agentic_resp.get("action")
    if isinstance(action_name, str) and action_name:
        planner_action_counts[action_name] += 1

    planned_expert = agentic_resp.get("expert")
    if isinstance(planned_expert, str):
        planned_expert = planned_expert.strip()
        if planned_expert in _ALLOWED_EXPERTS and planned_expert != expert_state.current_expert:
            _record_expert_switch(expert_stats, planned_expert)
            expert_state.current_expert = planned_expert
            expert_state.steps_in_expert = 0

    subgoal_name = agentic_resp.get("subgoal")
    if isinstance(subgoal_name, str) and subgoal_name:
        expert_state.current_subgoal = subgoal_name

    hint_prompt = agentic_resp.get("prompt_suggested")
    if isinstance(hint_prompt, str) and hint_prompt.strip():
        expert_state.current_prompt = hint_prompt.strip()
    elif expert_state.current_expert == DEFAULT_VLA_EXPERT and not expert_state.current_prompt:
        expert_state.current_prompt = default_prompt

    hints = agentic_resp.get("client_hints")
    if isinstance(hints, dict):
        try:
            expert_state.agentic_action_scale = float(hints.get("action_scale", expert_state.agentic_action_scale))
        except Exception:
            expert_state.agentic_action_scale = expert_state.agentic_action_scale
        try:
            expert_state.agentic_apply_steps = int(hints.get("apply_steps", expert_state.agentic_apply_steps))
        except Exception:
            expert_state.agentic_apply_steps = expert_state.agentic_apply_steps
        expert_state.agentic_action_scale = float(np.clip(expert_state.agentic_action_scale, 0.2, 1.5))
        expert_state.agentic_apply_steps = int(np.clip(expert_state.agentic_apply_steps, 0, 200))
        if expert_state.agentic_apply_steps > 0 and expert_state.current_subgoal == "execute":
            expert_state.current_subgoal = "recover"


def _verify_expert_progress(
    expert_state: ExpertState,
    *,
    failure_event: str,
    state_gap_active: bool = False,
) -> VerifierDecision:
    failure_event = str(failure_event or "").strip().lower()
    if expert_state.current_expert == DEFAULT_VLA_EXPERT:
        return VerifierDecision(status="continue", reason="default executor remains active")
    if expert_state.current_expert == TRANSITION_EXPERT:
        if not state_gap_active:
            return VerifierDecision(status="resolved", reason="state gap no longer active")
        if expert_state.steps_in_expert >= max(TRANSITION_CHUNK_STEPS * 3, 12):
            return VerifierDecision(status="escalate", reason="transition expert exceeded step budget")
        return VerifierDecision(status="continue", reason="transition expert still active")
    if expert_state.current_expert == RECOVERY_EXPERT:
        if not failure_event and expert_state.subgoal_success_window >= 6:
            return VerifierDecision(status="resolved", reason="recovery window is stable")
        if expert_state.steps_in_expert >= max(CRITIC_RECOVERY_STEPS * 3, 18):
            return VerifierDecision(status="replan", reason="recovery expert exceeded step budget")
        return VerifierDecision(status="continue", reason="recovery expert still active")
    if expert_state.current_expert == REGRASP_EXPERT:
        if failure_event not in {"misgrasp", "slip"} and expert_state.subgoal_success_window >= 6:
            return VerifierDecision(status="resolved", reason="grasp state appears stable")
        if expert_state.steps_in_expert >= max(CRITIC_RECOVERY_STEPS * 4, 24):
            return VerifierDecision(status="escalate", reason="regrasp expert exceeded step budget")
        return VerifierDecision(status="continue", reason="regrasp expert still active")
    return VerifierDecision(status="continue", reason="no verifier rule matched")


def _sync_expert_runtime_state(
    expert_state: ExpertState,
    *,
    current_prompt: str,
    current_subgoal: str,
    agentic_action_scale: float,
    agentic_apply_steps: int,
    planner_cooldown_steps: int,
    subgoal_success_window: int,
) -> None:
    expert_state.current_prompt = current_prompt
    expert_state.current_subgoal = current_subgoal
    expert_state.agentic_action_scale = float(agentic_action_scale)
    expert_state.agentic_apply_steps = int(agentic_apply_steps)
    expert_state.planner_cooldown_steps = int(planner_cooldown_steps)
    expert_state.subgoal_success_window = int(subgoal_success_window)


def _build_regrasp_prompt(task_description: str, priors: dict | None) -> str:
    task_lower = task_description.lower()
    priors = priors or {}
    if "microwave" in task_lower:
        return (
            "carefully re-align the object, close the gripper again, "
            "keep it upright, and re-approach the microwave opening"
        )
    if "stove" in task_lower or "burner" in task_lower:
        return (
            "carefully re-grasp the object, keep it upright, "
            "and re-center above the burner before placement"
        )
    if priors.get("stability_hint") == "keep_upright_while_transporting":
        return "carefully re-grasp, stabilize the object, and keep it upright before continuing"
    return "carefully re-grasp and stabilize the object before continuing the task"


def _build_policy_payload(
    *,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
    episode_id: str | None = None,
    timestep: int | None = None,
    agentic_req: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "observation/image": base_img_p,
        "observation/wrist_image": wrist_img_p,
        "observation/state": np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )),
        "prompt": str(prompt),
    }
    if episode_id is not None:
        payload["episode_id"] = episode_id
    if timestep is not None:
        payload["timestep"] = int(timestep)
    if agentic_req is not None:
        payload["agentic"] = agentic_req
    return payload


def _execute_prompt_chunk(
    *,
    env: Any,
    client: Any,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
    chunk_steps: int,
    max_actions_per_infer: int = 3,
) -> tuple[dict[str, Any], int, bool]:
    total_steps = 0
    done = False
    current_obs = obs
    for _ in range(int(chunk_steps)):
        payload = _build_policy_payload(
            obs=current_obs,
            base_img_p=base_img_p,
            wrist_img_p=wrist_img_p,
            prompt=prompt,
        )
        try:
            inf = _infer_with_retry(client, payload, max_retries=3)
        except Exception:
            break
        if "actions" not in inf:
            break
        for action in inf["actions"][:max_actions_per_infer]:
            current_obs, _, done_local, _ = env.step(action.tolist())
            total_steps += 1
            if done_local:
                done = True
                break
        if done:
            break
    return current_obs, total_steps, done


def _execute_transition_expert(
    *,
    env: Any,
    client: Any,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
) -> tuple[dict[str, Any], int, bool]:
    return _execute_prompt_chunk(
        env=env,
        client=client,
        obs=obs,
        base_img_p=base_img_p,
        wrist_img_p=wrist_img_p,
        prompt=prompt,
        chunk_steps=TRANSITION_CHUNK_STEPS,
    )


def _execute_recovery_expert(
    *,
    env: Any,
    client: Any,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
) -> tuple[dict[str, Any], int, bool]:
    return _execute_prompt_chunk(
        env=env,
        client=client,
        obs=obs,
        base_img_p=base_img_p,
        wrist_img_p=wrist_img_p,
        prompt=prompt,
        chunk_steps=CRITIC_RECOVERY_STEPS,
    )


def _execute_regrasp_expert(
    *,
    env: Any,
    client: Any,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
) -> tuple[dict[str, Any], int, bool]:
    return _execute_prompt_chunk(
        env=env,
        client=client,
        obs=obs,
        base_img_p=base_img_p,
        wrist_img_p=wrist_img_p,
        prompt=prompt,
        chunk_steps=max(CRITIC_RECOVERY_STEPS, 8),
    )


def _apply_runtime_from_expert_state(expert_state: ExpertState) -> tuple[str, str, float, int]:
    return (
        expert_state.current_subgoal,
        expert_state.current_prompt,
        expert_state.agentic_action_scale,
        expert_state.agentic_apply_steps,
    )


def _handle_agentic_response(
    agentic_resp: dict[str, Any],
    *,
    expert_state: ExpertState,
    planner_stats: dict[str, Any],
    planner_action_counts: collections.Counter,
    planner_expert_counts: collections.Counter,
    expert_stats: dict[str, collections.Counter],
    default_prompt: str,
) -> tuple[str, str, float, int]:
    if float(agentic_resp.get("plan_ms", 0.0)) > 0.0:
        planner_stats["plans_generated"] += 1
    expert_name = agentic_resp.get("expert")
    if isinstance(expert_name, str) and expert_name:
        planner_expert_counts[expert_name] += 1
    _apply_agentic_response(
        agentic_resp,
        expert_state=expert_state,
        planner_action_counts=planner_action_counts,
        expert_stats=expert_stats,
        default_prompt=default_prompt,
    )
    return _apply_runtime_from_expert_state(expert_state)


def _request_planner(
    *,
    client: Any,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
    episode_id: str,
    timestep: int,
    event: str,
    reason: str,
    context: str,
    planner_stats: dict[str, Any],
    planner_event_counts: collections.Counter,
    planner_action_counts: collections.Counter,
    planner_expert_counts: collections.Counter,
    expert_state: ExpertState,
    expert_stats: dict[str, collections.Counter],
    default_prompt: str,
    current_cooldown_steps: int,
    cooldown_steps: int,
) -> tuple[bool, int, str, str, float, int]:
    if current_cooldown_steps > 0:
        return False, current_cooldown_steps, * _apply_runtime_from_expert_state(expert_state)
    planner_stats["total_requests"] += 1
    planner_event_counts[str(event)] += 1
    next_cooldown_steps = max(int(current_cooldown_steps), int(cooldown_steps))
    payload = _build_policy_payload(
        obs=obs,
        base_img_p=base_img_p,
        wrist_img_p=wrist_img_p,
        prompt=prompt,
        episode_id=episode_id,
        timestep=timestep,
        agentic_req={
            "request_plan": True,
            "event": str(event),
            "reason": str(reason),
            "context": str(context),
        },
    )
    try:
        plan_inf = _infer_with_retry(client, payload, max_retries=3)
    except Exception:
        return True, next_cooldown_steps, * _apply_runtime_from_expert_state(expert_state)
    if isinstance(plan_inf.get("agentic"), dict):
        current_subgoal, current_prompt, action_scale, apply_steps = _handle_agentic_response(
            plan_inf["agentic"],
            expert_state=expert_state,
            planner_stats=planner_stats,
            planner_action_counts=planner_action_counts,
            planner_expert_counts=planner_expert_counts,
            expert_stats=expert_stats,
            default_prompt=default_prompt,
        )
        return True, next_cooldown_steps, current_subgoal, current_prompt, action_scale, apply_steps
    return True, next_cooldown_steps, * _apply_runtime_from_expert_state(expert_state)


def _keyword_match(task_lower: str, keyword: str) -> bool:
    if keyword in task_lower:
        return True
    if keyword.endswith("y") and f"{keyword[:-1]}ies" in task_lower:
        return True
    if f"{keyword}s" in task_lower:
        return True
    return False


def _build_task_control_profile(task_description: str) -> dict:
    profile = TASK_CONTROL_PROFILES["default"].copy()
    task_lower = task_description.lower()
    for keyword, keyword_profile in TASK_CONTROL_PROFILES.items():
        if keyword == "default":
            continue
        if _keyword_match(task_lower, keyword):
            profile.update(keyword_profile)
    return profile


def _build_transition_prompt(task_description: str, priors: dict | None) -> str:
    task_lower = task_description.lower()
    priors = priors or {}
    if "microwave" in task_lower:
        return (
            "slightly back away from the door frame, keep the object upright, "
            "re-center with the opening, and maintain the current grasp"
        )
    if "stove" in task_lower or "burner" in task_lower:
        return (
            "slightly lift, keep the object upright, re-center above the burner, "
            "and maintain the current grasp before continuing placement"
        )
    if priors.get("stability_hint") == "keep_upright_while_transporting":
        return "slightly lift, keep the object upright, and stabilize the grasp before continuing"
    return TRANSITION_PROMPT


def _build_recovery_prompt(task_description: str, priors: dict | None) -> str:
    task_lower = task_description.lower()
    priors = priors or {}
    if "microwave" in task_lower:
        return (
            "carefully retract a little, keep the object upright, "
            "and stabilize before re-aligning with the microwave opening"
        )
    if "stove" in task_lower or "burner" in task_lower:
        return (
            "carefully lift a little, keep the object upright, "
            "and re-center above the burner before continuing placement"
        )
    if priors.get("stability_hint") == "keep_upright_while_transporting":
        return "slightly lift and stabilize the grasp while keeping the object upright"
    return CRITIC_RECOVERY_PROMPT


def _parse_int_list(value: str | None) -> list[int]:
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return [int(part) for part in parts]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real LIBERO evaluation for Agentic-VLA ablation.")
    parser.add_argument("--host", default="127.0.0.1", help="Websocket policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="Websocket policy server port.")
    parser.add_argument(
        "--task-suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite name.",
    )
    parser.add_argument("--task-id", type=int, default=None, help="Single task id to run.")
    parser.add_argument("--task-ids", type=_parse_int_list, default=[], help="Comma-separated task ids to run.")
    parser.add_argument(
        "--skip-task-ids",
        type=_parse_int_list,
        default=[],
        help="Comma-separated task ids to skip.",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per task.")
    parser.add_argument("--replan-steps", type=int, default=10, help="Action chunk replanning interval.")
    parser.add_argument("--resize-size", type=int, default=224, help="Image resize used by policy.")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="Warmup sim steps before control.")
    parser.add_argument("--video-dir", default=str(PROJECT_ROOT / "results" / "videos"), help="Video output directory.")
    parser.add_argument(
        "--results-json",
        default=str(PROJECT_ROOT / "results" / "libero_eval_results.json"),
        help="Path to save measured success metrics as JSON.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Ablation flags
    parser.add_argument("--transition", action="store_true", help="A2: Enable Transition Agent.")
    parser.add_argument("--graph-rag", action="store_true", help="A3: Enable Graph RAG + Memory.")
    parser.add_argument("--critic", action="store_true", help="A4: Enable Critic/Retry with Qwen3-VL.")
    parser.add_argument(
        "--agentic-planner",
        action="store_true",
        help="Enable server-side VLM planner on-demand (request plan only when stalled).",
    )
    parser.add_argument(
        "--planner-cooldown-steps",
        type=int,
        default=50,
        help="Minimum steps between two planner requests within one episode (on-demand planner only).",
    )

    # Legacy flag (appendix supplement only)
    parser.add_argument("--vision-prompt", action="store_true", help="Legacy: overlay colored masks (appendix only).")
    parser.add_argument("--mask-alpha", type=float, default=0.35, help="Alpha for colored mask overlay.")

    # Meta
    parser.add_argument("--ablation-tag", default="", help="Tag for this ablation experiment.")
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL model or local path for Critic.",
    )
    parser.add_argument("--qwen-quant", default="4bit", choices=["none", "4bit", "8bit"],
                        help="Quantization mode for Qwen3-VL (none/4bit/8bit). Default: 4bit for RTX 4090.")
    parser.add_argument("--critic-temp", type=float, default=0.3, help="Temperature for Qwen3-VL Critic.")
    return parser.parse_args()


def _require_runtime():
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy
    except Exception as exc:
        raise RuntimeError(
            "Missing runtime dependency for real LIBERO evaluation."
        ) from exc
    return benchmark, get_libero_path, OffScreenRenderEnv, SegmentationRenderEnv, image_tools, websocket_client_policy


def _max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _make_env(task, get_libero_path, env_class, seed: int):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = env_class(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def _build_results_payload(
    args: argparse.Namespace,
    ablation_tag: str,
    total_successes: int,
    total_episodes: int,
    episode_lengths: list[int],
    success_episode_lengths: list[int],
    task_metrics: dict,
    transition_stats: dict | None,
    critic_stats: dict | None,
    planner_stats: dict | None,
    expert_stats: dict | None,
    status: str,
    current_task: dict | None,
) -> dict:
    overall = total_successes / total_episodes if total_episodes else 0.0
    avg_episode_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    avg_success_episode_length = float(np.mean(success_episode_lengths)) if success_episode_lengths else 0.0
    payload = {
        "task_suite": args.task_suite,
        "trials_per_task": args.trials,
        "ablation_tag": ablation_tag,
        "status": status,
        "current_task": current_task,
        "evaluated_task_ids": getattr(args, "evaluated_task_ids", None),
        "skipped_task_ids": getattr(args, "skipped_task_ids", None),
        "flags": {
            "transition": args.transition,
            "graph_rag": args.graph_rag,
            "critic": args.critic,
            "vision_prompt": args.vision_prompt,
            "agentic_planner": getattr(args, "agentic_planner", False),
        },
        "overall_success_rate": overall,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "avg_episode_length": avg_episode_length,
        "avg_success_episode_length": avg_success_episode_length,
        "task_metrics": task_metrics,
        "transition_stats": transition_stats if args.transition else None,
        "critic_stats": critic_stats if args.critic else None,
        "planner_stats": planner_stats if getattr(args, "agentic_planner", False) else None,
        "expert_stats": expert_stats,
        "mechanism_summary": {
            "avg_transitions_per_episode": (
                transition_stats["total_transitions"] / total_episodes if args.transition and total_episodes else 0.0
            ),
            "transition_success_rate_when_used": (
                transition_stats["transitions_leading_to_success"] / max(1, sum(
                    metrics["transition_episode_count"] for metrics in task_metrics.values()
                ))
                if args.transition else 0.0
            ),
            "avg_retries_per_episode": (
                critic_stats["retries_triggered"] / total_episodes if args.critic and total_episodes else 0.0
            ),
            "retry_success_rate_when_used": (
                critic_stats["retries_leading_to_success"] / max(1, sum(
                    metrics["retry_episode_count"] for metrics in task_metrics.values()
                ))
                if args.critic else 0.0
            ),
            "avg_episode_length": avg_episode_length,
            "avg_success_episode_length": avg_success_episode_length,
        },
    }
    if payload["evaluated_task_ids"] is None:
        payload.pop("evaluated_task_ids", None)
    if payload["skipped_task_ids"] is None:
        payload.pop("skipped_task_ids", None)
    return payload


# ===== Vision Prompt (Legacy) =====
def _apply_vision_prompt(image, segmentation_image, obj_of_interest, instance_to_id, alpha=0.35):
    if segmentation_image.ndim == 3:
        segmentation_image = segmentation_image.squeeze(-1)
    result = image.astype(np.float32)
    for i, obj_name in enumerate(obj_of_interest):
        if obj_name not in instance_to_id:
            continue
        seg_id = instance_to_id[obj_name]
        mask = segmentation_image == seg_id
        if not mask.any():
            continue
        color = VISION_PROMPT_COLORS[i % len(VISION_PROMPT_COLORS)]
        result[mask] = (1 - alpha) * result[mask] + alpha * color
    return np.clip(result, 0, 255).astype(np.uint8)


# ===== Transition Agent (A2) =====
class TransitionAgent:
    """Detects State Gap and triggers prompt-based transition via VLA."""

    def __init__(self, stall_threshold=TRANSITION_STALL_THRESHOLD,
                 stall_window=TRANSITION_STALL_WINDOW,
                 max_transitions=MAX_TRANSITIONS_PER_EPISODE):
        self.default_stall_threshold = stall_threshold
        self.default_stall_window = stall_window
        self.stall_threshold = stall_threshold
        self.stall_window = stall_window
        self.max_transitions = max_transitions
        self.ee_pos_history = []
        self.transition_count = 0
        self.in_transition = False

    def reset(self):
        self.ee_pos_history = []
        self.transition_count = 0
        self.in_transition = False
        self.stall_threshold = self.default_stall_threshold
        self.stall_window = self.default_stall_window

    def configure_for_task(self, control_profile: dict):
        self.stall_threshold = control_profile.get("transition_stall_threshold", self.default_stall_threshold)
        self.stall_window = control_profile.get("transition_stall_window", self.default_stall_window)

    def record_ee_pos(self, ee_pos):
        self.ee_pos_history.append(ee_pos.copy())

    def detect_state_gap(self):
        if self.transition_count >= self.max_transitions:
            return False
        if len(self.ee_pos_history) < self.stall_window:
            return False
        recent = self.ee_pos_history[-self.stall_window:]
        displacements = [np.linalg.norm(recent[i+1] - recent[i]) for i in range(len(recent)-1)]
        avg_disp = np.mean(displacements)
        return avg_disp < self.stall_threshold

    def trigger_transition(self):
        self.in_transition = True
        self.transition_count += 1
        self.ee_pos_history = []
        logger.info("[Transition] State Gap detected! Triggering transition #%d", self.transition_count)

    def finish_transition(self):
        self.in_transition = False


# ===== Failure Taxonomy (rule-based, no extra VLM) =====
class FailureTaxonomy:
    def __init__(
        self,
        *,
        window: int = 12,
        stall_disp: float = TRANSITION_STALL_THRESHOLD,
        collision_action_mag: float = 0.06,
        misgrasp_gripper_open: float = 0.75,
        misgrasp_close_cmd: float = -0.2,
        slip_open_high: float = 0.7,
        slip_open_low: float = 0.35,
    ) -> None:
        self.window = int(window)
        self.stall_disp = float(stall_disp)
        self.collision_action_mag = float(collision_action_mag)
        self.misgrasp_gripper_open = float(misgrasp_gripper_open)
        self.misgrasp_close_cmd = float(misgrasp_close_cmd)
        self.slip_open_high = float(slip_open_high)
        self.slip_open_low = float(slip_open_low)
        self.ee_pos_hist: list[np.ndarray] = []
        self.action_hist: list[np.ndarray] = []
        self.gripper_hist: list[np.ndarray] = []

    def reset(self) -> None:
        self.ee_pos_hist = []
        self.action_hist = []
        self.gripper_hist = []

    def update(self, *, ee_pos: np.ndarray, action: np.ndarray, gripper_qpos: np.ndarray) -> None:
        self.ee_pos_hist.append(np.asarray(ee_pos, dtype=np.float32).copy())
        self.action_hist.append(np.asarray(action, dtype=np.float32).copy())
        self.gripper_hist.append(np.asarray(gripper_qpos, dtype=np.float32).copy())
        if len(self.ee_pos_hist) > self.window:
            self.ee_pos_hist = self.ee_pos_hist[-self.window :]
        if len(self.action_hist) > self.window:
            self.action_hist = self.action_hist[-self.window :]
        if len(self.gripper_hist) > self.window:
            self.gripper_hist = self.gripper_hist[-self.window :]

    def _avg_disp(self) -> float:
        if len(self.ee_pos_hist) < 2:
            return 0.0
        recent = self.ee_pos_hist
        displacements = [float(np.linalg.norm(recent[i + 1] - recent[i])) for i in range(len(recent) - 1)]
        return float(np.mean(displacements)) if displacements else 0.0

    def _avg_action_mag(self) -> float:
        if not self.action_hist:
            return 0.0
        mags = [float(np.linalg.norm(a[:3])) for a in self.action_hist]
        return float(np.mean(mags)) if mags else 0.0

    def classify(self) -> tuple[str, str]:
        if len(self.ee_pos_hist) < self.window:
            return "", ""

        avg_disp = self._avg_disp()
        avg_act = self._avg_action_mag()

        last_a = self.action_hist[-1] if self.action_hist else np.zeros(7, dtype=np.float32)
        open_fracs = []
        for g in self.gripper_hist:
            gg = np.asarray(g, dtype=np.float32).reshape(-1)
            if gg.size:
                open_fracs.append(float(np.clip(np.mean(gg), 0.0, 1.0)))
        open_frac = open_fracs[-1] if open_fracs else 0.0

        if (
            float(last_a[6]) <= self.misgrasp_close_cmd
            and open_fracs
            and float(np.min(open_fracs)) <= self.slip_open_low
            and open_frac >= self.slip_open_high
        ):
            return "slip", f"close_cmd={float(last_a[6]):.3f} gripper_open={open_frac:.2f}"

        if avg_disp < self.stall_disp:
            if avg_act > self.collision_action_mag:
                return "collision", f"avg_disp={avg_disp:.4f} avg_act={avg_act:.4f}"
            return "stall", f"avg_disp={avg_disp:.4f}"

        if float(last_a[6]) <= self.misgrasp_close_cmd and open_frac >= self.misgrasp_gripper_open:
            return "misgrasp", f"close_cmd={float(last_a[6]):.3f} gripper_open={open_frac:.2f}"

        return "", ""


# ===== Graph RAG + Memory (A3) =====
class GraphRAGMemory:
    """Injects scene priors from Topo-Graph RAG and Evo-KAM Memory."""

    def __init__(self):
        self.priors_db = SCENE_PRIORS.copy()
        self.episode_memory = {}

    def extract_priors_for_task(self, task_description):
        task_lower = task_description.lower()
        object_choice = None
        object_priority = -1
        target_hints = []
        release_hints = []
        matched_keywords = []
        priors = {
            "matched_keywords": matched_keywords,
            "target_hints": target_hints,
            "release_hints": release_hints,
        }
        for obj_name, obj_priors in self.priors_db.items():
            if not _keyword_match(task_lower, obj_name):
                continue
            matched_keywords.append(obj_name)
            role = obj_priors.get("role", "manipulated_object")
            logger.info("[GraphRAG] Matched prior '%s': %s", obj_name, obj_priors)
            if role == "manipulated_object":
                priority = obj_priors.get("priority", 0)
                if priority > object_priority:
                    object_priority = priority
                    object_choice = obj_priors
            else:
                if target_hint := obj_priors.get("target_hint"):
                    target_hints.append(target_hint)
                if release_hint := obj_priors.get("release_hint"):
                    release_hints.append(release_hint)
        if object_choice:
            priors.update({
                key: value
                for key, value in object_choice.items()
                if key not in {"role", "priority"}
            })
        if task_description in self.episode_memory:
            priors["memory_hint"] = self.episode_memory[task_description]
        return priors

    def augment_prompt(self, task_description, priors):
        if not priors or len(priors) <= 3:
            return task_description
        augmentations = []
        if "z_offset" in priors:
            augmentations.append(f"keep a slight height margin of about {priors['z_offset']:.3f} meters")
        if "force" in priors:
            augmentations.append(f"use {priors['force']}")
        if "yaw_hint" in priors:
            augmentations.append(priors["yaw_hint"].replace("_", " "))
        if "stability_hint" in priors:
            augmentations.append(priors["stability_hint"].replace("_", " "))
        for target_hint in priors.get("target_hints", []):
            augmentations.append(target_hint.replace("_", " "))
        for release_hint in priors.get("release_hints", []):
            augmentations.append(release_hint.replace("_", " "))
        if "memory_hint" in priors:
            augmentations.append(priors["memory_hint"])
        if augmentations:
            return f"{task_description}, {', '.join(augmentations)}"
        return task_description

    def record_success(self, task_description, priors):
        memory_parts = []
        if "yaw_hint" in priors:
            memory_parts.append(priors["yaw_hint"].replace("_", " "))
        if "stability_hint" in priors:
            memory_parts.append(priors["stability_hint"].replace("_", " "))
        if priors.get("target_hints"):
            memory_parts.append(priors["target_hints"][0].replace("_", " "))
        memory_hint = "repeat the previously successful stable strategy"
        if memory_parts:
            memory_hint = f"repeat the previously successful stable strategy: {', '.join(memory_parts)}"
        self.episode_memory[task_description] = memory_hint
        logger.info("[EvoKAM] Recorded success for '%s': %s", task_description, memory_hint)


# ===== Critic Agent (A4) =====
class CriticAgent:
    """Reflective Critic with Qwen3-VL for error attribution and retry."""

    def __init__(self, qwen_model_name="Qwen/Qwen3-VL-8B-Instruct", temp=0.3,
                 quant_mode="4bit",
                 check_interval=CRITIC_CHECK_INTERVAL,
                 max_retries=CRITIC_MAX_RETRIES):
        self.qwen_model_name = qwen_model_name
        self.temp = temp
        self.quant_mode = quant_mode
        self.check_interval = check_interval
        self.max_retries = max_retries
        self.model = None
        self.processor = None
        self.retry_count = 0
        self.min_control_steps = CRITIC_MIN_CONTROL_STEPS

    def reset(self):
        self.retry_count = 0
        self.min_control_steps = CRITIC_MIN_CONTROL_STEPS

    def configure_for_task(self, control_profile: dict):
        self.min_control_steps = control_profile.get("critic_min_steps", CRITIC_MIN_CONTROL_STEPS)

    def load_model(self):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            import torch

            logger.info("[Critic] Loading Qwen3-VL: %s (quant=%s)...", self.qwen_model_name, self.quant_mode)

            # Build quantization config
            kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
            if self.quant_mode == "4bit":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("[Critic] Using bitsandbytes NF4 4-bit quantization.")
            elif self.quant_mode == "8bit":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                logger.info("[Critic] Using bitsandbytes 8-bit quantization.")
            else:
                logger.info("[Critic] No quantization (full precision).")

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.qwen_model_name, **kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(self.qwen_model_name)
            logger.info("[Critic] Qwen3-VL loaded successfully (quant=%s).", self.quant_mode)
            return True
        except Exception as e:
            logger.error("[Critic] Failed to load Qwen3-VL: %s", e)
            return False

    def unload_model(self):
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[Critic] Qwen3-VL unloaded, GPU memory freed.")
        except Exception as e:
            logger.warning("[Critic] Unload error: %s", e)

    def judge_subtask(self, image, task_description):
        if self.model is None or self.processor is None:
            return self._heuristic_judge()

        try:
            from PIL import Image as PILImage
            import torch

            pil_img = PILImage.fromarray(image)
            prompt_text = (
                f"You are a robotic task critic. Task: '{task_description}'.\n"
                "Be conservative. Only report stuck=true if the robot is clearly stalled or trembling "
                "for an extended moment rather than making slow but valid progress.\n"
                f"Analyze the image and answer:\n"
                f"1. Is the current subtask completed? (yes/no)\n"
                f"2. Is the gripper clearly stuck or trembling? (yes/no)\n"
                f"3. If not completed, error type? (collision/slip/misgrasp/other)\n"
                f'Respond JSON: {{"completed": bool, "stuck": bool, "error_type": str}}'
            )

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt_text},
                ]}
            ]

            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_input],
                images=[pil_img],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=256, temperature=self.temp, do_sample=self.temp > 0,
                )

            output_text = self.processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            )[0].strip()

            logger.info("[Critic] Qwen3-VL judgment: %s", output_text)
            return self._parse_critic_output(output_text)
        except Exception as e:
            logger.warning("[Critic] Inference failed: %s, falling back to heuristic.", e)
            return self._heuristic_judge()

    def _heuristic_judge(self):
        return {"completed": False, "stuck": False, "error_type": "unknown"}

    def _parse_critic_output(self, text):
        import re
        result = {"completed": False, "stuck": False, "error_type": "unknown"}
        try:
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                parsed = json.loads(json_match.group())
                result["completed"] = parsed.get("completed", False)
                result["stuck"] = parsed.get("stuck", False)
                result["error_type"] = parsed.get("error_type", "unknown")
                return result
        except json.JSONDecodeError:
            pass
        text_lower = text.lower()
        result["completed"] = "yes" in text_lower or "completed" in text_lower
        result["stuck"] = "stuck" in text_lower or "trembling" in text_lower
        for etype in ["collision", "slip", "misgrasp"]:
            if etype in text_lower:
                result["error_type"] = etype
                break
        return result

    def get_retry_prompt(self, task_description, judgment):
        error_type = judgment.get("error_type", "unknown")
        retry_prompts = {
            "collision": f"carefully avoid obstacles and {task_description}",
            "slip": f"grip more firmly and {task_description}",
            "misgrasp": f"reposition and carefully {task_description}",
            "unknown": f"try again carefully: {task_description}",
        }
        return retry_prompts.get(error_type, retry_prompts["unknown"])

    def should_retry(self, judgment):
        if self.retry_count >= self.max_retries:
            return False
        if judgment.get("completed", False):
            return False
        if judgment.get("stuck", False):
            return True
        return False


def _infer_with_retry(client, payload, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.infer(payload)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                logger.warning("Inference attempt %d failed: %s, retrying in %ds...", attempt + 1, e, wait)
                time.sleep(wait)
            else:
                raise


def _create_client(host, port, max_attempts=10):
    import websockets.sync.client
    from openpi_client import msgpack_numpy as _msgpack_numpy
    from openpi_client import websocket_client_policy as _ws_policy
    uri = f"ws://{host}:{port}"
    for attempt in range(max_attempts):
        try:
            conn = websockets.sync.client.connect(
                uri, compression=None, max_size=None,
                ping_interval=120, ping_timeout=120, close_timeout=30,
            )
            metadata = _msgpack_numpy.unpackb(conn.recv())
            client = _ws_policy.WebsocketClientPolicy.__new__(_ws_policy.WebsocketClientPolicy)
            client._uri = uri
            client._api_key = None
            client._packer = _msgpack_numpy.Packer()
            client._ws = conn
            client._server_metadata = metadata
            logger.info("Created client at %s:%d", host, port)
            return client
        except Exception as e:
            wait = min(2 ** attempt, 30)
            logger.warning("Client creation attempt %d failed: %s, retrying in %ds...", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to create client at {host}:{port} after {max_attempts} attempts")


def _restart_policy_server(openpi_root, port, env="LIBERO"):
    import subprocess
    logger.warning("Restarting policy server on port %d...", port)
    try:
        subprocess.run(["pkill", "-f", f"serve_policy.py.*--port.*{port}"], capture_output=True, timeout=10)
        time.sleep(3)
    except Exception:
        pass
    env_vars = os.environ.copy()
    env_vars["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    cmd = [
        str(pathlib.Path(openpi_root) / ".venv" / "bin" / "python"),
        str(pathlib.Path(openpi_root) / "scripts" / "serve_policy.py"),
        "--env", env, "--port", str(port),
    ]
    proc = subprocess.Popen(cmd, cwd=str(openpi_root), env=env_vars, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info("Policy server process started (PID %d), waiting...", proc.pid)
    time.sleep(30)
    return proc


def _reconnect_client(host, port, max_attempts=10):
    return _create_client(host, port, max_attempts)


def _build_ablation_tag(args):
    if args.ablation_tag:
        return args.ablation_tag
    parts = []
    if args.transition:
        parts.append("A2-Transition")
    if args.graph_rag:
        parts.append("A3-GraphRAG")
    if args.critic:
        parts.append("A4-Critic")
    if args.vision_prompt:
        parts.append("VP")
    if not parts:
        return "A1-baseline"
    return "+".join(parts)


def evaluate_real_libero(args: argparse.Namespace) -> dict:
    benchmark, get_libero_path, offscreen_render_env, segmentation_render_env, image_tools, websocket_client_policy = _require_runtime()

    np.random.seed(args.seed)
    pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    # Initialize agents based on ablation flags
    transition_agent = TransitionAgent() if args.transition else None
    planner_trigger_agent = TransitionAgent() if args.agentic_planner and transition_agent is None else None
    failure_taxonomy = FailureTaxonomy(window=TRANSITION_STALL_WINDOW, stall_disp=TRANSITION_STALL_THRESHOLD)
    graph_rag_memory = GraphRAGMemory() if args.graph_rag else None
    critic_agent = CriticAgent(
        qwen_model_name=args.qwen_model, temp=args.critic_temp,
        quant_mode=args.qwen_quant,
    ) if args.critic else None

    # Load Critic model if needed
    if critic_agent is not None:
        if not critic_agent.load_model():
            logger.warning("[A4] Qwen3-VL load failed, Critic will use heuristic fallback.")

    ablation_tag = _build_ablation_tag(args)
    logger.info("=== Ablation: %s | Suite: %s ===", ablation_tag, args.task_suite)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    max_steps = _max_steps_for_suite(args.task_suite)
    if args.task_id is not None:
        task_ids = [int(args.task_id)]
    elif args.task_ids:
        task_ids = [int(t) for t in args.task_ids]
    else:
        task_ids = list(range(task_suite.n_tasks))
    skip_ids = set(int(t) for t in getattr(args, "skip_task_ids", []) or [])
    task_ids = [task_id for task_id in task_ids if int(task_id) not in skip_ids]
    if not task_ids:
        raise ValueError("No tasks selected after applying --task-id/--task-ids and --skip-task-ids.")
    args.evaluated_task_ids = [int(t) for t in task_ids]
    args.skipped_task_ids = sorted(skip_ids) if skip_ids else []

    client = _create_client(args.host, args.port)

    env_class = segmentation_render_env if args.vision_prompt else offscreen_render_env
    if args.vision_prompt:
        logger.info("Vision Prompt ENABLED (appendix) | mask_alpha=%.2f", args.mask_alpha)

    results_path = pathlib.Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    completed_tasks_file = results_path.parent / "_completed_tasks.json"
    resume_state_file = results_path.parent / "_resume_state.json"
    partial_summary_file = results_path.parent / "_partial_summary.json"

    completed_tasks = set()
    resume_state = None
    if resume_state_file.exists():
        try:
            resume_state = json.loads(resume_state_file.read_text())
            logger.info("Loaded resume state from %s", resume_state_file)
        except Exception:
            resume_state = None

    if completed_tasks_file.exists():
        try:
            completed_tasks = set(json.loads(completed_tasks_file.read_text()))
            logger.info("Resuming: skipping already completed tasks %s", sorted(completed_tasks))
        except Exception:
            completed_tasks = set()
    if resume_state is not None:
        completed_tasks.update(str(task_id) for task_id in resume_state.get("completed_tasks", []))

    total_episodes = int(resume_state.get("total_episodes", 0)) if resume_state is not None else 0
    total_successes = int(resume_state.get("total_successes", 0)) if resume_state is not None else 0
    task_metrics = dict(resume_state.get("task_metrics", {})) if resume_state is not None else {}
    episode_lengths = list(resume_state.get("episode_lengths", [])) if resume_state is not None else []
    success_episode_lengths = list(resume_state.get("success_episode_lengths", [])) if resume_state is not None else []
    transition_stats = {"total_transitions": 0, "transitions_leading_to_success": 0}
    critic_stats = {"total_checks": 0, "retries_triggered": 0, "retries_leading_to_success": 0}
    planner_stats = {"total_requests": 0, "plans_generated": 0}
    expert_stats = _build_empty_expert_stats()
    planner_event_counts: dict[str, int] = collections.Counter()
    planner_action_counts: dict[str, int] = collections.Counter()
    planner_expert_counts: dict[str, int] = collections.Counter()
    if resume_state is not None and resume_state.get("transition_stats"):
        transition_stats.update(resume_state["transition_stats"])
    if resume_state is not None and resume_state.get("critic_stats"):
        critic_stats.update(resume_state["critic_stats"])
    if resume_state is not None and resume_state.get("expert_stats"):
        for key, values in resume_state["expert_stats"].items():
            if key in expert_stats and isinstance(values, dict):
                expert_stats[key].update(values)
    current_task_resume = resume_state.get("current_task") if resume_state is not None else None

    def _persist_progress(current_task: dict | None, status: str = "running") -> None:
        partial_results = _build_results_payload(
            args=args,
            ablation_tag=ablation_tag,
            total_successes=total_successes,
            total_episodes=total_episodes,
            episode_lengths=episode_lengths,
            success_episode_lengths=success_episode_lengths,
            task_metrics=task_metrics,
            transition_stats=transition_stats,
            critic_stats=critic_stats,
            planner_stats={
                **planner_stats,
                "event_counts": dict(planner_event_counts),
                "action_counts": dict(planner_action_counts),
                "expert_counts": dict(planner_expert_counts),
            },
            expert_stats=_serialize_expert_stats(expert_stats),
            status=status,
            current_task=current_task,
        )
        partial_summary_file.write_text(json.dumps(partial_results, indent=2), encoding="utf-8")
        resume_payload = {
            "task_suite": args.task_suite,
            "ablation_tag": ablation_tag,
            "completed_tasks": sorted(completed_tasks),
            "total_successes": total_successes,
            "total_episodes": total_episodes,
            "episode_lengths": episode_lengths,
            "success_episode_lengths": success_episode_lengths,
            "task_metrics": task_metrics,
            "transition_stats": transition_stats if args.transition else None,
            "critic_stats": critic_stats if args.critic else None,
            "expert_stats": _serialize_expert_stats(expert_stats),
            "current_task": current_task,
        }
        resume_state_file.write_text(json.dumps(resume_payload, indent=2), encoding="utf-8")

    for task_id in task_ids:
        task_key = str(task_id)
        if task_key in completed_tasks:
            logger.info("Task %s already completed, skipping", task_id)
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env = _make_env(task, get_libero_path, env_class, args.seed)
        task_description = task.language

        # A3: Graph RAG prompt augmentation
        task_control_profile = _build_task_control_profile(task_description)
        if transition_agent is not None:
            transition_agent.configure_for_task(task_control_profile)
        if critic_agent is not None:
            critic_agent.configure_for_task(task_control_profile)

        task_priors = {}
        effective_prompt = task_description
        if graph_rag_memory is not None:
            task_priors = graph_rag_memory.extract_priors_for_task(task_description)
            effective_prompt = graph_rag_memory.augment_prompt(task_description, task_priors)

        agentic_context = ""
        if task_priors:
            matched = task_priors.get("matched_keywords") if isinstance(task_priors.get("matched_keywords"), list) else []
            target_hints = task_priors.get("target_hints") if isinstance(task_priors.get("target_hints"), list) else []
            release_hints = task_priors.get("release_hints") if isinstance(task_priors.get("release_hints"), list) else []
            agentic_context = (
                f"priors.matched={','.join([str(x) for x in matched[:6]])}; "
                f"priors.target={';'.join([str(x) for x in target_hints[:3]])}; "
                f"priors.release={';'.join([str(x) for x in release_hints[:3]])}"
            ).strip()

        transition_prompt = _build_transition_prompt(task_description, task_priors)
        recovery_prompt = _build_recovery_prompt(task_description, task_priors)
        regrasp_prompt = _build_regrasp_prompt(task_description, task_priors)

        if args.vision_prompt:
            env.reset()
            logger.info("Task %s | obj_of_interest=%s | instance_to_id=%s",
                        task_id, env.obj_of_interest, env.instance_to_id)

        resume_for_task = current_task_resume if (
            current_task_resume is not None and int(current_task_resume.get("task_id", -1)) == task_id
        ) else None

        start_episode_idx = int(resume_for_task.get("next_episode_idx", 0)) if resume_for_task else 0
        task_successes = int(resume_for_task.get("task_successes", 0)) if resume_for_task else 0
        task_episode_lengths = list(resume_for_task.get("task_episode_lengths", [])) if resume_for_task else []
        task_transition_total = int(resume_for_task.get("task_transition_total", 0)) if resume_for_task else 0
        task_transition_episodes = int(resume_for_task.get("task_transition_episodes", 0)) if resume_for_task else 0
        task_transition_successes = int(resume_for_task.get("task_transition_successes", 0)) if resume_for_task else 0
        task_retry_total = int(resume_for_task.get("task_retry_total", 0)) if resume_for_task else 0
        task_retry_episodes = int(resume_for_task.get("task_retry_episodes", 0)) if resume_for_task else 0
        task_retry_successes = int(resume_for_task.get("task_retry_successes", 0)) if resume_for_task else 0
        task_critic_checks = int(resume_for_task.get("task_critic_checks", 0)) if resume_for_task else 0
        resumed_task_expert_stats = _materialize_expert_stats(
            resume_for_task.get("task_expert_stats") if resume_for_task else None
        )
        task_expert_stats_before = _clone_expert_stats(expert_stats)
        if resume_for_task:
            logger.info(
                "Resuming task %s from trial %d/%d",
                task_id,
                min(start_episode_idx + 1, args.trials),
                args.trials,
            )

        for episode_idx in range(start_episode_idx, args.trials):
            logger.info("Task %s | Trial %d/%d | %s | Prompt: %s",
                        task_id, episode_idx + 1, args.trials, task_description, effective_prompt)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
            action_plan = collections.deque()
            replay_images = []
            done = False
            current_prompt = effective_prompt
            episode_steps = 0
            agentic_action_scale = 1.0
            agentic_apply_steps = 0
            planner_cooldown_steps = 0
            current_subgoal = "execute"
            subgoal_success_window = 0
            expert_state = ExpertState(current_prompt=current_prompt)
            expert_state.reset_for_episode(current_prompt)
            _sync_expert_runtime_state(
                expert_state,
                current_prompt=current_prompt,
                current_subgoal=current_subgoal,
                agentic_action_scale=agentic_action_scale,
                agentic_apply_steps=agentic_apply_steps,
                planner_cooldown_steps=planner_cooldown_steps,
                subgoal_success_window=subgoal_success_window,
            )
            _record_expert_switch(expert_stats, expert_state.current_expert)

            if transition_agent is not None:
                transition_agent.reset()
            if planner_trigger_agent is not None:
                planner_trigger_agent.reset()
            if critic_agent is not None:
                critic_agent.reset()
            failure_taxonomy.reset()

            for t in range(max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    episode_steps += 1
                    if done:
                        break
                    continue
                if planner_cooldown_steps > 0:
                    planner_cooldown_steps -= 1
                if agentic_apply_steps <= 0 and abs(agentic_action_scale - 1.0) > 1e-6:
                    agentic_action_scale = 1.0
                if agentic_apply_steps <= 0 and current_subgoal != "execute":
                    current_subgoal = "execute"
                if subgoal_success_window > 0:
                    subgoal_success_window -= 1
                _sync_expert_runtime_state(
                    expert_state,
                    current_prompt=current_prompt,
                    current_subgoal=current_subgoal,
                    agentic_action_scale=agentic_action_scale,
                    agentic_apply_steps=agentic_apply_steps,
                    planner_cooldown_steps=planner_cooldown_steps,
                    subgoal_success_window=subgoal_success_window,
                )

                base_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Legacy Vision Prompt
                if args.vision_prompt:
                    seg_key = "agentview_segmentation_instance"
                    if seg_key in obs:
                        seg_image = obs[seg_key][::-1, ::-1]
                        base_img = _apply_vision_prompt(
                            base_img, seg_image, env.obj_of_interest,
                            env.instance_to_id, alpha=args.mask_alpha,
                        )

                base_img_p = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(base_img, args.resize_size, args.resize_size)
                )
                wrist_img_p = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                replay_images.append(base_img)

                # Record EE position for Transition detection
                ee_pos = obs["robot0_eef_pos"].copy()
                if transition_agent is not None:
                    transition_agent.record_ee_pos(ee_pos)
                if planner_trigger_agent is not None:
                    planner_trigger_agent.record_ee_pos(ee_pos)

                # A4: Critic check at intervals
                if (critic_agent is not None and t >= critic_agent.min_control_steps
                        and t % critic_agent.check_interval == 0 and not done):
                    critic_stats["total_checks"] += 1
                    task_critic_checks += 1
                    judgment = critic_agent.judge_subtask(base_img, task_description)
                    logger.info("[Critic] Step %d: judgment=%s", t, judgment)

                    if critic_agent.should_retry(judgment):
                        decision = _route_expert(
                            failure_event="critic",
                            reason=str(judgment),
                            expert_state=expert_state,
                            critic_stuck=bool(judgment.get("stuck", False)),
                            request_plan=bool(args.agentic_planner and planner_cooldown_steps <= 0),
                        )
                        _apply_router_decision(
                            expert_state,
                            decision,
                            expert_stats,
                            default_prompt=effective_prompt,
                        )
                        current_prompt = expert_state.current_prompt
                        current_subgoal = expert_state.current_subgoal
                        critic_stats["retries_triggered"] += 1
                        critic_agent.retry_count += 1
                        retry_prompt = critic_agent.get_retry_prompt(task_description, judgment)
                        logger.info("[Critic] Recovery + retry: %s", retry_prompt)
                        if args.agentic_planner and planner_cooldown_steps <= 0:
                            (
                                _requested,
                                planner_cooldown_steps,
                                current_subgoal,
                                current_prompt,
                                agentic_action_scale,
                                agentic_apply_steps,
                            ) = _request_planner(
                                client=client,
                                obs=obs,
                                base_img_p=base_img_p,
                                wrist_img_p=wrist_img_p,
                                prompt=current_prompt,
                                episode_id=f"{task_id}:{episode_idx}",
                                timestep=episode_steps,
                                event="critic",
                                reason=str(judgment),
                                context=f"{agentic_context}; subgoal={current_subgoal}",
                                planner_stats=planner_stats,
                                planner_event_counts=planner_event_counts,
                                planner_action_counts=planner_action_counts,
                                planner_expert_counts=planner_expert_counts,
                                expert_state=expert_state,
                                expert_stats=expert_stats,
                                default_prompt=effective_prompt,
                                current_cooldown_steps=planner_cooldown_steps,
                                cooldown_steps=int(args.planner_cooldown_steps),
                            )

                        # Execute recovery action
                        recovery_executor_prompt = (
                            regrasp_prompt if expert_state.current_expert == REGRASP_EXPERT else recovery_prompt
                        )
                        obs, recovery_steps, done = (
                            _execute_regrasp_expert(
                                env=env,
                                client=client,
                                obs=obs,
                                base_img_p=base_img_p,
                                wrist_img_p=wrist_img_p,
                                prompt=recovery_executor_prompt,
                            )
                            if expert_state.current_expert == REGRASP_EXPERT
                            else _execute_recovery_expert(
                                env=env,
                                client=client,
                                obs=obs,
                                base_img_p=base_img_p,
                                wrist_img_p=wrist_img_p,
                                prompt=recovery_executor_prompt,
                            )
                        )
                        episode_steps += recovery_steps

                        current_prompt = retry_prompt
                        expert_state.current_prompt = current_prompt
                        action_plan.clear()

                # A2: Transition Agent - detect State Gap
                if (transition_agent is not None and t >= task_control_profile["transition_min_steps"]
                        and not transition_agent.in_transition
                        and transition_agent.detect_state_gap()):
                    decision = _route_expert(
                        failure_event="state_gap",
                        reason="transition agent detected state gap",
                        expert_state=expert_state,
                        request_plan=bool(args.agentic_planner and planner_cooldown_steps <= 0),
                    )
                    _apply_router_decision(
                        expert_state,
                        decision,
                        expert_stats,
                        default_prompt=effective_prompt,
                    )
                    current_subgoal = expert_state.current_subgoal
                    transition_agent.trigger_transition()
                    transition_stats["total_transitions"] += 1
                    action_plan.clear()

                    # Execute transition via VLA prompt
                    obs, transition_steps, done = _execute_transition_expert(
                        env=env,
                        client=client,
                        obs=obs,
                        base_img_p=base_img_p,
                        wrist_img_p=wrist_img_p,
                        prompt=transition_prompt,
                    )
                    episode_steps += transition_steps

                    transition_agent.finish_transition()
                    verifier_decision = _verify_expert_progress(
                        expert_state,
                        failure_event="",
                        state_gap_active=False,
                    )
                    if verifier_decision.status == "resolved":
                        expert_stats["success_counts"][expert_state.current_expert] += 1
                        _apply_router_decision(
                            expert_state,
                            RouterDecision(
                                expert=DEFAULT_VLA_EXPERT,
                                trigger="verifier",
                                reason=verifier_decision.reason,
                                subgoal="execute",
                            ),
                            expert_stats,
                            default_prompt=effective_prompt,
                        )
                    current_prompt = effective_prompt
                    expert_state.current_prompt = current_prompt
                    current_subgoal = expert_state.current_subgoal
                    action_plan.clear()
                    if args.agentic_planner and planner_cooldown_steps <= 0:
                        (
                            _requested,
                            planner_cooldown_steps,
                            current_subgoal,
                            current_prompt,
                            agentic_action_scale,
                            agentic_apply_steps,
                        ) = _request_planner(
                            client=client,
                            obs=obs,
                            base_img_p=base_img_p,
                            wrist_img_p=wrist_img_p,
                            prompt=current_prompt,
                            episode_id=f"{task_id}:{episode_idx}",
                            timestep=episode_steps,
                            event="state_gap",
                            reason="transition agent detected state gap; provide next-step strategy",
                            context=f"{agentic_context}; subgoal={current_subgoal}",
                            planner_stats=planner_stats,
                            planner_event_counts=planner_event_counts,
                            planner_action_counts=planner_action_counts,
                            planner_expert_counts=planner_expert_counts,
                            expert_state=expert_state,
                            expert_stats=expert_stats,
                            default_prompt=effective_prompt,
                            current_cooldown_steps=planner_cooldown_steps,
                            cooldown_steps=int(args.planner_cooldown_steps),
                        )

                # Normal VLA inference
                if not action_plan:
                    agentic_req = None
                    if planner_trigger_agent is not None and t >= task_control_profile["transition_min_steps"]:
                        failure_event, failure_reason = failure_taxonomy.classify()
                        if failure_event and planner_cooldown_steps <= 0:
                            decision = _route_expert(
                                failure_event=failure_event,
                                reason=failure_reason,
                                expert_state=expert_state,
                                request_plan=True,
                            )
                            _apply_router_decision(
                                expert_state,
                                decision,
                                expert_stats,
                                default_prompt=effective_prompt,
                            )
                            current_subgoal = expert_state.current_subgoal
                            current_prompt = expert_state.current_prompt
                            planner_trigger_agent.trigger_transition()
                            agentic_req = {
                                "request_plan": True,
                                "event": failure_event,
                                "reason": failure_reason,
                                "context": f"{agentic_context}; subgoal={current_subgoal}",
                            }
                            planner_stats["total_requests"] += 1
                            planner_event_counts[failure_event] += 1
                            planner_cooldown_steps = max(planner_cooldown_steps, int(args.planner_cooldown_steps))
                    payload = _build_policy_payload(
                        obs=obs,
                        base_img_p=base_img_p,
                        wrist_img_p=wrist_img_p,
                        prompt=current_prompt,
                        episode_id=f"{task_id}:{episode_idx}",
                        timestep=episode_steps,
                        agentic_req=agentic_req,
                    )
                    try:
                        inference = _infer_with_retry(client, payload, max_retries=5)
                    except Exception as infer_err:
                        logger.warning("Inference failed: %s — restarting server", infer_err)
                        try:
                            _restart_policy_server(OPENPI_ROOT, args.port)
                            client = _reconnect_client(args.host, args.port)
                            inference = _infer_with_retry(client, payload, max_retries=5)
                        except Exception as reconnect_err:
                            logger.error("Server restart failed: %s — aborting episode", reconnect_err)
                            break
                    if "actions" not in inference:
                        raise RuntimeError(f"Malformed response: {inference}")
                    if isinstance(inference.get("agentic"), dict):
                        agentic_resp = inference["agentic"]
                        (
                            current_subgoal,
                            current_prompt,
                            agentic_action_scale,
                            agentic_apply_steps,
                        ) = _handle_agentic_response(
                            agentic_resp,
                            expert_state=expert_state,
                            planner_stats=planner_stats,
                            planner_action_counts=planner_action_counts,
                            planner_expert_counts=planner_expert_counts,
                            expert_stats=expert_stats,
                            default_prompt=effective_prompt,
                        )
                        if agentic_apply_steps > 0:
                            planner_cooldown_steps = max(planner_cooldown_steps, agentic_apply_steps)
                    action_chunk = inference["actions"]
                    if len(action_chunk) < args.replan_steps:
                        raise RuntimeError(
                            f"Policy predicted {len(action_chunk)} steps, need {args.replan_steps}."
                        )
                    action_plan.extend(action_chunk[:args.replan_steps])

                action = action_plan.popleft()
                if agentic_apply_steps > 0 and abs(agentic_action_scale - 1.0) > 1e-6:
                    a = np.asarray(action, dtype=np.float32).copy()
                    a[:6] *= agentic_action_scale
                    action = a
                    agentic_apply_steps -= 1
                failure_taxonomy.update(
                    ee_pos=obs["robot0_eef_pos"],
                    action=np.asarray(action, dtype=np.float32),
                    gripper_qpos=np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                )
                if current_subgoal != "execute":
                    fe, _ = failure_taxonomy.classify()
                    if not fe:
                        subgoal_success_window += 1
                    else:
                        subgoal_success_window = 0
                    expert_state.subgoal_success_window = subgoal_success_window
                    if subgoal_success_window >= 6:
                        verifier_decision = _verify_expert_progress(
                            expert_state,
                            failure_event=fe,
                            state_gap_active=False,
                        )
                        if verifier_decision.status == "resolved":
                            expert_stats["success_counts"][expert_state.current_expert] += 1
                            _apply_router_decision(
                                expert_state,
                                RouterDecision(
                                    expert=DEFAULT_VLA_EXPERT,
                                    trigger="verifier",
                                    reason=verifier_decision.reason,
                                    subgoal="execute",
                                ),
                                expert_stats,
                                default_prompt=effective_prompt,
                            )
                            current_prompt = expert_state.current_prompt
                            current_subgoal = expert_state.current_subgoal
                        elif verifier_decision.status == "escalate":
                            expert_stats["escalation_counts"][expert_state.current_expert] += 1
                        elif verifier_decision.status == "replan":
                            expert_stats["replan_counts"][expert_state.current_expert] += 1
                _sync_expert_runtime_state(
                    expert_state,
                    current_prompt=current_prompt,
                    current_subgoal=current_subgoal,
                    agentic_action_scale=agentic_action_scale,
                    agentic_apply_steps=agentic_apply_steps,
                    planner_cooldown_steps=planner_cooldown_steps,
                    subgoal_success_window=subgoal_success_window,
                )
                obs, _, done, _ = env.step(action.tolist())
                episode_steps += 1
                expert_state.steps_in_expert += 1
                if done:
                    break

            total_episodes += 1
            episode_lengths.append(episode_steps)
            task_episode_lengths.append(episode_steps)
            episode_had_transition = transition_agent is not None and transition_agent.transition_count > 0
            episode_had_retry = critic_agent is not None and critic_agent.retry_count > 0
            if episode_had_transition:
                task_transition_total += transition_agent.transition_count
                task_transition_episodes += 1
            if episode_had_retry:
                task_retry_total += critic_agent.retry_count
                task_retry_episodes += 1
            if done:
                task_successes += 1
                total_successes += 1
                success_episode_lengths.append(episode_steps)
                if graph_rag_memory is not None:
                    graph_rag_memory.record_success(task_description, task_priors)
                if episode_had_transition:
                    transition_stats["transitions_leading_to_success"] += 1
                    task_transition_successes += 1
                if episode_had_retry:
                    critic_stats["retries_leading_to_success"] += 1
                    task_retry_successes += 1

            suffix = "success" if done else "failure"
            safe_task = task_description.replace(" ", "_").replace("/", "_")
            video_path = pathlib.Path(args.video_dir) / f"task{task_id}_trial{episode_idx}_{suffix}_{safe_task}.mp4"
            if replay_images:
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
            logger.info("Episode done=%s | running success=%.3f", done, total_successes / total_episodes)
            task_expert_stats_running = _serialize_expert_stats(
                _merge_expert_stats(
                    resumed_task_expert_stats,
                    _diff_expert_stats(expert_stats, task_expert_stats_before),
                )
            )
            _persist_progress(
                current_task={
                    "task_id": task_id,
                    "task_description": task_description,
                    "next_episode_idx": episode_idx + 1,
                    "task_successes": task_successes,
                    "task_episode_lengths": task_episode_lengths,
                    "task_transition_total": task_transition_total,
                    "task_transition_episodes": task_transition_episodes,
                    "task_transition_successes": task_transition_successes,
                    "task_retry_total": task_retry_total,
                    "task_retry_episodes": task_retry_episodes,
                    "task_retry_successes": task_retry_successes,
                    "task_critic_checks": task_critic_checks,
                    "task_expert_stats": task_expert_stats_running,
                },
                status="running",
            )

        task_expert_stats = _serialize_expert_stats(
            _merge_expert_stats(
                resumed_task_expert_stats,
                _diff_expert_stats(expert_stats, task_expert_stats_before),
            )
        )
        task_metrics[task_key] = {
            "success_rate": task_successes / args.trials,
            "episodes": float(args.trials),
            "avg_episode_length": float(np.mean(task_episode_lengths)) if task_episode_lengths else 0.0,
            "transition_total": int(task_transition_total) if args.transition else 0,
            "transition_episode_count": int(task_transition_episodes) if args.transition else 0,
            "transition_success_count": int(task_transition_successes) if args.transition else 0,
            "critic_checks": int(task_critic_checks) if args.critic else 0,
            "retry_total": int(task_retry_total) if args.critic else 0,
            "retry_episode_count": int(task_retry_episodes) if args.critic else 0,
            "retry_success_count": int(task_retry_successes) if args.critic else 0,
            "expert_stats": task_expert_stats,
        }
        logger.info("Task %s success rate: %.3f", task_id, task_metrics[task_key]["success_rate"])
        completed_tasks.add(task_key)
        completed_tasks_file.write_text(json.dumps(sorted(completed_tasks)), encoding="utf-8")
        current_task_resume = None
        _persist_progress(current_task=None, status="running")

    overall = total_successes / total_episodes if total_episodes else 0.0
    logger.info("Overall success rate: %.3f (%d/%d)", overall, total_successes, total_episodes)

    results = _build_results_payload(
        args=args,
        ablation_tag=ablation_tag,
        total_successes=total_successes,
        total_episodes=total_episodes,
        episode_lengths=episode_lengths,
        success_episode_lengths=success_episode_lengths,
        task_metrics=task_metrics,
        transition_stats=transition_stats,
        critic_stats=critic_stats,
        planner_stats={
            **planner_stats,
            "event_counts": dict(planner_event_counts),
            "action_counts": dict(planner_action_counts),
            "expert_counts": dict(planner_expert_counts),
        },
        expert_stats=_serialize_expert_stats(expert_stats),
        status="completed",
        current_task=None,
    )
    if args.vision_prompt:
        results["mask_alpha"] = args.mask_alpha

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    partial_summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    resume_state_file.write_text(json.dumps({
        "task_suite": args.task_suite,
        "ablation_tag": ablation_tag,
        "evaluated_task_ids": getattr(args, "evaluated_task_ids", None),
        "skipped_task_ids": getattr(args, "skipped_task_ids", None),
        "completed_tasks": sorted(completed_tasks),
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "episode_lengths": episode_lengths,
        "success_episode_lengths": success_episode_lengths,
        "task_metrics": task_metrics,
        "transition_stats": transition_stats if args.transition else None,
        "critic_stats": critic_stats if args.critic else None,
        "expert_stats": _serialize_expert_stats(expert_stats),
        "current_task": None,
        "status": "completed",
    }, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", results_path)

    # Unload Critic model if loaded
    if critic_agent is not None:
        critic_agent.unload_model()

    return results


if __name__ == "__main__":
    args = _parse_args()
    try:
        evaluate_real_libero(args)
    except Exception as exc:
        logger.error("Real LIBERO evaluation aborted: %s", exc)
        raise SystemExit(1)
