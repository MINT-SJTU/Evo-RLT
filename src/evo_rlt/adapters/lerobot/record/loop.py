# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core recording loop used by `lerobot_record.py`."""

import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import torch

from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.robots import Robot
from evo_rlt.adapters.lerobot.record.hil import (
    INTERVENTION_STATE_ACTIVE,
    INTERVENTION_STATE_POLICY,
    INTERVENTION_STATE_RELEASE,
    ACPInferenceConfig,
    PolicySyncDualArmExecutor,
    _capture_policy_runtime_state,
    set_teleop_manual_control as apply_teleop_manual_control,
    _predict_policy_action_with_acp_inference,
)
from evo_rlt.adapters.lerobot.record.vla_rtc import reset_vla_only_rtc_runtimes
from lerobot.teleoperators import Teleoperator, koch_leader, omx_leader, so_leader
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from evo_rlt.adapters.lerobot.record.annotations import (
    COLLECTOR_HUMAN,
    COLLECTOR_POLICY,
    EPISODE_FAILURE,
    EPISODE_SUCCESS,
    PHASE_CRITICAL,
    PHASE_PREFIX,
    SOURCE_HUMAN,
    SOURCE_VLA,
    resolve_collector_policy_id,
    resolve_rlt_collector_policy_id,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import log_rerun_data

T = TypeVar("T")


def _clone_robot_action(action: RobotAction) -> RobotAction:
    cloned: RobotAction = {}
    for key, value in action.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = value
    return cloned


def _blend_robot_actions(
    action_feature_names: list[str],
    start_action: RobotAction,
    target_action: RobotAction,
    alpha: float,
) -> RobotAction:
    clipped_alpha = min(max(alpha, 0.0), 1.0)
    blended: RobotAction = {}
    for name in action_feature_names:
        start_value = start_action.get(name)
        target_value = target_action.get(name)
        if start_value is None:
            blended[name] = target_value
            continue
        if target_value is None:
            blended[name] = start_value
            continue

        start_array = np.asarray(start_value, dtype=np.float32)
        target_array = np.asarray(target_value, dtype=np.float32)
        blended_value = (1.0 - clipped_alpha) * start_array + clipped_alpha * target_array
        if blended_value.shape == ():
            blended[name] = float(blended_value)
        else:
            blended[name] = blended_value.astype(np.float32)
    return blended


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


def _validate_policy_image_features(
    policy: PreTrainedPolicy, dataset_features: dict[str, dict]
) -> None:
    """Check that dataset features include all image features the policy expects.

    Raises a clear error if images are missing - the most common cause is
    `--dataset.video=false` which silently drops all image features from the
    dataset due to an upstream lerobot limitation in
    `aggregate_pipeline_dataset_features`.
    """
    policy_image_keys = [
        k for k, ft in policy.config.input_features.items()
        if ft.type.value == "VISUAL"
    ]
    if not policy_image_keys:
        return

    ds_image_keys = [
        k for k, ft in dataset_features.items()
        if ft.get("dtype") in ("image", "video")
    ]
    missing = [k for k in policy_image_keys if k not in ds_image_keys]
    if not missing:
        return

    hint = (
        "This usually means --dataset.video=false was set, which disables ALL "
        "image features in the dataset (upstream lerobot limitation). "
        "Set --dataset.video=true (the default) to fix this."
    )
    if ds_image_keys:
        hint += (
            f"\n  Dataset has images: {ds_image_keys}"
            f"\n  Policy expects:     {policy_image_keys}"
            "\n  Check camera naming - BiSOFollower auto-prepends left_/right_ "
            "to each arm's camera names."
        )
    raise ValueError(
        f"Policy expects image features {missing} but they are not in "
        f"the dataset features. {hint}"
    )


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    policy_sync_executor: PolicySyncDualArmExecutor | None = None,
    intervention_state_machine_enabled: bool = True,
    collector_policy_id_policy: int = COLLECTOR_POLICY,
    collector_policy_id_human: int = COLLECTOR_HUMAN,
    acp_inference: ACPInferenceConfig | None = None,
    communication_retry_timeout_s: float = 2.0,
    communication_retry_interval_s: float = 0.1,
    rlt_online_collector: Any | None = None,
    critical_phase_tracker: Any | None = None,
    rlt_intervention_tracker: Any | None = None,
    skip_prefix_recording: bool = False,
    rl_phase_key_toggles_episode: bool = False,
    rl_phase_key_toggles_critical_phase: bool = False,
    rl_phase_double_tap_window_s: float = 1.0,
    start_in_teleop: bool = False,
    intervention_action_blend_time_s: float = 0.0,
):
    if intervention_action_blend_time_s < 0:
        raise ValueError("intervention_action_blend_time_s must be >= 0")
    if acp_inference is None:
        acp_inference = ACPInferenceConfig()

    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    if dataset is None and policy is not None:
        raise ValueError("Policy-driven recording requires a dataset for feature mapping.")

    # Early check: verify dataset features include all image features the policy expects.
    if policy is not None and dataset is not None:
        _validate_policy_image_features(policy, dataset.features)

    action_feature_names = dataset.features[ACTION]["names"] if dataset is not None else None
    if action_feature_names is None:
        if hasattr(robot.action_features, "keys"):
            action_feature_names = list(robot.action_features.keys())
        else:
            action_feature_names = list(robot.action_features)
    zero_policy_action = dict.fromkeys(action_feature_names, 0.0)
    has_teleop = isinstance(teleop, (Teleoperator, list))
    # Duck-type RLT phase control: if policy has set_rl_mode, it's an RLT policy
    rlt = policy if policy is not None and hasattr(policy, "set_rl_mode") else None
    has_autonomous_source = policy is not None
    intervention_enabled = intervention_state_machine_enabled and has_autonomous_source and has_teleop
    # start_in_teleop: episode begins in human-teleop mode (no policy actions
    # are sent to the robot) until the user presses r to enter RL. Used by
    # the wo_prefix HIL recorder where VLA should never drive.
    if start_in_teleop and intervention_enabled:
        intervention_state = INTERVENTION_STATE_ACTIVE
    else:
        intervention_state = INTERVENTION_STATE_POLICY
    last_teleop_action: RobotAction | None = None
    last_policy_action_for_blend: RobotAction | None = None
    intervention_blend_start_t: float | None = None
    intervention_blend_start_action: RobotAction | None = None
    teleop_fallback_warned = False

    teleop_arm_for_mode_switch: Any | None = None
    if isinstance(teleop, Teleoperator):
        teleop_arm_for_mode_switch = teleop
    elif isinstance(teleop, list):
        teleop_arm_for_mode_switch = teleop_arm

    def set_teleop_manual_control(enabled: bool) -> None:
        if teleop_arm_for_mode_switch is not None:
            apply_teleop_manual_control(teleop_arm_for_mode_switch, enabled)

    if policy is None:
        # During reset/teleop-only loops keep leader backdrivable for manual dragging.
        set_teleop_manual_control(True)

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        reset_vla_only_rtc_runtimes(policy)

    cond_policy_runtime_state: dict[str, Any] | None = None
    uncond_policy_runtime_state: dict[str, Any] | None = None
    if policy is not None and acp_inference.enable and acp_inference.use_cfg:
        cond_policy_runtime_state = _capture_policy_runtime_state(policy)
        uncond_policy_runtime_state = _capture_policy_runtime_state(policy)

    if intervention_enabled:
        if intervention_state == INTERVENTION_STATE_ACTIVE:
            # start_in_teleop mode: leader is backdrivable, follower mirrors leader.
            set_teleop_manual_control(True)
        else:
            # S0: policy drives both arms, teleop arm should accept feedback commands.
            set_teleop_manual_control(False)

    def run_with_connection_retry(action_name: str, fn: Callable[[], T]) -> T:
        timeout_s = max(communication_retry_timeout_s, 0.0)
        interval_s = max(communication_retry_interval_s, 0.0)
        deadline_t = time.perf_counter() + timeout_s
        attempts = 0
        first_error: ConnectionError | None = None

        while True:
            attempts += 1
            try:
                result = fn()
                if attempts > 1:
                    elapsed_s = timeout_s - max(deadline_t - time.perf_counter(), 0.0)
                    logging.warning(
                        "%s recovered after %d retries in %.2fs.",
                        action_name,
                        attempts - 1,
                        elapsed_s,
                    )
                return result
            except ConnectionError as error:
                if first_error is None:
                    first_error = error
                    logging.warning(
                        "%s failed with transient communication error; retrying for up to %.2fs (%s)",
                        action_name,
                        timeout_s,
                        error,
                    )

                if timeout_s <= 0.0:
                    raise

                remaining_s = deadline_t - time.perf_counter()
                if remaining_s <= 0.0:
                    raise

                sleep_s = interval_s if interval_s > 0.0 else remaining_s
                time.sleep(min(sleep_s, remaining_s))

    def build_action_tensor(values: RobotAction) -> torch.Tensor:
        return torch.tensor(
            [float(np.asarray(values[name]).reshape(-1)[0]) for name in action_feature_names],
            dtype=torch.float32,
        )

    # Open sidecar JSONL for crash-recovery (state/action per frame)
    _recovery_fh = None
    _frame_counter = 0
    if dataset is not None and hasattr(dataset, "root") and dataset.root is not None:
        _recovery_path = dataset.root / "recovery_frames.jsonl"
        _recovery_fh = open(_recovery_path, "a")  # noqa: SIM115

    def _is_image_key(key: str) -> bool:
        return "image" in key or (dataset is not None and key in dataset.features
                                  and dataset.features[key].get("dtype") in ("video", "image"))

    timestamp = 0
    start_episode_t = time.perf_counter()
    prev_phase = PHASE_PREFIX
    rl_phase_started = False
    pending_end_press_time: float | None = None
    final_outcome: str | None = None
    _frame_idx = 0
    _cuda_cleanup_interval = 500  # defrag CUDA allocator every N frames

    def get_episode_frame_index() -> int:
        if dataset is None or dataset.episode_buffer is None:
            return 0
        return dataset.episode_buffer["size"]

    def _start_intervention() -> None:
        nonlocal intervention_state, intervention_blend_start_t, intervention_blend_start_action
        nonlocal pending_end_press_time
        intervention_state = INTERVENTION_STATE_ACTIVE
        set_teleop_manual_control(True)
        if rlt_intervention_tracker is not None:
            rlt_intervention_tracker.start(get_episode_frame_index())
        if intervention_action_blend_time_s > 0 and last_policy_action_for_blend is not None:
            intervention_blend_start_t = time.perf_counter()
            intervention_blend_start_action = _clone_robot_action(last_policy_action_for_blend)
            logging.info("Intervention action blend started for %.2fs.", intervention_action_blend_time_s)
        else:
            intervention_blend_start_t = None
            intervention_blend_start_action = None
        if rlt is not None:
            rlt.interrupt_chunk()
            log_say("intervene", play_sounds=True)
        pending_end_press_time = None
        logging.info("Intervention enabled (S1): teleop actions now override policy execution.")

    def _reset_policy_after_intervention_release() -> None:
        nonlocal cond_policy_runtime_state, uncond_policy_runtime_state
        if policy is None or preprocessor is None or postprocessor is None:
            return
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        reset_vla_only_rtc_runtimes(policy)
        if acp_inference.enable and acp_inference.use_cfg:
            cond_policy_runtime_state = _capture_policy_runtime_state(policy)
            uncond_policy_runtime_state = _capture_policy_runtime_state(policy)
        logging.info("Policy cache reset on release: next policy action is recomputed.")

    def _release_intervention() -> None:
        nonlocal intervention_state, intervention_blend_start_t, intervention_blend_start_action
        if rlt_intervention_tracker is not None:
            rlt_intervention_tracker.stop(get_episode_frame_index())
        intervention_state = INTERVENTION_STATE_RELEASE
        intervention_blend_start_t = None
        intervention_blend_start_action = None
        set_teleop_manual_control(False)
        _reset_policy_after_intervention_release()
        if rlt is not None:
            if rl_phase_started:
                rlt.set_rl_mode()
            else:
                rlt.interrupt_chunk()
            log_say("resume", play_sounds=True)
            logging.info("RLT chunk interrupted on release: next action recomputed.")
        logging.info("Intervention release requested (S2): returning control to policy.")

    def _handle_intervention_toggle() -> None:
        if not events.get("toggle_intervention", False):
            return
        events["toggle_intervention"] = False
        if not intervention_enabled:
            logging.info("Intervention toggle ignored because policy+teleop are not both active.")
            return
        if intervention_state == INTERVENTION_STATE_POLICY:
            _start_intervention()
            return
        _release_intervention()

    def _handle_critical_phase_events() -> None:
        if events.get("toggle_critical_phase", False):
            events["toggle_critical_phase"] = False
            if rlt is not None:
                rlt.trigger_critical_phase()
            if critical_phase_tracker is not None and dataset is not None:
                critical_phase_tracker.toggle(dataset.episode_buffer["size"])
                if critical_phase_tracker.is_active:
                    from lerobot.utils.audio_feedback import say_start
                    say_start()
        if events.get("cp_mark_success", False):
            events["cp_mark_success"] = False
            if critical_phase_tracker is not None and dataset is not None:
                critical_phase_tracker.mark_success(dataset.episode_buffer["size"])
                from lerobot.utils.audio_feedback import say_success
                say_success()
        if events.get("cp_mark_failure", False):
            events["cp_mark_failure"] = False
            if critical_phase_tracker is not None and dataset is not None:
                critical_phase_tracker.mark_failure(dataset.episode_buffer["size"])
                from lerobot.utils.audio_feedback import say_failure
                say_failure()

    def _finish_active_rl_phase(toggles_episode: bool, toggles_cp: bool) -> None:
        nonlocal final_outcome, pending_end_press_time, rl_phase_started
        if pending_end_press_time is None:
            pending_end_press_time = time.perf_counter()
            log_say("RL end", play_sounds=True)
            logging.info("RL end pending - tap r again within %.1fs to mark failure", rl_phase_double_tap_window_s)
            return
        if toggles_episode:
            final_outcome = EPISODE_FAILURE
            events["exit_early"] = True
        elif toggles_cp:
            if rlt is not None:
                rlt.set_vla_mode()
            if critical_phase_tracker is not None and dataset is not None:
                critical_phase_tracker.mark_failure(dataset.episode_buffer["size"])
            rl_phase_started = False
        pending_end_press_time = None
        log_say("failure", play_sounds=True)
        logging.info("RL phase ended via double-tap (failure)")

    def _start_rl_phase_from_key() -> None:
        nonlocal intervention_state, intervention_blend_start_t, intervention_blend_start_action
        nonlocal rl_phase_started, pending_end_press_time
        if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
            intervention_state = INTERVENTION_STATE_RELEASE
            intervention_blend_start_t = None
            intervention_blend_start_action = None
            set_teleop_manual_control(False)
        if rlt is not None:
            rlt.set_rl_mode()
        if critical_phase_tracker is not None and dataset is not None:
            critical_phase_tracker.toggle(dataset.episode_buffer["size"])
        rl_phase_started = True
        pending_end_press_time = None
        log_say("RL start", play_sounds=True)
        logging.info("RL phase started (r key)")

    def _handle_rl_phase_start_event() -> None:
        if not events.get("start_rl_phase", False):
            return
        events["start_rl_phase"] = False
        r_key_active = rlt is not None or rl_phase_key_toggles_episode or rl_phase_key_toggles_critical_phase
        active_intervention = intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE
        if rl_phase_started and active_intervention:
            logging.info("Ignoring r key: human intervention is active")
            return
        if not r_key_active:
            return
        toggles_episode = rl_phase_key_toggles_episode
        toggles_cp = rl_phase_key_toggles_critical_phase
        if (toggles_episode or toggles_cp) and rl_phase_started:
            _finish_active_rl_phase(toggles_episode, toggles_cp)
            return
        _start_rl_phase_from_key()

    def _resolve_pending_rl_phase_end() -> None:
        nonlocal final_outcome, pending_end_press_time, rl_phase_started
        if pending_end_press_time is None or final_outcome is not None:
            return
        if (time.perf_counter() - pending_end_press_time) < rl_phase_double_tap_window_s:
            return
        if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
            return
        if rl_phase_key_toggles_episode:
            final_outcome = EPISODE_SUCCESS
            events["exit_early"] = True
        elif rl_phase_key_toggles_critical_phase and rlt is not None:
            rlt.set_vla_mode()
            if critical_phase_tracker is not None and dataset is not None:
                critical_phase_tracker.mark_success(dataset.episode_buffer["size"])
            rl_phase_started = False
        pending_end_press_time = None
        log_say("success", play_sounds=True)
        logging.info("RL phase ended via single press (success)")

    def _release_active_intervention_after_phase_end() -> None:
        nonlocal intervention_state, intervention_blend_start_t, intervention_blend_start_action
        if not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE):
            return
        if rlt_intervention_tracker is not None:
            rlt_intervention_tracker.stop(get_episode_frame_index())
        intervention_state = INTERVENTION_STATE_RELEASE
        intervention_blend_start_t = None
        intervention_blend_start_action = None
        set_teleop_manual_control(False)

    def _handle_end_phase_event(event_name: str, outcome: str) -> None:
        if not events.get(event_name, False):
            return
        events[event_name] = False
        if rlt is None:
            return
        rlt.set_vla_mode()
        if critical_phase_tracker is not None and dataset is not None:
            marker = critical_phase_tracker.mark_success if outcome == EPISODE_SUCCESS else critical_phase_tracker.mark_failure
            marker(dataset.episode_buffer["size"])
        _release_active_intervention_after_phase_end()
        log_say(outcome, play_sounds=True)
        logging.info("RL phase ended (%s)", outcome)

    def _select_action_values(
        act_processed_policy: RobotAction | None,
        act_processed_teleop: RobotAction | None,
    ) -> tuple[float, RobotAction]:
        nonlocal teleop_fallback_warned
        if not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE):
            action = act_processed_policy if act_processed_policy is not None else act_processed_teleop
            return 0.0, action
        if act_processed_teleop is not None:
            return 1.0, act_processed_teleop
        if last_teleop_action is not None:
            if not teleop_fallback_warned:
                logging.warning("Intervention is active but no fresh teleop action is available; reusing last teleop action.")
                teleop_fallback_warned = True
            return 1.0, last_teleop_action
        if act_processed_policy is not None:
            if not teleop_fallback_warned:
                logging.warning("Intervention is active but teleop action is unavailable; falling back to policy action.")
                teleop_fallback_warned = True
            return 1.0, act_processed_policy
        if not teleop_fallback_warned:
            logging.warning("Intervention is active but no teleop/policy action is available; sending zero action.")
            teleop_fallback_warned = True
        return 1.0, zero_policy_action

    def _apply_intervention_blend(is_intervention: float, action_values: RobotAction) -> RobotAction:
        nonlocal intervention_blend_start_t, intervention_blend_start_action
        if not is_intervention or intervention_blend_start_t is None or intervention_blend_start_action is None:
            return action_values
        elapsed_s = time.perf_counter() - intervention_blend_start_t
        alpha = min(elapsed_s / intervention_action_blend_time_s, 1.0)
        if alpha < 1.0:
            return _blend_robot_actions(action_feature_names, intervention_blend_start_action, action_values, alpha)
        intervention_blend_start_t = None
        intervention_blend_start_action = None
        return action_values

    def _write_recovery_row(frame: dict[str, Any]) -> None:
        nonlocal _frame_counter
        if _recovery_fh is None:
            return
        recovery_row = {}
        for key, value in frame.items():
            if _is_image_key(key) or key == "task":
                continue
            if isinstance(value, np.ndarray):
                recovery_row[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool)):
                recovery_row[key] = value
        _recovery_fh.write(json.dumps(recovery_row) + "\n")
        _recovery_fh.flush()
        _frame_counter += 1

    def _collector_policy_code(
        is_intervention: float,
        selected_from_policy: bool,
        rlt_source: float,
    ) -> int:
        if rlt is not None:
            return resolve_rlt_collector_policy_id(
                is_intervention=bool(is_intervention),
                source_type=rlt_source,
            )
        return resolve_collector_policy_id(
            intervention_enabled=intervention_enabled,
            is_intervention=bool(is_intervention),
            selected_from_policy=selected_from_policy,
            policy_id=collector_policy_id_policy,
            human_id=collector_policy_id_human,
        )

    # Per-frame timing instrumentation -> /tmp/frame_timing.csv
    _perf_fh = open("/tmp/frame_timing.csv", "w")  # noqa: SIM115
    _perf_fh.write("frame,total_ms,obs_ms,infer_ms,send_ms,dataset_ms,sleep_ms\n")

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        _t_infer = 0.0  # only set on inference frames
        _t_send = 0.0
        _t_dataset = 0.0

        if events["exit_early"]:
            events["exit_early"] = False
            break

        _handle_intervention_toggle()
        _handle_critical_phase_events()
        _handle_rl_phase_start_event()
        _resolve_pending_rl_phase_end()
        _handle_end_phase_event("end_phase_success", EPISODE_SUCCESS)
        _handle_end_phase_event("end_phase_failure", EPISODE_FAILURE)

        # Get robot observation
        _t0 = time.perf_counter()
        obs = robot.get_observation()
        _t_obs = (time.perf_counter() - _t0) * 1000

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from policy and/or teleop
        act_processed_policy: RobotAction | None = None
        act_processed_teleop: RobotAction | None = None
        if (
            policy is not None
            and preprocessor is not None
            and postprocessor is not None
            and not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE)
        ):
            _t0 = time.perf_counter()
            policy_action = _predict_policy_action_with_acp_inference(
                observation_frame=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
                acp_inference=acp_inference,
                cond_runtime_state=cond_policy_runtime_state,
                uncond_runtime_state=uncond_policy_runtime_state,
                fps=fps,
            )
            _t_infer = (time.perf_counter() - _t0) * 1000
            act_processed_policy = make_robot_action(policy_action, dataset.features)

        if isinstance(teleop, Teleoperator):
            act = run_with_connection_retry("teleop.get_action", teleop.get_action)

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        elif isinstance(teleop, list):
            arm_action = run_with_connection_retry("teleop_arm.get_action", teleop_arm.get_action)
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))

        if act_processed_policy is None and act_processed_teleop is None:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        if act_processed_teleop is not None:
            last_teleop_action = act_processed_teleop
            teleop_fallback_warned = False

        policy_action_for_storage = (
            act_processed_policy if act_processed_policy is not None else zero_policy_action
        )

        is_intervention, action_values = _select_action_values(act_processed_policy, act_processed_teleop)
        action_values = _apply_intervention_blend(is_intervention, action_values)

        # Applies a pipeline to the action, default is IdentityProcessor
        robot_action_to_send = robot_action_processor((action_values, obs))

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        selected_from_policy = act_processed_policy is not None and action_values is act_processed_policy
        if selected_from_policy:
            last_policy_action_for_blend = _clone_robot_action(action_values)
        _t0 = time.perf_counter()
        if policy_sync_executor is not None and selected_from_policy:
            _sent_action = run_with_connection_retry(
                "policy_sync_executor.send_action",
                lambda robot_action_to_send=robot_action_to_send: policy_sync_executor.send_action(
                    robot_action_to_send
                ),
            )
        else:
            _sent_action = run_with_connection_retry(
                "robot.send_action",
                lambda robot_action_to_send=robot_action_to_send: robot.send_action(robot_action_to_send),
            )
        _t_send = (time.perf_counter() - _t0) * 1000

        # Compute RLT metadata for both dataset writing and online collector.
        # Only pop metadata when policy action was actually executed (not during intervention)
        # to keep _meta_queue in sync with _action_queue.
        rlt_meta = None
        if rlt is not None and not is_intervention:
            rlt_meta = rlt.pop_step_metadata()

        if rlt is None and skip_prefix_recording:
            # Pure-teleop mode with r-key-driven episode boundaries: derive
            # the phase gate from rl_phase_started so skip_prefix_recording
            # drops pre-r frames even though no rlt policy is emitting
            # per-step phase metadata.
            rlt_phase = PHASE_CRITICAL if rl_phase_started else PHASE_PREFIX
        else:
            rlt_phase = rlt_meta.phase if rlt_meta is not None else prev_phase
        rlt_source = SOURCE_HUMAN if is_intervention else (rlt_meta.source_type if rlt_meta else SOURCE_VLA)
        rlt_is_critical = float(rlt_phase == PHASE_CRITICAL)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            policy_action_frame = build_dataset_frame(
                dataset.features, policy_action_for_storage, prefix="complementary_info.policy_action"
            )
            frame = {**observation_frame, **action_frame, **policy_action_frame, "task": single_task}

            if "complementary_info.is_intervention" in dataset.features:
                frame["complementary_info.is_intervention"] = np.array([is_intervention], dtype=np.float32)
            if "complementary_info.state" in dataset.features:
                frame["complementary_info.state"] = np.array([intervention_state], dtype=np.float32)
            if "complementary_info.collector_policy_id" in dataset.features:
                collector_code = _collector_policy_code(is_intervention, selected_from_policy, rlt_source)
                frame["complementary_info.collector_policy_id"] = np.array([collector_code], dtype=np.int64)
            if "complementary_info.phase" in dataset.features:
                frame["complementary_info.phase"] = np.array([rlt_phase], dtype=np.float32)
            prev_phase = rlt_phase
            skip_frame = skip_prefix_recording and rlt_phase == PHASE_PREFIX
            if not skip_frame:
                _t0 = time.perf_counter()
                dataset.add_frame(frame)
                _t_dataset = (time.perf_counter() - _t0) * 1000

                _write_recovery_row(frame)

        if rlt_online_collector is not None:
            action_tensor = build_action_tensor(action_values)
            rlt_online_collector.on_frame(
                action=action_tensor,
                state_vec=None,
                ref_chunk=None,
                source_type=rlt_source,
                is_critical=rlt_is_critical,
            )

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        if intervention_state == INTERVENTION_STATE_RELEASE:
            intervention_state = INTERVENTION_STATE_POLICY

        # Periodically defragment CUDA allocator and trigger Python GC to prevent
        # progressive inference slowdown from allocator fragmentation + GC pressure.
        # (KV cache alloc/dealloc every n_action_steps fragments the CUDA free list;
        # episode_buffer accumulates ~20 numpy arrays/frame -> 250K+ objects by 10 min.)
        _frame_idx += 1
        if policy is not None and _frame_idx % _cuda_cleanup_interval == 0:
            torch.cuda.empty_cache()

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))
        _t_total = (time.perf_counter() - start_loop_t) * 1000
        _t_sleep = _t_total - dt_s * 1000
        _perf_fh.write(
            f"{_frame_idx},{_t_total:.1f},{_t_obs:.1f},{_t_infer:.1f},"
            f"{_t_send:.1f},{_t_dataset:.1f},{_t_sleep:.1f}\n"
        )
        if _frame_idx % 100 == 0:
            _perf_fh.flush()

        timestamp = time.perf_counter() - start_episode_t

    # Finalize toggle-mode episode end: stop intervention, switch RLT back to
    # VLA mode, and tag both the critical phase interval and the episode with
    # the resolved success/failure outcome.
    if final_outcome is not None:
        if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
            if rlt_intervention_tracker is not None:
                rlt_intervention_tracker.stop(get_episode_frame_index())
            intervention_state = INTERVENTION_STATE_RELEASE
            set_teleop_manual_control(False)
        if rlt is not None:
            rlt.set_vla_mode()
        if critical_phase_tracker is not None and dataset is not None:
            ep_size = dataset.episode_buffer["size"]
            if final_outcome == EPISODE_SUCCESS:
                critical_phase_tracker.mark_success(ep_size)
            else:
                critical_phase_tracker.mark_failure(ep_size)
        events["episode_outcome"] = final_outcome

    # Close timing file
    if _perf_fh is not None:
        _perf_fh.close()
        logging.info("[Timing] Wrote %d frame timings to /tmp/frame_timing.csv", _frame_idx)

    # Close sidecar file
    if _recovery_fh is not None:
        logging.info("[Recovery] Wrote %d frames to recovery_frames.jsonl", _frame_counter)
        _recovery_fh.close()
