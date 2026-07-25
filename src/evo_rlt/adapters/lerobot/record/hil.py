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

"""Human-in-loop recording helpers used by `lerobot_record.py`."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, RobotAction
from evo_rlt.adapters.lerobot.record.acp_tags import build_acp_tagged_task
from evo_rlt.adapters.lerobot.record.vla_rtc import get_vla_only_rtc_runtime
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.control_utils import predict_action, prepare_observation_for_inference


@dataclass
class ACPInferenceConfig:
    enable: bool = False
    use_cfg: bool = False
    cfg_beta: float = 1.0


POLICY_RUNTIME_STATE_KEYS = ("_action_queue", "_queues", "_prev_mean")


INTERVENTION_STATE_POLICY = 0.0
INTERVENTION_STATE_ACTIVE = 1.0
INTERVENTION_STATE_RELEASE = 2.0


def _get_torch_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _set_torch_rng_state(
    device: torch.device, cpu_state: torch.Tensor, cuda_state: torch.Tensor | None
) -> None:
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _clone_runtime_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, deque):
        return deque((_clone_runtime_value(item) for item in value), maxlen=value.maxlen)
    if isinstance(value, dict):
        return {key: _clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    return deepcopy(value)


def _capture_policy_runtime_state(policy: PreTrainedPolicy) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key in POLICY_RUNTIME_STATE_KEYS:
        if hasattr(policy, key):
            state[key] = _clone_runtime_value(getattr(policy, key))
    return state


def _restore_policy_runtime_state(policy: PreTrainedPolicy, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(policy, key, _clone_runtime_value(value))


def _policy_uses_rtc_chunking(policy: PreTrainedPolicy) -> bool:
    rtc_config = getattr(policy.config, "rtc_config", None)
    return bool(getattr(rtc_config, "enabled", False))


def _predict_action_with_optional_rtc(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    fps: float,
) -> PolicyAction:
    if not _policy_uses_rtc_chunking(policy):
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )

    with torch.no_grad():
        observation = prepare_observation_for_inference(copy(observation_frame), device, task, robot_type)
        observation = preprocessor(observation)
        runtime = get_vla_only_rtc_runtime(
            policy, stream_id=task, fps=fps, device=device, use_amp=use_amp
        )
        action = runtime.select_action(observation)
        return postprocessor(action)


def _predict_policy_action_with_runtime_state(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    runtime_state: dict[str, Any],
    fps: float,
) -> PolicyAction:
    _restore_policy_runtime_state(policy, runtime_state)
    action = _predict_action_with_optional_rtc(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
        fps=fps,
    )
    runtime_state.clear()
    runtime_state.update(_capture_policy_runtime_state(policy))
    return action


def _predict_policy_action_with_acp_inference(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    acp_inference: ACPInferenceConfig,
    cond_runtime_state: dict[str, Any] | None = None,
    uncond_runtime_state: dict[str, Any] | None = None,
    fps: float = 30.0,
) -> PolicyAction:
    if not acp_inference.enable:
        return _predict_action_with_optional_rtc(
            observation_frame=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
            fps=fps,
        )

    conditional_task = build_acp_tagged_task(task, is_positive=True)
    if not acp_inference.use_cfg:
        return _predict_action_with_optional_rtc(
            observation_frame=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=conditional_task,
            robot_type=robot_type,
            fps=fps,
        )

    if cond_runtime_state is None or uncond_runtime_state is None:
        raise ValueError("CFG inference requires cond/uncond runtime states.")

    cpu_state, cuda_state = _get_torch_rng_state(device)
    action_cond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=conditional_task,
        robot_type=robot_type,
        runtime_state=cond_runtime_state,
        fps=fps,
    )
    _set_torch_rng_state(device, cpu_state, cuda_state)
    action_uncond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
        runtime_state=uncond_runtime_state,
        fps=fps,
    )
    return action_uncond + acp_inference.cfg_beta * (action_cond - action_uncond)


def _is_so_leader_arm(teleop: Teleoperator) -> bool:
    bus = getattr(teleop, "bus", None)
    return (
        callable(getattr(bus, "sync_write", None))
        and callable(getattr(bus, "enable_torque", None))
        and callable(getattr(bus, "disable_torque", None))
    )


def _is_bi_so_leader(teleop: Teleoperator) -> bool:
    left_arm = getattr(teleop, "left_arm", None)
    right_arm = getattr(teleop, "right_arm", None)
    return _is_so_leader_arm(left_arm) and _is_so_leader_arm(right_arm)


def _raise_so_leader_connection_error(teleop: Teleoperator, operation: str, error: ConnectionError) -> None:
    from lerobot.utils.errors import DeviceDroppedConnectionError

    label = getattr(teleop, "diagnostic_label", "leader arm")
    port = getattr(getattr(teleop, "config", None), "port", "unknown")
    raise DeviceDroppedConnectionError(label, port, operation, error) from error


def _set_so_leader_manual_control(teleop: Teleoperator, enabled: bool) -> None:
    manual_control_enabled = getattr(teleop, "_evo_rlt_manual_control_enabled", True)
    if enabled:
        if not manual_control_enabled:
            try:
                teleop.bus.disable_torque()
            except ConnectionError as error:
                _raise_so_leader_connection_error(teleop, "enabling leader manual control", error)
            teleop._evo_rlt_manual_control_enabled = True
        return

    if manual_control_enabled:
        try:
            teleop.bus.enable_torque()
        except ConnectionError as error:
            _raise_so_leader_connection_error(teleop, "disabling leader manual control", error)
        teleop._evo_rlt_manual_control_enabled = False


def set_teleop_manual_control(teleop: Teleoperator, enabled: bool) -> None:
    set_manual_control = getattr(teleop, "set_manual_control", None)
    if callable(set_manual_control):
        set_manual_control(enabled)
        return

    if _is_bi_so_leader(teleop):
        set_teleop_manual_control(teleop.left_arm, enabled)
        set_teleop_manual_control(teleop.right_arm, enabled)
        return

    if _is_so_leader_arm(teleop):
        _set_so_leader_manual_control(teleop, enabled)


def _send_so_leader_feedback(teleop: Teleoperator, feedback: dict[str, float]) -> None:
    set_teleop_manual_control(teleop, False)
    goal_pos = {key.removesuffix(".pos"): val for key, val in feedback.items() if key.endswith(".pos")}
    if not goal_pos:
        return

    try:
        teleop.bus.sync_write("Goal_Position", goal_pos)
    except ConnectionError as error:
        _raise_so_leader_connection_error(teleop, "sending leader feedback goal position", error)


def send_teleop_feedback(teleop: Teleoperator, feedback: RobotAction) -> None:
    if _is_bi_so_leader(teleop):
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }
        _send_so_leader_feedback(teleop.left_arm, left_feedback)
        _send_so_leader_feedback(teleop.right_arm, right_feedback)
        return

    if _is_so_leader_arm(teleop):
        _send_so_leader_feedback(teleop, feedback)
        return

    teleop.send_feedback(feedback)


class PolicySyncDualArmExecutor:
    """Broadcast one policy-derived robot action to follower + teleop arm."""

    def __init__(self, robot: Robot, teleop: Teleoperator, parallel_dispatch: bool = True):
        self.robot = robot
        self.teleop = teleop
        self.parallel_dispatch = parallel_dispatch
        self._pool = ThreadPoolExecutor(max_workers=2) if parallel_dispatch else None

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._pool is None:
            sent_action = self.robot.send_action(action)
            send_teleop_feedback(self.teleop, action)
            return sent_action

        robot_future = self._pool.submit(self.robot.send_action, action)
        teleop_future = self._pool.submit(send_teleop_feedback, self.teleop, action)
        sent_action = robot_future.result()
        teleop_future.result()
        return sent_action

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
