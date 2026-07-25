import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evo_rlt.adapters.lerobot.record.cli import build_parser
from evo_rlt.adapters.lerobot.record import runner
from evo_rlt.adapters.lerobot.record.runner import (
    _collect_external_episode_outcome_key,
    _patch_double_tap_episode_outcome_listener,
    _patch_skip_policyless_reset_loop,
    build_default_collect_record_argv,
    build_segment_record_argv,
    build_vla_only_record_argv,
)


def test_initial_source_rejects_rlt():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "segment",
            "--initial-source",
            "rlt",
            "--critical-source",
            "rlt",
        ])


def test_segment_defaults_to_rtc_enabled():
    parser = build_parser()
    args = parser.parse_args([
        "segment",
        "--initial-source",
        "teleop",
        "--critical-source",
        "rlt",
        "--policy-path",
        "/tmp/ac",
    ])
    assert args.rtc is True


def test_segment_rlt_argv_marks_key_segment_with_teleop_start_and_rtc():
    args = SimpleNamespace(
        critical_source="rlt",
        initial_source="teleop",
        policy_path="/tmp/ac",
        vla_path="/tmp/vla",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        reset_time_s=None,
        fps=30,
        vcodec="h264",
        double_tap_window_s=0.6,
        intervention_action_blend_time_s=0.4,
        rtc=True,
        rtc_execution_horizon=10,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=None,
        vla_rtc_execution_horizon=None,
        vla_ref=True,
        chunk_exec_steps=25,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={},
        right_cameras={},
    )
    paths = SimpleNamespace(
        dataset_name="local/test",
        dataset_root="/tmp/dataset",
    )
    argv = build_segment_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--rlt.rl_phase_key_toggles_episode=true" in argv
    assert "--rlt.start_in_teleop=true" in argv
    assert "--rlt.rtc_enabled=true" in argv
    assert "--enable_episode_outcome_labeling=true" in argv
    assert "--policy_sync_to_teleop=true" in argv
    assert "--policy.path=/tmp/ac" in argv


def test_pedal_listener_routes_record_events_and_episode_outcome(monkeypatch):
    control_utils = pytest.importorskip("lerobot.utils.control_utils")
    from evo_rlt.adapters.lerobot.record import pedal_listener

    captured = {}

    class FakePedalListener:
        def __init__(self, on_press):
            captured["on_press"] = on_press

        def start(self):
            return True

        def stop(self):
            captured["stopped"] = True

    def original_init_keyboard_listener(*args, **kwargs):
        return None, {"episode_outcome": None, "exit_early": False}

    monkeypatch.setattr(control_utils, "is_headless", lambda: True)
    monkeypatch.setattr(control_utils, "init_keyboard_listener", original_init_keyboard_listener)
    monkeypatch.setattr(pedal_listener, "PedalListener", FakePedalListener)

    _patch_double_tap_episode_outcome_listener(0.01, "e")
    listener, events = control_utils.init_keyboard_listener(
        intervention_toggle_key=" ",
        rl_phase_key="r",
    )

    captured["on_press"]("space")
    assert events["toggle_intervention"] is True

    captured["on_press"]("r")
    assert events["start_rl_phase"] is True

    captured["on_press"]("e")
    time.sleep(0.03)
    assert events["episode_outcome"] == "success"
    assert events["exit_early"] is True

    events["episode_outcome"] = None
    events["exit_early"] = False
    captured["on_press"]("e")
    captured["on_press"]("e")
    assert events["episode_outcome"] == "failure"
    assert events["exit_early"] is True
    listener.stop()


def test_skip_policyless_reset_loop_keeps_recording_loop(monkeypatch):
    from evo_rlt.adapters.lerobot.record import backend as lerobot_rlt_record

    calls = []

    def original_record_loop(*args, **kwargs):
        calls.append((args, kwargs))
        return "called"

    monkeypatch.setattr(lerobot_rlt_record, "record_loop", original_record_loop)

    _patch_skip_policyless_reset_loop()

    assert lerobot_rlt_record.record_loop(teleop=object(), control_time_s=10) is None
    assert calls == []
    assert lerobot_rlt_record.record_loop(policy=object(), dataset=object()) == "called"
    assert len(calls) == 1


def test_save_episode_patch_preserves_official_background_video_encoding(monkeypatch):
    calls = []

    class FakeMeta:
        total_episodes = 0
        video_keys = ["observation.images.wrist"]

        def save_episode(self, episode_index, episode_length, episode_tasks, episode_stats, episode_metadata):
            calls.append(("meta", dict(episode_metadata)))
            self.total_episodes += 1

    class FakeWriter:
        def __init__(self):
            self._batch_encoding_size = 6

    class FakeLeRobotDataset:
        def __init__(self):
            self.meta = FakeMeta()
            self.writer = FakeWriter()

        def save_episode(self, *args, **kwargs):
            calls.append(("save", self.writer._batch_encoding_size, kwargs))
            self.meta.save_episode(0, 1, ["task"], {}, {"base": "metadata"})
            return "saved"

    fake_lerobot_dataset = type(sys)("lerobot.datasets.lerobot_dataset")
    fake_lerobot_dataset.LeRobotDataset = FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", fake_lerobot_dataset)

    runner._patch_save_episode_extra_metadata()

    dataset = FakeLeRobotDataset()
    assert dataset.save_episode(extra_episode_metadata={"episode_success": "success"}) == "saved"
    assert ("save", 6, {}) in calls
    assert ("meta", {"base": "metadata", "episode_success": "success"}) in calls
    assert dataset.writer._batch_encoding_size == 6


def test_default_collect_parser_requires_user_policy_path():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["collect"])


def test_default_collect_parser_uses_open_source_safe_defaults():
    parser = build_parser()
    args = parser.parse_args(["collect", "--policy-path", "/tmp/ac"])

    assert args.policy_path == "/tmp/ac"
    assert args.vla_path is None
    assert args.rl_token_path is None
    assert args.dataset_tag == "vla_rlt_vla_test"
    assert args.num_episodes == 5
    assert args.rlt_toggle_key == "r"
    assert args.teleop_toggle_key == "space"
    assert args.start_with_teleop is False
    assert args.only_critical is False
    assert args.rtc is True
    assert args.rtc_execution_horizon == 10
    assert args.vla_rtc_execution_horizon == 25
    assert args.rtc_action_queue_size_to_get_new_actions == 30


def test_default_collect_full_mode_uses_r_key_as_episode_outcome():
    parser = build_parser()
    args = parser.parse_args(["collect", "--policy-path", "/tmp/ac", "--rlt-toggle-key", "r"])

    assert _collect_external_episode_outcome_key(args) == "r"


def test_default_collect_argv_matches_best_real_robot_rtc_chunks():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        fps=30,
        vcodec="h264",
        double_tap_window_s=0.6,
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        default_episode_success=None,
        start_with_teleop=False,
        only_critical=False,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={
            "wrist": {
                "type": "opencv",
                "index_or_path": "/tmp/left-camera",
                "width": 640,
                "height": 480,
                "fps": 30,
                "fourcc": "MJPG",
            }
        },
        right_cameras={"wrist": {}, "front": {}},
    )
    paths = SimpleNamespace(dataset_name="local/eval_vla_rlt_vla_123456", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--policy.phase_mode=manual" in argv
    assert "--rlt.enable=true" in argv
    assert "--rlt.rl_phase_key=r" in argv
    assert "--rlt.start_in_teleop=false" in argv
    assert "--rlt.rl_phase_key_toggles_critical_phase=true" not in argv
    assert "--rlt.rl_phase_key_toggles_episode=true" not in argv
    assert "--rlt.skip_prefix_recording=true" not in argv
    assert "--rlt.rtc_execution_horizon=10" in argv
    assert "--rlt.vla_rtc_execution_horizon=25" in argv
    assert "--rlt.rtc_action_queue_size_to_get_new_actions=30" in argv
    assert "--enable_episode_outcome_labeling=true" in argv
    assert "--require_episode_success_label=true" in argv
    assert "--dataset.video_encoding_batch_size=6" in argv
    assert "--dataset.streaming_encoding=true" in argv
    assert "--policy_sync_to_teleop=true" in argv
    assert "--vla_ref=true" in argv


def test_default_collect_only_critical_starts_recording_on_first_r_and_ends_on_second_r():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        fps=30,
        vcodec="h264",
        double_tap_window_s=0.6,
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        start_with_teleop=False,
        only_critical=True,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={},
        right_cameras={},
    )
    paths = SimpleNamespace(dataset_name="local/eval_vla_rlt_vla_123456", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--rlt.skip_prefix_recording=true" in argv
    assert "--rlt.rl_phase_key_toggles_episode=true" in argv
    assert "--rlt.start_in_teleop=false" in argv
    assert "--rlt.rl_phase_key_toggles_critical_phase=true" not in argv
    assert "--enable_episode_outcome_labeling=true" in argv
    assert "--require_episode_success_label=true" in argv
    assert "--policy_sync_to_teleop=true" in argv


def test_default_collect_start_with_teleop_sets_episode_initial_source():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        fps=30,
        vcodec="h264",
        double_tap_window_s=0.6,
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        start_with_teleop=True,
        only_critical=False,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={},
        right_cameras={},
    )
    paths = SimpleNamespace(dataset_name="local/eval_vla_rlt_vla_123456", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--rlt.start_in_teleop=true" in argv
    assert "--rlt.rl_phase_key_toggles_critical_phase=true" not in argv
    assert "--rlt.rl_phase_key_toggles_episode=true" not in argv


def test_full_vla_pedal_outcome_parser():
    parser = build_parser()
    args = parser.parse_args([
        "full",
        "--initial-source",
        "vla",
        "--policy-path",
        "/tmp/ac",
        "--vla-path",
        "/tmp/base.pt",
        "--phase-mode",
        "always_vla",
        "--chunk-exec-steps",
        "25",
        "--pedal-outcome",
        "--episode-outcome-key",
        "e",
        "--reset-time-s",
        "0",
    ])

    assert args.rtc is True
    assert args.pedal_outcome is True
    assert args.episode_outcome_key == "e"
    assert args.phase_mode == "always_vla"
    assert args.chunk_exec_steps == 25
    assert args.reset_time_s == 0


def test_evo_rlt_recording_does_not_import_lerobot_fork_only_modules():
    source_root = Path(__file__).parents[2] / "src" / "evo_rlt"
    banned_imports = [
        "lerobot.scripts.lerobot_rlt_record",
        "lerobot.scripts.recording_hil",
        "lerobot.scripts.recording_loop",
        "lerobot.scripts.robot_config_loader",
        "lerobot.utils.recording_annotations",
        "lerobot.rl.acp_tags",
        "lerobot.policies.rlt",
    ]

    offenders = []
    for py_file in source_root.rglob("*.py"):
        text = py_file.read_text()
        for banned in banned_imports:
            if banned in text:
                offenders.append(f"{py_file.relative_to(source_root)}: {banned}")

    assert offenders == []


class _FakeLeaderBus:
    def __init__(self):
        self.calls = []

    def enable_torque(self):
        self.calls.append(("enable_torque",))

    def disable_torque(self):
        self.calls.append(("disable_torque",))

    def sync_write(self, data_name, values):
        self.calls.append(("sync_write", data_name, dict(values)))


class _FakeLeaderArm:
    def __init__(self):
        self.bus = _FakeLeaderBus()
        self.config = SimpleNamespace(port="/dev/fake")


def test_official_so_leader_feedback_is_sent_through_bus():
    from evo_rlt.adapters.lerobot.record.hil import send_teleop_feedback, set_teleop_manual_control

    leader = _FakeLeaderArm()

    send_teleop_feedback(leader, {"shoulder_pan.pos": 1.0, "ignored": 2.0})
    send_teleop_feedback(leader, {"shoulder_lift.pos": 3.0})
    set_teleop_manual_control(leader, True)

    assert leader.bus.calls == [
        ("enable_torque",),
        ("sync_write", "Goal_Position", {"shoulder_pan": 1.0}),
        ("sync_write", "Goal_Position", {"shoulder_lift": 3.0}),
        ("disable_torque",),
    ]


def test_official_bi_so_leader_feedback_splits_prefixed_actions():
    from evo_rlt.adapters.lerobot.record.hil import send_teleop_feedback

    teleop = SimpleNamespace(left_arm=_FakeLeaderArm(), right_arm=_FakeLeaderArm())

    send_teleop_feedback(
        teleop,
        {
            "left_shoulder_pan.pos": 1.0,
            "right_elbow_flex.pos": 2.0,
            "action_is_pad": 0.0,
        },
    )

    assert teleop.left_arm.bus.calls == [
        ("enable_torque",),
        ("sync_write", "Goal_Position", {"shoulder_pan": 1.0}),
    ]
    assert teleop.right_arm.bus.calls == [
        ("enable_torque",),
        ("sync_write", "Goal_Position", {"elbow_flex": 2.0}),
    ]


def test_default_collect_argv_accepts_headless_default_episode_success():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=1,
        episode_time_s=10,
        fps=30,
        vcodec="h264",
        double_tap_window_s=0.6,
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        default_episode_success="success",
        start_with_teleop=False,
        only_critical=False,
    )
    setup = SimpleNamespace(followers=[{"port": "left"}, {"port": "right"}], left_cameras={}, right_cameras={})
    paths = SimpleNamespace(dataset_name="local/test", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(args, setup, paths, "/tmp/cal", ["--teleop.type=bi_so_leader"])

    assert "--default_episode_success=success" in argv
    assert "--require_episode_success_label=true" in argv




def test_rtc_policy_prediction_passes_leftover_chunk_to_background_replan():
    from evo_rlt.adapters.lerobot.record.hil import ACPInferenceConfig, _predict_policy_action_with_acp_inference
    from evo_rlt.adapters.lerobot.record.vla_rtc import reset_vla_only_rtc_runtimes

    class FakePolicy:
        def __init__(self):
            self.config = SimpleNamespace(
                rtc_config=SimpleNamespace(enabled=True, execution_horizon=2),
                device="cpu",
                use_amp=False,
                chunk_size=3,
            )
            self.chunk_calls = 0
            self.chunk_kwargs = []

        def select_action(self, batch):
            raise AssertionError("select_action should not be used when RTC is enabled")

        def predict_action_chunk(self, batch, **kwargs):
            self.chunk_calls += 1
            self.chunk_kwargs.append(kwargs)
            assert batch["task"] == "task"
            return torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    policy = FakePolicy()
    preprocessor = lambda observation: observation
    postprocessor = lambda action: action
    observation = {"observation.state": np.array([0.0], dtype=np.float32)}

    first_action = _predict_policy_action_with_acp_inference(
        observation_frame=observation,
        policy=policy,
        device=torch.device("cpu"),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=False,
        task="task",
        robot_type="bi_so_follower",
        acp_inference=ACPInferenceConfig(),
    )
    second_action = _predict_policy_action_with_acp_inference(
        observation_frame=observation,
        policy=policy,
        device=torch.device("cpu"),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=False,
        task="task",
        robot_type="bi_so_follower",
        acp_inference=ACPInferenceConfig(),
    )

    assert torch.equal(first_action, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(second_action, torch.tensor([[3.0, 4.0]]))
    reset_vla_only_rtc_runtimes(policy)
    assert policy.chunk_calls == 2
    assert policy.chunk_kwargs[0] == {
        "inference_delay": 0,
        "prev_chunk_left_over": None,
        "execution_horizon": 2,
    }
    assert torch.equal(policy.chunk_kwargs[1]["prev_chunk_left_over"], torch.tensor([[5.0, 6.0]]))
    assert policy.chunk_kwargs[1]["execution_horizon"] == 2

def test_vla_only_parser_uses_deployed_checkpoint_by_default():
    parser = build_parser()
    args = parser.parse_args(["vla-only"])

    assert args.policy_path.endswith("vla_online_base_0528_rec_20260610_filtered_lr1e5_ep2")
    assert args.dataset_tag == "vla_online_base_0528_ft"
    assert args.device == "cuda"
    assert args.dtype == "bfloat16"
    assert args.n_action_steps is None
    assert args.rtc is True
    assert args.rtc_execution_horizon == 25
    assert args.rtc_max_guidance_weight == 10.0
    assert args.rtc_prefix_attention_schedule == "EXP"
    assert args.rtc_debug is False
    assert args.no_teleop is True
    assert args.preflight is True
    assert args.pedal_outcome is False


def test_vla_only_parser_can_enable_teleop():
    parser = build_parser()
    args = parser.parse_args(["vla-only", "--teleop"])

    assert args.no_teleop is False


def test_vla_only_record_argv_does_not_deploy_rlt_policy():
    args = SimpleNamespace(
        policy_path="/models/vla-ft",
        task="task",
        num_episodes=1,
        episode_time_s=3000,
        reset_time_s=None,
        fps=30,
        vcodec="h264",
        device="cuda",
        dtype="bfloat16",
        n_action_steps=25,
        rtc=True,
        rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_debug=False,
        pedal_outcome=False,
        default_episode_success=None,
        teleop_toggle_key="space",
        play_sounds=False,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={},
        right_cameras={},
    )
    paths = SimpleNamespace(
        dataset_name="local/vla_only",
        dataset_root="/tmp/vla_only",
    )

    argv = build_vla_only_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--policy.path=/models/vla-ft" in argv
    assert "--policy.device=cuda" in argv
    assert "--policy.dtype=bfloat16" in argv
    assert "--policy.n_action_steps=25" in argv
    assert "--policy.rtc_config.enabled=true" in argv
    assert "--policy.rtc_config.execution_horizon=25" in argv
    assert "--policy.rtc_config.max_guidance_weight=10.0" in argv
    assert "--policy.rtc_config.prefix_attention_schedule=EXP" in argv
    assert "--policy_sync_to_teleop=true" in argv
    assert "--intervention_toggle_key=space" in argv
    assert not any(item.startswith("--rlt.") for item in argv)
    assert not any("rl_token" in item for item in argv)
    assert not any("vla_pretrained_path" in item for item in argv)
