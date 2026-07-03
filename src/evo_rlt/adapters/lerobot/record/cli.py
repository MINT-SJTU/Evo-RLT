from __future__ import annotations

import argparse
import sys

from evo_rlt.adapters.lerobot.record.runner import run_collect, run_full, run_live, run_segment, run_vla_only


DEFAULT_COLLECT_DATASET_TAG = "vla_rlt_vla_test"
DEFAULT_COLLECT_TASK = "Insert the copper screw into the black sleeve."
DEFAULT_VLA_ONLY_POLICY_PATH = "/home/kye/rlt_deploy/vla_online_base_0528_rec_20260610_filtered_lr1e5_ep2"
DEFAULT_VLA_ONLY_DATASET_TAG = "vla_online_base_0528_ft"


def add_common_record_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial-source", choices=["vla", "teleop"], required=True)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--vla-path", default=None)
    parser.add_argument("--rl-token-path", default=None)
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve.")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument("--reset-time-s", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--no-teleop", action="store_true", default=False)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true", default=False)


def add_rtc_args(
    parser: argparse.ArgumentParser,
    *,
    execution_horizon_default: int = 10,
    vla_execution_horizon_default: int | None = None,
    action_queue_default: int | None = None,
) -> None:
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtc-execution-horizon", type=int, default=execution_horizon_default)
    parser.add_argument("--vla-rtc-execution-horizon", type=int, default=vla_execution_horizon_default)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        default="EXP",
        choices=["EXP", "LINEAR", "ONES", "ZEROS"],
    )
    parser.add_argument(
        "--rtc-action-queue-size-to-get-new-actions",
        type=int,
        default=action_queue_default,
    )


def add_default_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--vla-path", default=None)
    parser.add_argument("--rl-token-path", default=None)
    parser.add_argument("--task", default=DEFAULT_COLLECT_TASK)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--dataset-tag", default=DEFAULT_COLLECT_DATASET_TAG)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--no-teleop", action="store_true", default=False)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--double-tap-window-s", type=float, default=0.6)
    parser.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--play-sounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rlt-toggle-key", default="r")
    parser.add_argument("--teleop-toggle-key", default="space")
    parser.add_argument("--default-episode-success", choices=["success", "failure"], default=None)
    parser.add_argument(
        "--start-with-teleop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start each episode in teleop instead of VLA.",
    )
    parser.add_argument(
        "--only-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record only the RLT critical segment. The first RLT key press starts "
            "recording and enters RLT; the next RLT key press saves the segment. "
            "The default records the full trajectory immediately and uses the RLT key "
            "as the full-episode outcome key."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", default=False)


def add_vla_only_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-path", default=DEFAULT_VLA_ONLY_POLICY_PATH)
    parser.add_argument("--task", default=DEFAULT_COLLECT_TASK)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument("--reset-time-s", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--dataset-tag", default=DEFAULT_VLA_ONLY_DATASET_TAG)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--no-teleop", action="store_true", default=False)
    parser.add_argument("--teleop-toggle-key", default="space")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtc-execution-horizon", type=int, default=25)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        default="EXP",
        choices=["EXP", "LINEAR", "ONES", "ZEROS"],
    )
    parser.add_argument("--rtc-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pedal-outcome", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episode-outcome-key", default="r")
    parser.add_argument("--double-tap-window-s", type=float, default=0.6)
    parser.add_argument("--default-episode-success", choices=["success", "failure"], default=None)
    parser.add_argument("--play-sounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true", default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified real-robot recording entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect",
        help="Run the default VLA-RLT-VLA real-robot data collection script.",
    )
    add_default_collect_args(collect)
    add_rtc_args(
        collect,
        vla_execution_horizon_default=25,
        action_queue_default=30,
    )
    collect.set_defaults(func=run_collect)

    vla_only = subparsers.add_parser(
        "vla-only",
        help="Run a fine-tuned VLA directly on the real robot without RLT policy wrapping.",
    )
    add_vla_only_args(vla_only)
    vla_only.set_defaults(func=run_vla_only)

    segment = subparsers.add_parser(
        "segment",
        help="Record only the key segment. Success/failure labels apply to the segment.",
    )
    add_common_record_args(segment)
    add_rtc_args(segment)
    segment.add_argument("--critical-source", choices=["rlt", "vla"], required=True)
    segment.add_argument("--double-tap-window-s", type=float, default=0.6)
    segment.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True)
    segment.add_argument("--chunk-exec-steps", type=int, default=25)
    segment.add_argument("--intervention-action-blend-time-s", type=float, default=0.4)
    segment.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True)
    segment.set_defaults(func=run_segment)

    full = subparsers.add_parser(
        "full",
        help="Record the full trajectory. Success/failure labels apply to the trajectory.",
    )
    add_common_record_args(full)
    add_rtc_args(full)
    full.add_argument("--phase-mode", default=None)
    full.add_argument("--chunk-exec-steps", type=int, default=None)
    full.add_argument("--pedal-outcome", action=argparse.BooleanOptionalAction, default=False)
    full.add_argument("--episode-outcome-key", default="r")
    full.add_argument("--double-tap-window-s", type=float, default=0.6)
    full.set_defaults(func=run_full)

    live = subparsers.add_parser("live", help="Run policy live on the robot without saving a dataset.")
    live.add_argument("--policy-path", required=True)
    live.add_argument("--eval-script", required=True)
    live.add_argument("--vla-path", default=None)
    live.add_argument("--rl-token-path", default=None)
    live.add_argument("--phase-mode", default="always_vla")
    live.add_argument("--chunk-exec-steps", type=int, default=25)
    live.add_argument("--task", default="Insert the copper screw into the black sleeve.")
    live.add_argument("--duration", type=float, default=120.0)
    live.add_argument("--fps", type=float, default=30.0)
    live.add_argument("--setup-json", default=None)
    live.add_argument("--dry-run", action="store_true", default=False)
    add_rtc_args(live)
    live.set_defaults(func=run_live)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"segment", "full"} and args.dataset_tag is None:
        args.dataset_tag = f"{args.initial_source}_{args.command}"
    args.func(args)


def collect_default_main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    main(["collect", *args])


def vla_only_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tuned VLA-only real-robot inference")
    add_vla_only_args(parser)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    run_vla_only(args)


if __name__ == "__main__":
    main()
