<h1 align="center">Evo-RLT</h1>

<p align="center">
  <a href="https://github.com/huggingface/lerobot"><img alt="lerobot version" src="https://img.shields.io/badge/LeRobot-0.5.1-f59e0b"/></a>
  <a href="https://huggingface.co/datasets/Elvinky/bi-so101-insert-screw-562ep"><img alt="training dataset" src="https://img.shields.io/static/v1?label=Dataset&message=562ep&color=22c55e"/></a>
  <a href="https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset"><img alt="RW-RL dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-RW--RL-ffcc4d"/></a>
  <a href="https://huggingface.co/Shiki42/pi05_screw_c_mix_cont15k_fp16/tree/main"><img alt="model" src="https://img.shields.io/static/v1?label=Model&message=pi0.5&color=0ea5e9"/></a>
  <a href="https://huggingface.co/Shiki42/pi05_screw_c_mix_cont15k_fp16/tree/main"><img alt="checkpoint" src="https://img.shields.io/static/v1?label=Ckpt&message=Available&color=6366f1"/></a>
  <a href="./LICENSE"><img alt="license" src="https://img.shields.io/badge/License-Apache--2.0-ef4444"/></a>
</p>

<p align="center"><strong>SJTU-MINT</strong></p>

<p align="center">
  <strong>A LeRobot-based reproduction of <a href="https://www.pi.website/research/rlt">RLT</a>, covering RL-token learning, transition-cache generation, actor-critic training, and real-robot rollout.</strong>
</p>

<p align="center"><strong>RLT Pipeline</strong></p>

<p align="center">
  <img alt="RLT training pipeline" src="./website/assets/images/rlt_pipeline.png" width="96%"/>
</p>

<p align="center"><strong>Real-Robot Rollout Demo</strong></p>

<p align="center">
  <img alt="RLT real-robot rollout demo" src="./website/assets/images/rlt_rollout.gif" width="96%"/>
</p>

<p align="center"><strong>Collect Human Demonstrations</strong></p>

<p align="center">
  <img alt="Collect human demonstrations demo" src="./website/assets/images/rlt_collect_human_demonstrations.gif" width="96%"/>
</p>

<p align="center"><strong>Policy Rollout with Human Intervention</strong></p>

<p align="center">
  <img alt="Policy rollout with human intervention demo" src="./website/assets/images/rlt_rollout_human_intervention.gif" width="96%"/>
</p>

## 🎯 Evo-RLT Focus

- **RLT reproduction:** this repository presents RLT as an independent LeRobot-based reproduction for the pi paper.
- **Open training path:** the code covers VLA finetuning, RL-token learning, transition-cache generation, and chunk actor-critic training.
- **Real-robot deployment path:** the recording wrapper supports VLA/RLT rollout, RTC defaults, pedal labels, and human-in-the-loop collection.

## 📰 News

- **[2026-06-29]** Released Evo-RLT.
- **[2026-06-26]** Added training dataset and checkpoint links.

## 🧭 Table of Contents

| Getting Started | Training Pipeline | Project Info |
| -------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| [⚡ Quick Start](#quick-start) | [🧪 Training Pipeline](#training-pipeline) | [🤗 Model & Dataset](#model-dataset) |
| [1) Installation](#installation) | [3) Finetune VLA](#finetune-vla) | [🗂️ Repository Layout](#repository-layout) |
| [2) Hardware Setup](#hardware-setup) | [4) Train RL Token](#train-rl-token) | [✅ Development Checks](#development-checks) |
| [🤖 Real-Robot Recording and Deployment](#real-robot-recording-and-deployment) | [5) Build Transition Cache](#build-transition-cache) | [🧭 Future TODO](#future-todo) |
| | [6) Train Chunk Actor-Critic](#train-chunk-actor-critic) | [💬 Community Channels](#community-channels) / [🏫 Affiliations](#affiliations) / [📄 License](#license) |

<a id="quick-start"></a>

## ⚡ Quick Start

<a id="installation"></a>

### 1) Installation

Evo-RLT depends on LeRobot `v0.5.1`, which currently ships from the official GitHub tag and requires Python 3.12+.

```bash
git clone https://github.com/MINT-SJTU/Evo-RLT.git
cd evo-rlt

conda create -y -n evo-rlt python=3.12
conda activate evo-rlt

python -m pip install -e ".[lerobot]"
```

Do not put a local LeRobot source checkout on `PYTHONPATH`; Evo-RLT is tested against the official LeRobot package installed by the `lerobot` extra.

Evo-RLT keeps policy registration out of LeRobot source files. Before using LeRobot factory helpers with RLT policy types, register the adapter once:

```python
from evo_rlt.adapters.lerobot import register

register()
```

Registered policy types:

```text
rlt_token    # RL-token reconstruction policy
rlt_ac       # chunk actor-critic policy
rlt          # deployment policy wrapper
```

### LeRobot 0.5.1 Normalization

Evo-RLT follows the LeRobot `>=0.5` processor-pipeline runtime:

```text
raw observation -> policy_preprocessor -> policy -> policy_postprocessor -> robot action
```

Checkpoints trained or migrated for LeRobot `>=0.5` are expected to include `policy_preprocessor.json`, `policy_postprocessor.json`, and processor weight files such as `NormalizerProcessorStep` / `UnnormalizerProcessorStep` statistics. The presence of `NormalizerProcessorStep` is normal in LeRobot `0.5.1`; it is not a legacy workaround.

Only migrate normalization for checkpoints trained before LeRobot's processor-pipeline migration. For those checkpoints, verify normalization is not applied twice: model weights should not contain embedded normalization keys such as `normalize_inputs.*`, and the external pre/postprocessor stats must match the training normalization modes.

<a id="hardware-setup"></a>

### 2) Hardware Setup

Use the [Evo-RL hardware setup](https://github.com/MINT-SJTU/Evo-RL#2-hardware-setup) for the shared SO-series robot bring-up steps: assembly, stable serial/camera paths, camera validation, and basic teleoperation checks. PiPER/PiPER-X support is planned; see [Future TODO](#future-todo).

This repository only differs at the recording/deployment configuration layer:

- `evo-rlt-record` reads a setup manifest from `--setup-json`, or from `~/.roboclaw/workspace/embodied/manifest.json` when the flag is omitted.
- Arm entries point to per-device `calibration_dir` folders. The wrapper looks for `<calibration_dir>/<folder-name>.json`, then stages those files into temporary LeRobot-compatible names at runtime.
- Follower calibrations are staged as `bimanual_left.json` and `bimanual_right.json` under a temporary robot calibration directory.
- Leader calibrations are staged as `bimanual_leader_left.json` and `bimanual_leader_right.json` under a temporary teleop calibration directory.
- Dataset paths are created under `<datasets.root>/<MMDD>_<dataset-tag>/<prefix>_<HHMMSS>`. If `datasets.root` is omitted, the default is `~/.roboclaw/workspace/embodied/datasets`.

Example setup manifest:

```json
{
  "datasets": {"root": "/path/to/lerobot_datasets"},
  "arms": [
    {
      "alias": "left_follower",
      "type": "follower",
      "port": "/dev/serial/by-id/<left-follower>",
      "calibration_dir": "/path/to/calibration/<left-follower-serial>"
    },
    {
      "alias": "right_follower",
      "type": "follower",
      "port": "/dev/serial/by-id/<right-follower>",
      "calibration_dir": "/path/to/calibration/<right-follower-serial>"
    },
    {
      "alias": "left_leader",
      "type": "leader",
      "port": "/dev/serial/by-id/<left-leader>",
      "calibration_dir": "/path/to/calibration/<left-leader-serial>"
    },
    {
      "alias": "right_leader",
      "type": "leader",
      "port": "/dev/serial/by-id/<right-leader>",
      "calibration_dir": "/path/to/calibration/<right-leader-serial>"
    }
  ],
  "cameras": [
    {
      "alias": "left_wrist",
      "port": "/dev/v4l/by-path/<left-wrist>",
      "width": 640,
      "height": 480,
      "fps": 30,
      "fourcc": "MJPG"
    },
    {
      "alias": "right_wrist",
      "port": "/dev/v4l/by-path/<right-wrist>",
      "width": 640,
      "height": 480,
      "fps": 30,
      "fourcc": "MJPG"
    },
    {
      "alias": "right_front",
      "port": "/dev/v4l/by-path/<right-front>",
      "width": 640,
      "height": 480,
      "fps": 30,
      "fourcc": "MJPG"
    }
  ]
}
```

<a id="training-pipeline"></a>

## 🧪 Training Pipeline

The typical RLT workflow has four stages. Video datasets require FFmpeg shared libraries for LeRobot `0.5.1` / TorchCodec decoding:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

For saved checkpoints, LeRobot `0.5.1` writes numeric checkpoint directories such as `checkpoints/000001/pretrained_model`. Use the latest numeric directory when `checkpoints/last/pretrained_model` is not present.

<a id="finetune-vla"></a>

### 3) Finetune VLA

Use LeRobot's training entrypoint to finetune a pi0.5 VLA checkpoint on a LeRobot dataset.

```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=<HF_ORG>/<DATASET> \
  --dataset.root=<LOCAL_DATASET_ROOT> \
  --policy.path=<BASE_PI05_CHECKPOINT_DIR> \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=16 \
  --steps=30000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir=outputs/vla_ft \
  --job_name=vla_ft
```

<a id="train-rl-token"></a>

### 4) Train RL Token

```bash
python -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.repo_id=<HF_ORG>/<DATASET> \
  --dataset.root=<LOCAL_DATASET_ROOT> \
  --policy.type=rlt_token \
  --policy.repo_id=<HF_ORG>/rlt_token \
  --policy.push_to_hub=false \
  --policy.vla_pretrained_path=outputs/vla_ft/checkpoints/last/pretrained_model \
  --policy.vla_dtype=bfloat16 \
  --policy.rl_token_num_rl_tokens=1 \
  --policy.tokenizer_path=/path/to/paligemma-3b-pt-224-snapshot \
  --policy.token_pool_size=0 \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=10000 \
  --save_freq=2000 \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir=outputs/rl_token \
  --job_name=rl_token
```

<a id="build-transition-cache"></a>

### 5) Build Transition Cache

```bash
evo-rlt-build-transition-cache \
  --demo-dataset-repo-id <HF_ORG>/<DATASET> \
  --demo-dataset-root <LOCAL_DATASET_ROOT> \
  --rl-token-policy-path outputs/rl_token/checkpoints/last/pretrained_model \
  --norm-stats-path <RL_TOKEN_NORM_STATS_PATH> \
  --vla-pretrained-path outputs/vla_ft/checkpoints/last/pretrained_model \
  --tokenizer-path /path/to/paligemma-3b-pt-224-snapshot \
  --output-dir outputs/cache \
  --task-instruction "<TASK>" \
  --chunk-length 10 \
  --frame-stride 2 \
  --batch-size 8 \
  --num-workers 2 \
  --train-ratio 0.9 \
  --tolerance-s 0.04 \
  --device cuda
```

The source dataset must contain an `episode_success` value (`success` or
`failure`) for every episode. The cache uses sparse binary rewards: only the
last valid step of a successful terminal chunk receives reward `1`; all other
steps receive `0`.

<a id="train-chunk-actor-critic"></a>

### 6) Train Chunk Actor-Critic

`outputs/cache` must contain `chunk_transitions_train.pt`. The Evo-RLT registry detects this cache directory through `--dataset.repo_id`.

```bash
python -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.repo_id=outputs/cache \
  --policy.type=rlt_ac \
  --policy.repo_id=<HF_ORG>/rlt_ac \
  --policy.push_to_hub=false \
  --policy.vla_pretrained_path=outputs/vla_ft/checkpoints/last/pretrained_model \
  --policy.rl_token_pretrained_path=outputs/rl_token/checkpoints/last/pretrained_model \
  --policy.vla_dtype=bfloat16 \
  --policy.tokenizer_path=/path/to/paligemma-3b-pt-224-snapshot \
  --policy.rl_token_num_rl_tokens=1 \
  --policy.chunk_length=10 \
  --policy.chunk_exec_steps=25 \
  --policy.phase_mode=always_rl \
  --policy.device=cuda \
  --batch_size=256 \
  --steps=50000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --output_dir=outputs/ac \
  --job_name=rlt_ac
```

<a id="real-robot-recording-and-deployment"></a>

## 🤖 Real-Robot Recording and Deployment

Set up the environment before running robot commands:

```bash
cd /path/to/evo-rlt
source ~/miniconda3/etc/profile.d/conda.sh
conda activate evo-rlt
python -m pip install -e ".[lerobot]"
export HF_HUB_OFFLINE=1
```

Default VLA-RLT-VLA real-robot collection uses the official LeRobot `0.5.1` streaming encoder. The wrapper keeps the foreground recording loop responsive and expands the dataset settings to `--dataset.vcodec=h264`, `--dataset.video_encoding_batch_size=<num_episodes + 1>`, and `--dataset.streaming_encoding=true`.

Shared collection arguments:

```bash
COMMON_ARGS=(
  --setup-json /path/to/robot_manifest.json \
  --policy-path /path/to/rlt_ac_policy \
  --vla-path /path/to/pi05_vla_checkpoint_or_dir \
  --rl-token-path /path/to/rl_token_policy \
  --dataset-tag vla_rlt_vla_test \
  --num-episodes 5 \
  --episode-time-s 3000 \
  --fps 30 \
  --vcodec h264 \
  --rlt-toggle-key r \
  --teleop-toggle-key space
)
```

Start in VLA mode and record the full trajectory:

```bash
evo-rlt-record collect "${COMMON_ARGS[@]}"
```

Start in VLA mode and record only the critical segment:

```bash
evo-rlt-record collect "${COMMON_ARGS[@]}" --only-critical
```

Start in teleoperation mode and record the full trajectory:

```bash
evo-rlt-record collect "${COMMON_ARGS[@]}" --start-with-teleop
```

Start in teleoperation mode and record only the critical segment:

```bash
evo-rlt-record collect "${COMMON_ARGS[@]}" --start-with-teleop --only-critical
```

The same collection entrypoint is exposed as `evo-rlt-collect-default` after reinstalling package entry points, but checkpoint and setup paths still need to be supplied by the caller.

Validated RTC defaults for this collection mode:

```text
RLT RTC execution horizon: 10
VLA RTC execution horizon: 25
RTC action queue refill threshold: 30
RTC max guidance weight: 10.0
RTC prefix attention schedule: EXP
```

Default collection controls:

```text
Full-trajectory mode:
r              save the full episode as success after the double-tap window
r+r            save the full episode as failure
space          toggle teleop intervention; pressing again exits teleop
left arrow     rerecord the current episode
Esc            stop data collection

Critical-segment mode (`--only-critical`):
r              enter RLT mode and start recording the critical segment
r              save the segment as success, exit RLT mode, then end the episode
r+r            save the segment as failure, exit RLT mode, then end the episode
space          toggle teleop intervention; pressing again exits teleop
left arrow     rerecord the current episode
Esc            stop data collection
```

VLA-only full-process recording with pedal outcome labels:

```bash
evo-rlt-record full \
  --initial-source vla \
  --setup-json <ROBOT_SETUP_JSON> \
  --policy-path <AC_OR_VLA_POLICY_PATH> \
  --vla-path <BASE_OR_FINETUNED_VLA_PT> \
  --phase-mode always_vla \
  --chunk-exec-steps 25 \
  --pedal-outcome \
  --double-tap-window-s 0.6 \
  --num-episodes 5 \
  --episode-time-s 3000 \
  --reset-time-s 0 \
  --fps 30 \
  --vcodec h264 \
  --dataset-tag vla_full_pedal \
  --no-teleop
```

For headless SSH runs where no keyboard or pedal outcome will be provided, add
`--default-episode-success success` or `--default-episode-success failure`.

Pedal semantics in this mode:

```text
single tap    success, end current episode, start next episode
double tap    failure, end current episode, start next episode
```

<a id="repository-layout"></a>

## 🗂️ Repository Layout

```text
src/evo_rlt/core                  # algorithm core, torch-only
src/evo_rlt/adapters/lerobot      # LeRobot/pi0.5/dataset/policy/record adapters
src/evo_rlt/cli                   # training and cache CLIs
tests/rlt                         # focused RLT unit and integration tests
```

<a id="development-checks"></a>

## ✅ Development Checks

```bash
PYTHONPATH=src pytest -q tests/rlt
PYTHONPATH=src python -m compileall -q src/evo_rlt tests/rlt
```

<a id="model-dataset"></a>

## 🤗 Model & Dataset

- Training dataset: [Elvinky/bi-so101-insert-screw-562ep](https://huggingface.co/datasets/Elvinky/bi-so101-insert-screw-562ep).
- Real-world RL dataset: [MINT-SJTU/RW-RL-Dataset](https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset).
- Checkpoint repo: [Shiki42/pi05_screw_c_mix_cont15k_fp16](https://huggingface.co/Shiki42/pi05_screw_c_mix_cont15k_fp16/tree/main).

<a id="future-todo"></a>

## 🧭 Future TODO

- PiPER/PiPER-X real-robot deployment support.

<a id="community-channels"></a>

## 💬 Community Channels

- Email: business@evomind-tech.com
- WeChat group QR code:

<p align="center">
  <img alt="EvoMind WeChat QR" src="./website/assets/images/rlgroup.jpg" width="220"/>
</p>

<a id="affiliations"></a>

## 🏫 Affiliations

<p align="center">
  <img alt="SJTU community visual" src="./website/assets/images/sjtu.png" height="68"/>
  <img alt="EvoMind" src="./website/assets/images/evomind1.png" height="60"/>
</p>

<a id="license"></a>

## 📄 License

Apache-2.0. See [LICENSE](./LICENSE).
