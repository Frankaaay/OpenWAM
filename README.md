<p align="center">
  <img src="assets/repo_images/openwam_title.png" alt="OpenWAM" width="60%">
</p>

<p align="center"><strong>An Open, Modular Exploration Towards Systematic World–Action Model Pretraining</strong></p>

<p align="center">
  <a href="https://openwam-official.github.io/"><img src="https://img.shields.io/badge/Project%20Page-OPENWAM--OFFICIAL.GITHUB.IO-blue?style=flat&amp;logo=github" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2609.07398"><img src="https://img.shields.io/badge/arXiv-2609.07398-red?style=flat&amp;logo=arxiv" alt="Paper on arXiv"></a>
  <a href="https://huggingface.co/OpenWAM"><img src="https://img.shields.io/badge/HuggingFace-Model%20%26%20Data-orange?style=flat&amp;logo=huggingface" alt="Model &amp; Data on Hugging Face"></a>
</p>

<p align="center">
  <img src="assets/repo_images/teaser.jpg" alt="OpenWAM teaser">
</p>

## ORION 双臂真机 v3 部署（本分支）

代码分支：`frank/realrobot-v3-inference`。2026-11-13 已通过 GPU smoke 的权重为 **Wan 叠杯、Wan 交接、Stereo 叠杯**；Stereo 交接按用户要求停止，未完成验证。三项均通过真实记录归一化/TCP 往返、严格加载和两次 10 步推理；这不代表真机闭环已验证。

### 1. 准备代码、环境和资产

在 GPU 机器使用本分支 OpenWAM 和已配置的 CUDA/PyTorch 环境（安装方法见下方 Installation），以及以下版本的依赖源码：

- orion-0：`783b602d128a43814af32285e60a60f270ec2703`。
- FastWAM：`491e518fac6e04646ecfbd464027dd575ed99bde`，Stereo 必需。
- 本次 H100 验证环境：`/mnt/workdir/frank/envs/openwam-baselines-20261014/bin/python`，PyTorch 2.7.1+cu128。该路径在既有运行容器内使用，不在登录节点运行模型。

H100 资产根目录为 `/gpfs/jiuquyun/projects/Frank/realrobot-v3-posttrain-20261111/`：

| 模型/任务 | 相对根目录的最终权重路径 | GPU smoke |
|---|---|---|
| Wan 叠杯 | `wan/formal/cup_stacking/2026-11-11_23-31-51/checkpoint_step_9090.safetensors` | 通过 |
| Wan 交接 | `wan/formal/object_handover/2026-11-12_04-35-36/checkpoint_step_10930.safetensors` | 通过 |
| Stereo 叠杯 | `stereo-r2/formal/cup_stacking/2026-11-12_10-48-57/checkpoint_step_9090.safetensors` | 通过 |
| Stereo 交接 | `stereo-r2/formal/object_handover/2026-11-12_16-33-52/checkpoint_step_10930.safetensors` | 未完成 |

配置、统计量和 tokenizer 已补齐。迁往现场 GPU 时，保留对应权重同目录的 `config.yaml`、`realrobot_stats.json`、`data-contract.json`、`tokenizer/`，Stereo 另带 `stereotok.json`。不要混用其他任务的统计量。机器人端沿用现有 AIRBOT SDK 环境，需要 NumPy、PyYAML、OpenCV、`airbot_py` 和上述 ORION 源码；不需要加载模型。

### 2. 配置任务和现场设备

编辑本分支的 [`configs/realrobot_v3_inference.yaml`](configs/realrobot_v3_inference.yaml)，GPU 端和机器人端使用相同内容：

```yaml
task: cup_stacking       # 交接改为 object_handover
input_mode: mono-left    # Wan 用 mono-left；Stereo 用 stereo
```

Stereo 需填写 `client.right_camera_devices`，顺序为 head-right、left-wrist-right、right-wrist-right，必须是三个真实右目设备。Wan 使用三个安装位的左目。

沿用现场 ORION `deployment_config.yaml`，填写实际机器人 IP 和左目设备，并核对这些字段（保留其余现场设置）：

```yaml
action_space: tcp
policy:
  host: 127.0.0.1
  port: 8000
cameras:
  devices: [<head-left>, <left-wrist-left>, <right-wrist-left>]
  width: 256
  height: 192
control:
  hz: 20
async:
  enabled: false
```

`<...>` 替换为实际设备路径。图像必须为全视野 RGB uint8、256×192；实际采集尺寸和视野要与训练记录一致。沿用同一套 ORION 双臂硬件，现场核对的是记录与 SDK 的 TCP 坐标、原始夹爪量纲和相机对应。不要套用二值夹爪或重复归一化。

### 3. GPU 端启动服务

先激活 GPU 环境，将以下路径替换为该机器的实际路径：

```bash
export OPENWAM=/absolute/path/to/OpenWAM
export ORION=/absolute/path/to/orion-0
export FASTWAM=/absolute/path/to/FastWAM
export CHECKPOINT=/absolute/path/to/final-run/checkpoint_step_9090.safetensors
export PYTHONPATH="$OPENWAM:$ORION/src:$FASTWAM/src${PYTHONPATH:+:$PYTHONPATH}"

python -m openwam.deploy.realrobot_v3 \
  --checkpoint "$CHECKPOINT" \
  --config "$OPENWAM/configs/realrobot_v3_inference.yaml"
```

默认监听 `127.0.0.1:8000`。GPU 与机器人分机部署时，在机器人端另开终端并保持 SSH 隧道运行，目标必须是实际运行服务的 GPU 主机：

```bash
ssh -N -L 8000:127.0.0.1:8000 user@gpu-host
```

ORION policy 地址保持 `127.0.0.1:8000`。协议使用 pickle，只用于可信连接。

### 4. 机器人端先检查，再开启运动

在机器人端激活原有 AIRBOT 环境，并设置本机路径：

```bash
export OPENWAM=/absolute/path/to/OpenWAM
export ORION=/absolute/path/to/orion-0
export ROBOT_CONFIG=/absolute/path/to/site-deployment-config.yaml
export PYTHONPATH="$ORION/src${PYTHONPATH:+:$PYTHONPATH}"

python "$OPENWAM/scripts/realrobot_v3_client.py" \
  --config "$OPENWAM/configs/realrobot_v3_inference.yaml" \
  --robot-config "$ROBOT_CONFIG"
```

默认仅采集并推理一次，不复位、不切换运动模式、不发送动作。确认观测、TCP 坐标、夹爪数值和图像对应正确，按 SDK 实际范围填写 `client.raw_gripper_limits`（默认 `[0,1]` 为待核实值），再将 `client.interfaces_verified` 设为 `true`。开启运动：

```bash
python "$OPENWAM/scripts/realrobot_v3_client.py" \
  --config "$OPENWAM/configs/realrobot_v3_inference.yaml" \
  --robot-config "$ROBOT_CONFIG" \
  --enable-motion
```

程序先预热，按 Enter 后重新采集观测再运动，不执行预热输出。默认每轮执行 4 个目标，最多 20 个目标点。保留默认位姿和 anchor 漂移检查；客户端稳态往返上限为 2 秒。

H100 实测预热后推理 0.608/0.618/0.630 秒（依次为 Wan 叠杯、Wan 交接、Stereo 叠杯）；峰值显存 Wan 约 23.4 GiB、Stereo 约 37.2 GiB。Stereo 首次推理约 30 秒，需等待预热完成。当前行为是“同步等待推理 → 以 20 Hz 执行 4 步”，并非持续 20 Hz 闭环；网络往返耗时需现场观察。

更多接口约定、验证记录和日志路径见 [真机 v3 推理记录](docs/26-11-13/26-11-13-realrobot-v3-inference.md)。

## What is OpenWAM

OpenWAM is an open research stack for systematically developing **World-Action Models (WAMs)**. It turns tightly coupled design choices into modular components and controlled experiments. It consists of:

- **OpenWAM-Infra:** A modular infrastructure for composing and comparing model, representation, training, inference, deployment, and evaluation choices.
- **OpenWAM-Study:** Controlled studies that derive practical principles for inheriting world knowledge, coupling world and action learning, and scaling across domains.
- **OpenWAM-α:** An open pretrained WAM that applies these principles at scale, trained on 518.5M frames (about 6,400 hours) of egocentric human and robot data.

<p align="center">
  <img src="assets/repo_images/infra.jpg" alt="OpenWAM infrastructure overview">
</p>

<!--
## Repository Layout

```text
OpenWAM/
├── openwam/
│   ├── dataloader/    # Dataset adapters (RoboTwin), transforms, processors, registry
│   ├── model/
│   │   ├── architectures/    # WAM families: dual_system, single_system, tri_system
│   │   ├── action_backbone/  # ActionBackbone ABCs, separate ActionDiT, shared action backbone,
│   │   │                     #   latent action encoder/decoder, scheduler
│   │   ├── video_backbone/   # VideoBackbone ABC, Wan backbones, encoder/ (VAE / DINOv3 / V-JEPA 2.1)
│   │   └── vlm_backbone/     # VlmBackbone ABC, Qwen3-VL backbone
│   ├── train/         # OpenWAMTrainer, flow-match loss, checkpointing, optimizer utils
│   └── deploy/        # Policy server, model loader, inference engine, executors, optimizations
├── scripts/           # Entrypoints: train.sh, deploy.sh, inference tooling, SVAE / LAPA tooling
├── configs/           # Hydra configs for model, dataloader, training, deploy
├── benchmarks/
│   ├── robotwin/      # RoboTwin eval client, single / multi eval scripts
│   ├── libero/        # LIBERO WebSocket eval client
│   ├── libero-plus/   # LIBERO-plus perturbation-suite eval client
│   ├── robocasa365/   # RoboCasa365 native-action eval client
│   ├── robocasa_gr1/  # RoboCasa GR1 tabletop eval client
│   ├── vlabench/      # VLABench eval client, single / multi-GPU track sweeps
│   ├── ebench/        # EBench (GenManip) eval bridge
│   └── robodojo/      # RoboDojo: training in OpenWAM, evaluation via XPolicyLab
├── assets/            # Base-model checkpoints (created by the download script; git-ignored)
└── third_party/       # Vendored externals (Cosmos-Predict2.5 submodule)
```

## Support Status

### Architectures

| Architecture | Variant | Description |
|---|---|---|
| `single_system` | `vanilla` | Single shared DiT carries video + action + state tokens in one sequence |
| `single_system` | `moe` | Shared DiT with mixture-of-experts FFN layers (expert FFN on the bridge layers) |
| `dual_system` | `joint_self_attn` | Separate ActionDiT + video DiT, fused per layer via one mixed self-attention (MoT driver). |
| `dual_system` | `joint_cross_attn` | Video DiT runs to completion → bridge features → ActionDiT runs once with cross-attention to them. Sub-variants via `detach_bridge`: `false` lets action gradients flow back into the video DiT, `true` blocks them (ActionDiT trains on detached video features) |
| `dual_system` | `idm` | Inverse-dynamics-style teacher-forcing training + two-stage inference; Wan, Cosmos-Predict2.5 and Cosmos3-Edge |
| `tri_system` | `joint_self_attn` | Adds a frozen VLM understanding expert to the joint self-attention sequence (`[video + action + understanding]`) |

All architectures are selected via `configs/model/<framework>.yaml` with `architecture.variant`. The video backbone is composed from the Hydra `video_backbone` group (default `wan22_ti2v_5b`).

### Benchmarks and Evaluation

| Benchmark | Status | Notes |
|---|---|---|
| RoboTwin eval | Supported | All 50 tasks; see `benchmarks/robotwin/` |
| SimplerEnv eval | Planned | Requires external environment setup |
| LIBERO eval | Supported | See `benchmarks/libero/` |
| LIBERO-plus eval | Supported | Perturbation-robustness suite over LIBERO; see `benchmarks/libero-plus/` |
| RoboCasa365 eval | Supported | Native state19/action15 contract; see `benchmarks/robocasa365/` |
| RoboCasa GR1 eval | Supported | GR1 tabletop tasks; see `benchmarks/robocasa_gr1/` |
| VLABench eval | Supported | 10 primitive tasks across 6 evaluation tracks; see `benchmarks/vlabench/` |
| EBench eval | Supported | GenManip generalist tasks; see `benchmarks/ebench/` |
| RoboDojo eval | External | Trains in OpenWAM (sim + real); evaluates via [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) |
| Calvin eval | Planned | Requires external environment setup |
-->

## News
- **[2026/09/09]** 📄 OpenWAM [Paper](https://arxiv.org/pdf/2609.07398) is released on arXiv.
- **[2026/09/06]** 🤖 OpenWAM is integrated into [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab).
- **[2026/09/06]** 🤗 We release all pretrained and finetuned models on [huggingface](https://huggingface.co/OpenWAM).
- **[2026/09/06]** 🔥 OpenWAM Codebase Release！

## Installation

Choose one installation route:

| Route | Use it when you want to… |
|---|---|
| [Docker](assets/openwam_usage_docs/docker.md) | run a CUDA policy server, train, develop in a container, or deploy to an offline GPU host |
| [Native Python environment](#native-python-environment) | manage Python and CUDA dependencies directly on your machine |

Docker includes the project's Python environment. GPU hosts still need an
NVIDIA driver and NVIDIA Container Toolkit; see the guide's host requirements.

### Native Python environment

Create an environment with **conda**:

```bash
# Requires Python >= 3.10
conda create -n openwam python=3.10
conda activate openwam
```

or with **venv**:

```bash
# Requires Python >= 3.10 (check with `python3 --version`)
python3 -m venv .venv
source .venv/bin/activate
```

We recommend using PyTorch 2.7.1 with CUDA 12.8 (others may also work):

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install OpenWAM:

```bash
pip install -e .
```

> `deepspeed` ships as a source distribution and is compiled during `pip install`, so a C compiler (`gcc`) must be on the PATH.

<details>
<summary><b>Cosmos-Predict2.5 Extras (Optional)</b> — needed only for experiments with the <code>cosmos_predict25_2b</code> video backbone</summary>

With your environment activated:

```bash
git submodule update --init third_party/cosmos-predict2.5
bash scripts/install_cosmos_predict25.sh
```

The script installs the upstream cosmos packages into the active environment and compiles `transformer-engine` (CUDA toolkit with `nvcc` required), then automatically restores the package versions OpenWAM pins.

</details>

## Assets Preparation

The downloaders are interactive by default; every menu step also has a flag (`--name`, `--source`, `--root`, `--yes`, see `--help`) so they can run unattended, and the default storage location is resolved relative to the repository regardless of the working directory. Component downloaders store assets under
`assets/` and update the matching YAML path; the released-checkpoint downloader
keeps each checkpoint's self-contained config unchanged.

### 1. Video Backbone

~~~bash
python scripts/download_assets/download_video_backbone.py
~~~

**Supported video backbones**

<table>
<tr>
<td>Wan2.2-TI2V-5B ✅</td>
<td>Wan2.1-VACE-1.3B ✅</td>
<td>Wan2.1-I2V-14B-480P ✅</td>
</tr>
<tr>
<td>Cosmos-Predict2.5-2B ✅</td>
<td>Cosmos3-Edge ✅</td>
<td></td>
</tr>
</table>

Weights are saved under `assets/video_backbone_ckpt/` and the selected
`configs/model/video_backbone/` file is updated with the downloaded path.

### 2. Benchmark Data

~~~bash
python scripts/download_assets/download_benchmark_data.py
~~~

**Supported benchmarks**

<table>
<tr>
<td>RoboTwin2.0 ✅</td>
<td>RoboDojo ✅</td>
<td>RoboDojo-Real ✅</td>
</tr>
<tr>
<td>LIBERO ✅</td>
<td>VLABench ✅</td>
<td>EBench ✅</td>
</tr>
<tr>
<td>RoboCasa365 ✅</td>
<td>RoboCasa_GR1 ✅</td>
<td></td>
</tr>
</table>

Data is saved under `assets/benchmark_data/<benchmark>/`. Normalization
statistics are prepared when needed, and the selected dataloader configuration
is updated.

### 3. VLM Backbone (Optional)

Required only by tri_system:

~~~bash
python scripts/download_assets/download_vlm_backbone.py
~~~

**Supported VLM backbones**

<table>
<tr>
<td>Qwen3-VL-2B-Instruct ✅</td>
</tr>
</table>

Weights are saved under `assets/vlm_backbone_ckpt/`, and the selected configuration is updated.

### 4. Visual Encoders (Optional)

Required only for video backbones that use an external encoder:

~~~bash
python scripts/download_assets/download_visual_encoder.py
~~~

**Supported visual encoders**

<table>
<tr>
<td>DINOv3 ViT-B/16 ✅</td>
<td>V-JEPA 2.1 ViT-G/16 ✅</td>
</tr>
<tr>
<td>Wan2.2 VAE ✅</td>
<td>FLUX.2 VAE ✅</td>
</tr>
</table>

Weights are saved under `assets/visual_encoder_ckpt/`, and the selected encoder configuration is
updated.

### 5. Released OpenWAM Checkpoints

Use this downloader to obtain OpenWAM-Alpha releases or OpenWAM-Study
checkpoints from the OpenWAM collection:

~~~bash
python scripts/download_assets/download_openwam_checkpoints.py
~~~

Checkpoints are saved under `assets/openwam_ckpt/openwam_alpha/` or
`assets/openwam_ckpt/openwam_study/<type>/`. Each checkpoint directory contains
its own config and can be deployed directly with:

~~~bash
bash scripts/deploy.sh <ckpt_dir_path>
~~~

For fine-tuning, set training.finetune_ckpt_path in `configs/train.yaml` to the
downloaded checkpoint directory. Benchmark data is still required.


## Quick Start

Quick Start provides a minimal end-to-end example: prepare the assets, train
or fine-tune a policy, deploy its checkpoint, and run a first inference check.

The example uses the DualSystem JointSelfAttention architecture, the
Wan2.2-TI2V-5B video backbone, and the Mutual attention mask.

| Component | Selection | Configuration |
|---|---|---|
| Architecture | dual_system / joint_self_attn | [`configs/model/dual_system.yaml`](configs/model/dual_system.yaml) |
| Video backbone | wan22_ti2v_5b | [`configs/model/video_backbone/wan22_ti2v_5b.yaml`](configs/model/video_backbone/wan22_ti2v_5b.yaml) |
| Attention mask | mutual | model.architecture.attention_mask_mode |
| Dataset | libero | [`configs/dataloader/libero.yaml`](configs/dataloader/libero.yaml) |

> **Resource recommendation:** We recommend 8 GPUs with 80 GB VRAM each for
> training. This configuration supports normal training for all architectures
> using Wan2.2-5B or smaller video backbones. More GPUs are better when
> available and can further improve training throughput.

### From Scratch Training

1. Download LIBERO and let the downloader update its dataloader configuration:

   ~~~bash
   python scripts/download_assets/download_benchmark_data.py
   ~~~

   Select LIBERO in the interactive menu.

2. Download Wan2.2-TI2V-5B:

   ~~~bash
   python scripts/download_assets/download_video_backbone.py
   ~~~

   Select Wan2.2-TI2V-5B and the desired model source.

3. Start a debug run with the complete model selection:

   ~~~bash
   bash scripts/train.sh \
     dataloader=libero \
     model=dual_system \
     model/video_backbone=wan22_ti2v_5b \
     model.architecture.variant=joint_self_attn \
     model.architecture.attention_mask_mode=mutual \
     training.debug=true
   ~~~

   training.debug=true runs 20 steps, saves at steps 10 and 20, and uses a
   constant learning rate. Check the run output, then set
   training.debug=false for normal training. Training defaults and CLI
   overrides are defined in [`configs/train.yaml`](configs/train.yaml).
   Debug outputs use training.output_path, whose default is
   `outputs/openwam_checkpoints`.

4. Deploy the debug checkpoint and inspect one input-output cycle:

   ~~~bash
   bash scripts/deploy.sh <debug_ckpt_dir_path>
   ~~~

   Keep the server running, then open another terminal and run the two
   inference helpers. Deployment enables compile by default, so the first
   inference may take longer while compilation warms up; later requests are
   typically faster:

   ~~~bash
   python scripts/inference_test/inference_single_test.py \
     --server ws://127.0.0.1:8848 --test --state-dim 10

   python scripts/inference_test/inference_continuous_test.py \
     --server ws://127.0.0.1:8848 --test --state-dim 10
   ~~~

   The single-request helper checks ping, one prediction, and reset. The
   continuous helper sends repeated predictions over one connection and
   reports the returned action dimension and latency. Stop the deployment
   process after the checks.

### OpenWAM-α Fine-Tuning

1. Download LIBERO as shown above.

2. Download the OpenWAM-Alpha foundation checkpoint:

   ~~~bash
   python scripts/download_assets/download_openwam_checkpoints.py
   ~~~

   Select OpenWAM_Alpha and OpenWAM-Alpha-Pretrain-Foundation-Model.
   Keep the resulting directory as <foundation_ckpt_dir_path>.

3. Start fine-tuning from that directory:

   ~~~bash
   bash scripts/train.sh \
     dataloader=libero \
     training.finetune_ckpt_path=<foundation_ckpt_dir_path>
   ~~~

   The default model configuration already matches the required setup above.
   You can set the same field in [`configs/train.yaml`](configs/train.yaml)
   instead of passing it on the command line. num_frames=33 and
   video_stride=4 in [`configs/dataloader/libero.yaml`](configs/dataloader/libero.yaml)
   produce the 32-step action horizon expected by the sampler; no separate
   action_chunk override is needed.

4. Deploy the resulting checkpoint directory:

   ~~~bash
   bash scripts/deploy.sh <ckpt_dir_path>
   ~~~

   Install and run the LIBERO client according to the
   [LIBERO evaluation guide](benchmarks/libero/README.md).


## OpenWAM Usage Guidance

OpenWAM is configured through composable Hydra YAML files. Select an architecture,
backbone, dataloader, and runtime behavior by changing configuration values or
overriding them on the command line. The task guides below are the maintained
entry points for using and extending the repository:

| Guide | Use it when you want to… |
|---|---|
| [Docker and offline deployment](assets/openwam_usage_docs/docker.md) | build an image, run a first policy, train, develop with Compose, or deploy to an offline GPU host |
| [Training and deployment](assets/openwam_usage_docs/train-and-deploy.md) | choose a model/dataloader, prepare assets, train, fine-tune, resume, or deploy a policy |
| [Architecture extension](assets/openwam_usage_docs/architecture-extension.md) | extend a video, visual, VLM, action backbone, or WAM architecture |
| [Benchmark integration](assets/openwam_usage_docs/benchmark-integration.md) | extend a dataloader and connect a benchmark client to the WebSocket protocol |
| [OpenWAM-α fine-tuning](assets/openwam_usage_docs/openwam-alpha-finetuning.md) | fine-tune the released foundation checkpoint to execute downstream task |

> 💡 **Agent tip:** Pick the guide that matches your task and feed it directly
> to your agent — less explaining, more building.

Installation and Assets Preparation above cover environment setup and model or
dataset downloads. Benchmark-specific environment and evaluation details remain
in the benchmarks directory.

## Projects Based on OpenWAM

- **[LD4WAM](https://github.com/stubborn111/LD4WAM)** — *LD4WAM: Learning Latent Dynamics from Human Videos for World Action Models* ([arXiv:2608.22403](https://arxiv.org/abs/2608.22403)). Learns a Latent Dynamics Model (LDM) from human videos and uses it to build a World Dynamics Action Model (WDAM) — a Wan2.2 video DiT coupled with a latent-dynamics expert and an action expert — on top of the OpenWAM framework, evaluated on RoboTwin 2.0.

If you build on OpenWAM, feel free to open a PR to add your project here.

## Star History

<p align="center">
  <a href="https://star-history.com/#OpenWAM-Official/OpenWAM&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=OpenWAM-Official/OpenWAM&amp;type=Date&amp;theme=dark">
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=OpenWAM-Official/OpenWAM&amp;type=Date">
      <img src="https://api.star-history.com/svg?repos=OpenWAM-Official/OpenWAM&amp;type=Date" alt="Star history of OpenWAM-Official/OpenWAM" width="70%">
    </picture>
  </a>
</p>

## License

OpenWAM is released under the [Apache License 2.0](LICENSE).

## Citation

If you use OpenWAM, please cite:

```bibtex
@article{wang2026openwam,
  title   = {OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining},
  author  = {Yuran Wang and Siqiao Huang and Mingleyang Li and Chenhao Zhang and Jiaqi Liang and Weiyang Jin and Yue Chen and Xuemin Chi and Donghao Zhou and Qize Yu and Yu-Kai Wang and Yuhan Rui and Shenzhe Yao and Zhen Yuan and Zhenhao Shen and Kefei Zhu and Zijie Zhu and Ning Gao and Xiaowei Chi and Guanqi He and Shanghang Zhang and Hao Dong and Lin Shao and Hang Zhao},
  year    = {2026},
  journal = {arXiv preprint arXiv: 2609.07398}
}
```

## Contact

Have a question, want to share results, or just follow development? Join us:

<table align="center">
  <tr>
    <td align="center" width="360">
      <img src="assets/repo_images/WeChat_Group.png" alt="OpenWAM WeChat group QR code" height="300">
      <br><br>
      <b>WeChat Group</b>
      <br>
      Scan to join
    </td>
    <td align="center" width="360">
      <a href="https://discord.gg/yRSqhpNXu"><img src="assets/repo_images/Discord_QRCode.png" alt="OpenWAM Discord QR code" height="300"></a>
      <br><br>
      <b>Discord</b>
      <br>
      <a href="https://discord.gg/yRSqhpNXu">discord.gg/yRSqhpNXu</a>
    </td>
  </tr>
</table>
