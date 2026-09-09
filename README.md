# tele_alpa — Alpamayo 1.5 ROS 2 (Trajectory + VQA)

ROS 2 package within **alpamayo-autoware**. It sits next to `alpamayo_ros` and
reuses the same `alpamayo1_5` inference library from this repo. It provides
**two** nodes:

| Node | Executable | Purpose |
|------|------------|---------|
| Trajectory planner | `alpamayo_node` | End-to-end trajectory planning (verbatim copy of `alpamayo_ros`, only the package name changed) |
| Visual QA | `vqa_node` | Ask the Alpamayo model natural-language questions about the live camera feed |

Plus a small CLI helper, `ask`, for one-off questions.

## Architecture

```text
Camera Topics (CompressedImage × 4)          Question (std_msgs/String, optional)
        │                                             │
        ▼                                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     tele_alpa / vqa_node                     │
│                                                             │
│  GPU JPEG Decode ──► Tokenizer ──► VLM (BF16) ──► generate_text
│  (torchvision)                        │                      │
│                              <|answer_start|> … <|answer_end|>
└──────────────────────────────┬──────────────────────────────┘
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
                /tele_alpa/answer   /tele_alpa/answer_stamped
```

## Setup

Same environment as `alpamayo_ros` — there is nothing extra to install. From the
`alpamayo-autoware` repo root:

```bash
# 1. Python deps (shared with alpamayo_ros)
uv venv a1_5_venv --python python3.10
source a1_5_venv/bin/activate
uv sync --active
huggingface-cli login          # needs access to nvidia/Alpamayo-1.5-10B

# 2. Build the ROS 2 package
source /opt/ros/humble/setup.bash
colcon build --packages-select tele_alpa --symlink-install
source install/setup.bash
```

`colcon build` only installs the thin ROS 2 wrapper; the heavy `alpamayo1_5`
dependencies live in `a1_5_venv`, so keep that venv activated when launching
either node.

## Visual Question Answering

```bash
source /opt/ros/humble/setup.bash
source a1_5_venv/bin/activate
source install/setup.bash

# default question from the launch file
ros2 launch tele_alpa vqa.launch.py

# or set it at launch time
ros2 launch tele_alpa vqa.launch.py question:="Is it safe to change into the left lane?"
```

Once the camera buffers fill, the node answers the current question every
`inference_period_sec` and publishes to `/tele_alpa/answer`.

### Ask questions at runtime

```bash
# one-shot helper (vqa_node must be running)
ros2 run tele_alpa ask "How many pedestrians are near the crosswalk?"

# or publish directly
ros2 topic pub --once /tele_alpa/question std_msgs/String \
  "{data: 'What is the vehicle in front doing?'}"
ros2 topic echo /tele_alpa/answer
```

### Direct execution (no colcon)

```bash
source /opt/ros/humble/setup.bash
source a1_5_venv/bin/activate

python3 ./src/tele_alpa/tele_alpa/vqa_node.py --ros-args \
  -p question:="Describe the current traffic scene." \
  -p camera_topics:="['/sensing/camera/camera3/image_raw/compressed', \
  '/sensing/camera/camera1/image_raw/compressed', \
  '/sensing/camera/camera4/image_raw/compressed', \
  '/sensing/camera/camera2/image_raw/compressed']" \
  -p camera_indices:="[0, 1, 2, 6]"
```

### VQA parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `question` | `"What should the ego vehicle do next, and why?"` | Question answered on the timer |
| `question_topic` | `/tele_alpa/question` | Input topic to change the question at runtime (`std_msgs/String`) |
| `answer_topic` | `/tele_alpa/answer` | Output answer topic (`std_msgs/String`) |
| `answer_stamped_topic` | `/tele_alpa/answer_stamped` | Timestamped answer (`autoware_internal_debug_msgs/StringStamped`) |
| `camera_topics` | (required) | Camera image topics (`CompressedImage`) |
| `camera_indices` | (required) | Camera index per topic (0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear, 5=Rear right, 6=Front telephoto) |
| `inference_period_sec` | `2.0` | How often the current question is (re-)answered |
| `answer_once` | `false` | Answer once per question change instead of continuously |
| `use_greedy_decode` | `true` | Greedy decode (faster, deterministic) |
| `top_p` | `0.98` | Nucleus sampling threshold (when `use_greedy_decode:=false`) |
| `temperature` | `0.6` | Sampling temperature (when `use_greedy_decode:=false`) |
| `max_generation_length` | `256` | Token budget for the answer |
| `num_samples` | `1` | Answer samples to generate per tick |

## Trajectory planning

`alpamayo_node` is a verbatim copy of `alpamayo_ros`'s node; see the
`alpamayo_ros` docs / the alpamayo-autoware top-level README for the full
parameter and output-topic list. Quick start:

```bash
ros2 launch tele_alpa alpamayo.launch.py                 # live
ros2 launch tele_alpa alpamayo.launch.py use_sim_time:=true   # + `ros2 bag play <bag> --clock --rate 0.5`
```

Optimized (TRT expert) mode, same as `alpamayo_ros`:

```bash
ros2 launch tele_alpa alpamayo.launch.py \
  expert_onnx_path:=/path/to/engines/expert_step.int8.qdq.onnx \
  num_diffusion_steps:=5 use_greedy_decode:=true
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`** — recreate the venv with Python 3.10.
- **Empty VQA answer** — the model did not emit `<|answer_end|>` within the token budget; increase `max_generation_length`.
- **CUDA OOM** — use a GPU with 24 GB+ VRAM, or increase `inference_period_sec`.
- **Flash Attention issues** — set `config.attn_implementation = "sdpa"` as a fallback.

## License

Inference code: Apache License 2.0. Model weights: non-commercial
([HuggingFace model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B)).
Alpamayo 1.5 is a research model, not a complete autonomous-driving stack, and is
not intended for production use.
