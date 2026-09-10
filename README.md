# tele_alpa — Alpamayo 1.5 ROS 2 (Trajectory + VQA)

ROS 2 package within **alpamayo-autoware**. It sits next to `alpamayo_ros` and
reuses the same `alpamayo1_5` inference library from this repo.

## `tele_alpa_node` — one model, both jobs

The main node. It loads the Alpamayo 1.5 model **once** and runs, from the same
buffered input:

- **trajectory planning** — Autoware `Trajectory` + chain-of-thought + RViz
  markers + navigation text (identical to `alpamayo_ros`), and
- **VQA** — answers to a free-form question about the live scene.

The two inference paths share one worker thread, so they never run on the GPU at
the same time. Both consume the 4-camera × 4-frame image stack **plus the ego
trajectory history** (fused `<|traj_history|>` tokens — the same trajectory
input the planner uses; VQA side is toggled with `vqa_use_trajectory_input`).

Inference only runs when **new camera data has arrived** since the last run
(`require_new_data`, default `true`), so replaying a bag and letting it end no
longer produces trajectories/answers on a frozen buffer. A question change still
forces one fresh answer.

```bash
ros2 launch tele_alpa tele_alpa.launch.py question:="Is it safe to merge left?"
```

```text
Camera × 4 (CompressedImage)   Odometry            Question (std_msgs/String)
      │                            │                       │
      ▼                            ▼                       ▼
┌───────────────────────────────────────────────────────────────────┐
│                          tele_alpa_node                            │
│                     one Alpamayo 1.5 model                         │
│  GPU JPEG decode ─┬─► create_message  ─► sample_trajectories ─┐    │
│  + ego history ───┤                                           │    │
│                   └─► create_vqa_message ─► generate_text ────┤    │
│           (shared single-worker executor — never concurrent)  │    │
└───────────────────────────────────────────────┬──────────────┘    │
        /alpamayo/predicted_trajectory (+ _markers, /alpamayo/reasoning,
        /alpamayo/nav_text)   +   /tele_alpa/answer (+ _stamped)
```

### Other executables (optional)

| Executable | Purpose |
|------------|---------|
| `alpamayo_node` | Trajectory only — loads its own model (`alpamayo.launch.py`) |
| `vqa_node` | VQA only — loads its own model (`vqa.launch.py`) |
| `ask` | CLI helper: `ros2 run tele_alpa ask "..."` |

Use these only if you specifically want a single capability in isolation;
`tele_alpa_node` is the one to run normally.

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
dependencies live in `a1_5_venv`, so keep that venv activated when launching the
node.

## Running

```bash
source /opt/ros/humble/setup.bash
source a1_5_venv/bin/activate
source install/setup.bash

ros2 launch tele_alpa tele_alpa.launch.py \
  question:="Is it safe to change into the left lane?"

# rosbag replay
ros2 launch tele_alpa tele_alpa.launch.py use_sim_time:=true
ros2 bag play <bag_path> --clock --rate 0.5
```

Trajectory goes to `/alpamayo/predicted_trajectory` (+ `/alpamayo/reasoning`,
`_markers`, `/alpamayo/nav_text`); answers to `/tele_alpa/answer`.

### Ask questions at runtime

```bash
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

python3 ./src/tele_alpa/tele_alpa/tele_alpa_node.py --ros-args \
  -p question:="Describe the current traffic scene." \
  -p camera_topics:="['/sensing/camera/camera3/image_raw/compressed', \
  '/sensing/camera/camera1/image_raw/compressed', \
  '/sensing/camera/camera4/image_raw/compressed', \
  '/sensing/camera/camera2/image_raw/compressed']" \
  -p camera_indices:="[0, 1, 2, 6]"
```

## Parameters (`tele_alpa_node`)

All `alpamayo_ros` trajectory parameters (`camera_topics`, `camera_indices`,
`odometry_topic`, `route_topic`, `trajectory_topic`, `cot_topic`,
`num_diffusion_steps`, `expert_onnx_path`, `use_greedy_decode`, `top_p`,
`temperature`, `max_generation_length`, `lanelet2_map_path`, `use_sim_time`) plus:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `require_new_data` | `true` | Only infer when a new camera frame has arrived since the last run — stops trajectory/answer output on a frozen buffer after a bag ends |
| `inference_period_sec` | `1.0` | Trajectory inference trigger period |
| `vqa_enabled` | `true` | Run the VQA path too |
| `vqa_period_sec` | `2.0` | VQA trigger period |
| `question` | `"What should the ego vehicle do next, and why?"` | Initial question |
| `question_topic` | `/tele_alpa/question` | Runtime question input (`std_msgs/String`) |
| `answer_topic` / `answer_stamped_topic` | `/tele_alpa/answer` / `_stamped` | Answer outputs (`String` / `StringStamped`) |
| `answer_once` | `false` | Answer once per question change instead of every `vqa_period_sec` |
| `vqa_use_trajectory_input` | `true` | Fuse ego trajectory-history tokens into the VQA prompt (same input the planner uses) |
| `vqa_max_generation_length` | `256` | Token budget for the answer |
| `vqa_num_samples` | `1` | Answer samples per VQA run |

Sampling (`use_greedy_decode` / `top_p` / `temperature`) is shared by both paths.

## Single-capability nodes

`alpamayo_node` / `vqa_node` match `alpamayo_ros` and load their own model each.
Both also honour `require_new_data`; `vqa_node` additionally takes
`odometry_topic` + `use_trajectory_input`. Quick start:

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
