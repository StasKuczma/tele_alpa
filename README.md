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
forces one fresh answer, and while a question is unanswered the VQA path
**preempts** the trajectory tick (both share the one GPU worker), so a new
question is always picked up within one `vqa_period_sec`.

The question the node is actually using is echoed (latched) on
`<question_topic>/active` — `ros2 topic echo /tele_alpa/question/active` confirms
a runtime `/tele_alpa/question` was received.

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

## Operator assist — stop → describe → propose → approve

When the ego's longitudinal velocity goes **non-positive** (`twist.twist.linear.x
<= stopped_speed_threshold`, default 0.1 m/s) and stays there for
`stopped_duration_sec` (default 5 s), the node runs one assist pass:

1. **Describe** — VQA with `assist_question` → what is blocking the path.
2. **Propose** — a nav-conditioned trajectory (`assist_nav_text`, default *"The
   lane ahead is blocked. Plan a path around the obstacle."*) → a way around.
3. **Ping** — a JSON message to the operator on `/tele_alpa/operator/ping`
   (latched, so a tool that attaches late still sees it).
4. **Approve** — the proposal is published on
   `/tele_alpa/operator/proposed_trajectory` only. It reaches
   `/tele_alpa/operator/approved_trajectory` **only after** the operator approves.

The assist pass preempts the routine trajectory and VQA ticks (they share one GPU
worker). It's all plain ROS topics — wiring `approved_trajectory` into Autoware's
planning/control mux is deliberately left out.

### Topics

| Topic | Type | Direction |
|---|---|---|
| `/tele_alpa/operator/ping` | `std_msgs/String` (JSON, latched) | out — the operator request |
| `/tele_alpa/operator/ping_stamped` | `StringStamped` | out |
| `/tele_alpa/operator/status` | `std_msgs/String` (JSON, latched) | out — state machine |
| `/tele_alpa/operator/proposed_trajectory` | `autoware_planning_msgs/Trajectory` (latched) | out — **suggestion only** |
| `/tele_alpa/operator/proposed_trajectory_markers` | `MarkerArray` (latched) | out — orange path + caption for RViz |
| `/tele_alpa/operator/approval` | `std_msgs/String` | **in** — `approve[:<id>]` / `reject[:<id>]` |
| `/tele_alpa/operator/approved_trajectory` | `autoware_planning_msgs/Trajectory` | out — published on approval |

State machine: `IDLE → STOPPED → ASSESSING → AWAITING_OPERATOR → APPROVED / REJECTED / EXPIRED`,
returning to `IDLE` as soon as the ego moves again.

### Using it

```bash
# watch for a request
ros2 topic echo /tele_alpa/operator/ping
ros2 topic echo /tele_alpa/operator/status

# the ping's JSON carries a copy-paste approve/reject command, e.g.
ros2 topic pub --once /tele_alpa/operator/approval std_msgs/String "{data: 'approve:1a2b3c4d'}"
ros2 topic pub --once /tele_alpa/operator/approval std_msgs/String "{data: 'reject:1a2b3c4d'}"

# bare 'approve' / 'reject' (no id) act on the outstanding proposal
ros2 topic pub --once /tele_alpa/operator/approval std_msgs/String "{data: 'approve'}"

# then
ros2 topic echo /tele_alpa/operator/approved_trajectory
```

A proposal that gets no answer within `proposal_ttl_sec` (default 120 s) EXPIREs.
After an approve / reject / expiry the node waits `assist_cooldown_sec`
(default 30 s) before proposing again while the ego is still stopped.

> The proposed path is **not** collision-checked, drivable-area-checked, or
> feasibility-checked. Validate it before anything acts on it — operator approval
> must not be the only safety gate.

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
| `vqa_use_trajectory_input` | `true` | Fuse ego trajectory-history tokens into the VQA prompt (same input the planner uses); biases answers toward driving actions — set `false` for description-style answers |
| `vqa_system_prompt` | `"You are a driving assistant. …answer the question about the scene directly…"` | System prompt for the VQA path; empty string keeps the upstream *"generates safe and accurate actions"* prompt |
| `vqa_max_generation_length` | `384` | Token budget for the answer (too small → empty answer) |
| `vqa_num_samples` | `1` | Answer samples per VQA run |

Operator assist:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `operator_assist_enabled` | `true` | Run the stop → describe → propose → approve flow |
| `stopped_speed_threshold` | `0.1` | m/s; `linear.x` at or below this counts as stopped |
| `stopped_duration_sec` | `5.0` | Debounce, so ordinary traffic-light stops don't ping |
| `assist_check_period_sec` | `0.5` | How often the stop condition / TTL is evaluated |
| `proposal_ttl_sec` | `120.0` | Approval window before the proposal EXPIREs |
| `assist_cooldown_sec` | `30.0` | Wait after approve/reject/expiry before re-proposing |
| `assist_question` | *"…describe what is blocking the path ahead…"* | VQA prompt for the description |
| `assist_nav_text` | *"The lane ahead is blocked. Plan a path around the obstacle."* | Route hint for the proposed path; empty → reuse the lanelet2 nav text |
| `operator_ping_topic` / `operator_ping_stamped_topic` / `operator_status_topic` | `/tele_alpa/operator/…` | Outputs |
| `proposed_trajectory_topic` / `approval_topic` / `approved_trajectory_topic` | `/tele_alpa/operator/…` | Proposal, operator input, approved output |

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

## Reasoning vs. VQA

Both are text from the same model over the same camera frames, but they are
produced by different prompts and different decode paths:

| | Reasoning (chain-of-thought) | VQA |
|---|---|---|
| Prompt | `helper.create_message` — seeds `<|cot_start|>`, asks for "the chain-of-thought reasoning of the driving process, then the future trajectory" | `helper.create_vqa_message` — wraps your text in `<|question_start|>..<|question_end|>`, seeds `<|answer_start|>` |
| Decode | `model.sample_trajectories_*` — VLM rollout **+** diffusion over `<i..>` trajectory tokens | `model.generate_text` — plain autoregressive decode, no diffusion, no trajectory tokens |
| Output | an Autoware `Trajectory` (+ markers); the text is a by-product returned as `extra["cot"]` / `extra["meta_action"]` | text only, returned as `extra["answer"]` |
| Answers your question? | **No** — the planning prompt is fixed; CoT explains *why this trajectory* | **Yes** |
| Topic | `cot_topic` (`/alpamayo/reasoning`) | `answer_topic` (`/tele_alpa/answer`) |

Shared: the 10B model, the 4-camera × 4-frame image stack, and the optional
fused `<|traj_history|>` ego-motion tokens.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`** — recreate the venv with Python 3.10.
- **VQA never answers, only the trajectory runs** — both timers share one GPU
  worker; the node now lets an unanswered question preempt the trajectory tick.
  Confirm the question arrived with `ros2 topic echo /tele_alpa/question/active`,
  and look for `Running VQA for question:` in the node log.
- **Empty VQA answer** — the model did not emit `<|answer_end|>` within the token
  budget. The node now publishes an explicit `[no answer: ...]` string instead of
  an empty one; raise `vqa_max_generation_length` (default 384).
- **Answer comes back as `[source=cot]` / `[source=meta_action]` or reads like a
  driving instruction** (`Q: Is it safe to change left?` → `A: Nudge left to
  clear the stopped van`) — with `vqa_use_trajectory_input:=true` (default) the
  fused `<|traj_history|>` tokens + the *"...generates safe and accurate
  actions"* system prompt put the model into a **trajectory-aware rollout**: it
  answers in short driving-action form and closes `<|cot_end|>` rather than
  `<|answer_end|>`. The reply is usually still on-topic; it's just planning-style.
  For plain descriptive Q&A run:

  ```bash
  ros2 launch tele_alpa tele_alpa.launch.py \
    question:="Describe the scene." vqa_use_trajectory_input:=false
  ```

  which gives the model a clean `<|answer_start|>..<|answer_end|>` VQA prompt.
  The node also overrides the system prompt (`vqa_system_prompt`, blank = keep
  upstream). Confirm the question is actually steering the output by publishing
  two different questions and diffing the answers, and check
  `ros2 topic echo /tele_alpa/question/active` shows the one you sent.
- Note: Alpamayo 1.5 is a **driving** VLA, not a general VQA model — even in pure
  VQA mode its answers stay terse and driving-centric. That is a model capability
  limit, not a node bug.
- **CUDA OOM** — use a GPU with 24 GB+ VRAM, or increase `inference_period_sec`.
- **Flash Attention issues** — set `config.attn_implementation = "sdpa"` as a fallback.

## License

Inference code: Apache License 2.0. Model weights: non-commercial
([HuggingFace model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B)).
Alpamayo 1.5 is a research model, not a complete autonomous-driving stack, and is
not intended for production use.
