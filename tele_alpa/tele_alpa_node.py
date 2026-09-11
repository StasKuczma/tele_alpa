#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""One ROS 2 node, one Alpamayo 1.5 model, everything ``alpamayo_ros`` does
plus scene question answering.

Outputs
-------
* an Autoware ``Trajectory`` (+ RViz markers),
* the planner's chain-of-thought "reasoning" and derived navigation text,
* a free-form answer to a question about the live scene (VQA),
* an **operator assist** request: when the ego's velocity goes non-positive for
  ``stopped_duration_sec``, the model describes what is blocking the path and
  drafts a way around it, pings a human on ``operator_ping_topic``, and only
  republishes that path on ``approved_trajectory_topic`` once the operator
  approves on ``approval_topic``. Plain ROS topics — no execution mux, and the
  proposal is *not* collision- or feasibility-checked.

The 10B model is loaded **once**. The trajectory path and the VQA path submit to
the same single-worker executor, so they never touch the GPU at the same time,
and both consume the same buffered input: a multi-camera x N-frame image stack
plus the ego trajectory history.

Reasoning vs. VQA
-----------------
Both are text from the same model over the same images, but they come from
different prompts and different decode routines:

* **Reasoning (chain-of-thought)** is a by-product of *planning*.
  ``helper.create_message`` seeds ``<|cot_start|>`` and asks the model to
  "output the chain-of-thought reasoning of the driving process, then output the
  future trajectory". ``sample_trajectories_*`` runs that rollout, decodes the
  ``<i..>`` trajectory tokens into a path, and returns the free text as
  ``extra["cot"]`` / ``extra["meta_action"]``. It explains *why this trajectory*;
  it does not answer a question. Published on ``cot_topic``.
* **VQA** is standalone Q&A. ``helper.create_vqa_message`` wraps the question in
  ``<|question_start|>..<|question_end|>`` and seeds ``<|answer_start|>``;
  ``model.generate_text`` does plain autoregressive decoding (no diffusion, no
  trajectory tokens) and returns ``extra["answer"]``. Text only, no trajectory.
  Published on ``answer_topic``.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
import torch
from autoware_internal_debug_msgs.msg import StringStamped
from autoware_planning_msgs.msg import LaneletRoute, Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

try:  # navigation text is optional — needs the lanelet2 python bindings
    import lanelet2
    from autoware_lanelet2_extension_python.projection import MGRSProjector

    _HAS_LANELET2 = True
except ImportError:
    _HAS_LANELET2 = False

DEFAULT_QUESTION = "What should the ego vehicle do next, and why?"
DEFAULT_VQA_SYSTEM_PROMPT = (
    "You are a driving assistant. Look at the camera images and answer the "
    "question about the scene directly, concisely, and factually."
)

# --- operator assist ------------------------------------------------------- #
DEFAULT_ASSIST_QUESTION = (
    "The ego vehicle has come to a stop. In one or two sentences, describe what "
    "is blocking the path ahead and why the vehicle cannot proceed. Answer in "
    "words, not coordinates."
)
DEFAULT_ASSIST_NAV_TEXT = "The lane ahead is blocked. Plan a path around the obstacle."

# Assist state machine. IDLE while the ego is moving; the rest is one stop event.
ASSIST_IDLE = "IDLE"
ASSIST_STOPPED = "STOPPED"  # below speed threshold, waiting out stopped_duration_sec
ASSIST_ASSESSING = "ASSESSING"  # model is describing the scene + drafting a path
ASSIST_AWAITING = "AWAITING_OPERATOR"  # proposal published, waiting for approve/reject
ASSIST_APPROVED = "APPROVED"
ASSIST_REJECTED = "REJECTED"
ASSIST_EXPIRED = "EXPIRED"

# Ego trajectory-history placeholder, identical to the one
# ``helper.create_message`` builds for planning. ``fuse_traj_tokens`` swaps these
# pad tokens for the encoded ego motion so the VLM gets the same trajectory input
# during VQA that it gets during planning.
_NUM_TRAJ_HISTORY_TOKENS = 48
_HIST_TRAJ_PLACEHOLDER = (
    "<|traj_history_start|>"
    + "<|traj_history|>" * _NUM_TRAJ_HISTORY_TOKENS
    + "<|traj_history_end|>"
)

_MODEL_INPUT_SIZE = (560, 1008)  # (H, W) the VLM image tower expects


class TeleAlpaNode(Node):
    """Alpamayo 1.5: Autoware trajectory planning + chain-of-thought + scene VQA."""

    def __init__(self) -> None:
        super().__init__("tele_alpa_node")

        self.model_name = "nvidia/Alpamayo-1.5-10B"
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16
        self._frame_id = "base_link"

        # /localization/kinematic_state is ~50 Hz; the model expects 10 Hz input.
        self._kinematic_hz = 50.0
        self._model_input_hz = 10.0
        self._odom_skip = int(self._kinematic_hz / self._model_input_hz)
        self._num_history_steps = 16
        self._num_frames = 4

        self._declare_parameters()

        # ------------------------------------------------------------------ #
        # Publishers                                                        #
        # ------------------------------------------------------------------ #
        qs = 10
        traj_topic = self._param("trajectory_topic")
        self._traj_pub = self.create_publisher(Trajectory, traj_topic, qs)
        self._marker_pub = self.create_publisher(MarkerArray, f"{traj_topic}_markers", qs)
        self._reasoning_pub = self.create_publisher(String, self._param("cot_topic"), qs)
        self._reasoning_stamped_pub = self.create_publisher(
            StringStamped, self._param("cot_with_stamped_topic"), qs
        )
        self._nav_text_pub = self.create_publisher(String, self._param("nav_text_topic"), qs)
        self._answer_pub = self.create_publisher(String, self._param("answer_topic"), qs)
        self._answer_stamped_pub = self.create_publisher(
            StringStamped, self._param("answer_stamped_topic"), qs
        )
        # Latched echo of the question the node is actually using — lets
        # `ros2 topic echo <question_topic>/active` confirm a runtime question
        # published on `question_topic` was received.
        self._active_question_pub = self.create_publisher(
            String,
            f"{self._param('question_topic')}/active",
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        # Operator assist. The ping and the status are latched so an operator
        # tool that attaches late still sees the outstanding request.
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._ping_pub = self.create_publisher(
            String, self._param("operator_ping_topic"), latched
        )
        self._ping_stamped_pub = self.create_publisher(
            StringStamped, self._param("operator_ping_stamped_topic"), qs
        )
        self._assist_status_pub = self.create_publisher(
            String, self._param("operator_status_topic"), latched
        )
        proposed_topic = self._param("proposed_trajectory_topic")
        self._proposed_traj_pub = self.create_publisher(Trajectory, proposed_topic, latched)
        self._proposed_marker_pub = self.create_publisher(
            MarkerArray, f"{proposed_topic}_markers", latched
        )
        self._approved_traj_pub = self.create_publisher(
            Trajectory, self._param("approved_trajectory_topic"), qs
        )

        # ------------------------------------------------------------------ #
        # Runtime config                                                    #
        # ------------------------------------------------------------------ #
        self._require_new_data = bool(self._param("require_new_data"))
        self._vqa_enabled = bool(self._param("vqa_enabled"))
        self._answer_once = bool(self._param("answer_once"))
        self._vqa_use_traj = bool(self._param("vqa_use_trajectory_input"))
        self._vqa_system_prompt = str(self._param("vqa_system_prompt") or "").strip()
        self._cot_max_gen_len = int(self._param("max_generation_length"))
        self._vqa_max_gen_len = int(self._param("vqa_max_generation_length"))
        self._vqa_num_samples = max(1, int(self._param("vqa_num_samples")))

        self._question = str(self._param("question") or DEFAULT_QUESTION).strip() or DEFAULT_QUESTION
        self._last_answered_question: Optional[str] = None

        # Monotonic count of camera messages; frozen when the stream stops.
        self._rx_count = 0
        self._last_traj_rx = -1
        self._last_vqa_rx = -1
        self._nav_text: Optional[str] = None
        self._warned_no_odom = False

        # --- operator assist state ---
        self._assist_enabled = bool(self._param("operator_assist_enabled"))
        self._stopped_speed_threshold = float(self._param("stopped_speed_threshold"))
        self._stopped_duration_sec = float(self._param("stopped_duration_sec"))
        self._proposal_ttl_sec = float(self._param("proposal_ttl_sec"))
        self._assist_cooldown_sec = float(self._param("assist_cooldown_sec"))
        self._assist_question = str(self._param("assist_question") or DEFAULT_ASSIST_QUESTION)
        self._assist_nav_text = str(self._param("assist_nav_text") or "").strip()
        self._ego_speed = 0.0
        self._stopped_since: Optional[float] = None
        self._assist_state = ASSIST_IDLE
        self._proposal: Optional[Dict[str, Any]] = None
        self._assist_blocked_until = 0.0

        # ------------------------------------------------------------------ #
        # Subscriptions                                                     #
        # ------------------------------------------------------------------ #
        self._camera_topics: List[str] = [
            t for t in self._param("camera_topics") if t
        ]
        if not self._camera_topics:
            raise ValueError("camera_topics must list at least one image topic.")
        camera_indices = list(self._param("camera_indices"))
        if len(camera_indices) != len(self._camera_topics):
            raise ValueError(
                f"camera_indices length ({len(camera_indices)}) must match "
                f"camera_topics length ({len(self._camera_topics)})."
            )
        self._camera_indices = torch.tensor(camera_indices, dtype=torch.int64)
        self._camera_buffers = {
            topic: deque(maxlen=self._num_frames * 3) for topic in self._camera_topics
        }

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
        )
        for topic in self._camera_topics:
            self.create_subscription(
                CompressedImage, topic, lambda m, t=topic: self._on_image(t, m), sensor_qos
            )
            self.get_logger().info(f"camera <- {topic}")

        self._odom_buffer: deque = deque(maxlen=self._num_history_steps * self._odom_skip + 10)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50
        )
        self.create_subscription(
            Odometry, self._param("odometry_topic"), self._on_odometry, odom_qos
        )
        self.get_logger().info(f"odometry <- {self._param('odometry_topic')}")

        self.create_subscription(
            String, self._param("question_topic"), self._on_question, qs
        )
        self.create_subscription(
            String, self._param("approval_topic"), self._on_approval, qs
        )

        # ------------------------------------------------------------------ #
        # Lanelet2 map + route  ->  navigation text                         #
        # ------------------------------------------------------------------ #
        self._lanelet_map: Optional[dict] = None
        self._route_lanelet_ids: List[int] = []
        map_path = self._param("lanelet2_map_path")
        if map_path:
            self._load_lanelet2_map(map_path)
        route_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            LaneletRoute, self._param("route_topic"), self._on_route, route_qos
        )

        # ------------------------------------------------------------------ #
        # Model — loaded ONCE, shared by both paths                         #
        # ------------------------------------------------------------------ #
        self.get_logger().info(
            f"Loading {self.model_name} on {self._device} ({self._dtype}) ..."
        )
        self._model = Alpamayo1_5.from_pretrained(self.model_name, dtype=self._dtype).to(
            self._device
        )
        self._model.eval()
        self._configure_expert()
        num_steps = int(self._param("num_diffusion_steps"))
        self._model.diffusion.num_inference_steps = num_steps
        self.get_logger().info(f"diffusion steps: {num_steps}")

        if bool(self._param("use_greedy_decode")):
            self._top_p, self._temperature = 1.0, 1.0
            self.get_logger().info("decode: GREEDY (top_p=1.0, temperature=1.0)")
        else:
            self._top_p = float(self._param("top_p"))
            self._temperature = float(self._param("temperature"))
            self.get_logger().info(
                f"decode: NUCLEUS (top_p={self._top_p}, temperature={self._temperature})"
            )

        self._processor = helper.get_processor(self._model.tokenizer)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        # ------------------------------------------------------------------ #
        # Executor + timers                                                 #
        # ------------------------------------------------------------------ #
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None

        self._traj_timer = self.create_timer(
            float(self._param("inference_period_sec")), self._traj_timer_callback
        )
        if self._vqa_enabled:
            self._vqa_timer = self.create_timer(
                float(self._param("vqa_period_sec")), self._vqa_timer_callback
            )
        if self._assist_enabled:
            self._assist_timer = self.create_timer(
                float(self._param("assist_check_period_sec")), self._assist_timer_callback
            )
        self._publish_active_question()
        self._set_assist_state(ASSIST_IDLE)
        self.get_logger().info(
            f"tele_alpa ready — trajectory on {traj_topic}, "
            f"answers on {self._param('answer_topic')}, "
            f"VQA {'enabled' if self._vqa_enabled else 'disabled'}, "
            f"operator assist {'enabled' if self._assist_enabled else 'disabled'} "
            f"(stop < {self._stopped_speed_threshold} m/s for "
            f"{self._stopped_duration_sec}s -> {self._param('operator_ping_topic')}), "
            f"question={self._question!r}"
        )

    # ------------------------------------------------------------------ #
    # Parameters                                                        #
    # ------------------------------------------------------------------ #

    def _declare_parameters(self) -> None:
        # --- trajectory (mirrors alpamayo_ros) ---
        self.declare_parameter("trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("cot_topic", "/alpamayo/reasoning")
        self.declare_parameter("cot_with_stamped_topic", "/alpamayo/reasoning_stamped")
        self.declare_parameter("nav_text_topic", "/alpamayo/nav_text")
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        self.declare_parameter("route_topic", "/planning/mission_planning/route")
        self.declare_parameter("lanelet2_map_path", "")
        self.declare_parameter("inference_period_sec", 1.0)
        # ROS 2 needs a non-empty default for array params to infer the type.
        self.declare_parameter("camera_topics", [""])
        # 0=Front left 1=Front 2=Front right 3=Rear left 4=Rear 5=Rear right
        # 6=Front telephoto — one index per camera topic.
        self.declare_parameter("camera_indices", [0])
        self.declare_parameter("num_diffusion_steps", 5)
        self.declare_parameter("use_greedy_decode", True)
        self.declare_parameter("top_p", 0.98)
        self.declare_parameter("temperature", 0.6)
        self.declare_parameter("max_generation_length", 64)
        self.declare_parameter("expert_onnx_path", "")
        # Only infer when a new camera frame has arrived since the last run —
        # stops output on a frozen buffer after a bag ends.
        self.declare_parameter("require_new_data", True)

        # --- VQA ---
        self.declare_parameter("vqa_enabled", True)
        self.declare_parameter("question", DEFAULT_QUESTION)
        self.declare_parameter("question_topic", "/tele_alpa/question")
        self.declare_parameter("answer_topic", "/tele_alpa/answer")
        self.declare_parameter("answer_stamped_topic", "/tele_alpa/answer_stamped")
        self.declare_parameter("vqa_period_sec", 2.0)
        # Answer once per question change, then idle (vs. re-answer every tick).
        self.declare_parameter("answer_once", False)
        # Token budget for the answer. Too small -> the model never closes
        # <|answer_end|> and the parsed answer is empty.
        self.declare_parameter("vqa_max_generation_length", 500)
        self.declare_parameter("vqa_num_samples", 1)
        # Fuse the ego trajectory-history tokens into the VQA prompt. Leaving this
        # on biases answers toward planning-style output; set false for
        # description-style answers.
        self.declare_parameter("vqa_use_trajectory_input", True)
        # System prompt for the VQA path. helper.create_vqa_message bakes in
        # "...generates safe and accurate actions", which pushes the model to
        # reply with a driving action instead of addressing the question. An
        # empty string keeps that upstream default.
        self.declare_parameter("vqa_system_prompt", DEFAULT_VQA_SYSTEM_PROMPT)

        # --- operator assist ---
        # When the ego stops (non-positive velocity) for `stopped_duration_sec`,
        # ask the model what is blocking the path, draft a way around it, and
        # ping a human operator who must approve before the path is re-published
        # on `approved_trajectory_topic`. Everything is plain ROS topics.
        self.declare_parameter("operator_assist_enabled", True)
        # "not positive" velocity, with a small deadband for sensor noise (m/s).
        self.declare_parameter("stopped_speed_threshold", 0.1)
        # Debounce, so ordinary traffic-light stops do not ping the operator.
        self.declare_parameter("stopped_duration_sec", 5.0)
        self.declare_parameter("assist_check_period_sec", 0.5)
        # How long a proposal stays approvable before it EXPIREs.
        self.declare_parameter("proposal_ttl_sec", 120.0)
        # After a reject / expiry, wait this long before proposing again while
        # the ego is still stopped.
        self.declare_parameter("assist_cooldown_sec", 30.0)
        self.declare_parameter("assist_question", DEFAULT_ASSIST_QUESTION)
        # Route hint used for the proposed trajectory. Empty -> reuse the current
        # lanelet2-derived nav text.
        self.declare_parameter("assist_nav_text", DEFAULT_ASSIST_NAV_TEXT)
        self.declare_parameter("operator_ping_topic", "/tele_alpa/operator/ping")
        self.declare_parameter(
            "operator_ping_stamped_topic", "/tele_alpa/operator/ping_stamped"
        )
        self.declare_parameter("operator_status_topic", "/tele_alpa/operator/status")
        self.declare_parameter(
            "proposed_trajectory_topic", "/tele_alpa/operator/proposed_trajectory"
        )
        self.declare_parameter("approval_topic", "/tele_alpa/operator/approval")
        self.declare_parameter(
            "approved_trajectory_topic", "/tele_alpa/operator/approved_trajectory"
        )

    def _param(self, name: str):
        return self.get_parameter(name).value

    def _configure_expert(self) -> None:
        onnx = str(self._param("expert_onnx_path") or "")
        if onnx and Path(onnx).exists():
            from alpamayo1_5.trt.expert_runtime import TrtExpertEngine

            engine = TrtExpertEngine(
                onnx_model_path=onnx,
                engine_cache_dir=str(Path(onnx).parent / "engine_cache"),
                enable_int8=False,
                enable_fp16=True,
            )
            self._model.set_expert_step_runner(engine)
            self.get_logger().info(f"TRT expert engine: {onnx}")
        else:
            self.get_logger().info("TRT expert engine: off (native PyTorch denoiser).")

    # ------------------------------------------------------------------ #
    # Subscription callbacks                                            #
    # ------------------------------------------------------------------ #

    def _on_image(self, topic: str, msg: CompressedImage) -> None:
        jpeg = torch.frombuffer(bytearray(msg.data), dtype=torch.uint8)
        self._camera_buffers[topic].append((msg.header.stamp, jpeg))
        self._rx_count += 1

    def _on_odometry(self, msg: Odometry) -> None:
        self._odom_buffer.append(msg)
        # Longitudinal speed in base_link. "Not positive" (<= threshold) starts
        # the stopped clock that arms the operator-assist trigger.
        self._ego_speed = float(msg.twist.twist.linear.x)
        if self._ego_speed > self._stopped_speed_threshold:
            self._stopped_since = None
        elif self._stopped_since is None:
            self._stopped_since = time.monotonic()

    def _on_route(self, msg: LaneletRoute) -> None:
        self._route_lanelet_ids = [seg.preferred_primitive.id for seg in msg.segments]
        self.get_logger().info(f"route: {len(self._route_lanelet_ids)} segments")

    def _on_question(self, msg: String) -> None:
        q = msg.data.strip()
        if not q:
            return
        if q != self._question:
            self.get_logger().info(f"new question: {q!r}")
        self._question = q
        self._last_answered_question = None  # force a fresh answer
        self._publish_active_question()

    def _publish_active_question(self) -> None:
        self._active_question_pub.publish(String(data=self._question))

    # ------------------------------------------------------------------ #
    # Shared input preparation                                          #
    # ------------------------------------------------------------------ #

    def _decode_latest_frames(self) -> Optional[torch.Tensor]:
        """GPU-decode the newest frames -> ``[n_cams, n_frames, 3, H, W]`` uint8."""
        if not all(len(b) >= self._num_frames for b in self._camera_buffers.values()):
            return None

        import torchvision

        jpegs: List[torch.Tensor] = []
        for topic in self._camera_topics:
            jpegs.extend(f for _, f in list(self._camera_buffers[topic])[-self._num_frames :])

        stacked = torch.stack(
            [torchvision.io.decode_jpeg(j, device="cuda") for j in jpegs]
        )  # [n_cams * n_frames, 3, H, W]
        if stacked.shape[-2:] != _MODEL_INPUT_SIZE:
            stacked = (
                torch.nn.functional.interpolate(
                    stacked.float(), size=_MODEL_INPUT_SIZE, mode="bicubic", align_corners=False
                )
                .clamp(0, 255)
                .to(torch.uint8)
            )
        n_cams = len(self._camera_topics)
        return torch.stack(
            [stacked[i * self._num_frames : (i + 1) * self._num_frames] for i in range(n_cams)],
            dim=0,
        )

    def _build_ego_history(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray]]:
        """Ego motion in the t0 local frame + raw map-frame position at t0.

        ``(xyz, rot, ego_pos_map)`` shaped ``(1, 1, T, 3)`` / ``(1, 1, T, 3, 3)``,
        or ``(None, None, None)`` if not enough odometry has been buffered.
        """
        need = self._num_history_steps * self._odom_skip
        if len(self._odom_buffer) < need:
            return None, None, None

        history = list(self._odom_buffer)[-need :: self._odom_skip]
        positions, rotations = [], []
        for msg in history:
            p = msg.pose.pose
            positions.append([p.position.x, p.position.y, p.position.z])
            rotations.append(
                Rotation.from_quat(
                    [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
                ).as_matrix()
            )
        positions = np.asarray(positions, dtype=np.float32)
        rotations = np.asarray(rotations, dtype=np.float32)
        t0_rot_inv = np.linalg.inv(rotations[-1])
        xyz_local = (positions - positions[-1]) @ t0_rot_inv.T
        rot_local = np.einsum("ij,njk->nik", t0_rot_inv, rotations)

        xyz = torch.from_numpy(xyz_local).unsqueeze(0).unsqueeze(0)
        rot = torch.from_numpy(rot_local).unsqueeze(0).unsqueeze(0)
        return xyz, rot, positions[-1]

    def _tokenize(self, messages: list) -> dict:
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            device="cuda",
        )
        return {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

    @staticmethod
    def _span(extra: dict, key: str) -> str:
        """Stripped text for ``key``, robust to the ``[B,ns,nj]`` (trajectory)
        and ``[B,ns]`` (generate_text) array shapes."""
        if not extra or key not in extra:
            return ""
        try:
            value = np.asarray(extra[key]).reshape(-1)[0]
        except Exception:
            return ""
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return str(value).strip()

    # ------------------------------------------------------------------ #
    # Trajectory path                                                   #
    # ------------------------------------------------------------------ #

    def _traj_timer_callback(self) -> None:
        if self._active_future and not self._active_future.done():
            return
        # An outstanding operator request outranks routine planning: the ego is
        # stopped anyway, and re-planning would only fight the pending proposal.
        if self._assist_pending():
            return
        # Yield the shared GPU worker to an unanswered question. Both timers gate
        # on `_active_future`, and this one fires more often, so without this it
        # wins every race and VQA never runs.
        if self._vqa_enabled and self._question and self._question != self._last_answered_question:
            return
        if self._require_new_data and self._rx_count == self._last_traj_rx:
            return

        frames = self._decode_latest_frames()
        if frames is None:
            return
        ego_xyz, ego_rot, ego_pos_map = self._build_ego_history()
        if ego_xyz is None:
            return

        nav_text = self._compute_nav_text(ego_pos_map)
        if nav_text and nav_text != self._nav_text:
            self._nav_text = nav_text
            self.get_logger().info(f"navigation: {nav_text}")

        self._last_traj_rx = self._rx_count
        self._active_future = self._executor.submit(
            self._run_trajectory, frames, ego_xyz, ego_rot, self._nav_text
        )
        self._active_future.add_done_callback(self._on_future_done)

    def _infer_trajectory(
        self,
        frames: torch.Tensor,
        ego_xyz: torch.Tensor,
        ego_rot: torch.Tensor,
        nav_text: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Run the planning rollout. Returns ``(xyz, rot, extra)`` on CPU."""
        messages = helper.create_message(
            frames.flatten(0, 1),
            camera_indices=self._camera_indices,
            num_frames_per_camera=self._num_frames,
            nav_text=nav_text,
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": self._tokenize(messages),
                "ego_history_xyz": ego_xyz,
                "ego_history_rot": ego_rot,
            },
            device=self._device,
        )

        kwargs = dict(
            top_p=self._top_p,
            temperature=self._temperature,
            num_traj_samples=1,
            num_traj_sets=1,
            max_generation_length=self._cot_max_gen_len,
            return_extra=True,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sampler = (
                self._model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav
                if nav_text
                else self._model.sample_trajectories_from_data_with_vlm_rollout
            )
            pred_xyz, pred_rot, extra = sampler(data=model_inputs, **kwargs)
        return pred_xyz.detach().cpu()[0, 0, 0], pred_rot.detach().cpu()[0, 0, 0], extra

    def _run_trajectory(
        self,
        frames: torch.Tensor,
        ego_xyz: torch.Tensor,
        ego_rot: torch.Tensor,
        nav_text: Optional[str],
    ) -> dict:
        start = time.time()
        trajectory, rotation, extra = self._infer_trajectory(
            frames, ego_xyz, ego_rot, nav_text
        )
        traj_msg = self._to_autoware_trajectory(trajectory, rotation)
        self._traj_pub.publish(traj_msg)
        self._marker_pub.publish(self._trajectory_markers(trajectory))

        if nav_text:
            self._nav_text_pub.publish(String(data=nav_text))

        reasoning = self._span(extra, "cot")
        if reasoning:
            self._reasoning_pub.publish(String(data=reasoning))
            self._reasoning_stamped_pub.publish(
                StringStamped(stamp=traj_msg.header.stamp, data=reasoning)
            )

        return {
            "kind": "trajectory",
            "duration_sec": time.time() - start,
            "num_poses": len(traj_msg.points),
            "reasoning_chars": len(reasoning),
        }

    # ------------------------------------------------------------------ #
    # VQA path                                                          #
    # ------------------------------------------------------------------ #

    def _vqa_timer_callback(self) -> None:
        if not self._vqa_enabled or not self._question:
            return
        if self._active_future and not self._active_future.done():
            return
        if self._assist_pending():
            return
        if self._answer_once and self._question == self._last_answered_question:
            return
        question_pending = self._question != self._last_answered_question
        if (
            self._require_new_data
            and not question_pending
            and self._rx_count == self._last_vqa_rx
        ):
            return

        frames = self._decode_latest_frames()
        if frames is None:
            return

        ego_xyz = ego_rot = None
        if self._vqa_use_traj:
            ego_xyz, ego_rot, _ = self._build_ego_history()
            if ego_xyz is None and not self._warned_no_odom:
                self.get_logger().warning(
                    "no ego-motion history yet — answering without trajectory input"
                )
                self._warned_no_odom = True

        question = self._question
        self._last_vqa_rx = self._rx_count
        self._last_answered_question = question
        self._active_future = self._executor.submit(
            self._run_vqa, question, frames, ego_xyz, ego_rot
        )
        self._active_future.add_done_callback(self._on_future_done)

    def _infer_answer(
        self,
        question: str,
        frames: torch.Tensor,
        ego_xyz: Optional[torch.Tensor],
        ego_rot: Optional[torch.Tensor],
    ) -> Tuple[str, Optional[str]]:
        """Run the VQA rollout. Returns ``(answer_text, source_span_or_None)``."""
        with_traj = ego_xyz is not None and ego_rot is not None

        messages = helper.create_vqa_message(
            frames.flatten(0, 1),
            question=question,
            camera_indices=self._camera_indices,
            num_frames_per_camera=self._num_frames,
        )
        if self._vqa_system_prompt:
            for item in messages[0]["content"]:
                if item.get("type") == "text":
                    item["text"] = self._vqa_system_prompt
                    break
        if with_traj:
            # Prepend the ego trajectory-history placeholder to the question text.
            for item in reversed(messages[1]["content"]):
                if item.get("type") == "text":
                    item["text"] = _HIST_TRAJ_PLACEHOLDER + item["text"]
                    break

        tokenized = self._tokenize(messages)
        if with_traj:
            tokenized["input_ids"] = self._model.fuse_traj_tokens(
                tokenized["input_ids"],
                {
                    "ego_history_xyz": ego_xyz.to(self._device),
                    "ego_history_rot": ego_rot.to(self._device),
                },
            )
        model_inputs = helper.to_device({"tokenized_data": tokenized}, device=self._device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            extra = self._model.generate_text(
                data=model_inputs,
                top_p=self._top_p,
                temperature=self._temperature,
                num_samples=self._vqa_num_samples,
                max_generation_length=self._vqa_max_gen_len,
            )

        answer, source = self._pick_answer(extra)
        if source is None:
            answer = (
                f"[no answer: model produced no closed <|answer_end|> / <|cot_end|> "
                f"span within {self._vqa_max_gen_len} tokens — raise "
                f"vqa_max_generation_length, or set vqa_use_trajectory_input:=false]"
            )
            self.get_logger().warning(answer)
        elif source != "answer" and not with_traj:
            # Pure VQA mode: we expect a real `<|answer_start|>..<|answer_end|>`
            # span. Falling back to a planning span means the answer span never
            # closed — flag it and mark the text so it is not mistaken for a
            # direct reply.
            self.get_logger().warning(
                f"no <|answer_end|> span; using extra[{source!r}] "
                f"(raise vqa_max_generation_length for a clean answer)"
            )
            answer = f"[{source}] {answer}"
        # With `vqa_use_trajectory_input` on the model runs a trajectory-aware
        # rollout, so a `cot` / `meta_action` span *is* the expected reply — take
        # it as-is (source is still reported in the log + metrics).
        return answer, source

    def _run_vqa(
        self,
        question: str,
        frames: torch.Tensor,
        ego_xyz: Optional[torch.Tensor],
        ego_rot: Optional[torch.Tensor],
    ) -> dict:
        start = time.time()
        answer, source = self._infer_answer(question, frames, ego_xyz, ego_rot)

        stamp = self.get_clock().now().to_msg()
        self._answer_pub.publish(String(data=answer))
        self._answer_stamped_pub.publish(StringStamped(stamp=stamp, data=answer))

        return {
            "kind": "vqa",
            "duration_sec": time.time() - start,
            "question": question,
            "answer": answer,
            "answer_source": source or "none",
            "with_traj": ego_xyz is not None and ego_rot is not None,
        }

    @staticmethod
    def _pick_answer(extra: dict) -> Tuple[str, Optional[str]]:
        """First non-empty text span from ``generate_text``.

        Returns ``(text, source)``; ``source`` is the ``extra`` key the text came
        from (``"answer"`` is the real reply, ``"meta_action"`` / ``"cot"`` are
        planning fallbacks) or ``None`` if every field was empty.
        """
        for key in ("answer", "meta_action", "cot"):
            value = TeleAlpaNode._span(extra, key)
            if value:
                return value, key
        return "", None

    # ------------------------------------------------------------------ #
    # Operator assist                                                   #
    #                                                                    #
    # Ego stops (non-positive velocity) -> the model describes what is    #
    # blocking the path and drafts a way around it -> a ping goes to the  #
    # operator, who must approve before the path is republished on        #
    # `approved_trajectory_topic`. Plain ROS topics, no execution mux.    #
    # ------------------------------------------------------------------ #

    def _assist_pending(self) -> bool:
        """True while a stop event is being handled — used to hold off the
        routine trajectory / VQA ticks so the assist run gets the GPU."""
        return self._assist_state in (ASSIST_ASSESSING, ASSIST_AWAITING)

    def _set_assist_state(self, state: str, detail: str = "") -> None:
        self._assist_state = state
        payload = {
            "state": state,
            "detail": detail,
            "ego_speed_mps": round(self._ego_speed, 3),
            "proposal_id": (self._proposal or {}).get("id"),
        }
        self._assist_status_pub.publish(String(data=json.dumps(payload)))

    def _assist_timer_callback(self) -> None:
        if not self._assist_enabled:
            return
        now = time.monotonic()
        moving = self._ego_speed > self._stopped_speed_threshold

        # Vehicle is rolling again -> the stop event is over, whatever its state.
        if moving:
            if self._assist_state != ASSIST_IDLE:
                self._proposal = None
                self._set_assist_state(ASSIST_IDLE, "ego moving again")
            return

        if self._assist_state == ASSIST_AWAITING:
            proposal = self._proposal
            if proposal and now - proposal["created_at"] > self._proposal_ttl_sec:
                self.get_logger().warning(
                    f"proposal {proposal['id']} expired after {self._proposal_ttl_sec:.0f}s "
                    f"with no operator response"
                )
                self._proposal = None
                self._assist_blocked_until = now + self._assist_cooldown_sec
                self._set_assist_state(ASSIST_EXPIRED, "no operator response")
            return

        if self._assist_state == ASSIST_ASSESSING:
            return  # inference in flight

        # IDLE / APPROVED / REJECTED / EXPIRED and still stopped: re-arm once the
        # cooldown has passed and the ego has been stopped long enough.
        if now < self._assist_blocked_until:
            return
        if self._stopped_since is None:
            return
        stopped_for = now - self._stopped_since
        if stopped_for < self._stopped_duration_sec:
            if self._assist_state == ASSIST_IDLE:
                self._set_assist_state(ASSIST_STOPPED, f"stopped {stopped_for:.1f}s")
            return

        if self._active_future and not self._active_future.done():
            return  # GPU busy; try again next tick
        frames = self._decode_latest_frames()
        if frames is None:
            return
        ego_xyz, ego_rot, _ = self._build_ego_history()

        self._set_assist_state(ASSIST_ASSESSING, f"stopped {stopped_for:.1f}s")
        self.get_logger().info(
            f"operator assist: ego stopped {stopped_for:.1f}s "
            f"(v={self._ego_speed:.2f} m/s) — assessing scene"
        )
        self._active_future = self._executor.submit(
            self._run_operator_assist, frames, ego_xyz, ego_rot, stopped_for
        )
        self._active_future.add_done_callback(self._on_future_done)

    def _run_operator_assist(
        self,
        frames: torch.Tensor,
        ego_xyz: Optional[torch.Tensor],
        ego_rot: Optional[torch.Tensor],
        stopped_for: float,
    ) -> dict:
        start = time.time()

        # 1) Describe the blockage for the operator.
        description, source = self._infer_answer(
            self._assist_question, frames, ego_xyz, ego_rot
        )

        # 2) Draft a way around it. Needs ego history — the planner cannot run
        #    without it, so fall back to a description-only ping.
        trajectory = rotation = None
        reasoning = ""
        traj_msg: Optional[Trajectory] = None
        if ego_xyz is not None and ego_rot is not None:
            nav_text = self._assist_nav_text or self._nav_text
            trajectory, rotation, extra = self._infer_trajectory(
                frames, ego_xyz, ego_rot, nav_text
            )
            reasoning = self._span(extra, "cot")
            traj_msg = self._to_autoware_trajectory(trajectory, rotation)

        proposal_id = uuid.uuid4().hex[:8]
        stamp = self.get_clock().now().to_msg()
        approval_topic = self._param("approval_topic")
        ping = {
            "id": proposal_id,
            "event": "ego_stopped",
            "reason": (
                f"ego stopped for {stopped_for:.1f}s "
                f"(v={self._ego_speed:.2f} m/s <= {self._stopped_speed_threshold} m/s)"
            ),
            "description": description,
            "description_source": source or "none",
            "planner_reasoning": reasoning,
            "proposal": {
                "available": traj_msg is not None,
                "topic": self._param("proposed_trajectory_topic"),
                "num_points": len(traj_msg.points) if traj_msg else 0,
                "length_m": round(self._path_length(trajectory), 2) if traj_msg else 0.0,
                "frame_id": self._frame_id,
            },
            "ttl_sec": self._proposal_ttl_sec,
            "approve": (
                f"ros2 topic pub --once {approval_topic} std_msgs/String "
                f"\"{{data: 'approve:{proposal_id}'}}\""
            ),
            "reject": (
                f"ros2 topic pub --once {approval_topic} std_msgs/String "
                f"\"{{data: 'reject:{proposal_id}'}}\""
            ),
        }
        ping_json = json.dumps(ping, indent=2)

        if traj_msg is not None:
            self._proposed_traj_pub.publish(traj_msg)
            self._proposed_marker_pub.publish(
                self._trajectory_markers(
                    trajectory,
                    ns="operator_proposal",
                    color=ColorRGBA(r=1.0, g=0.55, b=0.0, a=1.0),
                    label=f"PROPOSAL {proposal_id}: {description[:80]}",
                )
            )

        self._proposal = {
            "id": proposal_id,
            "created_at": time.monotonic(),
            "trajectory": traj_msg,
            "description": description,
            "reasoning": reasoning,
        }
        self._ping_pub.publish(String(data=ping_json))
        self._ping_stamped_pub.publish(StringStamped(stamp=stamp, data=ping_json))
        self._set_assist_state(ASSIST_AWAITING, f"proposal {proposal_id} awaiting approval")

        return {
            "kind": "assist",
            "duration_sec": time.time() - start,
            "proposal_id": proposal_id,
            "description": description,
            "has_trajectory": traj_msg is not None,
            "approve_cmd": ping["approve"],
        }

    def _on_approval(self, msg: String) -> None:
        raw = msg.data.strip().lower()
        if not raw:
            return
        verb, _, target = raw.partition(":")
        verb = verb.strip()
        target = target.strip()
        if verb not in ("approve", "reject"):
            self.get_logger().warning(
                f"approval message {msg.data!r} not understood — "
                f"expected 'approve[:<id>]' or 'reject[:<id>]'"
            )
            return

        proposal = self._proposal
        if proposal is None or self._assist_state != ASSIST_AWAITING:
            self.get_logger().warning(f"'{verb}' received but no proposal is awaiting approval")
            return
        if target and target not in ("latest", proposal["id"]):
            self.get_logger().warning(
                f"'{verb}:{target}' does not match the outstanding proposal "
                f"{proposal['id']} — ignoring"
            )
            return

        if verb == "reject":
            self.get_logger().info(f"operator REJECTED proposal {proposal['id']}")
            self._proposal = None
            self._assist_blocked_until = time.monotonic() + self._assist_cooldown_sec
            self._set_assist_state(ASSIST_REJECTED, f"proposal {proposal['id']} rejected")
            return

        traj_msg = proposal["trajectory"]
        if traj_msg is None:
            self.get_logger().warning(
                f"proposal {proposal['id']} has no trajectory to approve (description only)"
            )
            self._proposal = None
            self._assist_blocked_until = time.monotonic() + self._assist_cooldown_sec
            self._set_assist_state(ASSIST_REJECTED, "no trajectory in proposal")
            return

        traj_msg.header.stamp = self.get_clock().now().to_msg()
        self._approved_traj_pub.publish(traj_msg)
        self.get_logger().info(
            f"operator APPROVED proposal {proposal['id']} — republished "
            f"{len(traj_msg.points)} points on {self._param('approved_trajectory_topic')}"
        )
        self._assist_blocked_until = time.monotonic() + self._assist_cooldown_sec
        self._set_assist_state(ASSIST_APPROVED, f"proposal {proposal['id']} approved")

    @staticmethod
    def _path_length(trajectory: Optional[torch.Tensor]) -> float:
        if trajectory is None:
            return 0.0
        pts = trajectory.numpy()[:, :2]
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    # ------------------------------------------------------------------ #
    # Completion + output builders                                      #
    # ------------------------------------------------------------------ #

    def _on_future_done(self, future: Future) -> None:
        try:
            metrics = future.result()
        except Exception as exc:  # noqa: BLE001 — surface any inference failure
            self.get_logger().error(f"inference failed: {exc}")
            # A failed assist run would otherwise leave the node stuck in
            # ASSESSING, which also holds off the trajectory and VQA ticks.
            if self._assist_state == ASSIST_ASSESSING:
                self._proposal = None
                self._assist_blocked_until = time.monotonic() + self._assist_cooldown_sec
                self._set_assist_state(ASSIST_IDLE, f"assist run failed: {exc}")
            return
        if not metrics:
            return
        if metrics["kind"] == "vqa":
            src = metrics["answer_source"]
            traj_note = "" if metrics["with_traj"] else " [no traj input]"
            src_note = "" if src == "answer" else f" [source={src}]"
            self.get_logger().info(
                f"VQA {metrics['duration_sec']:.2f}s{traj_note}{src_note}\n"
                f"  Q: {metrics['question']}\n"
                f"  A: {metrics['answer'] or '(empty)'}"
            )
        elif metrics["kind"] == "assist":
            traj_note = (
                "" if metrics["has_trajectory"] else " (no trajectory — missing ego history)"
            )
            self.get_logger().warning(
                f"OPERATOR PING {metrics['proposal_id']} in "
                f"{metrics['duration_sec']:.2f}s{traj_note}\n"
                f"  scene: {metrics['description'] or '(empty)'}\n"
                f"  approve: {metrics['approve_cmd']}"
            )
        else:
            self.get_logger().info(
                f"trajectory {metrics['duration_sec']:.2f}s "
                f"(points={metrics['num_poses']}, reasoning={metrics['reasoning_chars']} chars)"
            )

    def _to_autoware_trajectory(
        self, trajectory: torch.Tensor, rotations: Optional[torch.Tensor]
    ) -> Trajectory:
        traj_np = trajectory.numpy()
        rot_np = rotations.numpy() if rotations is not None else None

        msg = Trajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id

        dt = 0.1
        prev_xy = None
        for idx, point in enumerate(traj_np):
            tp = TrajectoryPoint()
            tp.pose.position.x = float(point[0])
            tp.pose.position.y = float(point[1])
            tp.pose.position.z = float(point[2])
            if rot_np is not None:
                q = Rotation.from_matrix(rot_np[idx]).as_quat()
                tp.pose.orientation.x = float(q[0])
                tp.pose.orientation.y = float(q[1])
                tp.pose.orientation.z = float(q[2])
                tp.pose.orientation.w = float(q[3])
            else:
                tp.pose.orientation.w = 1.0

            if prev_xy is None:
                speed = 0.0
            else:
                speed = math.hypot(point[0] - prev_xy[0], point[1] - prev_xy[1]) / dt
            tp.longitudinal_velocity_mps = float(speed)

            secs = idx * dt
            tp.time_from_start = Duration(sec=int(secs), nanosec=int((secs % 1.0) * 1e9))
            msg.points.append(tp)
            prev_xy = (point[0], point[1])
        return msg

    def _trajectory_markers(
        self,
        trajectory: torch.Tensor,
        ns: str = "trajectory",
        color: Optional[ColorRGBA] = None,
        label: str = "",
    ) -> MarkerArray:
        stamp = self.get_clock().now().to_msg()
        points = trajectory.numpy()

        line = Marker()
        line.header.stamp = stamp
        line.header.frame_id = self._frame_id
        line.ns = ns
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 1.0
        line.color = color or ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        line.pose.orientation.w = 1.0
        for point in points:
            line.points.append(Point(x=float(point[0]), y=float(point[1]), z=float(point[2])))

        markers = MarkerArray()
        markers.markers.append(line)

        if label:
            # Floating caption above the end of the path, so an operator looking
            # at RViz sees why the proposal exists without reading the topic.
            text = Marker()
            text.header.stamp = stamp
            text.header.frame_id = self._frame_id
            text.ns = f"{ns}_label"
            text.id = 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.scale.z = 1.5
            text.color = color or ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            tip = points[-1]
            text.pose.position = Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2]) + 3.0)
            text.pose.orientation.w = 1.0
            text.text = label
            markers.markers.append(text)
        return markers

    # ------------------------------------------------------------------ #
    # Lanelet2 map + navigation text                                    #
    # ------------------------------------------------------------------ #

    def _load_lanelet2_map(self, map_path: str) -> None:
        if not _HAS_LANELET2:
            self.get_logger().warning(
                "lanelet2 bindings unavailable — navigation text disabled"
            )
            return
        self.get_logger().info(f"loading lanelet2 map: {map_path}")
        projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
        ll2_map = lanelet2.io.load(map_path, projection)

        info: dict = {}
        for ll in ll2_map.laneletLayer:
            attrs = ll.attributes
            subtype = attrs["subtype"] if "subtype" in attrs else ""
            if subtype not in ("road", "highway", "road_shoulder", "bicycle_lane"):
                continue
            centerline = np.array([(p.x, p.y, p.z) for p in ll.centerline])
            info[ll.id] = {
                "centerline": centerline,
                "turn_direction": attrs["turn_direction"] if "turn_direction" in attrs else "",
                "center": np.mean(centerline[:, :2], axis=0),
            }
        self._lanelet_map = info
        self.get_logger().info(f"lanelet2 map: {len(info)} road lanelets")

    def _compute_nav_text(self, ego_pos_map: Optional[np.ndarray]) -> Optional[str]:
        if self._lanelet_map is None or not self._route_lanelet_ids or ego_pos_map is None:
            return None
        ego_xy = ego_pos_map[:2]

        best_idx, best_dist = 0, float("inf")
        for i, ll_id in enumerate(self._route_lanelet_ids):
            entry = self._lanelet_map.get(ll_id)
            if entry is None:
                continue
            dist = np.linalg.norm(entry["centerline"][0, :2] - ego_xy)
            if dist < best_dist:
                best_dist, best_idx = dist, i

        current = self._lanelet_map.get(self._route_lanelet_ids[best_idx])
        if current is not None and current["turn_direction"] in ("left", "right"):
            return f"Turn {current['turn_direction']}"

        cumulative, prev_pt = 0.0, ego_xy
        for i in range(best_idx + 1, len(self._route_lanelet_ids)):
            entry = self._lanelet_map.get(self._route_lanelet_ids[i])
            if entry is None:
                continue
            first_pt = entry["centerline"][0, :2]
            cumulative += np.linalg.norm(first_pt - prev_pt)
            prev_pt = first_pt
            if entry["turn_direction"] in ("left", "right"):
                return f"Turn {entry['turn_direction']} in {int(round(cumulative))}m"
        return "Continue straight"

    def destroy_node(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = TeleAlpaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
