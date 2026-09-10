#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""One ROS 2 node, one Alpamayo 1.5 model, two outputs:

* an Autoware trajectory (+ chain-of-thought / RViz markers / nav text), and
* answers to free-form questions about the live scene (VQA).

The 10B model is loaded **once**. The trajectory path and the VQA path share a
single-worker executor, so they never run on the GPU at the same time, and they
consume the same buffered input — a 4-camera x 4-frame image stack plus the ego
trajectory history.
"""

from __future__ import annotations

import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

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

try:
    import lanelet2
    from autoware_lanelet2_extension_python.projection import MGRSProjector

    _HAS_LANELET2 = True
except ImportError:
    _HAS_LANELET2 = False

DEFAULT_QUESTION = "What should the ego vehicle do next, and why?"

# Ego trajectory-history placeholder, identical to the one
# ``helper.create_message`` builds for planning. ``fuse_traj_tokens`` swaps these
# pad tokens for the encoded ego motion so the VLM gets the same trajectory
# input during VQA that it gets during planning.
_NUM_TRAJ_HISTORY_TOKENS = 48
_HIST_TRAJ_PLACEHOLDER = (
    "<|traj_history_start|>"
    + "<|traj_history|>" * _NUM_TRAJ_HISTORY_TOKENS
    + "<|traj_history_end|>"
)


class TeleAlpaNode(Node):
    """Runs Alpamayo 1.5 once and publishes both a trajectory and scene answers."""

    def __init__(self) -> None:
        super().__init__("tele_alpa_node")

        self.model_name: str = "nvidia/Alpamayo-1.5-10B"

        # --- trajectory params ---
        self.declare_parameter("trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("cot_topic", "/alpamayo/reasoning")
        self.declare_parameter("cot_with_stamped_topic", "/alpamayo/reasoning_stamped")
        self.declare_parameter("nav_text_topic", "/alpamayo/nav_text")
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        self.declare_parameter("route_topic", "/planning/mission_planning/route")
        self.declare_parameter("inference_period_sec", 1.0)
        # ROS2 Jazzy: non-empty defaults so string/int array param types infer.
        self.declare_parameter("camera_topics", [""])
        # 0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear,
        # 5=Rear right, 6=Front telephoto — one index per camera topic.
        self.declare_parameter("camera_indices", [0])
        self.declare_parameter("lanelet2_map_path", "")
        self.declare_parameter("num_diffusion_steps", 5)
        self.declare_parameter("use_greedy_decode", True)
        self.declare_parameter("top_p", 0.98)
        self.declare_parameter("temperature", 0.6)
        self.declare_parameter("max_generation_length", 64)
        self.declare_parameter("expert_onnx_path", "")

        # --- shared ---
        # Only run inference when a new camera frame has arrived since the last
        # run. Stops the node inferring on a frozen buffer after a bag ends.
        self.declare_parameter("require_new_data", True)

        # --- VQA params ---
        self.declare_parameter("vqa_enabled", True)
        self.declare_parameter("question", DEFAULT_QUESTION)
        self.declare_parameter("question_topic", "/tele_alpa/question")
        self.declare_parameter("answer_topic", "/tele_alpa/answer")
        self.declare_parameter("answer_stamped_topic", "/tele_alpa/answer_stamped")
        self.declare_parameter("vqa_period_sec", 2.0)
        # Answer once per question change, then idle (vs. re-answer every tick).
        self.declare_parameter("answer_once", False)
        self.declare_parameter("vqa_max_generation_length", 256)
        self.declare_parameter("vqa_num_samples", 1)
        # Feed the ego trajectory history into the VQA prompt too.
        self.declare_parameter("vqa_use_trajectory_input", True)

        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        KINEMATIC_STATE_HZ = 50.0
        ALPAMAYO_INPUT_HZ = 10.0  # model expects 10 Hz (time_step = 0.1 s)
        self.skip_num = int(KINEMATIC_STATE_HZ / ALPAMAYO_INPUT_HZ)
        self._num_history_steps = 16
        self._num_frames = 4
        self._frame_id = "base_link"

        # --- publishers ---
        qs = 10
        traj_topic = self.get_parameter("trajectory_topic").value
        self._trajectory_pub = self.create_publisher(Trajectory, traj_topic, qs)
        self._cot_pub = self.create_publisher(String, self.get_parameter("cot_topic").value, qs)
        self._cot_stamped_pub = self.create_publisher(
            StringStamped, self.get_parameter("cot_with_stamped_topic").value, qs
        )
        self._nav_text_pub = self.create_publisher(
            String, self.get_parameter("nav_text_topic").value, qs
        )
        self._marker_pub = self.create_publisher(MarkerArray, f"{traj_topic}_markers", qs)
        self._answer_pub = self.create_publisher(
            String, self.get_parameter("answer_topic").value, qs
        )
        self._answer_stamped_pub = self.create_publisher(
            StringStamped, self.get_parameter("answer_stamped_topic").value, qs
        )
        self.get_logger().info(
            f"Publishing trajectory on {traj_topic}, answers on "
            f"{self.get_parameter('answer_topic').value}"
        )

        # --- executor + gating state (shared by both paths) ---
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None
        self._require_new_data = bool(self.get_parameter("require_new_data").value)
        self._rx_count = 0  # monotonic count of camera msgs; frozen when stream stops
        self._last_traj_rx = -1
        self._last_vqa_rx = -1

        # --- camera subscriptions ---
        camera_topics = list(
            self.get_parameter("camera_topics").get_parameter_value().string_array_value
        )
        if not camera_topics or camera_topics == [""]:
            raise ValueError("camera_topics parameter must list at least one image topic.")
        self._camera_topics = camera_topics

        camera_indices = list(
            self.get_parameter("camera_indices").get_parameter_value().integer_array_value
        )
        if len(camera_indices) != len(camera_topics):
            raise ValueError(
                f"camera_indices length ({len(camera_indices)}) must match "
                f"camera_topics length ({len(camera_topics)})."
            )
        self._camera_indices = torch.tensor(camera_indices, dtype=torch.int64)
        self._camera_buffers = {
            topic: deque(maxlen=self._num_frames * 3) for topic in self._camera_topics
        }

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
        )
        for topic in self._camera_topics:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, t=topic: self._image_callback(t, msg),
                camera_qos,
            )
            self.get_logger().info(f"Subscribed to camera topic: {topic}")

        # --- odometry (ego trajectory history) ---
        self._odometry_buffer: deque = deque(
            maxlen=self._num_history_steps * self.skip_num + 10
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50
        )
        odom_topic = self.get_parameter("odometry_topic").value
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)
        self.get_logger().info(f"Subscribed to odometry topic: {odom_topic}")

        # --- lanelet2 map + route (nav text) ---
        self._lanelet_map = None
        self._route_lanelet_ids: List[int] = []
        self._nav_text: Optional[str] = None
        map_path = self.get_parameter("lanelet2_map_path").value
        if map_path:
            self._load_lanelet2_map(map_path)
        route_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            LaneletRoute, self.get_parameter("route_topic").value, self._route_callback, route_qos
        )

        # --- question input ---
        self._question = (
            str(self.get_parameter("question").value or DEFAULT_QUESTION).strip()
            or DEFAULT_QUESTION
        )
        self._answer_once = bool(self.get_parameter("answer_once").value)
        self._vqa_enabled = bool(self.get_parameter("vqa_enabled").value)
        self._vqa_use_traj = bool(self.get_parameter("vqa_use_trajectory_input").value)
        self._last_answered_question: Optional[str] = None
        self._warned_no_odom = False
        self.create_subscription(
            String, self.get_parameter("question_topic").value, self._question_callback, qs
        )

        # --- model: loaded ONCE, shared by both paths ---
        self.get_logger().info(
            f"Loading Alpamayo model {self.model_name} on device={self._device} dtype={self._dtype}"
        )
        self._model = Alpamayo1_5.from_pretrained(self.model_name, dtype=self._dtype).to(
            self._device
        )
        self._model.eval()

        expert_onnx = str(self.get_parameter("expert_onnx_path").value or "")
        if expert_onnx and Path(expert_onnx).exists():
            from alpamayo1_5.trt.expert_runtime import TrtExpertEngine

            engine = TrtExpertEngine(
                onnx_model_path=expert_onnx,
                engine_cache_dir=str(Path(expert_onnx).parent / "engine_cache"),
                enable_int8=False,
                enable_fp16=True,
            )
            self._model.set_expert_step_runner(engine)
            self.get_logger().info(f"TRT Expert loaded: {expert_onnx}")
        else:
            self.get_logger().info("TRT Expert: off (native PyTorch denoiser).")

        num_steps = int(self.get_parameter("num_diffusion_steps").value)
        self._model.diffusion.num_inference_steps = num_steps
        self.get_logger().info(f"Diffusion inference steps: {num_steps}")

        self._use_greedy = bool(self.get_parameter("use_greedy_decode").value)
        if self._use_greedy:
            self._top_p, self._temperature = 1.0, 1.0
            self.get_logger().info("Generation: GREEDY (top_p=1.0, temperature=1.0)")
        else:
            self._top_p = float(self.get_parameter("top_p").value)
            self._temperature = float(self.get_parameter("temperature").value)
            self.get_logger().info(
                f"Generation: NUCLEUS (top_p={self._top_p}, temperature={self._temperature})"
            )
        self._max_gen_len = int(self.get_parameter("max_generation_length").value)
        self._vqa_max_gen_len = int(self.get_parameter("vqa_max_generation_length").value)
        self._vqa_num_samples = max(1, int(self.get_parameter("vqa_num_samples").value))

        self._processor = helper.get_processor(self._model.tokenizer)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        # --- timers ---
        self._traj_timer = self.create_timer(
            float(self.get_parameter("inference_period_sec").value), self._traj_timer_callback
        )
        if self._vqa_enabled:
            self._vqa_timer = self.create_timer(
                float(self.get_parameter("vqa_period_sec").value), self._vqa_timer_callback
            )
            self.get_logger().info(f"VQA enabled. Current question: {self._question!r}")
        else:
            self.get_logger().info("VQA disabled (vqa_enabled:=false).")

        self.get_logger().info("tele_alpa node ready — one model, trajectory + VQA.")

    # ------------------------------------------------------------------ #
    # Subscription callbacks                                            #
    # ------------------------------------------------------------------ #

    def _image_callback(self, topic: str, msg: CompressedImage) -> None:
        jpeg_bytes = torch.frombuffer(bytearray(msg.data), dtype=torch.uint8)
        self._camera_buffers[topic].append((msg.header.stamp, jpeg_bytes))
        self._rx_count += 1

    def _odometry_callback(self, msg: Odometry) -> None:
        self._odometry_buffer.append(msg)

    def _question_callback(self, msg: String) -> None:
        question = msg.data.strip()
        if not question:
            return
        if question != self._question:
            self.get_logger().info(f"New question: {question!r}")
        self._question = question
        self._last_answered_question = None  # force a fresh answer

    def _route_callback(self, msg: LaneletRoute) -> None:
        self._route_lanelet_ids = [seg.preferred_primitive.id for seg in msg.segments]
        self.get_logger().info(f"Received route with {len(self._route_lanelet_ids)} segments.")

    # ------------------------------------------------------------------ #
    # Lanelet2 map + navigation text                                    #
    # ------------------------------------------------------------------ #

    def _load_lanelet2_map(self, map_path: str) -> None:
        if not _HAS_LANELET2:
            self.get_logger().warn(
                "lanelet2 / autoware_lanelet2_extension_python not available. "
                "Navigation instructions disabled."
            )
            return
        self.get_logger().info(f"Loading lanelet2 map from {map_path}...")
        projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
        ll2_map = lanelet2.io.load(map_path, projection)

        lanelet_info: dict[int, dict] = {}
        for ll in ll2_map.laneletLayer:
            subtype = ll.attributes.get("subtype", "") if hasattr(ll.attributes, "get") else ""
            if not subtype:
                subtype = ll.attributes["subtype"] if "subtype" in ll.attributes else ""
            if subtype not in ("road", "highway", "road_shoulder", "bicycle_lane"):
                continue
            centerline = np.array([(p.x, p.y, p.z) for p in ll.centerline])
            turn_dir_str = ll.attributes["turn_direction"] if "turn_direction" in ll.attributes else ""
            lanelet_info[ll.id] = {
                "centerline": centerline,
                "turn_direction": turn_dir_str,
                "center": np.mean(centerline[:, :2], axis=0),
            }
        self._lanelet_map = lanelet_info
        self.get_logger().info(f"Loaded {len(lanelet_info)} road lanelets from map.")

    def _compute_nav_text(self, ego_pos_map: np.ndarray) -> Optional[str]:
        if self._lanelet_map is None or not self._route_lanelet_ids:
            return None
        ego_xy = ego_pos_map[:2]

        best_idx, best_dist = 0, float("inf")
        for i, ll_id in enumerate(self._route_lanelet_ids):
            info = self._lanelet_map.get(ll_id)
            if info is None:
                continue
            dist = np.linalg.norm(info["centerline"][0, :2] - ego_xy)
            if dist < best_dist:
                best_dist, best_idx = dist, i

        current_info = self._lanelet_map.get(self._route_lanelet_ids[best_idx])
        if current_info is not None and current_info["turn_direction"] in ("left", "right"):
            return f"Turn {current_info['turn_direction']}"

        cumulative_dist = 0.0
        prev_pt = ego_xy
        for i in range(best_idx + 1, len(self._route_lanelet_ids)):
            info = self._lanelet_map.get(self._route_lanelet_ids[i])
            if info is None:
                continue
            first_pt = info["centerline"][0, :2]
            cumulative_dist += np.linalg.norm(first_pt - prev_pt)
            prev_pt = first_pt
            if info["turn_direction"] in ("left", "right"):
                return f"Turn {info['turn_direction']} in {int(round(cumulative_dist))}m"
        return "Continue straight"

    # ------------------------------------------------------------------ #
    # Shared input preparation                                          #
    # ------------------------------------------------------------------ #

    def _prepare_image_frames(self) -> Optional[torch.Tensor]:
        """GPU-decode the latest frames -> ``[n_cams, n_frames, 3, 560, 1008]`` uint8."""
        if not all(len(buf) >= self._num_frames for buf in self._camera_buffers.values()):
            return None

        import torchvision

        jpeg_buffers: list[torch.Tensor] = []
        for topic in self._camera_topics:
            frames = list(self._camera_buffers[topic])[-self._num_frames :]
            jpeg_buffers.extend([f for _, f in frames])

        decoded = [torchvision.io.decode_jpeg(buf, device="cuda") for buf in jpeg_buffers]
        stacked = torch.stack(decoded)
        if stacked.shape[-2:] != (560, 1008):
            stacked = (
                torch.nn.functional.interpolate(
                    stacked.float(), size=(560, 1008), mode="bicubic", align_corners=False
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
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray]]:
        """Ego motion in the t0 local frame + the raw map-frame position at t0.

        Returns ``(xyz, rot, ego_pos_map)`` or ``(None, None, None)`` if not
        enough odometry has been buffered yet.
        """
        need = self._num_history_steps * self.skip_num
        if len(self._odometry_buffer) < need:
            return None, None, None

        odom_history = list(self._odometry_buffer)[-need :: self.skip_num]
        positions, rotations = [], []
        for msg in odom_history:
            pose = msg.pose.pose
            positions.append([pose.position.x, pose.position.y, pose.position.z])
            rotations.append(
                Rotation.from_quat(
                    [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                ).as_matrix()
            )
        positions_np = np.asarray(positions, dtype=np.float32)
        rotations_np = np.asarray(rotations, dtype=np.float32)
        t0_rot_inv = np.linalg.inv(rotations_np[-1])
        history_xyz_local = (positions_np - positions_np[-1]) @ t0_rot_inv.T
        history_rot_local = np.einsum("ij,njk->nik", t0_rot_inv, rotations_np)

        xyz = torch.from_numpy(history_xyz_local).unsqueeze(0).unsqueeze(0)
        rot = torch.from_numpy(history_rot_local).unsqueeze(0).unsqueeze(0)
        return xyz, rot, positions_np[-1]

    # ------------------------------------------------------------------ #
    # Trajectory path                                                   #
    # ------------------------------------------------------------------ #

    def _traj_timer_callback(self) -> None:
        if self._active_future and not self._active_future.done():
            return
        if self._require_new_data and self._rx_count == self._last_traj_rx:
            return
        frames = self._prepare_image_frames()
        if frames is None:
            return
        ego_xyz, ego_rot, ego_pos_map = self._build_ego_history()
        if ego_xyz is None:
            return

        nav_text = self._compute_nav_text(ego_pos_map)
        if nav_text and nav_text != self._nav_text:
            self._nav_text = nav_text
            self.get_logger().info(f"Navigation instruction: {nav_text}")

        self._last_traj_rx = self._rx_count
        self.get_logger().info(f"Starting trajectory inference. nav_text={self._nav_text}")
        self._active_future = self._executor.submit(
            self._run_trajectory, frames, ego_xyz, ego_rot, self._nav_text
        )
        self._active_future.add_done_callback(self._on_future_done)

    def _run_trajectory(
        self,
        frames: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        nav_text: Optional[str],
    ) -> dict:
        start = time.time()
        messages = helper.create_message(
            frames.flatten(0, 1),
            camera_indices=self._camera_indices,
            num_frames_per_camera=self._num_frames,
            nav_text=nav_text,
        )
        processor_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            device="cuda",
        )
        processor_inputs = {
            k: v.to(self._device) if hasattr(v, "to") else v
            for k, v in processor_inputs.items()
        }
        model_inputs = helper.to_device(
            {
                "tokenized_data": processor_inputs,
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
            },
            device=self._device,
        )

        generation_kwargs = {
            "top_p": self._top_p,
            "temperature": self._temperature,
            "num_traj_samples": 1,
            "num_traj_sets": 1,
            "max_generation_length": self._max_gen_len,
            "return_extra": True,
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if nav_text:
                pred_xyz, pred_rot, extra = (
                    self._model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
                        data=model_inputs, **generation_kwargs
                    )
                )
            else:
                pred_xyz, pred_rot, extra = (
                    self._model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs, **generation_kwargs
                    )
                )

        trajectory = pred_xyz.detach().cpu()[0, 0, 0]
        rotation = pred_rot.detach().cpu()[0, 0, 0]
        traj_msg = self._to_autoware_trajectory(trajectory, rotation)
        self._trajectory_pub.publish(traj_msg)
        self._marker_pub.publish(self._trajectory_to_markers(trajectory))

        if nav_text:
            m = String()
            m.data = nav_text
            self._nav_text_pub.publish(m)

        cot_text = self._extract_text(extra, "cot")
        if cot_text:
            m = String()
            m.data = cot_text
            self._cot_pub.publish(m)
            s = StringStamped()
            s.stamp = traj_msg.header.stamp
            s.data = cot_text
            self._cot_stamped_pub.publish(s)

        return {
            "kind": "trajectory",
            "duration_sec": time.time() - start,
            "num_poses": len(traj_msg.points),
        }

    # ------------------------------------------------------------------ #
    # VQA path                                                          #
    # ------------------------------------------------------------------ #

    def _vqa_timer_callback(self) -> None:
        if not self._vqa_enabled or not self._question:
            return
        if self._active_future and not self._active_future.done():
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
        frames = self._prepare_image_frames()
        if frames is None:
            return

        ego_xyz = ego_rot = None
        if self._vqa_use_traj:
            ego_xyz, ego_rot, _ = self._build_ego_history()
            if ego_xyz is None and not self._warned_no_odom:
                self.get_logger().warn(
                    "No ego-motion history yet — answering without trajectory input."
                )
                self._warned_no_odom = True

        self._last_vqa_rx = self._rx_count
        self._last_answered_question = self._question
        question = self._question
        self.get_logger().info(f"Running VQA for question: {question!r}")
        self._active_future = self._executor.submit(
            self._run_vqa, question, frames, ego_xyz, ego_rot
        )
        self._active_future.add_done_callback(self._on_future_done)

    def _run_vqa(
        self,
        question: str,
        frames: torch.Tensor,
        ego_xyz: Optional[torch.Tensor],
        ego_rot: Optional[torch.Tensor],
    ) -> dict:
        start = time.time()
        with_traj = ego_xyz is not None and ego_rot is not None

        messages = helper.create_vqa_message(
            frames.flatten(0, 1),
            question=question,
            camera_indices=self._camera_indices,
            num_frames_per_camera=self._num_frames,
        )
        if with_traj:
            for item in reversed(messages[1]["content"]):
                if item.get("type") == "text":
                    item["text"] = _HIST_TRAJ_PLACEHOLDER + item["text"]
                    break

        processor_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            device="cuda",
        )
        processor_inputs = {
            k: v.to(self._device) if hasattr(v, "to") else v
            for k, v in processor_inputs.items()
        }
        if with_traj:
            traj_data = {
                "ego_history_xyz": ego_xyz.to(self._device),
                "ego_history_rot": ego_rot.to(self._device),
            }
            processor_inputs["input_ids"] = self._model.fuse_traj_tokens(
                processor_inputs["input_ids"], traj_data
            )
        model_inputs = helper.to_device(
            {"tokenized_data": processor_inputs}, device=self._device
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            extra = self._model.generate_text(
                data=model_inputs,
                top_p=self._top_p,
                temperature=self._temperature,
                num_samples=self._vqa_num_samples,
                max_generation_length=self._vqa_max_gen_len,
            )

        answer = self._pick_answer(extra)
        now = self.get_clock().now().to_msg()
        m = String()
        m.data = answer
        self._answer_pub.publish(m)
        s = StringStamped()
        s.stamp = now
        s.data = answer
        self._answer_stamped_pub.publish(s)

        return {
            "kind": "vqa",
            "duration_sec": time.time() - start,
            "question": question,
            "answer": answer,
            "with_traj": with_traj,
        }

    @staticmethod
    def _pick_answer(extra: dict) -> str:
        for key in ("answer", "meta_action", "cot"):
            if key not in extra:
                continue
            try:
                value = str(extra[key][0, 0]).strip()
            except Exception:
                continue
            if value:
                return value
        return ""

    # ------------------------------------------------------------------ #
    # Output helpers + completion                                       #
    # ------------------------------------------------------------------ #

    def _to_autoware_trajectory(
        self, trajectory: torch.Tensor, rotations: torch.Tensor | None
    ) -> Trajectory:
        traj_np = trajectory.numpy()
        rot_np = rotations.numpy() if rotations is not None else None

        traj_msg = Trajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.header.frame_id = self._frame_id

        dt = 0.1
        prev_xy = None
        for idx, point in enumerate(traj_np):
            tp = TrajectoryPoint()
            tp.pose.position.x = float(point[0])
            tp.pose.position.y = float(point[1])
            tp.pose.position.z = float(point[2])
            if rot_np is not None:
                quat = Rotation.from_matrix(rot_np[idx]).as_quat()
                tp.pose.orientation.x = float(quat[0])
                tp.pose.orientation.y = float(quat[1])
                tp.pose.orientation.z = float(quat[2])
                tp.pose.orientation.w = float(quat[3])
            else:
                tp.pose.orientation.w = 1.0

            if prev_xy is None:
                speed = 0.0
            else:
                speed = math.hypot(point[0] - prev_xy[0], point[1] - prev_xy[1]) / dt
            tp.longitudinal_velocity_mps = float(speed)
            tp.lateral_velocity_mps = 0.0
            tp.acceleration_mps2 = 0.0
            tp.heading_rate_rps = 0.0

            seconds_float = idx * dt
            seconds_int = int(seconds_float)
            tp.time_from_start = Duration(
                sec=seconds_int, nanosec=int((seconds_float - seconds_int) * 1e9)
            )
            traj_msg.points.append(tp)
            prev_xy = (point[0], point[1])
        return traj_msg

    def _trajectory_to_markers(self, trajectory: torch.Tensor) -> MarkerArray:
        traj_np = trajectory.numpy()
        line_marker = Marker()
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.header.frame_id = self._frame_id
        line_marker.ns = "trajectory"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 1.0
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        line_marker.pose.orientation.w = 1.0
        for point in traj_np:
            p = Point()
            p.x, p.y, p.z = float(point[0]), float(point[1]), float(point[2])
            line_marker.points.append(p)
        marker_array = MarkerArray()
        marker_array.markers.append(line_marker)
        return marker_array

    def _extract_text(self, extra: dict, key: str) -> Optional[str]:
        if not extra or key not in extra:
            return None
        try:
            text = extra[key][0, 0, 0]
        except Exception:
            return None
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        text = str(text).strip()
        return text or None

    def _on_future_done(self, future: Future) -> None:
        try:
            metrics = future.result()
        except Exception as exc:  # noqa: BLE001 — surface any inference failure
            self.get_logger().error(f"tele_alpa inference failed: {exc}")
            return
        if not metrics:
            return
        if metrics.get("kind") == "vqa":
            note = "" if metrics.get("with_traj") else " [no trajectory input]"
            self.get_logger().info(
                f"VQA answered in {metrics['duration_sec']:.2f}s{note}\n"
                f"  Q: {metrics['question']}\n"
                f"  A: {metrics['answer'] or '(empty — raise vqa_max_generation_length)'}"
            )
        else:
            self.get_logger().info(
                f"Trajectory inference completed in {metrics['duration_sec']:.2f}s "
                f"(points={metrics.get('num_poses', 0)})."
            )

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
