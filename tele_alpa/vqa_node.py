#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 node that answers natural-language questions about the live camera feed
using the Alpamayo 1.5 VLM (visual question answering).

The node buffers the same multi-camera / multi-frame image stack the trajectory
node uses, then feeds it — together with a question string — through
``model.generate_text`` and publishes the model's answer.

The question can be set three ways (later sources win):

1. the ``question`` parameter (a sensible default is baked in for setup);
2. a launch-file override of that parameter;
3. a ``std_msgs/String`` published on ``question_topic`` at runtime.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import rclpy
import torch
from autoware_internal_debug_msgs.msg import StringStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

DEFAULT_QUESTION = "What should the ego vehicle do next, and why?"

# Ego trajectory-history placeholder, identical to the one
# ``helper.create_message`` builds for the trajectory node. ``fuse_traj_tokens``
# replaces these pad tokens with the encoded ego-motion history so the VLM sees
# the same trajectory input during VQA that it gets during planning.
_NUM_TRAJ_HISTORY_TOKENS = 48
_HIST_TRAJ_PLACEHOLDER = (
    "<|traj_history_start|>"
    + "<|traj_history|>" * _NUM_TRAJ_HISTORY_TOKENS
    + "<|traj_history_end|>"
)


class AlpamayoVqaNode(Node):
    """ROS 2 node that consumes live camera topics and answers questions about them."""

    def __init__(self) -> None:
        super().__init__("vqa_node")

        self.model_name: str = "nvidia/Alpamayo-1.5-10B"

        self.declare_parameter("question", DEFAULT_QUESTION)
        self.declare_parameter("question_topic", "/tele_alpa/question")
        self.declare_parameter("answer_topic", "/tele_alpa/answer")
        self.declare_parameter("answer_stamped_topic", "/tele_alpa/answer_stamped")
        self.declare_parameter("inference_period_sec", 2.0)
        # Feed the ego-motion history into the model (same trajectory input the
        # planning node uses). Best-effort: if no odometry has been received the
        # node still answers, just without the trajectory tokens.
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        self.declare_parameter("use_trajectory_input", True)
        # When True, run one inference each time the question changes and then
        # stay idle. When False (default), keep re-answering the current
        # question every ``inference_period_sec`` as the scene evolves.
        self.declare_parameter("answer_once", False)
        # Only run when a new camera frame has arrived since the last inference.
        # Stops the node re-answering on a frozen buffer once a recording / bag
        # ends. A question change still forces one answer on the latest frames.
        self.declare_parameter("require_new_data", True)

        # ROS2 Jazzy: non-empty default for string-array params to infer type.
        self.declare_parameter("camera_topics", [""])
        # 0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear,
        # 5=Rear right, 6=Front telephoto — one index per camera topic.
        self.declare_parameter("camera_indices", [0])

        # Greedy decode gives a single deterministic answer. Set False to sample
        # with the nucleus preset (top_p / temperature) below.
        self.declare_parameter("use_greedy_decode", True)
        self.declare_parameter("top_p", 0.98)
        self.declare_parameter("temperature", 0.6)
        self.declare_parameter("max_generation_length", 256)
        self.declare_parameter("num_samples", 1)

        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        self._num_frames = 4
        self._num_history_steps = 16
        # /localization/kinematic_state is ~50 Hz; the model expects 10 Hz.
        self._odom_skip = 5

        self._question = str(self.get_parameter("question").value or DEFAULT_QUESTION)
        self._answer_once = bool(self.get_parameter("answer_once").value)
        self._require_new_data = bool(self.get_parameter("require_new_data").value)
        self._use_trajectory_input = bool(self.get_parameter("use_trajectory_input").value)
        self._warned_no_odom = False
        self._last_answered_question: Optional[str] = None
        # Monotonic count of camera messages received; frozen once the stream
        # stops. ``_last_run_rx_count`` is the value at the previous inference.
        self._rx_count = 0
        self._last_run_rx_count = -1

        queue_size = 10
        answer_topic = self.get_parameter("answer_topic").value
        self._answer_pub = self.create_publisher(String, answer_topic, queue_size)
        self.get_logger().info(f"Publishing answers on {answer_topic}")

        answer_stamped_topic = self.get_parameter("answer_stamped_topic").value
        self._answer_stamped_pub = self.create_publisher(
            StringStamped, answer_stamped_topic, queue_size
        )
        self.get_logger().info(f"Publishing stamped answers on {answer_stamped_topic}")

        question_topic = self.get_parameter("question_topic").value
        self.create_subscription(String, question_topic, self._question_callback, queue_size)
        self.get_logger().info(f"Listening for questions on {question_topic}")

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None

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
        self._camera_buffers: Dict[str, deque] = {
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

        # --- Odometry (ego trajectory history for the model) ---
        self._odometry_buffer: deque = deque(
            maxlen=self._num_history_steps * self._odom_skip + 10
        )
        if self._use_trajectory_input:
            odom_topic = self.get_parameter("odometry_topic").value
            odom_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=50,
            )
            self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)
            self.get_logger().info(f"Subscribed to odometry topic: {odom_topic}")
        else:
            self.get_logger().info("Trajectory input disabled (use_trajectory_input:=false).")

        # --- Model load ---
        self.get_logger().info(
            f"Loading Alpamayo model {self.model_name} on device={self._device} dtype={self._dtype}"
        )
        self._model = Alpamayo1_5.from_pretrained(self.model_name, dtype=self._dtype).to(
            self._device
        )
        self._model.eval()

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
        self._num_samples = max(1, int(self.get_parameter("num_samples").value))

        self._processor = helper.get_processor(self._model.tokenizer)

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        inference_period = float(self.get_parameter("inference_period_sec").value)
        self._timer = self.create_timer(inference_period, self._timer_callback)

        self.get_logger().info(
            f"Alpamayo VQA node ready. Current question: {self._question!r}"
        )

    # --- Callbacks ---

    def _image_callback(self, topic: str, msg: CompressedImage) -> None:
        jpeg_bytes = torch.frombuffer(bytearray(msg.data), dtype=torch.uint8)
        self._camera_buffers[topic].append((msg.header.stamp, jpeg_bytes))
        self._rx_count += 1

    def _odometry_callback(self, msg: Odometry) -> None:
        self._odometry_buffer.append(msg)

    def _build_ego_history(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Ego-motion history in the t0 local frame, as the trajectory node builds it.

        Returns ``(xyz, rot)`` shaped ``(1, 1, num_history_steps, 3)`` /
        ``(1, 1, num_history_steps, 3, 3)``, or ``(None, None)`` if not enough
        odometry has been buffered yet.
        """
        need = self._num_history_steps * self._odom_skip
        if len(self._odometry_buffer) < need:
            return None, None

        odom_history = list(self._odometry_buffer)[-need :: self._odom_skip]
        positions, rotations = [], []
        for msg in odom_history:
            pose = msg.pose.pose
            positions.append([pose.position.x, pose.position.y, pose.position.z])
            quat = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
            rotations.append(Rotation.from_quat(quat).as_matrix())

        positions_np = np.asarray(positions, dtype=np.float32)
        rotations_np = np.asarray(rotations, dtype=np.float32)
        t0_rot_inv = np.linalg.inv(rotations_np[-1])
        centered = positions_np - positions_np[-1]
        history_xyz_local = centered @ t0_rot_inv.T
        history_rot_local = np.einsum("ij,njk->nik", t0_rot_inv, rotations_np)

        xyz = torch.from_numpy(history_xyz_local).unsqueeze(0).unsqueeze(0)
        rot = torch.from_numpy(history_rot_local).unsqueeze(0).unsqueeze(0)
        return xyz, rot

    def _question_callback(self, msg: String) -> None:
        question = msg.data.strip()
        if not question:
            return
        if question != self._question:
            self.get_logger().info(f"New question: {question!r}")
        self._question = question
        # Force a fresh answer even in answer_once mode.
        self._last_answered_question = None

    def _timer_callback(self) -> None:
        if self._active_future and not self._active_future.done():
            return
        if self._answer_once and self._question == self._last_answered_question:
            return
        # Skip if no new frames since the last run — unless the question is still
        # unanswered, in which case answer once on whatever frames we have.
        question_pending = self._question != self._last_answered_question
        if (
            self._require_new_data
            and not question_pending
            and self._rx_count == self._last_run_rx_count
        ):
            return
        payload = self._prepare_payload()
        if payload is None:
            return
        self._last_run_rx_count = self._rx_count
        self._last_answered_question = payload["question"]
        self.get_logger().info(f"Running VQA for question: {payload['question']!r}")
        self._active_future = self._executor.submit(self._run_inference, payload)
        self._active_future.add_done_callback(self._on_future_done)

    # --- Core logic ---

    def _prepare_payload(self) -> Optional[dict]:
        if not all(len(buf) >= self._num_frames for buf in self._camera_buffers.values()):
            return None

        import torchvision

        jpeg_buffers: list[torch.Tensor] = []
        for topic in self._camera_topics:
            frames = list(self._camera_buffers[topic])[-self._num_frames :]
            jpeg_buffers.extend([f for _, f in frames])

        decoded = [torchvision.io.decode_jpeg(buf, device="cuda") for buf in jpeg_buffers]
        stacked = torch.stack(decoded)  # [N_total, 3, H, W] uint8 on GPU
        if stacked.shape[-2:] != (560, 1008):
            stacked = (
                torch.nn.functional.interpolate(
                    stacked.float(),
                    size=(560, 1008),
                    mode="bicubic",
                    align_corners=False,
                )
                .clamp(0, 255)
                .to(torch.uint8)
            )

        n_cams = len(self._camera_topics)
        camera_tensors = [
            stacked[i * self._num_frames : (i + 1) * self._num_frames] for i in range(n_cams)
        ]
        image_frames = torch.stack(camera_tensors, dim=0)  # [n_cams, n_frames, 3, H, W]

        ego_xyz, ego_rot = (None, None)
        if self._use_trajectory_input:
            ego_xyz, ego_rot = self._build_ego_history()
            if ego_xyz is None and not self._warned_no_odom:
                self.get_logger().warn(
                    "No ego-motion history yet — answering without trajectory input. "
                    "Check the odometry_topic parameter."
                )
                self._warned_no_odom = True

        return {
            "image_frames": image_frames,
            "question": self._question,
            "ego_history_xyz": ego_xyz,
            "ego_history_rot": ego_rot,
        }

    def _run_inference(self, payload: dict) -> dict:
        start = time.time()
        frames = payload["image_frames"]
        question = payload["question"]
        ego_xyz = payload.get("ego_history_xyz")
        ego_rot = payload.get("ego_history_rot")
        with_traj = ego_xyz is not None and ego_rot is not None

        messages = helper.create_vqa_message(
            frames.flatten(0, 1),
            question=question,
            camera_indices=self._camera_indices,
            num_frames_per_camera=self._num_frames,
        )
        if with_traj:
            # Prepend the ego trajectory-history placeholder to the question text
            # (same tokens helper.create_message builds for the planning node).
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
            # Replace the <|traj_history|> pad tokens with the encoded ego motion.
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
                num_samples=self._num_samples,
                max_generation_length=self._max_gen_len,
            )

        answer = self._pick_answer(extra)
        now = self.get_clock().now().to_msg()

        answer_msg = String()
        answer_msg.data = answer
        self._answer_pub.publish(answer_msg)

        stamped = StringStamped()
        stamped.stamp = now
        stamped.data = answer
        self._answer_stamped_pub.publish(stamped)

        return {
            "duration_sec": time.time() - start,
            "question": question,
            "answer": answer,
            "with_traj": with_traj,
        }

    @staticmethod
    def _pick_answer(extra: dict) -> str:
        """Return the first non-empty text field, preferring the answer span."""
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

    def _on_future_done(self, future: Future) -> None:
        try:
            metrics = future.result()
        except Exception as exc:  # noqa: BLE001 — surface any inference failure
            self.get_logger().error(f"Alpamayo VQA inference failed: {exc}")
            return
        if not metrics:
            return
        traj_note = "" if metrics.get("with_traj") else " [no trajectory input]"
        self.get_logger().info(
            f"VQA answered in {metrics['duration_sec']:.2f}s{traj_note}\n"
            f"  Q: {metrics['question']}\n"
            f"  A: {metrics['answer'] or '(empty — try a larger max_generation_length)'}"
        )

    def destroy_node(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = AlpamayoVqaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
