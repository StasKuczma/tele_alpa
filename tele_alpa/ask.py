#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Command-line helper to ask the running ``vqa_node`` a one-off question.

Usage::

    ros2 run tele_alpa ask "Is the traffic light red?"
    ros2 run tele_alpa ask --timeout 60 "How many pedestrians are crossing?"

Publishes the question on ``--question-topic`` and prints the next message that
arrives on ``--answer-topic``. The ``vqa_node`` must already be running.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class AskClient(Node):
    def __init__(self, question: str, question_topic: str, answer_topic: str) -> None:
        super().__init__("tele_alpa_ask")
        self._question = question
        self._answer: Optional[str] = None
        self._pub = self.create_publisher(String, question_topic, 10)
        self.create_subscription(String, answer_topic, self._on_answer, 10)

    def _on_answer(self, msg: String) -> None:
        self._answer = msg.data

    def publish_question(self) -> None:
        msg = String()
        msg.data = self._question
        self._pub.publish(msg)

    @property
    def answer(self) -> Optional[str]:
        return self._answer


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ask the Alpamayo VQA node a question.")
    parser.add_argument("question", help="Natural-language question about the scene.")
    parser.add_argument("--question-topic", default="/tele_alpa/question")
    parser.add_argument("--answer-topic", default="/tele_alpa/answer")
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Seconds to wait for an answer."
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    rclpy.init()
    node = AskClient(args.question, args.question_topic, args.answer_topic)

    # Give discovery a moment so the vqa_node sees our publisher, then send.
    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    published_at = None
    print(f"Q: {args.question}", flush=True)
    try:
        while rclpy.ok():
            now = node.get_clock().now().nanoseconds
            if now > deadline:
                print("Timed out waiting for an answer.", file=sys.stderr)
                sys.exit(1)
            if published_at is None or (now - published_at) > int(2e9):
                node.publish_question()
                published_at = now
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.answer is not None:
                print(f"A: {node.answer}", flush=True)
                return
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
