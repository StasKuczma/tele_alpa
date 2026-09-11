from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """One node, one Alpamayo 1.5 model: Autoware trajectory planning +
    chain-of-thought reasoning + navigation text + scene VQA.

    The model is loaded once; the trajectory and VQA inferences share a
    single-worker executor so they never run on the GPU at the same time. When the
    ego stops, a stop report (why + scene description + suggested trajectory) is
    published and preempts the routine ticks.

        ros2 launch tele_alpa tele_alpa.launch.py question:="Is it safe to merge left?"
    """
    # 0=Front left 1=Front 2=Front right 3=Rear left 4=Rear 5=Rear right 6=Front telephoto
    default_camera_topics = [
        "/sensing/camera/camera3/image_raw/compressed",   # cross_left  -> index 0
        "/sensing/camera/camera1/image_raw/compressed",   # front_wide  -> index 1
        "/sensing/camera/camera4/image_raw/compressed",   # cross_right -> index 2
        "/sensing/camera/camera2/image_raw/compressed",   # front_tele  -> index 6
    ]
    default_camera_indices = [0, 1, 2, 6]

    args = [
        DeclareLaunchArgument("question", default_value="Describe the scene."),
        DeclareLaunchArgument("vqa_enabled", default_value="true"),
        DeclareLaunchArgument("vqa_use_trajectory_input", default_value="true"),
        DeclareLaunchArgument("vqa_period_sec", default_value="2.0"),
        DeclareLaunchArgument("inference_period_sec", default_value="1.0"),
        DeclareLaunchArgument("num_diffusion_steps", default_value="5"),
        DeclareLaunchArgument("expert_onnx_path", default_value=""),
        DeclareLaunchArgument("lanelet2_map_path", default_value=""),
        DeclareLaunchArgument("require_new_data", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        # --- stop report ---
        DeclareLaunchArgument("stop_report_enabled", default_value="true"),
        DeclareLaunchArgument("stopped_speed_threshold", default_value="0.1"),
        DeclareLaunchArgument("stopped_duration_sec", default_value="5.0"),
        DeclareLaunchArgument("stop_report_cooldown_sec", default_value="30.0"),
        DeclareLaunchArgument("trajectory_history_size", default_value="10"),
    ]

    node = Node(
        package="tele_alpa",
        executable="tele_alpa_node",
        name="tele_alpa_node",
        output="screen",
        parameters=[
            {
                "camera_topics": default_camera_topics,
                "camera_indices": default_camera_indices,
                "odometry_topic": "/localization/kinematic_state",
                "route_topic": "/planning/mission_planning/route",
                "lanelet2_map_path": LaunchConfiguration("lanelet2_map_path"),
                # trajectory / reasoning outputs (drop-in for alpamayo_ros)
                "trajectory_topic": "/alpamayo/predicted_trajectory",
                "cot_topic": "/alpamayo/reasoning",
                "cot_with_stamped_topic": "/alpamayo/reasoning_stamped",
                "nav_text_topic": "/alpamayo/nav_text",
                # VQA I/O
                "question": LaunchConfiguration("question"),
                "question_topic": "/tele_alpa/question",
                "answer_topic": "/tele_alpa/answer",
                "answer_stamped_topic": "/tele_alpa/answer_stamped",
                # behaviour
                "vqa_enabled": LaunchConfiguration("vqa_enabled"),
                "vqa_use_trajectory_input": LaunchConfiguration("vqa_use_trajectory_input"),
                "vqa_period_sec": LaunchConfiguration("vqa_period_sec"),
                "vqa_max_generation_length": 500,
                "inference_period_sec": LaunchConfiguration("inference_period_sec"),
                "num_diffusion_steps": LaunchConfiguration("num_diffusion_steps"),
                "expert_onnx_path": LaunchConfiguration("expert_onnx_path"),
                "use_greedy_decode": True,
                "require_new_data": LaunchConfiguration("require_new_data"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                # stop report: ego stops -> why + scene description + trajectory
                "stop_report_enabled": LaunchConfiguration("stop_report_enabled"),
                "stopped_speed_threshold": LaunchConfiguration("stopped_speed_threshold"),
                "stopped_duration_sec": LaunchConfiguration("stopped_duration_sec"),
                "stop_report_cooldown_sec": LaunchConfiguration("stop_report_cooldown_sec"),
                "stop_report_topic": "/tele_alpa/stop_report",
                "stop_report_stamped_topic": "/tele_alpa/stop_report_stamped",
                "stop_trajectory_topic": "/tele_alpa/stop_trajectory",
                "trajectory_history_size": LaunchConfiguration("trajectory_history_size"),
                "stop_candidates_topic": "/tele_alpa/stop_candidates",
            }
        ],
    )

    return LaunchDescription([*args, node])
