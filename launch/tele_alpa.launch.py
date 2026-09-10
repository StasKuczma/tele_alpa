from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """One node, one Alpamayo 1.5 model: trajectory planning + trajectory-aware VQA.

    The model is loaded once; the trajectory and VQA inferences share a
    single-worker executor so they never run on the GPU at the same time.
    """
    # 0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear, 5=Rear right, 6=Front telephoto
    default_camera_topics = [
        "/sensing/camera/camera3/image_raw/compressed",   # cross_left  -> index 0
        "/sensing/camera/camera1/image_raw/compressed",   # front_wide  -> index 1
        "/sensing/camera/camera4/image_raw/compressed",   # cross_right -> index 2
        "/sensing/camera/camera2/image_raw/compressed",   # front_tele  -> index 6
    ]
    default_camera_indices = [0, 1, 2, 6]

    args = [
        DeclareLaunchArgument(
            "question",
            default_value="Describe the scene.",
            description="Question the node answers about the live scene.",
        ),
        DeclareLaunchArgument("vqa_enabled", default_value="true"),
        DeclareLaunchArgument("vqa_period_sec", default_value="2.0"),
        DeclareLaunchArgument("inference_period_sec", default_value="1.0"),
        DeclareLaunchArgument("require_new_data", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
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
                "trajectory_topic": "/alpamayo/predicted_trajectory",
                "cot_topic": "/alpamayo/reasoning",
                "cot_with_stamped_topic": "/alpamayo/reasoning_stamped",
                "question": LaunchConfiguration("question"),
                "question_topic": "/tele_alpa/question",
                "answer_topic": "/tele_alpa/answer",
                "answer_stamped_topic": "/tele_alpa/answer_stamped",
                "vqa_enabled": LaunchConfiguration("vqa_enabled"),
                "vqa_use_trajectory_input": True,
                "vqa_period_sec": LaunchConfiguration("vqa_period_sec"),
                "inference_period_sec": LaunchConfiguration("inference_period_sec"),
                "require_new_data": LaunchConfiguration("require_new_data"),
                "use_greedy_decode": True,
                "vqa_max_generation_length": 384,
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
    )

    return LaunchDescription([*args, node])
