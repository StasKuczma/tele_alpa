from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch the Alpamayo VQA node against live camera topics.

    Override the question at launch time, e.g.::

        ros2 launch tele_alpa vqa.launch.py question:="Is it safe to change lanes left?"
    """
    # 0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear, 5=Rear right, 6=Front telephoto
    default_camera_topics = [
        "/sensing/camera/camera3/image_raw/compressed",   # cross_left  -> index 0
        "/sensing/camera/camera1/image_raw/compressed",   # front_wide  -> index 1
        "/sensing/camera/camera4/image_raw/compressed",   # cross_right -> index 2
        "/sensing/camera/camera2/image_raw/compressed",   # front_tele  -> index 6
    ]
    default_camera_indices = [0, 1, 2, 6]

    question_arg = DeclareLaunchArgument(
        "question",
        default_value="What should the ego vehicle do next, and why?",
        description="Default question the VQA node answers about the live scene.",
    )

    return LaunchDescription(
        [
            question_arg,
            Node(
                package="tele_alpa",
                executable="vqa_node",
                name="vqa_node",
                output="screen",
                parameters=[
                    {
                        "question": LaunchConfiguration("question"),
                        "question_topic": "/tele_alpa/question",
                        "answer_topic": "/tele_alpa/answer",
                        "answer_stamped_topic": "/tele_alpa/answer_stamped",
                        "camera_topics": default_camera_topics,
                        "camera_indices": default_camera_indices,
                        "odometry_topic": "/localization/kinematic_state",
                        "use_trajectory_input": True,
                        "inference_period_sec": 2.0,
                        "answer_once": False,
                        "require_new_data": True,
                        "use_greedy_decode": True,
                        "max_generation_length": 256,
                    }
                ],
            ),
        ]
    )
