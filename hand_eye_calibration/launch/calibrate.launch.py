from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration('config')
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'config',
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare('hand_eye_calibration'),
                        'config',
                        'default.yaml',
                    ]
                ),
            ),
            Node(
                package='hand_eye_calibration',
                executable='hand_eye_backend',
                name='hand_eye_backend',
                output='screen',
                parameters=[config],
            ),
            Node(
                package='hand_eye_calibration',
                executable='hand_eye_frontend',
                name='hand_eye_frontend',
                output='screen',
            ),
        ]
    )
