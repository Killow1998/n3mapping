"""
Synthetic relocalization before/after visualization.

Run with:
  ros2 launch n3mapping synthetic_relocalization_visualization.launch.py \
    map:=/path/to/n3map.pbstream query_id:=320
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('n3mapping')
    rviz_config_path = os.path.join(pkg_dir, 'launch', 'synthetic_relocalization_visualization.rviz')

    map_arg = DeclareLaunchArgument(
        'map',
        default_value='',
        description='Path to n3map.pbstream'
    )
    query_id_arg = DeclareLaunchArgument(
        'query_id',
        default_value='-1',
        description='Keyframe id to visualize, -1 selects the middle keyframe'
    )
    dropout_arg = DeclareLaunchArgument('dropout', default_value='0.3')
    noise_arg = DeclareLaunchArgument('noise_sigma', default_value='0.02')
    fake_yaw_arg = DeclareLaunchArgument('fake_odom_yaw_deg', default_value='90')
    repeat_arg = DeclareLaunchArgument('repeat', default_value='0')

    visualizer_node = Node(
        package='n3mapping',
        executable='n3mapping_synthetic_relocalization_visualizer',
        name='n3mapping_synthetic_relocalization_visualizer',
        output='screen',
        arguments=[
            '--map', LaunchConfiguration('map'),
            '--query_id', LaunchConfiguration('query_id'),
            '--dropout', LaunchConfiguration('dropout'),
            '--noise_sigma', LaunchConfiguration('noise_sigma'),
            '--fake_odom_yaw_deg', LaunchConfiguration('fake_odom_yaw_deg'),
            '--repeat', LaunchConfiguration('repeat'),
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
    )

    return LaunchDescription([
        map_arg,
        query_id_arg,
        dropout_arg,
        noise_arg,
        fake_yaw_arg,
        repeat_arg,
        visualizer_node,
        rviz_node,
    ])
