"""
Synthetic relocalization before/after visualization.

Run with:
  ros2 launch n3mapping synthetic_relocalization_visualization.launch.py \
    map:=/path/to/n3map.pbstream max_tests:=20 interval_sec:=20
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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
        description='Fixed keyframe id, -1 randomly samples keyframes'
    )
    dropout_arg = DeclareLaunchArgument('dropout', default_value='0.3')
    noise_arg = DeclareLaunchArgument('noise_sigma', default_value='0.02')
    fake_yaw_arg = DeclareLaunchArgument('fake_odom_yaw_deg', default_value='90')
    fake_roll_arg = DeclareLaunchArgument('fake_odom_roll_deg', default_value='0')
    fake_pitch_arg = DeclareLaunchArgument('fake_odom_pitch_deg', default_value='0')
    fake_z_arg = DeclareLaunchArgument('fake_odom_tz', default_value='1')
    query_source_arg = DeclareLaunchArgument('query_source', default_value='local_submap')
    query_xy_arg = DeclareLaunchArgument('query_pose_xy_jitter_m', default_value='0.5')
    query_z_arg = DeclareLaunchArgument('query_pose_z_jitter_m', default_value='0.5')
    query_yaw_arg = DeclareLaunchArgument('query_pose_yaw_jitter_deg', default_value='8')
    query_rp_arg = DeclareLaunchArgument('query_pose_roll_pitch_jitter_deg', default_value='3')
    fov_az_arg = DeclareLaunchArgument('fov_azimuth_deg', default_value='0')
    fov_vert_arg = DeclareLaunchArgument('fov_vertical_deg', default_value='0')
    range_max_arg = DeclareLaunchArgument('range_max', default_value='30')
    occ_bins_arg = DeclareLaunchArgument('occlusion_dilation_bins', default_value='1')
    occ_tol_arg = DeclareLaunchArgument('occlusion_depth_tolerance', default_value='0.3')
    max_tests_arg = DeclareLaunchArgument('max_tests', default_value='0')
    interval_arg = DeclareLaunchArgument('interval_sec', default_value='20')
    random_seed_arg = DeclareLaunchArgument('random_seed', default_value='-1')
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to start RViz'
    )

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
            '--fake_odom_roll_deg', LaunchConfiguration('fake_odom_roll_deg'),
            '--fake_odom_pitch_deg', LaunchConfiguration('fake_odom_pitch_deg'),
            '--fake_odom_tz', LaunchConfiguration('fake_odom_tz'),
            '--query_source', LaunchConfiguration('query_source'),
            '--query_pose_xy_jitter_m', LaunchConfiguration('query_pose_xy_jitter_m'),
            '--query_pose_z_jitter_m', LaunchConfiguration('query_pose_z_jitter_m'),
            '--query_pose_yaw_jitter_deg', LaunchConfiguration('query_pose_yaw_jitter_deg'),
            '--query_pose_roll_pitch_jitter_deg', LaunchConfiguration('query_pose_roll_pitch_jitter_deg'),
            '--fov_azimuth_deg', LaunchConfiguration('fov_azimuth_deg'),
            '--fov_vertical_deg', LaunchConfiguration('fov_vertical_deg'),
            '--range_max', LaunchConfiguration('range_max'),
            '--occlusion_dilation_bins', LaunchConfiguration('occlusion_dilation_bins'),
            '--occlusion_depth_tolerance', LaunchConfiguration('occlusion_depth_tolerance'),
            '--max_tests', LaunchConfiguration('max_tests'),
            '--interval_sec', LaunchConfiguration('interval_sec'),
            '--random_seed', LaunchConfiguration('random_seed'),
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        map_arg,
        query_id_arg,
        dropout_arg,
        noise_arg,
        fake_yaw_arg,
        fake_roll_arg,
        fake_pitch_arg,
        fake_z_arg,
        query_source_arg,
        query_xy_arg,
        query_z_arg,
        query_yaw_arg,
        query_rp_arg,
        fov_az_arg,
        fov_vert_arg,
        range_max_arg,
        occ_bins_arg,
        occ_tol_arg,
        max_tests_arg,
        interval_arg,
        random_seed_arg,
        rviz_arg,
        visualizer_node,
        rviz_node,
    ])
