"""
N3Mapping 重定位模式 Launch 文件

Usage:
    ros2 launch n3mapping localization.launch.py \
        config_file:=<path-to-localization-config.yaml> \
        rviz:=true

The config file must set mode: "localization" and a non-empty map_path.
`config_file` is required on purpose: silently defaulting to a mapping config
would run the node in the wrong mode.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition, LaunchConfigurationNotEquals
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('n3mapping')

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        description='Path to the localization configuration file '
                    '(must set mode: "localization" and map_path)'
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Whether to start RViz'
    )

    rviz_config_path = os.path.join(pkg_dir, 'launch', 'n3.rviz')

    n3mapping_node = Node(
        package='n3mapping',
        executable='n3mapping_node',
        name='n3mapping_node',
        additional_env={'OMP_WAIT_POLICY': EnvironmentVariable('OMP_WAIT_POLICY', default_value='PASSIVE')},
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
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
        DeclareLaunchArgument('dds_profile',
            default_value=EnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE',
                default_value=os.path.join(pkg_dir, 'config', 'fastdds.xml')),
            description='Fast DDS XML profile; preserves an existing environment profile'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', LaunchConfiguration('dds_profile'),
            condition=LaunchConfigurationNotEquals('dds_profile', '')),
        config_file_arg,
        rviz_arg,
        n3mapping_node,
        rviz_node,
    ])
