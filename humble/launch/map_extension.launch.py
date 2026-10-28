"""
N3Mapping 地图续建模式 Launch 文件

Requirements: 10.4, 12.1, 12.2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition, LaunchConfigurationNotEquals
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # 获取包路径
    pkg_dir = get_package_share_directory('n3mapping')
    
    # 声明参数
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(pkg_dir, 'config', 'n3mapping.yaml'),
        description='Path to the configuration file'
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to start RViz'
    )
    
    rviz_config_path = os.path.join(pkg_dir, 'launch', 'n3.rviz')
    
    # N3Mapping node. Empty config map paths fall back to Config defaults.
    n3mapping_node = Node(
        package='n3mapping',
        executable='n3mapping_node',
        name='n3mapping_node',
        additional_env={'OMP_WAIT_POLICY': EnvironmentVariable('OMP_WAIT_POLICY', default_value='PASSIVE')},
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'mode': 'map_extension',
            }
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
