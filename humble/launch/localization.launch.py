"""
N3Mapping 重定位模式 Launch 文件

Requirements: 10.3, 9.6
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('n3mapping')
    bundle_arg = DeclareLaunchArgument(
        'bundle',
        description='Verified Product Map Bundle V1 directory'
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false',
        description='Whether to start RViz'
    )
    
    rviz_config_path = os.path.join(pkg_dir, 'launch', 'n3.rviz')
    
    n3mapping_node = Node(
        package='n3mapping',
        executable='n3mapping_product_runtime.py',
        name='n3mapping_node',
        output='screen',
        arguments=['--bundle', LaunchConfiguration('bundle')],
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
        bundle_arg,
        rviz_arg,
        n3mapping_node,
        rviz_node,
    ])
