"""
N3Mapping 地图续建模式 Launch 文件

Requirements: 10.4, 12.1, 12.2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
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
    
    rviz_config_path = os.path.join(pkg_dir, 'launch', 'n3.rviz')
    
    # N3Mapping 节点 (map_path 和 map_save_path 使用代码中的默认值，即源目录下的 map 文件夹)
    n3mapping_node = Node(
        package='n3mapping',
        executable='n3mapping_node',
        name='n3mapping_node',
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
        arguments=['-d', rviz_config_path] 
    )
    
    return LaunchDescription([
        config_file_arg,
        n3mapping_node,
        rviz_node,
    ])
