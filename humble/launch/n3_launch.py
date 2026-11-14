"""YAML-only composition of the standalone N3Mapping nodes."""
import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def describe(mode):
    package = Path(get_package_share_directory("n3mapping"))
    settings = yaml.safe_load((package / "config/launch_humble.yaml").read_text())
    synthetic = mode == "synthetic"
    if synthetic:
        node = Node(package="n3mapping", executable="n3mapping_synthetic_relocalization_visualizer",
                    name="n3mapping_synthetic_relocalization_visualizer", output="screen",
                    arguments=[str(value) for value in settings["synthetic_arguments"]])
    else:
        node = Node(package="n3mapping", executable="n3mapping_node", name="n3mapping_node",
                    output="screen", additional_env={"OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY", "PASSIVE")},
                    parameters=[str(package / "config/n3mapping.yaml"), str(package / "config" / (mode + ".yaml"))])
    actions = []
    profile = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", (str(package / "config" / settings["dds_profile"]) if settings["dds_profile"] else ""))
    if profile:
        actions.append(SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", profile))
    actions.append(node)
    visualization = settings["synthetic_visualization" if synthetic else "visualization"]
    if visualization["enabled"]:
        actions.append(Node(package="rviz2", executable="rviz2", name="rviz2", output="screen",
                            arguments=["-d", str(package / "launch" / visualization["configuration"])]))
    return LaunchDescription(actions)
