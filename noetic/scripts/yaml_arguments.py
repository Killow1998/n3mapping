#!/usr/bin/env python3
"""Load a standalone ROS 1 tool's YAML and exec it without a child supervisor."""
import os
import sys

import roslib.packages
import yaml
from roslaunch.substitution_args import resolve_args

with open(sys.argv[1], encoding="utf-8") as stream:
    settings = yaml.safe_load(stream)[sys.argv[2]]
if settings["enabled"]:
    package, executable = settings["program"]
    nodes = roslib.packages.find_node(package, executable)
    if not nodes:
        raise FileNotFoundError(package + "/" + executable)
    command = [nodes[0], *sys.argv[3:]]
    command.extend(resolve_args(str(value)) for value in settings["arguments"])
    os.execv(command[0], command)
