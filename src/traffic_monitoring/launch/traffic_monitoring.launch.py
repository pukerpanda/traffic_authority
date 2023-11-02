#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

node_name='cam0'
camera_id='traffic'

def generate_launch_description():

    default_config_file = os.path.join(
        get_package_share_directory('traffic_monitoring'),
        'config',
        'pylon.yaml'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('pylon_ros2_camera_wrapper'),
                                                        'launch/pylon_ros2_camera.launch.py')]),
            launch_arguments={'config_file': default_config_file}.items(),
        ),

    ])

