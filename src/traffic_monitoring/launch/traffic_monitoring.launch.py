#!/usr/bin/env python3

import os

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

    config_file = LaunchConfiguration('config_file', default=default_config_file)


    return LaunchDescription([
        Node(
        package='pylon_ros2_camera_wrapper',
        namespace=camera_id,
        executable='pylon_ros2_camera_wrapper',
        name=node_name,
        output='screen',
        respawn=False,
        emulate_tty=True,
        parameters=[
            config_file,
            {
                # 'gige/mtu_size': mtu_size,
                # 'startup_user_set': startup_user_set,
                # 'enable_status_publisher': enable_status_publisher,
                # 'enable_current_params_publisher': enable_current_params_publisher
            }
        ]
        ),
    ])

