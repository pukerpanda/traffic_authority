# traffic_authority
Ready To Take Control Of Your Traffic

https://github.com/roboflow/supervision
https://github.com/roboflow/supervision.git

## Fundamental Matrix

dataset: http://cmp.felk.cvut.cz/wbs/datasets/WxBS_v1.1.zip
matcher: https://github.com/Parskatt/RoMa.git


# ROS
## Isaac ROS Buildfarm

    https://nvidia-isaac-ros.github.io/getting_started/isaac_ros_buildfarm_cdn.html

    sudo apt install ros-humble-diagnostic-updater
    sudo apt install ros-humble-pcl-ros

## Pylon camera

   git clone -b humble https://github.com/basler/pylon-ros-camera
   git clone https://github.com/ros-perception/image_common.git -b humble

colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Debug
ros2 launch pylon_ros2_camera_wrapper pylon_ros2_camera.launch.py  node_name:=cam0 camera_id:=traffic

frame_id: pylon_camera
height: 960
width: 1280
encoding: rgb8

## Deepstream
Boost the clocks:
  sudo nvpmodel -m 0
  sudo jetson_clocks