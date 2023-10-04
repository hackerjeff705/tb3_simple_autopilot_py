# Autonomous Robot Project with TurtleBot3

## Overview

The provided code represents a Python-based autonomous robot project utilizing a TurtleBot3 robot. This project aims to achieve several functionalities, including object detection (specifically, stoplights and stop signs), waypoint-based navigation using the Pure Pursuit controller, and an automatic emergency braking system. The project consists of three main components: the desktop component (split into robot controls and waypoint logger) and the Raspberry Pi 4 component.

## Desktop Component

### Robot Controls (robot_controls.py)

This Python script is responsible for controlling the TurtleBot3 robot and integrates the following key features:

- **Object Detection**: Utilizes MobileNetV1, implemented using TensorFlow Lite, to detect and determine the state of stoplights and stop signs in real-time. It processes video frames from a camera and identifies objects based on pre-trained models.

- **Traffic Light State Determination**: The code segment specifically identifies the state of traffic lights (red or green) by analyzing the colors within the detected traffic light region.

- **Pure Pursuit Controller**: Implements the Pure Pursuit geometric controller, which guides the robot along a predefined path (waypoints). It calculates the appropriate velocity and steering angle to reach the target waypoint while considering the robot's current position and orientation.

- **Automatic Emergency Braking**: Utilizes data from the LDS-02 LiDAR sensor to implement an automatic emergency braking system. It monitors the robot's distance to obstacles and controls the robot's velocity to avoid collisions.

### Waypoint Logger (waypoint_logger.py)

This script is responsible for logging waypoints as the robot moves. It subscribes to the Odometry topic to track the robot's position and orientation, recording this information along with the robot's speed into a CSV file. These waypoints serve as a predefined path for the Pure Pursuit controller.

## Raspberry Pi 4 Component

### Raspberry Pi Camera Publisher (raspicam_pub.py)

This script runs on the Raspberry Pi 4 and serves as the bridge between the camera attached to the Raspberry Pi and the robot's control system. It captures video frames from the camera, converts them into ROS Image messages, and publishes them on the '/video_frames' topic.

## How the Robot Works

![alt text]( https://github.com/hackerjeff705/tb3_simple_autopilot_py/blob/main/burger_autonomous_stack.jpeg "Robot Flow Structure")

1. The Raspberry Pi Camera Publisher runs on the Raspberry Pi 4, capturing video frames from the camera.

2. The desktop component's Robot Controls script runs on a separate computer (not on the Raspberry Pi). It subscribes to the '/video_frames' topic, receiving the camera's video feed.

3. The Object Detection module within Robot Controls uses TensorFlow Lite and a pre-trained MobileNetV1 model to detect objects in the video frames. It specifically identifies stoplights and stop signs.

4. The detected objects' positions and labels are used to determine their states (e.g., red or green for traffic lights). The state of these objects is used to make navigation decisions.

5. The Pure Pursuit Controller calculates the robot's velocity and steering angle to navigate it along a predefined path of waypoints. These waypoints are logged using the Waypoint Logger and are typically created manually or by recording the robot's path.

6. The Automatic Emergency Braking system, using data from the LDS-02 LiDAR sensor, continuously monitors the robot's proximity to obstacles. If it detects a potential collision, it adjusts the robot's velocity to prevent accidents.

7. The robot's control commands (velocity and steering angle) are published to the '/cmd_vel' topic, which directs the TurtleBot3's movements.

In summary, this project combines object detection, path planning using Pure Pursuit, and safety features like automatic emergency braking to create an autonomous robot capable of navigating based on detected objects and predefined waypoints while avoiding obstacles. It demonstrates the integration of multiple components to achieve autonomous navigation.

## Usage

To use this project, follow the instructions provided in each of the Python scripts included in this repository. Make sure to run the appropriate scripts on your TurtleBot3 robot, Raspberry Pi 4B, or development environment.

## Dependencies

- ROS (Robot Operating System)
- Python 3
- OpenCV
- TensorFlow Lite
- TurtleBot3
- LDS-02 LiDAR Sensor

Feel free to install or copy the code for your own projects!

## Installation
* Create a package for your computer and raspberry pi4b.
* Download/clone the project
* Move the respective folders to their respective directories
* Build and run!

## Acknowledgments
- The project leverages the capabilities of TurtleBot3, MobileNetV1, and Pure Pursuit for autonomous navigation and object detection.
- Special thanks to the ROS community for their contributions to the robotics ecosystem.
