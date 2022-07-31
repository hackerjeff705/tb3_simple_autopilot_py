# tb3_simple_autopilot_py
A simple autonomous robot in Python that implements the following using the turtlebot3.
* MobileNetV1 that detects and determines the state of stoplights and stopsign.
* Pure Pursuit geometric controller along with a tool to create waypoints.
* Automatic emergency braking system using the lds-02 lidar on the tb3.

The code is not run directly from the raspberry pi 4b. MobileNetV1 performance is approx 15 fps.

![alt text]( https://github.com/hackerjeff705/tb3_simple_autopilot_py/blob/main/burger_autonomous_stack.jpeg "Robot Flow Structure")

Feel free to install or copy the code for your own projects!

## Installation
* Create a package for your computer and raspberry pi4b.
* Download/clone the project
* Move the respective folders to their respective directories
* Build and run!
