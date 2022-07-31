#!/usr/bin/env python3
# Imports
import os
import cv2
import time
import math
import rospy
import rospkg
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from rospkg import RosPack
from numpy import linalg as la
from matplotlib import patches
from race.msg import drive_param
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist, Point, PoseStamped, Quaternion, Pose, Point, Vector3

# ------ SSD MOBILENET TFLITE ------ #
class Vision:
	def __init__(self):
		self.br = CvBridge()
		self.obj = self.detected_object(label="empty", score=0, yx_min=[0, 0], yx_max=[0, 0], frame=None, ratio=0)
		self.trafficlight_state = 0

		# Load TensorFlow Lite Models and allocate tensors.
		r = rospkg.RosPack()
		self.package_path = r.get_path('burgerpilot_remote')
		self.interpreter = tf.lite.Interpreter(model_path=self.package_path + "/src/models/detect.tflite")
		self.interpreter.allocate_tensors()

		# Load label map
		self.labels = self.get_labels()

	def detected_object(self, label, score, yx_min, yx_max, frame, ratio):
		object = {
			"label": label,
			"score": score,
			"yx_min": np.array(yx_min),
			"yx_max": np.array(yx_max),
			"image": frame,
			"ratio": ratio
		}
		return object

	def get_labels(self):
		# Load the label map
		file_path = self.package_path + "/src/models/COCO_labels.txt"
		with open(file_path, 'r') as f:
			labels = [line.strip() for line in f.readlines()]

		# Have to do a weird fix for label map if using the COCO "starter model" from
		# https://www.tensorflow.org/lite/models/object_detection/overview
		# First label is 'unlabeled', which has to be removed.
		if labels[0] == 'unlabeled':
			del(labels[0])

		return labels

	def object_detection(self, frame):
		object_name, score, ymin, xmin, ymax, xmax = "empty", 0, 0, 0, 0, 0
		interpreter = self.interpreter
		labels = self.labels

		# Target labels
		target_labels = np.array(["trafficlight", "stopsign"])

		# Get input and output tensors.
		# [{'name': 'conv2d_input', 'index': 8, 'shape': array([ 1, 28, 28,  1]), 
		#   'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		# Get the width and height of the input image of the model
		_, height, width, _ = input_details[0]['shape']

		# Acquire image and resize to expected shape [1xHxWx3]
		img = frame.copy()
		imH, imW, _ = img.shape
		image_resized = cv2.resize(img, (width, height))
		input_data = np.expand_dims(image_resized, axis=0)

		# Check if the model is quantized or not
		floating_model = input_details[0]['dtype'] == np.float32

		# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
		if floating_model:
			# input_data = np.float32(input_data) / 255.
			input_data = (np.float32(input_data) - 127.5) / 127.5

		# Perform the actual detection by running the model with the image as input
		# Sets the value of the input tensor.
		interpreter.set_tensor(input_details[0]['index'], input_data)

		# Invoke the interpreter.
		# Be sure to set the input sizes, allocate tensors and fill values before calling this.
		interpreter.invoke()

		"""
		Retrieve results on detected objects
		- Bounding box coordinates
		- Class index of detected objects
		- Confidence of detected objects
		"""
		boxes = interpreter.get_tensor(output_details[0]['index'])[0]
		classes = interpreter.get_tensor(output_details[1]['index'])[0]
		scores = interpreter.get_tensor(output_details[2]['index'])[0]

		# Total number of detected objects (inaccurate and not needed)
		# num = interpreter.get_tensor(output_details[3]['index'])[0]
		min_conf_threshold = 0.5
		highest_score = 0.
		label = "empty"
		yx_min, yx_max = [0, 0], [0, 0]

		# Loop over all detections and draw detection box if confidence is above minimum threshold
		for i in range(len(scores)):
			score = scores[i]
			if (score > min_conf_threshold) and (scores[i] <= 1.0):
				# Look up object name from "labels" array using class index
				object_name = labels[int(classes[i])]

				if object_name in target_labels and score > highest_score:
					# Get bounding box coordinates and draw box
					# Interpreter can return coordinates that are outside of image dimensions, 
					# need to force them to be within image using max() and min()
					ymin = int(max(1,(boxes[i][0] * imH)))
					xmin = int(max(1,(boxes[i][1] * imW)))
					ymax = int(min(imH,(boxes[i][2] * imH)))
					xmax = int(min(imW,(boxes[i][3] * imW)))
					highest_score = score
					yx_min = [ymin, xmin]
					yx_max = [ymax, xmax]
					label = object_name

		self.obj["label"] = label
		self.obj["score"] = highest_score
		self.obj["yx_min"] = yx_min
		self.obj["yx_max"] = yx_max
		self.obj["image"] = frame

	def target_ratio(self):
		height = float(self.obj["image"].shape[0])
		width = float(self.obj["image"].shape[1])
		frame_area = height * width
		[ymin, xmin] = self.obj["yx_min"]
		[ymax, xmax] = self.obj["yx_max"]
		obj_area = (xmax - xmin) * (ymax - ymin)
		return obj_area / frame_area

	def run_cam(self, msg):
		try:
			# Output debugging information to the terminal
			rospy.loginfo("receiving video frame")

			# Convert ROS Image message to OpenCV image
			curr_frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

			# Start timer (for calculating frame rate)
			t1 = cv2.getTickCount()

			# Do object detection
			self.object_detection(curr_frame)
			if self.obj["label"] != "empty":
				self.draw_box()

			# Calculate framerate
			t2 = cv2.getTickCount()
			time1 = (t2-t1) / cv2.getTickFrequency()
			frame_rate = 1 / time1

			# Draw framerate in corner of frame
			cv2.putText(self.obj["image"], 'FPS: {0:.2f}'.format(frame_rate), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)

			# Display image
			cv2.imshow("raspi_cam", self.obj["image"])

			# Calculate ratio
			self.obj["ratio"] = self.target_ratio()

			if self.obj["label"] == "trafficlight" and self.obj["ratio"] > 0.15:
				self.trafficlight()

			cv2.waitKey(1)

		except CvBridgeError as e:
			print(e)

	def mask_ratio(self, mask):
		unique, pixels = np.unique(mask, return_counts=True)

		if len(pixels) > 1:
			return pixels[1] / sum(pixels)
		else:
			return 0


	def draw_box(self):
		"""
		Draw label
		- Example: 'person: 72%'
		- Get font size
		- Make sure not to draw label too close to top of window
		- Draw white box to put label text in
		- Draw label text
		"""
		object_name = self.obj["label"]
		score = self.obj["score"]
		[ymin, xmin] = self.obj["yx_min"]
		[ymax, xmax] = self.obj["yx_max"]

		cv2.rectangle(self.obj["image"], (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
		label = '%s: %d%%' % (object_name, int(score*100))
		labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
		label_ymin = max(ymin, labelSize[1] + 10)
		cv2.rectangle(self.obj["image"], (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
		cv2.putText(self.obj["image"], label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

	def trafficlight(self):
		[ymin, xmin] = self.obj["yx_min"]
		[ymax, xmax] = self.obj["yx_max"]
		detected_obj = self.obj["image"]

		# Convert target imag into HSV
		trafficlight_obj = detected_obj[ymin:ymax, xmin:xmax]
		hsv = cv2.cvtColor(trafficlight_obj, cv2.COLOR_BGR2HSV)

		# Crop red and green light
		height = trafficlight_obj.shape[0]
		width = trafficlight_obj.shape[1]
		h_athird = int(height/3)

		# Create a Mask for red and green light

		# RED LIGHT
		crop_red = hsv[:h_athird+20, 20:width-20]
		
		# Lower boundary RED color range values; Hue (0 - 10)
		red_lwr1 = np.array([0, 100, 201])
		red_upr1 = np.array([10, 255, 255])

		# Upper boundary RED color range values; Hue (160 - 180)
		red_lwr2 = np.array([160, 100, 20])
		red_upr2 = np.array([179, 255, 255])

		# Create Mask
		redMask1 = cv2.inRange(crop_red, red_lwr1, red_upr1)
		redMask2 = cv2.inRange(crop_red, red_lwr2, red_upr2)
		redMask = redMask1 + redMask2

		# GREEN LIGHT
		crop_grn = hsv[-h_athird-20:, 20:width-20]
		grn_lwr = np.array([45, 100, 50])
		grn_upr = np.array([75, 255, 255])
		grnMask = cv2.inRange(crop_grn, grn_lwr, grn_upr)

		# Calculate area of RED and GREEN
		redRatio = self.mask_ratio(redMask)
		grnRatio = self.mask_ratio(grnMask)

		if (redRatio > 0.25) and (grnRatio < 0.25):
			self.trafficlight_state = 1
		if (grnRatio > 0.25) and (redRatio < 0.25):
			self.trafficlight_state =  0


# ------ AUTOMATIC EMERGENCY BREAK ------ #
class Laser:
	def __init__(self):
		# self.VELOCITY = 0.05
		self.DISTANCE_THRESHOLD = 0.2
		self.ANGLE_RANGE = 360
		self.STEERING_ANGLE = 0


	def dist_control(self, distance, msg):
		kp_dist = 0.75

		# Calculate Distance to Collision Error
		if distance > self.DISTANCE_THRESHOLD:
			if distance <= 0.4:
				dist_error = distance - self.DISTANCE_THRESHOLD # only need distance error
				msg.linear.x = kp_dist * dist_error
				msg.angular.z = self.STEERING_ANGLE
				path_blocked = 1
			else:
				path_blocked = 0
		else:
			msg.linear.x = 0.0
			msg.angular.z = self.STEERING_ANGLE
			path_blocked = 1

		return path_blocked

	def get_index(self, angle, data):
		# For a given angle, return the corresponding index for the data.ranges array
		ilen = len(data.ranges)
		mid = angle / 2.0
		ipd = ilen / self.ANGLE_RANGE # index per degree
		lwr_bound = int(ipd * mid)
		upr_bound = int(ilen - lwr_bound)
		return np.array([i for i in range(ilen) if i <= lwr_bound or i >= upr_bound])
	
	def get_distance(self, data):
		angle_front = 20   # Range of angle in the front of the vehicle we want to observer
		avg_dist = 0

		# Get the corresponding list of indices for given range of angles
		index_front = self.get_index(angle_front, data)

		# Find the avg range distance
		ranges = np.array(data.ranges)
		avg_dist = np.average(ranges[index_front])

		return avg_dist


# ------ PURE PURSUIT ------ #
class Server:
	def __init__(self):
		# GLOBAL VARIABLES
		self.msg = Twist()
		self.vision = Vision()
		self.laser = Laser()
		self.waypoints = self.read_points()
		self.idx = 0
		self.xc = 0.0
		self.yc = 0.0
		self.yaw = 0.0
		self.vel = 0.0
		self.v_prev_error = 0.0

		# CAR VARIABLES
		self.LOOKAHEAD = 0.2
		self.WB = 0.04

		# OBJECT VARIABLES
		self.label = "empty"
		self.ratio = 0.
		self.stopsign_state = 0
		self.trafficlight_state = 0

		# LASER VARIABLE
		self.path_blocked = 0
		# self.distance = np.float('inf')

	def pose_callback(self, data):
		# Convert Quaternions to Eulers
		qx = data.pose.pose.orientation.x
		qy = data.pose.pose.orientation.y
		qz = data.pose.pose.orientation.z
		qw = data.pose.pose.orientation.w
		quaternion = (qx,qy,qz,qw)
		euler = euler_from_quaternion(quaternion)

		"""
		Get current state of the vehicle
		"""
		self.xc = data.pose.pose.position.x
		self.yc = data.pose.pose.position.y
		self.yaw = euler[2]
		self.vel = la.norm(np.array([data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z]),2)

	def camera_callback(self, msg):
		# Returns detected images
		self.vision.run_cam(msg)
		self.trafficlight_state = self.vision.trafficlight_state
		self.label = self.vision.obj["label"]
		self.ratio = self.vision.obj["ratio"]

	def laser_callback(self, data):
		distance = self.laser.get_distance(data)
		self.path_blocked = self.laser.dist_control(distance, self.msg)

	def read_points(self):
		"""
		CHANGE THIS PATH TO WHERE YOU HAVE SAVED YOUR CSV FILES
		"""
		r = rospkg.RosPack()
		package_path = r.get_path('burgerpilot_remote')
		file_name = 'wp_file.csv'
		file_path = package_path + '/src/waypoints/' + file_name
		with open(file_path) as f:
			path_points = np.loadtxt(file_path, delimiter = ',')
		return path_points

	def plot_arrow(self, x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
		"""
		Plot arrow
		"""
		if not isinstance(x, float):
			for ix, iy, iyaw in zip(x, y, yaw):
				plot_arrow(ix, iy, iyaw)
		else:
			plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width)
			plt.plot(x, y)
			patches.Rectangle((self.xc, self.yc), 0.35,0.2)

	def find_distance(self, x1, y1):
		distance = math.sqrt((x1 - self.xc) ** 2 + (y1 - self.yc) ** 2)
		return distance

	def find_distance_index_based(self, idx):
		waypoints = self.waypoints
		x1 = float(waypoints[idx][0])
		y1 = float(waypoints[idx][1])
		distance = math.sqrt((x1 - self.xc) ** 2 + (y1 - self.yc) ** 2)
		return distance

	def find_nearest_waypoint(self):
		"""
		Get closest idx to the vehicle
		"""
		curr_xy = np.array([self.xc, self.yc])
		waypoints_xy = self.waypoints[:, :2]
		nearest_idx = np.argmin(np.sum((curr_xy - waypoints_xy)**2, axis=1))
		return nearest_idx

	def idx_close_to_lookahead(self, idx):
		"""
		Get closest index to lookahead that is greater than the lookahead
		"""
		while self.find_distance_index_based(idx) < self.LOOKAHEAD:
			idx += 1
		return idx - 1

	def controller(self, freqs, nearest_idx, waypoints, curr_xy, target_xy, method="pure pursuit"):
		"""
		method = "pure pursuit" or "stanley"
		"""
		method = method.lower()

		if method == "pure pursuit":
			# Get coordinates
			xc, yc = curr_xy
			target_x, target_y = target_xy

			# Velocity PID controller
			kp = 1.0 # 1.0
			kd = 0
			ki = 0

			dt = 1.0 / freqs
			v_desired = float(waypoints[nearest_idx][3])
			v_error = v_desired - self.vel

			P_vel = kp * v_error
			I_vel = v_error * dt
			D_vel = kd * (v_error - self.v_prev_error) / dt

			velocity = P_vel + I_vel + D_vel
			self.v_prev_error = v_error
			# print(f"NEAREST INDEX = {nearest_idx}, output = {velocity}, velocity desired = {v_desired}, current = {vel}")

			# Pure Pursuit controller
			x_delta = target_x - xc
			y_delta = target_y - yc
			alpha = np.arctan(y_delta / x_delta) - self.yaw

			if alpha > np.pi / 2:
				alpha -= np.pi
			if alpha < -np.pi / 2:
				alpha += np.pi

			# Set the lookahead distance depending on the speed
			lookahead = self.find_distance(target_x, target_y)
			steering_angle = np.arctan((2 * self.WB * np.sin(alpha)) / lookahead)

			# Set max wheel turning angle
			if steering_angle > 0.5:
				steering_angle = 0.5
			elif steering_angle < -0.5:
				steering_angle = -0.5

		return velocity, steering_angle


	# MAIN #
	def main(self):
		# Initialize node
		rospy.init_node('pure_pursuit')

		# Initialize rate
		freqs = 10
		r = rospy.Rate(freqs)

		print("RUNNING PURE-PURSUIT CODE.... \n\n")
		time.sleep(2)

		# Program variables
		show_animation = True
		waypoints = self.waypoints

		# Initialize the message, subscriber and publisher
		rospy.Subscriber("odom", Odometry, self.pose_callback, queue_size=1)
		rospy.Subscriber('/video_frames', Image, self.camera_callback, queue_size=1)
		rospy.Subscriber("/scan", LaserScan, self.laser_callback, queue_size=1)
		pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

		# Plotting waypoints
		cx = waypoints[:, 0]; cy = waypoints[:, 1]

		try:
			while not rospy.is_shutdown():
				if self.path_blocked == 0:
					if self.label == "stopsign" and self.ratio > 0.13 and self.stopsign_state == 0:
						self.stopsign_state = 1
						self.msg.linear.x = 0.
						self.msg.angular.z = 0.
						pub.publish(self.msg)
						rospy.sleep(3)

					if self.trafficlight_state == 1:
						self.msg.linear.x = 0.
						self.msg.angular.z = 0.
					else: # Do pure pursuit
						# Get velocity and steering angle
						nearest_idx = self.find_nearest_waypoint()
						idx_near_lookahead = self.idx_close_to_lookahead(nearest_idx) 
						target_x = float(waypoints[idx_near_lookahead][0])
						target_y = float(waypoints[idx_near_lookahead][1])

						# Create a controller function with options to choose the type to use
						curr_xy = [self.xc, self.yc]
						target_xy = [target_x, target_y]
						velocity, steering_angle = self.controller(freqs, nearest_idx, waypoints, curr_xy, target_xy, method="pure pursuit")

						# Publish messages
						self.msg.linear.x = velocity
						self.msg.angular.z = steering_angle

						if self.stopsign_state == 1 and self.ratio < 0.13:
							self.stopsign_state = 0

				pub.publish(self.msg)

				# Plot map progression
				if show_animation:
					plt.cla()
					# For stopping simulation with the esc key.
					plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
					self.plot_arrow(float(self.xc), float(self.yc), float(self.yaw))
					plt.plot(cx, cy, "-r", label = "course")
					plt.plot(self.xc, self.yc, "-b", label = "trajectory")
					plt.plot(target_x, target_y, "xg", label = "target")
					plt.axis("equal")
					plt.grid(True)
					plt.title("Pure Pursuit Control" + str(1))
					plt.pause(0.001)

		except IndexError:
			# Close down the video stream when done
			cv2.destroyAllWindows()
		print("PURE PURSUIT COMPLETE --> COMPLETED ALL WAYPOINTS")
		

if __name__=='__main__':
	server = Server()
	server.main()
