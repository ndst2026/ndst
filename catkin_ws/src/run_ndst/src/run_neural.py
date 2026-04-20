#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading 
import cv2
import time
import rospy
import numpy as np
from std_msgs.msg import Int32, Float64
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import math

import sys

from dataset_utils import ImageConverter, DriveRun, ImageProcess, KalmanFilter1D
from config import Config

from fusion.msg import Control

config = Config.train_ndst
velocity = 0.0
g_steer = 0.0
goal_velocity = 0.0
class NeuralControl:
    def __init__(self, weight_file_name, base_weight_name=None):
        rospy.init_node('run_neural')
        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.rate = rospy.Rate(30)
        self.drive= DriveRun(weight_file_name, base_weight_name)
        rospy.Subscriber(Config.run_ndst['camera_image_topic'], Image, self._controller_cb)
        self.image = None
        self.image_processed = False
        #self.config = Config()
        self.braking = False

    def _controller_cb(self, image): 
        img = self.ic.imgmsg_to_opencv(image)
        cropped = img[Config.run_ndst['image_crop_y1']:Config.run_ndst['image_crop_y2'],
                      Config.run_ndst['image_crop_x1']:Config.run_ndst['image_crop_x2']]
        
        img = cv2.resize(cropped, (config['input_image_width'],
                                   config['input_image_height']))
                                  
        self.image = self.image_process.process(img)

        self.image_processed = True
        
    def _timer_cb(self):
        self.braking = False

    def apply_brake(self):
        self.braking = True
        timer = threading.Timer(Config.run_neural['brake_apply_sec'], self._timer_cb) 
        timer.start()

      
def pos_vel_cb(value):
    global velocity

    vel_x = value.twist.twist.linear.x 
    vel_y = value.twist.twist.linear.y
    vel_z = value.twist.twist.linear.z
    
    velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    
def goal_vel_cb(value):
    global goal_velocity

    goal_velocity = value.data

def steer_cb(value):
    global g_steer
    g_steer = value.data

def main(weight_file_name, base_weight_name=None):

    # ready for neural network
    neural_control = NeuralControl(weight_file_name, base_weight_name)
    
    rospy.Subscriber(Config.run_ndst['base_pose_topic'], Odometry, pos_vel_cb)
    rospy.Subscriber(Config.run_ndst['vehicle_steer_topic'], Float64, steer_cb)
    rospy.Subscriber(Config.run_ndst['goal_velocity'], Float64, goal_vel_cb)
    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher(Config.run_ndst['vehicle_control_topic'], Control, queue_size = 10)
    joy_data = Control()

    print('\nStart running. Vroom. Vroom. Vroooooom......')
    print('steer \tthrt: \tbrake \tvel \tgoal_v \tHz')
    
    while not rospy.is_shutdown():

        if neural_control.image_processed is False:
            continue
        
        start = time.time()
        # predicted steering angle from an input image

        steer, throttle, brake = neural_control.drive.run((neural_control.image, velocity, float((goal_velocity-velocity))))
        
        if throttle > 1: throttle = 1
        if brake > 0.5:
            joy_data.throttle = 0
            joy_data.brake = brake
            joy_data.steer = steer * 0.8
        elif brake > 0.2:
            joy_data.throttle = 0
            joy_data.brake = brake
            joy_data.steer = steer * 0.9
        else:
            joy_data.throttle = throttle
            joy_data.brake = 0
            joy_data.steer = steer
        
            
        joy_pub.publish(joy_data)
        end = time.time()
        ## print out
        cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f} \t{4:.3f} \t{5}\r'.format(joy_data.steer, 
                        joy_data.throttle, joy_data.brake, velocity, goal_velocity, start-end)

        sys.stdout.write(cur_output)
        sys.stdout.flush()
            
        neural_control.image_processed = False
        neural_control.rate.sleep()



if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            exit('Usage:\n$ rosrun run_neural run_neural.py style_model base_model')
        print(sys.argv[2])
        main(sys.argv[1], sys.argv[2])
            

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
        