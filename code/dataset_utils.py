#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import pandas as pd

from config import Config
from model import NetModel
from cv_bridge import CvBridge, CvBridgeError
###############################################################################
#
class DriveRun:
    def __init__(self, model_path, base_weight_file = None):
        
        #self.config = Config()
        self.net_model = NetModel(model_path, base_weight_file)
        self.net_model.load()

    def run(self, input):
        image = input[0]
        velocity = input[1]
        goal_velocity = input[2]
        
        np_img = np.expand_dims(image, axis=0)
        np_vel = np.array(velocity).reshape(-1, 1)
        np_goalvel = np.array(goal_velocity).reshape(-1, 1)
        predict = self.net_model.model.predict([np_img, np_vel, np_goalvel])
        steering_angle = predict[0][0]
        throttle = predict[0][1]
        brake = predict[0][2]
        
        steering_angle /= Config.train_ndst['steering_angle_scale']
        throttle /= Config.train_ndst['throttle_scale']
        brake /= Config.train_ndst['brake_scale']
        if throttle < 0:
            throttle = 0
        if brake < 0:
            brake = 0
        return steering_angle, throttle, brake

class ImageProcess:

    def process(self, img, bgr = True):
        return self._normalize(img, bgr=bgr)
        
    # img is expected as BGR         
    def _equalize_histogram(self, img, bgr = True):

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        if (bgr == True):
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return img
    
    def _normalize(self, img, bgr = True):

        img_norm = np.zeros_like(img)
        if (bgr != True): # if not bgr then assume it as RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
        return img_norm


class ImageConverter:
    def __init__(self):
        #load CvBridge for conversion to and from ros image messages
        self.bridge = CvBridge()

    def opencv_to_imgmsg(self, cv_img):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(cv_img, 'bgr8')
            return img_msg
        except CvBridgeError as err:
            return err

    def imgmsg_to_opencv(self, img_msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            return cv_img
        except CvBridgeError as err:
            return err


class ImageProcess:

    def process(self, img, bgr = True):
        return self._normalize(img, bgr=bgr)
        
    # img is expected as BGR         
    def _equalize_histogram(self, img, bgr = True):

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        if (bgr == True):
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return img
    
    def _normalize(self, img, bgr = True):

        img_norm = np.zeros_like(img)
        if (bgr != True): # if not bgr then assume it as RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
        return img_norm

class DataAugmentation():
    def __init__(self):
#        self.config = Config()
        self.bright_limit = (-0.5, 0.15)
        self.shift_range = (40,5)
        self.brightness_multiplier = None
        self.image_hsv = None
        self.rows = None
        self.cols = None
        self.ch = None
        self.shift_x = None
        self.shift_y = None
        self.shift_matrix = None

    def flipping(self, img, steering):
        flip_image = cv2.flip(img,1)
        flip_steering = steering*-1.0
        return flip_image, flip_steering

    def brightness(self, img):
        self.brightness_multiplier = 1.0 + np.random.uniform(low=self.bright_limit[0], high=self.bright_limit[1])
        self.image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        self.image_hsv[:,:,2] = self.image_hsv[:,:,2] * self.brightness_multiplier
        bright_image = cv2.cvtColor(self.image_hsv, cv2.COLOR_HSV2RGB)
        return bright_image
    
    def lstm_brightness(self,images):
        self.brightness_multiplier = 1.0 + np.random.uniform(low=self.bright_limit[0], high=self.bright_limit[1])
        images_hsv = []
        for i in range(len(images)):
            for j in range(len(images[i])):
                self.image_hsv = cv2.cvtColor(images[i][j], cv2.COLOR_RGB2HSV)
                self.image_hsv[:,:,2] = self.image_hsv[:,:,2] * self.brightness_multiplier
                bright_image = cv2.cvtColor(self.image_hsv, cv2.COLOR_HSV2RGB)
                images_hsv.append(bright_image)
        return images_hsv

    def shift(self, img, steering):
        self.rows, self.cols, self.ch = img.shape
        self.shift_x = self.shift_range[0]*np.random.uniform()-self.shift_range[0]/2
        shift_steering = steering + (self.shift_x/self.shift_range[0]*2*0.2) * -1
        self.shift_y = self.shift_range[1]*np.random.uniform()-self.shift_range[1]/2
        self.shift_matrix = np.float32([[1,0,self.shift_x],[0,1,self.shift_y]])
        shift_image = cv2.warpAffine(img, self.shift_matrix, (self.cols, self.rows))
        return shift_image, shift_steering

class KalmanFilter1D:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2, initial_value=0.0):
        self.x = initial_value
        self.p = 1.0
        self.q = process_variance
        self.r = measurement_variance
    
    def update(self, measurement):
        self.p = self.p + self.q

        k = self.p / (self.p+self.r)
        self.x = self.x + k*(measurement - self.x)
        self.p = (1.0-k)*self.p
        return self.x


class DriveData:
    csv_header = ['image_fname', 
                  'steering_angle', 'throttle', 'brake', 
                  'linux_time', 
                  'vel', 'vel_x', 'vel_y', 'vel_z',
                  'accel_x', 'accel_y', 
                  'pos_x', 'pos_y', 'pos_z',
                  'goal_vel']
    # , 
    #               'delta_steering_angle', 'delta_throttle', 'delta_brake']


    def __init__(self, csv_fname):
        self.csv_fname = csv_fname
        self.df = None
        self.image_names = []
        self.measurements = []
        self.time_stamps = []
        self.velocities = []
        self.goal_velocities = []
        self.velocities_xyz = []
        self.positions_xyz = []
        self.delta = []
        
    def read(self, read = True, show_statistics = True, normalize = True):
        self.df = pd.read_csv(self.csv_fname, names=self.csv_header, index_col=False)
        #self.fname = fname

        ############################################
        # show statistics
        if (show_statistics):
            print('\n####### data statistics #########')
            print('Steering Command Statistics:')
            print(self.df['steering_angle'].describe())

            print('\nThrottle Command Statistics:')
            # Throttle Command Statistics
            print(self.df['throttle'].describe())

        ############################################
        # normalize data
        # 'normalize' arg is for overriding 'normalize_data' config.
        if (Config.train_ndst['normalize_data'] and normalize):
            print('\nnormalizing... wait for a moment')
            num_bins = 50
            fig, (ax1, ax2) = plt.subplots(1, 2)
            #fig.suptitle('Data Normalization')
            hist, bins = np.histogram(self.df['steering_angle'], (num_bins))
            center = (bins[:-1] + bins[1:])*0.5
            ax1.bar(center, hist, width=0.05)
            ax1.set(title = 'original')

            remove_list = []
            samples_per_bin = Config.train_ndst['samples_per_bin']

            for j in range(num_bins):
                list_ = []
                for i in range(len(self.df['steering_angle'])):
                    if self.df.loc[i,'steering_angle'] >= bins[j] and self.df.loc[i,'steering_angle'] <= bins[j+1]:
                        list_.append(i)
                list_ = list_[samples_per_bin:]
                remove_list.extend(list_)
            
            print('\r####### data normalization #########')
            print('removed:', len(remove_list))
            self.df.drop(self.df.index[remove_list], inplace = True)
            self.df.reset_index(inplace = True)
            self.df.drop(['index'], axis = 1, inplace = True)
            print('remaining:', len(self.df))
            
            hist, _ = np.histogram(self.df['steering_angle'], (num_bins))
            ax2.bar(center, hist, width=0.05)
            ax2.plot((np.min(self.df['steering_angle']), np.max(self.df['steering_angle'])), 
                        (samples_per_bin, samples_per_bin))  
            ax2.set(title = 'normalized')          

            plt.tight_layout()
            plt.savefig(self.get_data_path() + '_normalized.png', dpi=150)
            plt.savefig(self.get_data_path() + '_normalized.pdf', dpi=150)

        ############################################ 
        # read out
        if (read): 
            num_data = len(self.df)
            
            bar = ProgressBar()
            
            for i in bar(range(num_data)): # we don't have a title
                self.image_names.append(self.df.loc[i]['image_fname'])
                self.measurements.append((float(self.df.loc[i]['steering_angle']),
                                        float(self.df.loc[i]['throttle']), 
                                        float(self.df.loc[i]['brake'])))
                self.time_stamps.append(float(self.df.loc[i]['linux_time']))
                self.velocities.append(float(self.df.loc[i]['vel']))
                self.velocities_xyz.append((float(self.df.loc[i]['vel_x']), 
                                            float(self.df.loc[i]['vel_y']), 
                                            float(self.df.loc[i]['vel_z']),
                                            float(self.df.loc[i]['accel_x']), 
                                            float(self.df.loc[i]['accel_y'])))
                self.positions_xyz.append((float(self.df.loc[i]['pos_x']), 
                                            float(self.df.loc[i]['pos_y']), 
                                            float(self.df.loc[i]['pos_z'])))
                self.goal_velocities.append(float(self.df.loc[i]['goal_vel']))


    def get_data_path(self):
        loc_slash = self.csv_fname.rfind('/')
        
        if loc_slash != -1: # there is '/' in the data path
            data_path = self.csv_fname[:loc_slash] # get folder name
            return data_path
        else:
            exit('ERROR: csv file path must have a separator.')



###############################################################################
#  for testing DriveData class only
def main(data_path):
    import const

    if data_path[-1] == '/':
        data_path = data_path[:-1]

    loc_slash = data_path.rfind('/')
    if loc_slash != -1: # there is '/' in the data path
        model_name = data_path[loc_slash + 1:] # get folder name
        #model_name = model_name.strip('/')
    else:
        model_name = data_path
    csv_path = data_path + '/' + model_name + const.DATA_EXT   
    
    data = DriveData(csv_path)
    data.read(read = False)



###############################################################################
#       
if __name__ == '__main__':
    import sys

    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ python {} data_path'.format(sys.argv[0]))

        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
