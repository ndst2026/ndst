#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from model import NetModel
from config import Config
from dataset_utils import ImageProcess, DataAugmentation, DriveData
from progressbar import ProgressBar

import const
config = Config.train_ndst

###############################################################################
#
class DriveTrain:
    
    ###########################################################################
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56/'
    def __init__(self, data_path, base_model_path=None):
        
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = data_path[loc_slash + 1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = data_path
        csv_path = data_path + '/' + model_name + const.DATA_EXT  # use it for csv file name 
        
        self.csv_path = csv_path
        self.train_generator = None
        self.valid_generator = None
        self.train_hist = None
        self.data = None
        
        self.model_name = data_path + '_' + Config.train_ndst_yaml_name \
            + '_N' + str(config['network_type'])
        self.model_ckpt_name = self.model_name + '_ckpt'

        self.data = DriveData(self.csv_path)
        self.data_path = data_path
        self.net_model = NetModel(data_path, base_model_path=base_model_path)
        self.image_process = ImageProcess()
        self.data_aug = DataAugmentation()
        
        
    ###########################################################################
    #
    def _prepare_data(self):
        
        self.data.read()
        samples = list(zip(self.data.image_names, self.data.velocities, self.data.measurements, self.data.goal_velocities))
        
        self.train_data, self.valid_data = train_test_split(samples, 
                                    test_size=config['validation_rate'])
            
            
        self.num_train_samples = len(self.train_data)
        self.num_valid_samples = len(self.valid_data)
        
        print('Train samples: ', self.num_train_samples)
        print('Valid samples: ', self.num_valid_samples)
    
                                          
    ###########################################################################
    #
    def _build_model(self, show_summary=True):

        def _data_augmentation(image, steering_angle):
            if config['data_aug_flip'] is True:    
                # Flipping the image
                return True, self.data_aug.flipping(image, steering_angle)

            if config['data_aug_bright'] is True:    
                # Changing the brightness of image
                if steering_angle > config['steering_angle_jitter_tolerance'] or \
                    steering_angle < -config['steering_angle_jitter_tolerance']:
                    image = self.data_aug.brightness(image)
                return True, image, steering_angle

            if config['data_aug_shift'] is True:    
                # Shifting the image
                return True, self.data_aug.shift(image, steering_angle)

            return False, image, steering_angle

        def _prepare_batch_samples(batch_samples):
            images = []
            velocities = []
            measurements = []
            goal_velocities = []
            for image_name, velocity, measurement, goal_velocity in batch_samples:
                # for image_name, velocity, measurement, delta in batch_samples:
                image_path = self.data_path + '/' + image_name
                # print(image_path)
                image = cv2.imread(image_path)
                # if collected data is not cropped then crop here
                # otherwise do not crop.
                image = image[Config.run_ndst['image_crop_y1']:Config.run_ndst['image_crop_y2'],
                            Config.run_ndst['image_crop_x1']:Config.run_ndst['image_crop_x2']]
                image = cv2.resize(image, 
                                    (config['input_image_width'],
                                    config['input_image_height']))
                image = self.image_process.process(image)
                images.append(image)
                
                velocities.append(velocity)
                goal_velocities.append(goal_velocity-velocity)
                # if no brake data in collected data, brake values are dummy
                steering_angle, throttle, brake = measurement
                
                if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                    steering_angle = 0
             
                measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale'], brake*config['brake_scale']))

                # data augmentation
                append, image, steering_angle = _data_augmentation(image, steering_angle)
                if append is True:
                    images.append(image)
                    velocities.append(velocity)
                    goal_velocities.append(goal_velocity-velocity)
                    measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale'], brake*config['brake_scale']))
            return images, velocities, measurements, goal_velocities


        def _generator(samples, batch_size=config['batch_size']):
            num_samples = len(samples)
            while True: # Loop forever so the generator never terminates
                 
                samples = sklearn.utils.shuffle(samples)

                for offset in range(0, num_samples, batch_size):
                    batch_samples = samples[offset:offset+batch_size]

                    images, velocities, measurements, goal_vel = _prepare_batch_samples(batch_samples)
                    X_train_str = np.array(images)
                    X_train_vel = np.array(velocities).reshape(-1, 1)
                    X_train_gvel = np.array(goal_vel).reshape(-1, 1)
                    X_train = [X_train_str, X_train_vel, X_train_gvel]
                    
                    y_train = np.array(measurements)
                        
                    yield X_train, y_train
                        
        self.train_generator = _generator(self.train_data)
        self.valid_generator = _generator(self.valid_data)
        
        if (show_summary):
            self.net_model.model.summary()
    
    ###########################################################################
    #
    def _start_training(self):
        
        if (self.train_generator == None):
            raise NameError('Generators are not ready.')
        
        ######################################################################
        # callbacks
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        
        # checkpoint
        callbacks = []
        #weight_filename = self.data_path + '_' + Config.config_yaml_name \
        #    + '_N' + str(config['network_type']) + '_ckpt'
        checkpoint = ModelCheckpoint(self.model_ckpt_name +'.{epoch:02d}-{val_loss:.3f}.h5',
                                     monitor='val_loss', 
                                     verbose=1, save_best_only=True, mode='min')
        callbacks.append(checkpoint)
        
        # early stopping
        patience = config['early_stopping_patience']
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, 
                                  verbose=1, mode='min')
        callbacks.append(earlystop)

        self.train_hist = self.net_model.model.fit_generator(
                self.train_generator, 
                steps_per_epoch=self.num_train_samples//config['batch_size'], 
                epochs=config['num_epochs'], 
                validation_data=self.valid_generator,
                validation_steps=self.num_valid_samples//config['batch_size'],
                verbose=1, callbacks=callbacks, 
                use_multiprocessing=True,
                workers=12)
        
    ###########################################################################
    #
    def _plot_training_history(self):
    
        print(self.train_hist.history.keys())
        
        plt.figure() # new figure window
        ### plot the training and validation loss for each epoch
        plt.plot(self.train_hist.history['loss'][1:])
        plt.plot(self.train_hist.history['val_loss'][1:])
        #plt.title('Mean Squared Error Loss')
        plt.ylabel('mse loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validatation set'], loc='upper right')
        plt.tight_layout()
        #plt.show()
        plt.savefig(self.model_name + '_model.png', dpi=150)
        plt.savefig(self.model_name + '_model.pdf', dpi=150)
        
        new_txt = []
        bar = ProgressBar()
        for i in bar(range(len(self.train_hist.history['loss']))):
            new_txt.append(
                str(i)
                + ', '
                + str(self.train_hist.history['loss'][i])
                + ', '
                + str(self.train_hist.history['val_loss'][i])+ '\n')
            
        new_txt_fh = open(self.model_name + '_loss.csv', 'w')
        for i in range(len(new_txt)):
            new_txt_fh.write(new_txt[i])
        new_txt_fh.close()
        
    ###########################################################################
    #
    def train(self, show_summary=True, load_model_name=None):
        
        self._prepare_data()
        if config['weight_load'] is True:
            self.net_model.weight_load(load_model_name)
        self._build_model(show_summary)
        self._start_training()
        self.net_model.save(self.model_name)
            
        self._plot_training_history()
        Config.summary()



###############################################################################
#
def train(data_folder_name, load_model_name=None, base_model_path=None):
    drive_train = DriveTrain(data_folder_name, base_model_path=base_model_path)
    drive_train.train(show_summary=False, load_model_name=load_model_name)


###############################################################################
#
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python {} data_path base_model_name'.format(sys.argv[0]))
        train(sys.argv[1], base_model_path=sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')