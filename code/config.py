#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#

import os
import yaml

class Config:
    try:
        config_name = os.environ['NDST_PATH'] + '/config/' + 'ndst_config.yaml'
    except:
        exit('ERROR: NDST_PATH not defined. Please source setup.bash.') 

    with open(config_name) as file:
        config_yaml = yaml.load(file, Loader=yaml.FullLoader)
        print('=======================================================')
        print('               Configuration Settings')
        print('=======================================================')
        train_ndst_yaml_name = config_yaml['train_ndst']
        print('Neural Net:     \t' + train_ndst_yaml_name)
        run_ndst_yaml_name = config_yaml['run_ndst']
        print('Run Neural:     \t' + run_ndst_yaml_name)
        print('\n')

    # train_ndst
    train_ndst_yaml = os.environ['NDST_PATH'] + '/config/train_ndst/' + train_ndst_yaml_name + '.yaml'
    with open(train_ndst_yaml) as file:
        train_ndst = yaml.load(file, Loader=yaml.FullLoader)

    # run_ndst
    run_ndst_yaml = os.environ['NDST_PATH'] + '/config/run_ndst/' + run_ndst_yaml_name + '.yaml'
    with open(run_ndst_yaml) as file:
        run_ndst = yaml.load(file, Loader=yaml.FullLoader)

    def __init__(self): # model_name):
        pass
    
    @staticmethod
    def summary():
        print('=======================================================')
        print('                 System Configuration ')
        print('=======================================================')
        print('                  ::: Neural Net :::')
        print(yaml.dump(Config.train_ndst))
        print('                  ::: Run Neural :::')
        print(yaml.dump(Config.run_ndst))


if __name__ == '__main__':
    Config.summary()