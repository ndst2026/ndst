#!/bin/bash

####
## Assumption: you're at the 'ndst' directory.

##
# ndst main folder location
export NDST_PATH=$(pwd)

export GAZEBO_MODEL_PATH=${NDST_PATH}/catkin_ws/src/fusion/models:${GAZEBO_MODEL_PATH}
source ${NDST_PATH}/catkin_ws/devel/setup.bash

export PYTHONPATH=${NDST_PATH}/code:${PYTHONPATH}
