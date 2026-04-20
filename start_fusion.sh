#!/bin/bash

cd catkin_ws
catkin_make
cd ..

source ./catkin_ws/devel/setup.bash

if [ "$1" == "track_style_train" ] ; then
    echo "Starting with $1..."
    roslaunch fusion sitl.launch world:=$1 x:=0 y:=0 z:=0.3 R:=0 P:=0 Y:=0
elif [ "$1" == "track_style_test" ] ; then
    echo "Starting with $1..."
    roslaunch fusion sitl.launch world:=$1 x:=0 y:=0 z:=0.3 R:=0 P:=0 Y:=0
else 
    echo "Error: no $1.world file exist." 
fi