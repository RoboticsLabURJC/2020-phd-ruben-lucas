#!/usr/bin/env sh
# generated from catkin/cmake/template/local_setup.sh.in

# since this file is sourced either use the provided _CATKIN_SETUP_DIR
# or fall back to the destination set at configure time
: ${_CATKIN_SETUP_DIR:=/home/rlucasz/Desktop/doctorado/RL_Unibotics/RL-Studio/robot_mesh/installation/catkin_ws/devel}
CATKIN_SETUP_UTIL_ARGS="--extend --local"
. "$_CATKIN_SETUP_DIR/setup.sh"
unset CATKIN_SETUP_UTIL_ARGS
