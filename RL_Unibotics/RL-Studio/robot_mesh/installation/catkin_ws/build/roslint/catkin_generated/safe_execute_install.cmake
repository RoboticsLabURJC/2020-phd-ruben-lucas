execute_process(COMMAND "/home/rlucasz/Desktop/doctorado/RL_Unibotics/RL-Studio/robot_mesh/installation/catkin_ws/build/roslint/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/rlucasz/Desktop/doctorado/RL_Unibotics/RL-Studio/robot_mesh/installation/catkin_ws/build/roslint/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
