execute_process(COMMAND "/home/ruben/Desktop/RL-Studio/rl_studio/installation/catkin_ws/build/xacro/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/ruben/Desktop/RL-Studio/rl_studio/installation/catkin_ws/build/xacro/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
