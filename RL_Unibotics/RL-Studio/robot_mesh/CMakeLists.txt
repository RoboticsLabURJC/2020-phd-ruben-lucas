cmake_minimum_required(VERSION 3.0.2)
gents
project(my_simulation)
gents

gents
## Compile as C++11, supported in ROS Kinetic and newer
gents
# add_compile_options(-std=c++11)
gents

gents
## Find catkin macros and libraries
gents
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
gents
## is used, also find other catkin packages
gents
find_package(catkin REQUIRED)
gents

gents
## System dependencies are found with CMake's conventions
gents
# find_package(Boost REQUIRED COMPONENTS system)
gents

gents

gents
## Uncomment this if the package has a setup.py. This macro ensures
gents
## modules and global scripts declared therein get installed
gents
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
gents
# catkin_python_setup()
gents

gents
################################################
gents
## Declare ROS messages, services and actions ##
gents
################################################
gents

gents
## To declare and build messages, services or actions from within this
gents
## package, follow these steps:
gents
## * Let MSG_DEP_SET be the set of packages whose message types you use in
gents
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
gents
## * In the file package.xml:
gents
##   * add a build_depend tag for "message_generation"
gents
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
gents
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
gents
##     but can be declared for certainty nonetheless:
gents
##     * add a exec_depend tag for "message_runtime"
gents
## * In this file (CMakeLists.txt):
gents
##   * add "message_generation" and every package in MSG_DEP_SET to
gents
##     find_package(catkin REQUIRED COMPONENTS ...)
gents
##   * add "message_runtime" and every package in MSG_DEP_SET to
gents
##     catkin_package(CATKIN_DEPENDS ...)
gents
##   * uncomment the add_*_files sections below as needed
gents
##     and list every .msg/.srv/.action file to be processed
gents
##   * uncomment the generate_messages entry below
gents
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)
gents

gents
## Generate messages in the 'msg' folder
gents
# add_message_files(
gents
#   FILES
gents
#   Message1.msg
gents
#   Message2.msg
gents
# )
gents

gents
## Generate services in the 'srv' folder
gents
# add_service_files(
gents
#   FILES
gents
#   Service1.srv
gents
#   Service2.srv
gents
# )
gents

gents
## Generate actions in the 'action' folder
gents
# add_action_files(
gents
#   FILES
gents
#   Action1.action
gents
#   Action2.action
gents
# )
gents

gents
## Generate added messages and services with any dependencies listed here
gents
# generate_messages(
gents
#   DEPENDENCIES
gents
#   std_msgs  # Or other packages containing msgs
gents
# )
gents

gents
################################################
gents
## Declare ROS dynamic reconfigure parameters ##
gents
################################################
gents

gents
## To declare and build dynamic reconfigure parameters within this
gents
## package, follow these steps:
gents
## * In the file package.xml:
gents
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
gents
## * In this file (CMakeLists.txt):
gents
##   * add "dynamic_reconfigure" to
gents
##     find_package(catkin REQUIRED COMPONENTS ...)
gents
##   * uncomment the "generate_dynamic_reconfigure_options" section below
gents
##     and list every .cfg file to be processed
gents

gents
## Generate dynamic reconfigure parameters in the 'cfg' folder
gents
# generate_dynamic_reconfigure_options(
gents
#   cfg/DynReconf1.cfg
gents
#   cfg/DynReconf2.cfg
gents
# )
gents

gents
###################################
gents
## catkin specific configuration ##
gents
###################################
gents
## The catkin_package macro generates cmake config files for your package
gents
## Declare things to be passed to dependent projects
gents
## INCLUDE_DIRS: uncomment this if your package contains header files
gents
## LIBRARIES: libraries you create in this project that dependent projects also need
gents
## CATKIN_DEPENDS: catkin_packages dependent projects also need
gents
## DEPENDS: system dependencies of this project that dependent projects also need
gents
catkin_package(
gents
#  INCLUDE_DIRS include
gents
#  LIBRARIES my_simulation
gents
#  CATKIN_DEPENDS other_catkin_pkg
gents
#  DEPENDS system_lib
gents
)
gents

gents
###########
gents
## Build ##
gents
###########
gents

gents
## Specify additional locations of header files
gents
## Your package locations should be listed before other locations
gents
include_directories(
gents
# include
gents
# ${catkin_INCLUDE_DIRS}
gents
)
gents

gents
## Declare a C++ library
gents
# add_library(${PROJECT_NAME}
gents
#   src/${PROJECT_NAME}/my_simulation.cpp
gents
# )
gents

gents
## Add cmake target dependencies of the library
gents
## as an example, code may need to be generated before libraries
gents
## either from message generation or dynamic reconfigure
gents
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
gents

gents
## Declare a C++ executable
gents
## With catkin_make all packages are built within a single CMake context
gents
## The recommended prefix ensures that target names across packages don't collide
gents
# add_executable(${PROJECT_NAME}_node src/my_simulation_node.cpp)
gents

gents
## Rename C++ executable without prefix
gents
## The above recommended prefix causes long target names, the following renames the
gents
## target back to the shorter version for ease of user use
gents
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
gents
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")
gents

gents
## Add cmake target dependencies of the executable
gents
## same as for the library above
gents
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
gents

gents
## Specify libraries to link a library or executable target against
gents
# target_link_libraries(${PROJECT_NAME}_node
gents
#   ${catkin_LIBRARIES}
gents
# )
gents

gents
#############
gents
## Install ##
gents
#############
gents

gents
# all install targets should use catkin DESTINATION variables
gents
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html
gents

gents
## Mark executable scripts (Python etc.) for installation
gents
## in contrast to setup.py, you can choose the destination
gents
# catkin_install_python(PROGRAMS
gents
#   scripts/my_python_script
gents
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
gents
# )
gents

gents
## Mark executables for installation
gents
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
gents
# install(TARGETS ${PROJECT_NAME}_node
gents
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
gents
# )
gents

gents
## Mark libraries for installation
gents
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
gents
# install(TARGETS ${PROJECT_NAME}
gents
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
gents
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
gents
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
gents
# )
gents

gents
## Mark cpp header files for installation
gents
# install(DIRECTORY include/${PROJECT_NAME}/
gents
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
gents
#   FILES_MATCHING PATTERN "*.h"
gents
#   PATTERN ".svn" EXCLUDE
gents
# )
gents

gents
## Mark other files for installation (e.g. launch and bag files, etc.)
gents
# install(FILES
gents
#   # myfile1
gents
#   # myfile2
gents
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
gents
# )
gents

gents
#############
gents
## Testing ##
gents
#############
gents

gents
## Add gtest based cpp test target and link libraries
gents
# catkin_add_gtest(${PROJECT_NAME}-test test/test_my_simulation.cpp)
gents
# if(TARGET ${PROJECT_NAME}-test)
gents
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
gents
# endif()
gents

gents
## Add folders to be run by python nosetests
gents
# catkin_add_nosetests(test)
gents
