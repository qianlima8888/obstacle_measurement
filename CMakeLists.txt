cmake_minimum_required(VERSION 2.8.3)
project(obstacle_measurement)

set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    message_filters
    sensor_msgs
    cv_bridge
    image_transport
)

find_package(OpenCV 3 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

catkin_package(
  CATKIN_DEPENDS
    roscpp
    message_filters
    sensor_msgs
    cv_bridge
    image_transport
)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_executable(obstacle_measurement src/obstacle_measurement.cpp)
target_link_libraries(obstacle_measurement
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_executable(trash_measurement src/trash_measurement.cpp)
target_link_libraries(trash_measurement
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_executable(chair_measurement src/chair_measurement.cpp)
target_link_libraries(chair_measurement
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_executable(vision_laser src/vision_laser.cpp)
target_link_libraries(vision_laser
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_executable(door src/door_measurement.cpp)
target_link_libraries(door
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)