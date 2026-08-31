cmake_minimum_required(VERSION 3.16)
project(hand_eye_calibration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/app/ui)

option(USE_SANITIZERS "Enable address and undefined-behaviour sanitizers" OFF)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Jazzy's cv_bridge is compiled against the distribution OpenCV. Prefer that
# exact architecture-specific CMake package when installed, avoiding accidental
# ABI mixing with an OpenCV copy under /usr/local.
set(HAND_EYE_SYSTEM_OPENCV_DIR
  "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake/opencv4")
if(EXISTS "${HAND_EYE_SYSTEM_OPENCV_DIR}/OpenCVConfig.cmake")
  set(OpenCV_DIR "${HAND_EYE_SYSTEM_OPENCV_DIR}" CACHE PATH
      "ROS-compatible OpenCV package" FORCE)
endif()
find_package(OpenCV REQUIRED COMPONENTS aruco calib3d core highgui imgcodecs imgproc)
message(STATUS "OpenCV version: ${OpenCV_VERSION} (${OpenCV_DIR})")
set(HAND_EYE_CORE_OPENCV_LIBS
  opencv_aruco
  opencv_calib3d
  opencv_core
  opencv_imgcodecs
  opencv_imgproc
)

find_package(Boost 1.65 REQUIRED COMPONENTS program_options)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ros_babel_fish REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(rviz_rendering REQUIRED)

# RViz in ROS 2 Jazzy is built against Qt 5. Loading it into a Qt 6 process is
# unsupported and can crash when the embedded render panel initializes.
find_package(Qt5 REQUIRED COMPONENTS Widgets)
set(HAND_EYE_QT_WIDGETS Qt5::Widgets)

if(USE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

add_library(hand_eye_core
  lib/core/calibration.cpp
  lib/core/pose_filter.cpp
  lib/core/target.cpp
  lib/core/capture_policy.cpp
  lib/io/dataset_repository.cpp
  lib/io/calibration_io.cpp
)
target_include_directories(hand_eye_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(hand_eye_core PUBLIC
  ${HAND_EYE_CORE_OPENCV_LIBS}
  yaml-cpp::yaml-cpp
)

add_library(hand_eye_ros
  lib/ros/capture_source.cpp
  lib/ros/target_marker.cpp
)
target_include_directories(hand_eye_ros PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
ament_target_dependencies(hand_eye_ros PUBLIC
  builtin_interfaces
  cv_bridge
  geometry_msgs
  rclcpp
  ros_babel_fish
  sensor_msgs
  visualization_msgs
)
target_link_libraries(hand_eye_ros PUBLIC hand_eye_core)

add_executable(dataset_collector cli/collect_dataset.cpp)
target_include_directories(dataset_collector PRIVATE ${Boost_INCLUDE_DIRS})
ament_target_dependencies(dataset_collector rclcpp)
target_link_libraries(dataset_collector
  hand_eye_ros
  ${Boost_LIBRARIES}
  opencv_highgui
)

add_executable(compute_calibration cli/compute_calibration.cpp)
target_include_directories(compute_calibration PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(compute_calibration PRIVATE
  hand_eye_core
  ${Boost_LIBRARIES}
)

add_executable(validate_calibration cli/validate_calibration.cpp)
target_include_directories(validate_calibration PRIVATE ${Boost_INCLUDE_DIRS})
ament_target_dependencies(validate_calibration
  cv_bridge
  geometry_msgs
  rclcpp
  sensor_msgs
)
target_link_libraries(validate_calibration
  hand_eye_core
  ${Boost_LIBRARIES}
  opencv_highgui
)

add_executable(hand_eye_backend
  app/backend_main.cpp
  app/backend_node.cpp
)
ament_target_dependencies(hand_eye_backend
  builtin_interfaces
  geometry_msgs
  rcl_interfaces
  rclcpp
  std_srvs
  tf2
  tf2_ros
  visualization_msgs
)
target_link_libraries(hand_eye_backend hand_eye_core hand_eye_ros)

add_executable(hand_eye_frontend
  app/frontend_main.cpp
  app/frontend_window.cpp
  app/frontend_backend_client.cpp
  app/camera_feed.cpp
  app/rviz_widget.cpp
  app/frontend_state.cpp
  app/ui/hand_eye_frontend.ui
  app/resources/resources.qrc
)
ament_target_dependencies(hand_eye_frontend
  cv_bridge
  rclcpp
  rviz_common
  rviz_default_plugins
  rviz_rendering
  sensor_msgs
  std_srvs
)
target_link_libraries(hand_eye_frontend
  hand_eye_core
  ${HAND_EYE_QT_WIDGETS}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  ament_add_gtest(test_calibration test/test_calibration.cpp)
  target_link_libraries(test_calibration hand_eye_core)

  ament_add_gtest(test_target test/test_target.cpp)
  target_link_libraries(test_target hand_eye_core)

  ament_add_gtest(test_capture_policy test/test_capture_policy.cpp)
  target_link_libraries(test_capture_policy hand_eye_core)

  ament_add_gtest(test_pose_filter test/test_pose_filter.cpp)
  target_link_libraries(test_pose_filter hand_eye_core)

  ament_add_gtest(test_target_marker test/test_target_marker.cpp)
  target_link_libraries(test_target_marker hand_eye_ros)

  ament_add_gtest(test_dataset_repository test/test_dataset_repository.cpp)
  target_link_libraries(test_dataset_repository hand_eye_core)

  ament_add_gtest(test_frontend_state
    test/test_frontend_state.cpp
    app/frontend_state.cpp
  )
  target_include_directories(test_frontend_state PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(test_frontend_state ${HAND_EYE_QT_WIDGETS})

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  # The package uses clang-format's C++ style. ROS uncrustify enforces a
  # conflicting brace/reference style, so it is intentionally not registered.
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
  dataset_collector
  compute_calibration
  validate_calibration
  hand_eye_backend
  hand_eye_frontend
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
install(FILES LICENSE README.md
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY docs
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
