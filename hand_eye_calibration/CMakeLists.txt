cmake_minimum_required(VERSION 3.16)
project(hand_eye_calibration)
include(FetchContent)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(USE_SANITIZERS_DEFAULT ON)
  add_compile_options(--coverage)
  add_link_options(--coverage)
else()
  set(USE_SANITIZERS_DEFAULT OFF)
endif()

option(USE_SANITIZERS "Whether or not using sanitizers" ${USE_SANITIZERS_DEFAULT})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/include/hand_eye_calibration/ui)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(ros_babel_fish REQUIRED)
find_package(console_bridge_vendor REQUIRED) # Provides console_bridge 0.4.0 on platforms without it.
find_package(console_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(AMENT_DEPENDENCIES
  geometry_msgs
  image_transport
  rclcpp
  ros_babel_fish
  sensor_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
)

find_package(OpenCV REQUIRED COMPONENTS
  aruco
  calib3d
  core
  highgui
  imgcodecs
  imgproc
)
message("OpenCV version: ${OpenCV_VERSION}")

set(HAND_EYE_OPENCV_LIBS
  opencv_aruco
  opencv_calib3d
  opencv_core
  opencv_highgui
  opencv_imgcodecs
  opencv_imgproc
)

find_package(Boost 1.65 REQUIRED COMPONENTS program_options)

find_package(Qt6 QUIET COMPONENTS Widgets)
if(Qt6_FOUND)
  set(HAND_EYE_QT_WIDGETS Qt6::Widgets)
else()
  find_package(Qt5 REQUIRED COMPONENTS Widgets)
  set(HAND_EYE_QT_WIDGETS Qt5::Widgets)
endif()

if(USE_SANITIZERS)
  set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/externals/sanitizers-cmake/cmake" ${CMAKE_MODULE_PATH})
  set(SANITIZE_ADDRESS ON)
  set(SANITIZE_MEMORY OFF)
  set(SANITIZE_UNDEFINED ON)
  set(SANITIZE_THREAD OFF)
  find_package(Sanitizers REQUIRED)
endif()

FetchContent_Declare(
  csv  
  GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
  GIT_SHALLOW TRUE 
  GIT_TAG 2.2.1
)

FetchContent_Declare(
  magic_enum  
  GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
  GIT_SHALLOW TRUE
  GIT_TAG v0.9.6
)

FetchContent_MakeAvailable(csv)
FetchContent_MakeAvailable(magic_enum)

add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

add_library(${PROJECT_NAME}_lib
  src/library/helpers/calibrate_helpers.cpp
  src/library/helpers/collect_helpers.cpp
  src/library/helpers/dataset_helpers.cpp
  src/library/helpers/ros_helpers.cpp
  src/library/helpers/validate_helpers.cpp
  src/library/pattern_detector.cpp
)
ament_target_dependencies(${PROJECT_NAME}_lib
  ${AMENT_DEPENDENCIES}
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_lib
  ${HAND_EYE_OPENCV_LIBS}
  csv
  magic_enum
)

add_executable(dataset_collector
  src/nodes/collect_dataset.cpp
)
ament_target_dependencies(dataset_collector
  ${AMENT_DEPENDENCIES}
)
target_include_directories(dataset_collector PRIVATE
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(dataset_collector
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  cv_bridge::cv_bridge
  console_bridge::console_bridge
)

add_executable(compute_calibration
  src/nodes/compute_calibration.cpp
)
ament_target_dependencies(compute_calibration
  ${AMENT_DEPENDENCIES}
)
target_include_directories(compute_calibration PRIVATE
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(compute_calibration
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  console_bridge::console_bridge
)

add_executable(validate_calibration
  src/nodes/validate_calibration.cpp
)
ament_target_dependencies(validate_calibration
  ${AMENT_DEPENDENCIES}
)
target_include_directories(validate_calibration PRIVATE
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(validate_calibration
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  cv_bridge::cv_bridge
  console_bridge::console_bridge
)

add_executable(hand_eye_backend
  src/app/backend.cpp
)
ament_target_dependencies(hand_eye_backend
  ${AMENT_DEPENDENCIES}
)
target_link_libraries(hand_eye_backend
  ${PROJECT_NAME}_lib
  cv_bridge::cv_bridge
)

add_executable(hand_eye_frontend
  src/app/frontend.cpp
  include/hand_eye_calibration/ui/hand_eye_frontend.ui
  resources/hand_eye_frontend.qrc
)
ament_target_dependencies(hand_eye_frontend
  cv_bridge
  rclcpp
  sensor_msgs
  std_srvs
)
target_link_libraries(hand_eye_frontend
  ${PROJECT_NAME}_lib
  ${HAND_EYE_OPENCV_LIBS}
  ${HAND_EYE_QT_WIDGETS}
)

if(USE_SANITIZERS)
  add_sanitizers(${PROJECT_NAME}_lib)
  add_sanitizers(dataset_collector)
  add_sanitizers(compute_calibration)
  add_sanitizers(validate_calibration)
  add_sanitizers(hand_eye_backend)
  add_sanitizers(hand_eye_frontend)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# install (
#     DIRECTORY ...
#     DESTINATION lib/${PROJECT_NAME}/
# )

install(TARGETS
    dataset_collector
    compute_calibration
    validate_calibration
    hand_eye_backend
    hand_eye_frontend
    DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY include/hand_eye_calibration/ui/
  DESTINATION share/${PROJECT_NAME}/ui
)

ament_export_include_directories(include)
ament_export_dependencies(
  ${AMENT_DEPENDENCIES}
)

ament_package()
