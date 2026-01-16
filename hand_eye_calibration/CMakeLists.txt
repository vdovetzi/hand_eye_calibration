cmake_minimum_required(VERSION 3.16)
project(hand_eye_calibration)
include(FetchContent)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# TODO: fix debug build without coverage
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ros_babel_fish REQUIRED)
find_package(console_bridge_vendor REQUIRED) # Provides console_bridge 0.4.0 on platforms without it.
find_package(console_bridge REQUIRED)
find_package(tf2 REQUIRED)


find_package(OpenCV REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS program_options)

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

add_executable(dataset_collector src/collect_dataset.cpp)
ament_target_dependencies(dataset_collector
    rclcpp
    image_transport
    cv_bridge
    sensor_msgs
    ros_babel_fish
)
target_include_directories(dataset_collector PRIVATE ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include/)
target_link_libraries(dataset_collector ${OpenCV_LIBS} ${Boost_LIBRARIES} csv magic_enum console_bridge::console_bridge)

add_executable(calibrator src/calibrate.cpp)
ament_target_dependencies(calibrator
    rclcpp
    geometry_msgs
    sensor_msgs
    tf2
)
target_include_directories(calibrator PRIVATE ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include/)
target_link_libraries(calibrator ${OpenCV_LIBS} ${Boost_LIBRARIES} csv magic_enum console_bridge::console_bridge)

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
    calibrator
    DESTINATION lib/${PROJECT_NAME}
)


ament_package()
