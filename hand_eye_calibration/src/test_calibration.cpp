// https://github.com/ros/class_loader/pull/199
#include <console_bridge/console.h>

#include "geometry_msgs/msg/transform.hpp"
#include <boost/program_options.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <hand_eye_calibration/calibrate_helpers.hpp>
#include <hand_eye_calibration/test_helpers.hpp>
#include <hand_eye_calibration/ui.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/calib3d.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using Level = rclcpp::Logger::Level;
using image_transport::ImageTransport;
using image_transport::TransportHints;
using sensor_msgs::image_encodings::BGR8;
using sensor_msgs::msg::Image;
using namespace std::chrono_literals;

namespace po = boost::program_options;

std::shared_ptr<rclcpp::Logger> logger;
rclcpp::Node::SharedPtr node;

void ret(int32_t code) {
  logger.reset();
  node.reset();
  rclcpp::shutdown();
  exit(code);
}

void signalHandler([[maybe_unused]] int32_t signal) {
  std::cout << "\nSIGINT (Ctrl+C) received. Shutting down gracefully..."
            << std::endl;
  ret(0);
}

int32_t main(int32_t argc, char **argv) {
  signal(SIGINT, signalHandler);
  rclcpp::init(argc, argv);

  node = rclcpp::Node::make_shared("dataset_collector");
  logger = std::make_shared<rclcpp::Logger>(node->get_logger());

  // Parser settings
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Shows help message")(
      "arm-topic,at", po::value<std::string>()->required(),
      "Arm topic for sending command")("image-topic,it",
                                       po::value<std::string>()->required(),
                                       "Input image topic")(
      "calibration-path,cp", po::value<std::string>()->default_value("."),
      "Path to the directory, where calibration.yaml is located")(
      "pattern-type,pt", po::value<std::string>()->required(),
      "Supported markers:\n1. ArUco '<dict in {4,5,6,7}> <id> <size in m>'\n"
      "2. Chessboard '<rows> <cols> <size in m>'\n")(
      "log-level,ll", po::value<std::string>()->default_value("Info"),
      "Log-level: Info, Debug, Error");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      ret(0);
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "Error: " << e.what() << "\n" << desc << "\n";
    ret(1);
  }

  // Getting parsed input
  const std::string armTopicName = vm["arm-topic"].as<std::string>();
  const std::string imageTopicName = vm["image-topic"].as<std::string>();
  const fs::path calibrationPath =
      fs::path(vm["calibration-path"].as<std::string>()) / CALIBRATION_FILENAME;
  const std::string patternInfo = vm["pattern-type"].as<std::string>();
  const std::string logLevel = vm["log-level"].as<std::string>();

  // Setting logger level;
  std::optional<Level> level = magic_enum::enum_cast<Level>(logLevel);

  if (!level) {
    RCLCPP_ERROR(*logger, "Error: there is no such logger level");
    ret(1);
  }

  // Setting logger level
  logger->set_level(*level);
  rcutils_ret_t res = rcutils_logging_set_logger_level(
      HELPER_LOGGERNAME, static_cast<int32_t>(*level));
  if (RCUTILS_RET_OK != res) {
    RCLCPP_WARN(*logger, "Error: unable to set desired logging level");
  }
  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_ERROR);

  std::string armTopicType;

  // TODO: do without sleep
  bool armTopicFound = false;
  bool imageTopicFound = false;
  rclcpp::sleep_for(2s); // Small sleep for fetching actual topics
  auto topics = node->get_topic_names_and_types();
  for (const auto &[topic_name, topic_types] : topics) {
    RCLCPP_DEBUG(*logger, "Topic: %s", topic_name.c_str());
    if (topic_name == armTopicName) {
      armTopicFound = true;
      armTopicType = topic_types[0];

      RCLCPP_DEBUG(*logger, "Arm topic has type: %s", armTopicType.c_str());
    } else if (topic_name == imageTopicName) {
      imageTopicFound = true;
    }
  }

  if (!armTopicFound) {
    RCLCPP_ERROR(*logger, "Error: provide existing arm-topic");
    ret(1);
  }

  if (!imageTopicFound) {
    RCLCPP_ERROR(*logger, "Error: provide existing image-topic");
    ret(1);
  }

  auto pattern = std::make_unique<CalibrationPattern>();

  // Getting stuff for pattern
  try {
    pattern->setPatternInfo(patternInfo);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(*logger, "%s", e.what());
    ret(1);
  }

  auto it = std::make_unique<ImageTransport>(node);
  auto th = std::make_unique<TransportHints>(node.get(), "compressed");
  cv::Mat image;
  std::mutex image_guard;
  std::mutex point_guard;

  std::optional<size_t> clicked_ind;
  std::string camera_frame;

  // Image callback
  auto image_sub = it->subscribe(
      imageTopicName, 1,
      [&image, &pattern, &image_guard,
       &camera_frame](const Image::ConstSharedPtr &img_msg) {
        if (camera_frame.empty()) {
          camera_frame = img_msg->header.frame_id;
        }
        std::lock_guard<std::mutex> lock(image_guard);
        image = cv_bridge::toCvCopy(*img_msg, BGR8)->image;
        pattern->detectOn(image);
      },
      nullptr, th.get());

  // Arm publisher
  auto fish = BabelFish::make_shared();
  BabelFishPublisher::SharedPtr posePub =
      fish->create_publisher(*node, armTopicName, armTopicType, 1);
  CompoundMessage::SharedPtr message =
      fish->create_message_shared(armTopicType);

  auto ui = std::make_unique<ui::cvUI>(pattern->getImgPoints(), clicked_ind,
                                       point_guard, image, image_guard);

  auto data = std::make_unique<CalibrationData>();

  if (!data->read(calibrationPath)) {
    RCLCPP_ERROR(*logger, "Error: cannot read calibration file");
    ret(1);
  }

  auto T_cam2gripper = cvMatToTF2Transform(data->T);

  // Main loop
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);

    ui->show();

    point_guard.lock();
    if (clicked_ind) {

      std::optional<PoseStamped> poseOpt =
          pattern->estimatePose(*data, *pattern, *clicked_ind);

      if (!poseOpt) {
        RCLCPP_WARN(*logger, "Pose estimation failed");
        continue;
      }

      // TODO: publish in RViz
      PoseStamped poseInCam = *poseOpt;
      poseInCam.header.frame_id = camera_frame;
      poseInCam.header.stamp = node->now();

      tf2::Transform T_point_in_cam;
      tf2::fromMsg(poseInCam.pose, T_point_in_cam);

      tf2::Transform T_point_in_gripper = T_cam2gripper * T_point_in_cam;

      geometry_msgs::msg::Transform tf = tf2::toMsg(T_point_in_gripper);

      // TODO: publish in RViz
      PoseStamped poseInGripper;
      poseInGripper.pose.position.x = tf.translation.x;
      poseInGripper.pose.position.y = tf.translation.y;
      poseInGripper.pose.position.z = tf.translation.z;
      poseInGripper.pose.orientation = tf.rotation;

      fillMessage(*message, armTopicType, poseInGripper);

      posePub->publish(*message);

      clicked_ind.reset();
    }
    point_guard.unlock();
  }

  ret(0);
}
