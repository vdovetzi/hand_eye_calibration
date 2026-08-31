#include "lib/core/target.hpp"
#include "lib/io/calibration_io.hpp"

#include <boost/program_options.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace po = boost::program_options;
using hand_eye::core::RigidTransform;

constexpr char kRawImageType[] = "sensor_msgs/msg/Image";
constexpr char kCompressedImageType[] = "sensor_msgs/msg/CompressedImage";
constexpr char kPoseType[] = "geometry_msgs/msg/Pose";
constexpr char kPoseStampedType[] = "geometry_msgs/msg/PoseStamped";
constexpr char kWindowName[] = "Hand-eye calibration validation";
constexpr double kClickDistancePixels = 18.0;

struct HelpRequested {};

struct Options {
  std::string armTopic;
  std::string imageTopic;
  std::string armFrame;
  std::filesystem::path calibrationPath;
  std::string pattern;
  std::string logLevel;
  double maximumReprojectionErrorPixels{1.0};
  bool publishArmCommand{false};
  double toolRollDegrees{0.0};
  double toolPitchDegrees{0.0};
  double toolYawDegrees{0.0};
};

struct FrameState {
  cv::Mat display;
  std::optional<hand_eye::core::TargetObservation> observation;
  std::optional<hand_eye::core::TargetPose> targetPose;
  std::string cameraFrame;
  builtin_interfaces::msg::Time stamp;
  std::optional<std::size_t> clickedPoint;
  std::uint64_t sequence{0};
};

std::filesystem::path calibrationFile(const std::filesystem::path &path) {
  if (path.extension() == ".yaml" || path.extension() == ".yml") {
    return path;
  }
  return path / "calibration.yaml";
}

rclcpp::Logger::Level parseLogLevel(const std::string &value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char value) { return std::tolower(value); });
  static const std::unordered_map<std::string, rclcpp::Logger::Level> levels{
      {"debug", rclcpp::Logger::Level::Debug},
      {"info", rclcpp::Logger::Level::Info},
      {"warn", rclcpp::Logger::Level::Warn},
      {"warning", rclcpp::Logger::Level::Warn},
      {"error", rclcpp::Logger::Level::Error},
      {"fatal", rclcpp::Logger::Level::Fatal}};
  const auto level = levels.find(normalized);
  if (level == levels.end()) {
    throw std::invalid_argument(
        "log level must be Debug, Info, Warn, Error or Fatal");
  }
  return level->second;
}

Options parseOptions(const int argc, char **argv) {
  po::options_description description("Allowed options");
  description.add_options()("help,h", "Show this help message")(
      "arm-topic,at", po::value<std::string>()->required(),
      "Pose or PoseStamped command topic (used only with "
      "--publish-arm-command)")(
      "image-topic,it", po::value<std::string>()->required(),
      "sensor_msgs/Image or sensor_msgs/CompressedImage topic")(
      "arm-frame,af", po::value<std::string>()->required(),
      "Fallback output frame when calibration metadata has no parent_frame")(
      "calibration-path,cp", po::value<std::string>()->default_value("."),
      "Calibration YAML or directory containing calibration.yaml")(
      "pattern-type,pt", po::value<std::string>()->required(),
      "Pattern: '1 dict id marker_size', '2 rows cols square_size', or "
      "'3 rows cols square_size marker_size dict [legacy_id]'")(
      "log-level,ll", po::value<std::string>()->default_value("Info"),
      "Log level: Debug, Info, Warn, Error or Fatal")(
      "max-reprojection-px", po::value<double>()->default_value(1.0),
      "Maximum target pose reprojection RMS in pixels")(
      "publish-arm-command", po::bool_switch()->default_value(false),
      "Publish clicked target positions to the arm topic; disabled by "
      "default and allowed only for eye-to-hand calibrations")(
      "tool-roll-deg", po::value<double>()->default_value(0.0),
      "Commanded tool roll in degrees")("tool-pitch-deg",
                                        po::value<double>()->default_value(0.0),
                                        "Commanded tool pitch in degrees")(
      "tool-yaw-deg", po::value<double>()->default_value(0.0),
      "Commanded tool yaw in degrees");

  po::variables_map variables;
  try {
    po::store(po::parse_command_line(argc, argv, description), variables);
    if (variables.count("help") != 0U) {
      std::cout << description << '\n';
      throw HelpRequested{};
    }
    po::notify(variables);
  } catch (const po::error &error) {
    std::cerr << "Error: " << error.what() << "\n\n" << description << '\n';
    throw;
  }

  Options options;
  options.armTopic = variables["arm-topic"].as<std::string>();
  options.imageTopic = variables["image-topic"].as<std::string>();
  options.armFrame = variables["arm-frame"].as<std::string>();
  options.calibrationPath =
      calibrationFile(variables["calibration-path"].as<std::string>());
  options.pattern = variables["pattern-type"].as<std::string>();
  options.logLevel = variables["log-level"].as<std::string>();
  options.maximumReprojectionErrorPixels =
      variables["max-reprojection-px"].as<double>();
  options.publishArmCommand = variables["publish-arm-command"].as<bool>();
  options.toolRollDegrees = variables["tool-roll-deg"].as<double>();
  options.toolPitchDegrees = variables["tool-pitch-deg"].as<double>();
  options.toolYawDegrees = variables["tool-yaw-deg"].as<double>();
  if (!std::isfinite(options.maximumReprojectionErrorPixels) ||
      options.maximumReprojectionErrorPixels <= 0.0) {
    throw std::invalid_argument("max-reprojection-px must be positive");
  }
  if (!std::isfinite(options.toolRollDegrees) ||
      !std::isfinite(options.toolPitchDegrees) ||
      !std::isfinite(options.toolYawDegrees)) {
    throw std::invalid_argument("tool orientation angles must be finite");
  }
  return options;
}

std::string waitForTopicType(const rclcpp::Node::SharedPtr &node,
                             const std::string &topic,
                             const std::vector<std::string> &supportedTypes) {
  const std::string resolved =
      node->get_node_topics_interface()->resolve_topic_name(topic);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
    const auto topics = node->get_topic_names_and_types();
    const auto found = topics.find(resolved);
    if (found != topics.end()) {
      for (const std::string &type : supportedTypes) {
        if (std::find(found->second.begin(), found->second.end(), type) !=
            found->second.end()) {
          return type;
        }
      }
      std::ostringstream message;
      message << "topic '" << resolved << "' has unsupported type(s):";
      for (const std::string &type : found->second) {
        message << ' ' << type;
      }
      throw std::runtime_error(message.str());
    }
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error("topic '" + resolved + "' was not found");
}

geometry_msgs::msg::Pose toPose(const RigidTransform &transform) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform.translation.at<double>(0);
  pose.position.y = transform.translation.at<double>(1);
  pose.position.z = transform.translation.at<double>(2);
  const cv::Quatd quaternion =
      cv::Quatd::createFromRotMat(transform.rotation).normalize();
  pose.orientation.x = quaternion.x;
  pose.orientation.y = quaternion.y;
  pose.orientation.z = quaternion.z;
  pose.orientation.w = quaternion.w;
  return pose;
}

geometry_msgs::msg::PoseStamped
toPoseStamped(const RigidTransform &transform, const std::string &frame,
              const builtin_interfaces::msg::Time &stamp) {
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = frame;
  pose.header.stamp = stamp;
  pose.pose = toPose(transform);
  return pose;
}

geometry_msgs::msg::Quaternion toolOrientation(const Options &options) {
  constexpr double degreesToRadians = CV_PI / 180.0;
  const double roll = options.toolRollDegrees * degreesToRadians;
  const double pitch = options.toolPitchDegrees * degreesToRadians;
  const double yaw = options.toolYawDegrees * degreesToRadians;
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);

  geometry_msgs::msg::Quaternion orientation;
  orientation.w = cr * cp * cy + sr * sp * sy;
  orientation.x = sr * cp * cy - cr * sp * sy;
  orientation.y = cr * sp * cy + sr * cp * sy;
  orientation.z = cr * cp * sy - sr * sp * cy;
  return orientation;
}

RigidTransform pointPose(const hand_eye::core::TargetPose &targetPose,
                         const cv::Point3f &point) {
  const cv::Mat translation =
      (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
  return targetPose.targetToCamera *
         RigidTransform(cv::Mat::eye(3, 3, CV_64F), translation);
}

double axisLength(const hand_eye::core::PatternConfig &pattern) {
  if (pattern.type == hand_eye::core::PatternType::ARUCO) {
    return pattern.markerSizeMeters * 0.5;
  }
  return pattern.squareSizeMeters * 2.0;
}

void mouseCallback(const int event, const int x, const int y, int,
                   void *context) {
  if (event != cv::EVENT_LBUTTONDOWN || context == nullptr) {
    return;
  }
  auto &state = *static_cast<FrameState *>(context);
  if (!state.observation || !state.targetPose) {
    return;
  }
  double closestDistance = std::numeric_limits<double>::infinity();
  std::size_t closestIndex = 0;
  for (std::size_t index = 0; index < state.observation->imagePoints.size();
       ++index) {
    const double distance =
        cv::norm(state.observation->imagePoints[index] - cv::Point2f(x, y));
    if (distance < closestDistance) {
      closestDistance = distance;
      closestIndex = index;
    }
  }
  if (closestDistance <= kClickDistancePixels) {
    state.clickedPoint = closestIndex;
  }
}

int run(const Options &options) {
  const auto node = rclcpp::Node::make_shared("calibration_validator");
  rclcpp::Logger logger = node->get_logger();
  logger.set_level(parseLogLevel(options.logLevel));

  const auto calibration =
      hand_eye::io::CalibrationIO::read(options.calibrationPath);
  if (options.publishArmCommand &&
      calibration.calibrationType != "eye_to_hand") {
    const std::string calibrationType = calibration.calibrationType.empty()
                                            ? "missing (legacy calibration)"
                                            : calibration.calibrationType;
    throw std::runtime_error(
        "--publish-arm-command requires calibration_type 'eye_to_hand'; "
        "loaded calibration type is " +
        calibrationType +
        ". Eye-in-hand validation does not produce base-frame arm commands");
  }
  const std::string outputFrame = calibration.parentFrame.empty()
                                      ? options.armFrame
                                      : calibration.parentFrame;
  if (options.publishArmCommand && outputFrame.empty()) {
    throw std::runtime_error(
        "arm command publishing requires a non-empty calibration parent_frame "
        "or --arm-frame");
  }
  const auto pattern = hand_eye::core::parsePattern(options.pattern);
  const hand_eye::core::TargetDetector detector(pattern);
  hand_eye::core::TargetPoseOptions poseOptions;
  poseOptions.maximumReprojectionErrorPixels =
      options.maximumReprojectionErrorPixels;

  const std::string imageType = waitForTopicType(
      node, options.imageTopic, {kRawImageType, kCompressedImageType});
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr armStamped;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr armPose;
  if (options.publishArmCommand) {
    const std::string armType =
        waitForTopicType(node, options.armTopic, {kPoseStampedType, kPoseType});
    if (armType == kPoseStampedType) {
      armStamped = node->create_publisher<geometry_msgs::msg::PoseStamped>(
          options.armTopic, 1);
    } else {
      armPose =
          node->create_publisher<geometry_msgs::msg::Pose>(options.armTopic, 1);
    }
  }
  const auto publishArm = [&](const geometry_msgs::msg::PoseStamped &pose) {
    if (armStamped) {
      armStamped->publish(pose);
    } else if (armPose) {
      armPose->publish(pose.pose);
    }
  };
  const geometry_msgs::msg::Quaternion commandedToolOrientation =
      toolOrientation(options);
  const auto cameraPosePublisher =
      node->create_publisher<geometry_msgs::msg::PoseStamped>(
          "calibration_pose_cam", 1);
  const auto outputPosePublisher =
      node->create_publisher<geometry_msgs::msg::PoseStamped>(
          "calibration_pose_gripper", 1);

  FrameState state;
  auto processImage = [&](cv::Mat image, const std::string &cameraFrame,
                          const builtin_interfaces::msg::Time &stamp) {
    state.display = std::move(image);
    state.cameraFrame = cameraFrame.empty() ? "camera" : cameraFrame;
    state.stamp = stamp;
    state.observation.reset();
    state.targetPose.reset();
    ++state.sequence;
    if (!calibration.childFrame.empty() && !cameraFrame.empty() &&
        calibration.childFrame != cameraFrame) {
      cv::putText(state.display, "Camera frame does not match calibration",
                  {16, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {0, 0, 255}, 2,
                  cv::LINE_AA);
      RCLCPP_WARN_THROTTLE(
          logger, *node->get_clock(), 5000,
          "Ignoring image frame '%s'; calibration expects '%s'",
          cameraFrame.c_str(), calibration.childFrame.c_str());
      return;
    }
    try {
      state.observation = detector.detect(state.display);
      if (!state.observation) {
        cv::putText(state.display, "Target not detected", {16, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 255}, 2, cv::LINE_AA);
        return;
      }
      detector.draw(state.display, *state.observation);
      auto estimation = hand_eye::core::TargetPoseEstimator::estimate(
          pattern, {*state.observation}, state.display.size(),
          calibration.cameraMatrix, calibration.distortionCoefficients,
          poseOptions);
      if (!estimation.poses.front()) {
        cv::putText(state.display, "Pose rejected by reprojection limit",
                    {16, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {0, 165, 255}, 2,
                    cv::LINE_AA);
        return;
      }
      state.targetPose = *estimation.poses.front();
      cv::Mat rotationVector;
      cv::Rodrigues(state.targetPose->targetToCamera.rotation, rotationVector);
      cv::drawFrameAxes(state.display, calibration.cameraMatrix,
                        calibration.distortionCoefficients, rotationVector,
                        state.targetPose->targetToCamera.translation,
                        static_cast<float>(axisLength(pattern)), 2);
      std::ostringstream status;
      status << "Reprojection RMS: " << std::fixed << std::setprecision(2)
             << state.targetPose->reprojectionErrorPixels
             << " px | click a target point";
      cv::putText(state.display, status.str(), {16, 30},
                  cv::FONT_HERSHEY_SIMPLEX, 0.65, {0, 255, 0}, 2, cv::LINE_AA);
    } catch (const std::exception &error) {
      RCLCPP_DEBUG(logger, "Target processing failed: %s", error.what());
      state.observation.reset();
      state.targetPose.reset();
    }
  };

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rawSubscription;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
      compressedSubscription;
  if (imageType == kRawImageType) {
    rawSubscription = node->create_subscription<sensor_msgs::msg::Image>(
        options.imageTopic, rclcpp::SensorDataQoS(),
        [&](const sensor_msgs::msg::Image::ConstSharedPtr message) {
          try {
            processImage(cv_bridge::toCvCopy(*message,
                                             sensor_msgs::image_encodings::BGR8)
                             ->image,
                         message->header.frame_id, message->header.stamp);
          } catch (const std::exception &error) {
            RCLCPP_WARN(logger, "Cannot decode image: %s", error.what());
          }
        });
  } else {
    compressedSubscription = node->create_subscription<
        sensor_msgs::msg::CompressedImage>(
        options.imageTopic, rclcpp::SensorDataQoS(),
        [&](const sensor_msgs::msg::CompressedImage::ConstSharedPtr message) {
          const cv::Mat image = cv::imdecode(message->data, cv::IMREAD_COLOR);
          if (image.empty()) {
            RCLCPP_WARN(logger, "Cannot decode compressed image");
            return;
          }
          processImage(image, message->header.frame_id, message->header.stamp);
        });
  }

  if (options.publishArmCommand) {
    RCLCPP_WARN(logger,
                "Validation ready on %s; clicking a detected point publishes "
                "an arm command to %s in %s. Press q or Esc to quit.",
                options.imageTopic.c_str(), options.armTopic.c_str(),
                outputFrame.c_str());
  } else {
    RCLCPP_INFO(logger,
                "Validation ready on %s; calibrated output frame is %s. "
                "Clicks publish visualization poses only; arm commands are "
                "disabled. Press q or Esc to quit.",
                options.imageTopic.c_str(), outputFrame.c_str());
  }
  cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(kWindowName, mouseCallback, &state);
  std::uint64_t publishedSequence = 0;
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);
    if (!state.display.empty()) {
      cv::imshow(kWindowName, state.display);
    }
    if (state.targetPose && state.sequence != publishedSequence) {
      const auto cameraPose = toPoseStamped(state.targetPose->targetToCamera,
                                            state.cameraFrame, state.stamp);
      cameraPosePublisher->publish(cameraPose);
      outputPosePublisher->publish(toPoseStamped(
          calibration.transform * state.targetPose->targetToCamera, outputFrame,
          state.stamp));
      publishedSequence = state.sequence;
    }
    const int key = cv::waitKey(10) & 0xff;
    if (key == 'q' || key == 'Q' || key == 27) {
      break;
    }
    if (state.clickedPoint && state.targetPose && state.observation) {
      const std::size_t index = *state.clickedPoint;
      state.clickedPoint.reset();
      if (index < state.observation->objectPoints.size()) {
        const RigidTransform cameraPoint = pointPose(
            *state.targetPose, state.observation->objectPoints[index]);
        cameraPosePublisher->publish(
            toPoseStamped(cameraPoint, state.cameraFrame, state.stamp));
        const auto outputPose = toPoseStamped(
            calibration.transform * cameraPoint, outputFrame, state.stamp);
        outputPosePublisher->publish(outputPose);
        if (options.publishArmCommand) {
          auto commandPose = outputPose;
          commandPose.pose.orientation = commandedToolOrientation;
          publishArm(commandPose);
          RCLCPP_WARN(logger, "Published target point %zu to arm topic %s",
                      index, options.armTopic.c_str());
        }
      }
    }
  }
  cv::destroyWindow(kWindowName);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  int result = 1;
  try {
    const Options options = parseOptions(argc, argv);
    rclcpp::init(argc, argv);
    result = run(options);
  } catch (const HelpRequested &) {
    return 0;
  } catch (const po::error &) {
    return 1;
  } catch (const std::exception &error) {
    std::cerr << "Validation failed: " << error.what() << '\n';
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return result;
}
