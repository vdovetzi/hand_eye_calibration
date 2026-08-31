#include "app/backend_node.hpp"

#include "lib/core/capture_policy.hpp"
#include "lib/core/pose_filter.hpp"
#include "lib/core/target.hpp"
#include "lib/io/calibration_io.hpp"
#include "lib/io/dataset_repository.hpp"
#include "lib/ros/capture_source.hpp"
#include "lib/ros/target_marker.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hand_eye::app {
namespace {

namespace fs = std::filesystem;
using Trigger = std_srvs::srv::Trigger;
using Response = std::shared_ptr<Trigger::Response>;

std::string escapeJson(const std::string &text) {
  std::ostringstream output;
  for (const char character : text) {
    switch (character) {
    case '\\':
      output << "\\\\";
      break;
    case '"':
      output << "\\\"";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      output << character;
      break;
    }
  }
  return output.str();
}

const char *jsonBool(const bool value) { return value ? "true" : "false"; }

std::string indexList(const std::vector<std::size_t> &indices) {
  if (indices.empty())
    return "none";
  std::ostringstream output;
  for (std::size_t position = 0; position < indices.size(); ++position) {
    if (position != 0)
      output << ',';
    output << indices[position];
  }
  return output.str();
}

std::string topicType(const rclcpp::Node &node, const std::string &name) {
  const auto topics = node.get_topic_names_and_types();
  const auto found = topics.find(name);
  return found == topics.end() || found->second.empty() ? std::string{}
                                                        : found->second.front();
}

fs::path normalizedPath(const fs::path &path) {
  return fs::absolute(path).lexically_normal();
}

std::string readable(std::string message) {
  const std::vector<std::pair<std::string, std::string>> replacements{
      {"dataset_path", "dataset folder"},
      {"image_topic", "camera topic"},
      {"arm_topic", "arm topic"},
      {"pattern_info", "pattern settings"},
      {"intrinsics_path", "intrinsics YAML"}};
  for (const auto &[from, to] : replacements) {
    for (std::size_t position = 0;
         (position = message.find(from, position)) != std::string::npos;) {
      message.replace(position, from.size(), to);
      position += to.size();
    }
  }
  return message;
}

geometry_msgs::msg::TransformStamped
transformMessage(const core::RigidTransform &value, const std::string &parent,
                 const std::string &child,
                 const builtin_interfaces::msg::Time &stamp) {
  geometry_msgs::msg::TransformStamped message;
  message.header.stamp = stamp;
  message.header.frame_id = parent;
  message.child_frame_id = child;
  message.transform.translation.x = value.translation.at<double>(0);
  message.transform.translation.y = value.translation.at<double>(1);
  message.transform.translation.z = value.translation.at<double>(2);
  const cv::Mat &rotation = value.rotation;
  tf2::Matrix3x3 matrix(rotation.at<double>(0, 0), rotation.at<double>(0, 1),
                        rotation.at<double>(0, 2), rotation.at<double>(1, 0),
                        rotation.at<double>(1, 1), rotation.at<double>(1, 2),
                        rotation.at<double>(2, 0), rotation.at<double>(2, 1),
                        rotation.at<double>(2, 2));
  tf2::Quaternion quaternion;
  matrix.getRotation(quaternion);
  quaternion.normalize();
  message.transform.rotation.x = quaternion.x();
  message.transform.rotation.y = quaternion.y();
  message.transform.rotation.z = quaternion.z();
  message.transform.rotation.w = quaternion.w();
  return message;
}

core::RigidTransform transformFromComponents(const double x, const double y,
                                             const double z, const double qx,
                                             const double qy, const double qz,
                                             const double qw) {
  for (const double value : {x, y, z, qx, qy, qz, qw}) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "initial camera transform values must be finite");
    }
  }
  tf2::Quaternion quaternion(qx, qy, qz, qw);
  if (quaternion.length2() <= 1.0e-12) {
    throw std::invalid_argument(
        "initial camera transform quaternion must be non-zero");
  }
  quaternion.normalize();
  const tf2::Matrix3x3 matrix(quaternion);
  const cv::Mat rotation =
      (cv::Mat_<double>(3, 3) << matrix[0][0], matrix[0][1], matrix[0][2],
       matrix[1][0], matrix[1][1], matrix[1][2], matrix[2][0], matrix[2][1],
       matrix[2][2]);
  const cv::Mat translation = (cv::Mat_<double>(3, 1) << x, y, z);
  return {rotation, translation};
}

} // namespace

class BackendNode::Impl final {
public:
  explicit Impl(BackendNode &node) : node_(node), source_(node) {
    declareParameters();
    parameterValidationCallback_ = node_.add_on_set_parameters_callback(
        [](const std::vector<rclcpp::Parameter> &parameters) {
          rcl_interfaces::msg::SetParametersResult result;
          result.successful = true;
          for (const auto &parameter : parameters) {
            if (parameter.get_name() != "target_pose_filter_time_constant_s") {
              continue;
            }
            if (parameter.get_type() !=
                rclcpp::ParameterType::PARAMETER_DOUBLE) {
              result.successful = false;
              result.reason =
                  "target_pose_filter_time_constant_s must be a double";
              break;
            }
            const double value = parameter.as_double();
            if (!std::isfinite(value) || value < 0.0) {
              result.successful = false;
              result.reason =
                  "target_pose_filter_time_constant_s must be finite and "
                  "non-negative (0 disables smoothing)";
              break;
            }
          }
          return result;
        });
    refreshConfig();
    lastValidationStatus_ =
        "waiting for " + std::to_string(config_.minimumSamples) + " samples";
    configureService_ = service(
        "configure", [this](const Response &response) { configure(response); });
    getStateService_ = service("get_state", [this](const Response &response) {
      refreshConfig();
      updateStatistics();
      response->success = true;
      response->message = stateJson();
    });
    prepareService_ =
        service("prepare_dataset",
                [this](const Response &response) { prepareDataset(response); });
    startService_ =
        service("start_collection", [this](const Response &response) {
          startCollection(response);
        });
    captureService_ =
        service("capture_sample",
                [this](const Response &response) { captureSample(response); });
    removeService_ = service("remove_sample", [this](const Response &response) {
      removeSample(response);
    });
    stopService_ = service("stop_collection", [this](const Response &response) {
      stopCollection(response);
    });
    calibrationService_ =
        service("run_calibration",
                [this](const Response &response) { runCalibration(response); });
    tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
    targetMarkerPublisher_ =
        node_.create_publisher<visualization_msgs::msg::MarkerArray>(
            ros::kTargetMarkersTopic, rclcpp::QoS(10));
    visualizationTimer_ = node_.create_wall_timer(
        std::chrono::milliseconds(100), [this]() { updateLiveTransforms(); });
    RCLCPP_INFO(node_.get_logger(), "Hand-eye backend v2 is ready");
  }

  ~Impl() { source_.stop(); }

private:
  struct Config {
    std::string armTopic, imageTopic, cameraInfoTopic;
    std::string baseFrame, effectorFrame, cameraFrame, targetFrame;
    std::string gripperVisualizationFrame{"hand_eye_gripper_pose"};
    std::string datasetPath, calibrationPath, patternInfo, intrinsicsPath;
    std::string method;
    int poseFormat{1};
    int minimumSamples{static_cast<int>(core::kMinimumCalibrationSamples)};
    bool eyeToHand{true}, requirePattern{true}, requireStamped{true};
    bool publishLiveTf{true};
    bool publishGripperTf{true};
    bool overwrite{false}, resume{true};
    bool translationStats{true}, rotationStats{true}, validationStats{true};
    double minimumRotationDeg{35.0}, minimumRotationAxisSeparationDeg{10.0};
    double maximumReprojectionPx{1.0};
    double maximumTimeDeltaMs{30.0}, settlingMs{500.0};
    double stationaryTranslationM{0.0005}, stationaryRotationDeg{0.25};
    double duplicateTranslationM{0.005}, duplicateRotationDeg{5.0};
    double targetPoseFilterTimeConstantS{0.25};
    double initialCameraX{1.0}, initialCameraY{0.0}, initialCameraZ{0.0};
    double initialCameraQx{0.0}, initialCameraQy{0.0}, initialCameraQz{0.0};
    double initialCameraQw{1.0};
  } config_;

  struct CachedIntrinsics {
    io::CameraIntrinsics value;
    std::string imageTopic;
    std::string cameraInfoTopic;
    std::string imageTopicType;
    std::string configuredCameraFrame;
    fs::path datasetPath;
    std::string acceptedImageFrame;
    std::string acceptedCameraInfoFrame;
    int width{0};
    int height{0};
  };

  template <typename Callback>
  rclcpp::Service<Trigger>::SharedPtr service(const std::string &name,
                                              Callback callback) {
    return node_.create_service<Trigger>(
        "~/" + name, [callback = std::move(callback)](
                         const std::shared_ptr<Trigger::Request> &,
                         Response response) { callback(response); });
  }

  void declareParameters() {
    node_.declare_parameter<std::string>("arm_topic", "");
    node_.declare_parameter<std::string>("image_topic", "");
    node_.declare_parameter<std::string>("camera_info_topic", "");
    node_.declare_parameter<std::string>("robot_base_frame", "");
    node_.declare_parameter<std::string>("robot_effector_frame", "");
    node_.declare_parameter<std::string>("camera_frame", "");
    node_.declare_parameter<std::string>("target_frame", "");
    node_.declare_parameter<std::string>("gripper_visualization_frame",
                                         "hand_eye_gripper_pose");
    node_.declare_parameter<std::string>("dataset_path", "dataset");
    node_.declare_parameter<std::string>("calibration_path", ".");
    node_.declare_parameter<std::string>("pattern_info", "");
    node_.declare_parameter<int>("poses_format", 1);
    node_.declare_parameter<bool>("eye_to_hand", true);
    node_.declare_parameter<std::string>("intrinsics_path", "");
    node_.declare_parameter<std::string>("calibration_method", "auto");
    node_.declare_parameter<bool>("require_pattern_detection", true);
    node_.declare_parameter<bool>("require_stamped_samples", true);
    node_.declare_parameter<bool>("publish_live_tf", true);
    node_.declare_parameter<bool>("publish_gripper_tf", true);
    node_.declare_parameter<double>("target_pose_filter_time_constant_s", 0.25);
    node_.declare_parameter<double>("initial_camera_x", 1.0);
    node_.declare_parameter<double>("initial_camera_y", 0.0);
    node_.declare_parameter<double>("initial_camera_z", 0.0);
    node_.declare_parameter<double>("initial_camera_qx", 0.0);
    node_.declare_parameter<double>("initial_camera_qy", 0.0);
    node_.declare_parameter<double>("initial_camera_qz", 0.0);
    node_.declare_parameter<double>("initial_camera_qw", 1.0);
    node_.declare_parameter<bool>("overwrite_dataset", false);
    node_.declare_parameter<bool>("resume_dataset", true);
    node_.declare_parameter<int>(
        "min_samples", static_cast<int>(core::kMinimumCalibrationSamples));
    node_.declare_parameter<double>("min_rotation_span_deg", 35.0);
    node_.declare_parameter<double>("min_rotation_axis_separation_deg", 10.0);
    node_.declare_parameter<double>("max_reprojection_px", 1.0);
    node_.declare_parameter<double>("max_time_delta_ms", 30.0);
    node_.declare_parameter<double>("settle_window_ms", 500.0);
    node_.declare_parameter<double>("stationary_translation_m", 0.0005);
    node_.declare_parameter<double>("stationary_rotation_deg", 0.25);
    node_.declare_parameter<double>("duplicate_translation_m", 0.005);
    node_.declare_parameter<double>("duplicate_rotation_deg", 5.0);
    node_.declare_parameter<bool>("stat_translation_spread", true);
    node_.declare_parameter<bool>("stat_rotation_spread", true);
    node_.declare_parameter<bool>("stat_validation_rms", false);
  }

  void refreshConfig() {
    Config next;
    next.armTopic = node_.get_parameter("arm_topic").as_string();
    next.imageTopic = node_.get_parameter("image_topic").as_string();
    next.cameraInfoTopic = node_.get_parameter("camera_info_topic").as_string();
    next.baseFrame = node_.get_parameter("robot_base_frame").as_string();
    next.effectorFrame =
        node_.get_parameter("robot_effector_frame").as_string();
    next.cameraFrame = node_.get_parameter("camera_frame").as_string();
    next.targetFrame = node_.get_parameter("target_frame").as_string();
    next.gripperVisualizationFrame =
        node_.get_parameter("gripper_visualization_frame").as_string();
    next.datasetPath = node_.get_parameter("dataset_path").as_string();
    next.calibrationPath = node_.get_parameter("calibration_path").as_string();
    next.patternInfo = node_.get_parameter("pattern_info").as_string();
    next.poseFormat =
        static_cast<int>(node_.get_parameter("poses_format").as_int());
    next.eyeToHand = node_.get_parameter("eye_to_hand").as_bool();
    next.intrinsicsPath = node_.get_parameter("intrinsics_path").as_string();
    next.method = node_.get_parameter("calibration_method").as_string();
    next.requirePattern =
        node_.get_parameter("require_pattern_detection").as_bool();
    next.requireStamped =
        node_.get_parameter("require_stamped_samples").as_bool();
    next.publishLiveTf = node_.get_parameter("publish_live_tf").as_bool();
    next.publishGripperTf = node_.get_parameter("publish_gripper_tf").as_bool();
    next.targetPoseFilterTimeConstantS =
        node_.get_parameter("target_pose_filter_time_constant_s").as_double();
    next.initialCameraX = node_.get_parameter("initial_camera_x").as_double();
    next.initialCameraY = node_.get_parameter("initial_camera_y").as_double();
    next.initialCameraZ = node_.get_parameter("initial_camera_z").as_double();
    next.initialCameraQx = node_.get_parameter("initial_camera_qx").as_double();
    next.initialCameraQy = node_.get_parameter("initial_camera_qy").as_double();
    next.initialCameraQz = node_.get_parameter("initial_camera_qz").as_double();
    next.initialCameraQw = node_.get_parameter("initial_camera_qw").as_double();
    next.overwrite = node_.get_parameter("overwrite_dataset").as_bool();
    next.resume = node_.get_parameter("resume_dataset").as_bool();
    next.minimumSamples =
        std::max<int>(static_cast<int>(core::kMinimumCalibrationSamples),
                      node_.get_parameter("min_samples").as_int());
    next.minimumRotationDeg =
        node_.get_parameter("min_rotation_span_deg").as_double();
    next.minimumRotationAxisSeparationDeg =
        node_.get_parameter("min_rotation_axis_separation_deg").as_double();
    next.maximumReprojectionPx =
        node_.get_parameter("max_reprojection_px").as_double();
    next.maximumTimeDeltaMs =
        node_.get_parameter("max_time_delta_ms").as_double();
    next.settlingMs = node_.get_parameter("settle_window_ms").as_double();
    next.stationaryTranslationM =
        node_.get_parameter("stationary_translation_m").as_double();
    next.stationaryRotationDeg =
        node_.get_parameter("stationary_rotation_deg").as_double();
    next.duplicateTranslationM =
        node_.get_parameter("duplicate_translation_m").as_double();
    next.duplicateRotationDeg =
        node_.get_parameter("duplicate_rotation_deg").as_double();
    next.translationStats =
        node_.get_parameter("stat_translation_spread").as_bool();
    next.rotationStats = node_.get_parameter("stat_rotation_spread").as_bool();
    next.validationStats = node_.get_parameter("stat_validation_rms").as_bool();

    const bool sourceConfigurationChanged =
        next.armTopic != config_.armTopic ||
        next.imageTopic != config_.imageTopic ||
        next.cameraInfoTopic != config_.cameraInfoTopic ||
        next.baseFrame != config_.baseFrame ||
        next.effectorFrame != config_.effectorFrame;
    const bool sourceChanged = collecting_ && sourceConfigurationChanged;
    const bool intrinsicsSourceChanged =
        next.imageTopic != config_.imageTopic ||
        next.cameraInfoTopic != config_.cameraInfoTopic ||
        next.cameraFrame != config_.cameraFrame ||
        next.intrinsicsPath != config_.intrinsicsPath;
    const bool datasetChanged =
        datasetPrepared_ && (next.datasetPath != config_.datasetPath ||
                             next.poseFormat != config_.poseFormat);
    const bool calibrationChanged =
        next.datasetPath != config_.datasetPath ||
        next.poseFormat != config_.poseFormat ||
        next.patternInfo != config_.patternInfo ||
        next.eyeToHand != config_.eyeToHand ||
        next.intrinsicsPath != config_.intrinsicsPath ||
        next.method != config_.method || next.baseFrame != config_.baseFrame ||
        next.effectorFrame != config_.effectorFrame ||
        next.cameraFrame != config_.cameraFrame ||
        next.minimumSamples != config_.minimumSamples ||
        next.minimumRotationDeg != config_.minimumRotationDeg ||
        next.minimumRotationAxisSeparationDeg !=
            config_.minimumRotationAxisSeparationDeg ||
        next.maximumReprojectionPx != config_.maximumReprojectionPx;
    const bool capturePolicyChanged =
        next.requirePattern != config_.requirePattern ||
        next.requireStamped != config_.requireStamped ||
        next.maximumTimeDeltaMs != config_.maximumTimeDeltaMs ||
        next.settlingMs != config_.settlingMs ||
        next.stationaryTranslationM != config_.stationaryTranslationM ||
        next.stationaryRotationDeg != config_.stationaryRotationDeg ||
        next.duplicateTranslationM != config_.duplicateTranslationM ||
        next.duplicateRotationDeg != config_.duplicateRotationDeg;
    const bool liveDisplayChanged =
        next.targetFrame != config_.targetFrame ||
        next.publishLiveTf != config_.publishLiveTf ||
        next.publishGripperTf != config_.publishGripperTf ||
        next.gripperVisualizationFrame != config_.gripperVisualizationFrame;
    const bool visualizationTopologyChanged =
        next.targetFrame != config_.targetFrame ||
        next.gripperVisualizationFrame != config_.gripperVisualizationFrame;
    const bool visualizationPublicationEnabled =
        (!config_.publishLiveTf && next.publishLiveTf) ||
        (!config_.publishGripperTf && next.publishGripperTf);
    const bool targetPoseFilterChanged = next.targetPoseFilterTimeConstantS !=
                                         config_.targetPoseFilterTimeConstantS;
    const bool configurationChanged =
        sourceConfigurationChanged || intrinsicsSourceChanged ||
        calibrationChanged || capturePolicyChanged ||
        visualizationTopologyChanged || visualizationPublicationEnabled;
    config_ = std::move(next);
    if (targetPoseFilterChanged) {
      if (std::isfinite(config_.targetPoseFilterTimeConstantS) &&
          config_.targetPoseFilterTimeConstantS >= 0.0) {
        targetPoseFilter_.setTimeConstantSeconds(
            config_.targetPoseFilterTimeConstantS);
      } else {
        targetPoseFilter_.setTimeConstantSeconds(0.0);
      }
      lastFilteredTargetStamp_ = 0;
    }
    if (configurationChanged) {
      configured_ = false;
      targetVisible_ = false;
      clearTargetMarkersRequested_ = true;
      lastVisualizedImageStamp_ = 0;
      resetTargetPoseFilter();
    }
    if (sourceChanged) {
      source_.stop();
      collecting_ = false;
      targetVisible_ = false;
      clearTargetMarkersRequested_ = true;
    }
    if (intrinsicsSourceChanged) {
      cachedIntrinsics_.reset();
      configuredIntrinsics_.reset();
    }
    if (datasetChanged) {
      datasetPrepared_ = false;
    }
    if (calibrationChanged || sourceConfigurationChanged ||
        intrinsicsSourceChanged) {
      lastResult_.reset();
      lastValidationStatus_ =
          "configuration changed; calibration metrics are stale";
    }
    if (liveDisplayChanged || !config_.publishLiveTf) {
      targetVisible_ = false;
      clearTargetMarkersRequested_ = true;
      if (!config_.publishLiveTf) {
        resetTargetPoseFilter();
      }
    }
  }

  io::PoseFormat poseFormat() const {
    return io::parsePoseFormat(config_.poseFormat);
  }
  io::DatasetRepository repository() const {
    return io::DatasetRepository(config_.datasetPath);
  }

  fs::path calibrationFile() const {
    fs::path path(config_.calibrationPath);
    if (path.empty() || path == ".")
      return "calibration.yaml";
    if (path.extension() == ".yaml" || path.extension() == ".yml")
      return path;
    return path / "calibration.yaml";
  }

  core::PatternConfig pattern() const {
    if (config_.patternInfo.empty()) {
      throw std::invalid_argument("pattern_info parameter is required");
    }
    return core::parsePattern(config_.patternInfo);
  }

  std::string resolvedTargetFrame(const core::PatternConfig &target) const {
    if (!config_.targetFrame.empty()) {
      return config_.targetFrame;
    }
    if (target.type == core::PatternType::ARUCO) {
      return "aruco_" + std::to_string(target.markerId);
    }
    return target.type == core::PatternType::CHARUCO ? "charuco_board"
                                                     : "calibration_target";
  }

  std::string stateTargetFrame() const {
    if (!config_.targetFrame.empty() || config_.patternInfo.empty()) {
      return config_.targetFrame;
    }
    try {
      return resolvedTargetFrame(pattern());
    } catch (const std::exception &) {
      return "calibration_target";
    }
  }

  std::string cameraParentFrame() const {
    return config_.eyeToHand ? config_.baseFrame : config_.effectorFrame;
  }

  core::RigidTransform initialCameraTransform() const {
    return transformFromComponents(
        config_.initialCameraX, config_.initialCameraY, config_.initialCameraZ,
        config_.initialCameraQx, config_.initialCameraQy,
        config_.initialCameraQz, config_.initialCameraQw);
  }

  bool cameraFramesConfigured() const {
    if (config_.cameraFrame.empty() || cameraParentFrame().empty() ||
        config_.cameraFrame == config_.baseFrame ||
        config_.cameraFrame == config_.effectorFrame ||
        (!config_.baseFrame.empty() &&
         config_.baseFrame == config_.effectorFrame)) {
      return false;
    }
    return true;
  }

  bool initialCameraTfConfigured() const {
    if (!cameraFramesConfigured()) {
      return false;
    }
    try {
      (void)initialCameraTransform();
      return true;
    } catch (const std::exception &) {
      return false;
    }
  }

  bool targetPoseFilterSettingValid() const {
    return std::isfinite(config_.targetPoseFilterTimeConstantS) &&
           config_.targetPoseFilterTimeConstantS >= 0.0;
  }

  bool targetPoseFilterEnabled() const {
    return targetPoseFilterSettingValid() &&
           config_.targetPoseFilterTimeConstantS > 0.0;
  }

  void resetTargetPoseFilter() {
    targetPoseFilter_.reset();
    lastFilteredTargetStamp_ = 0;
  }

  core::RigidTransform
  filteredTargetPose(const core::RigidTransform &measurement,
                     const std::int64_t timestampNanoseconds) {
    constexpr std::int64_t kMaximumFilterGapNanoseconds = 1'000'000'000;
    if (lastFilteredTargetStamp_ != 0) {
      const std::int64_t delta =
          timestampNanoseconds - lastFilteredTargetStamp_;
      if (delta <= 0 || delta > kMaximumFilterGapNanoseconds) {
        resetTargetPoseFilter();
      }
    }
    lastFilteredTargetStamp_ = timestampNanoseconds;
    return targetPoseFilter_.update(measurement, timestampNanoseconds);
  }

  void validateVisualizationFrames(
      const std::optional<core::PatternConfig> &configuredPattern) const {
    if (!targetPoseFilterSettingValid()) {
      throw std::invalid_argument(
          "target_pose_filter_time_constant_s must be finite and "
          "non-negative (0 disables smoothing)");
    }
    if (!config_.publishLiveTf) {
      return;
    }
    if (!initialCameraTfConfigured()) {
      throw std::invalid_argument(
          "live camera TF requires distinct non-empty parent/camera frames "
          "and a finite non-zero initial camera quaternion");
    }
    if (config_.baseFrame == config_.effectorFrame ||
        config_.cameraFrame == config_.baseFrame ||
        config_.cameraFrame == config_.effectorFrame) {
      throw std::invalid_argument(
          "robot base, robot effector and camera frames must be distinct to "
          "avoid TF cycles or duplicate child publishers");
    }
    const std::string target = configuredPattern
                                   ? resolvedTargetFrame(*configuredPattern)
                                   : config_.targetFrame;
    if (!target.empty() &&
        (target == config_.baseFrame || target == config_.effectorFrame ||
         target == config_.cameraFrame)) {
      throw std::invalid_argument(
          "target_frame must be distinct from the robot base, effector and "
          "camera frames");
    }
    if (!config_.publishGripperTf) {
      return;
    }
    if (config_.baseFrame.empty() || config_.effectorFrame.empty() ||
        config_.gripperVisualizationFrame.empty()) {
      throw std::invalid_argument(
          "gripper visualization requires robot base, effector and alias "
          "frames");
    }
    for (const std::string *reserved :
         {&config_.baseFrame, &config_.effectorFrame, &config_.cameraFrame,
          &target}) {
      if (!reserved->empty() &&
          config_.gripperVisualizationFrame == *reserved) {
        throw std::invalid_argument(
            "gripper_visualization_frame must be a unique TF alias; do not "
            "reuse the robot effector, base, camera or target frame");
      }
    }
  }

  void clearTargetMarkers() {
    clearTargetMarkersRequested_ = false;
    if (!targetMarkersPublished_ || !targetMarkerPublisher_) {
      targetMarkersPublished_ = false;
      return;
    }
    const std::string frame = lastPublishedTargetFrame_.empty()
                                  ? stateTargetFrame()
                                  : lastPublishedTargetFrame_;
    if (!frame.empty()) {
      targetMarkerPublisher_->publish(ros::deleteTargetMarkers(
          frame, static_cast<builtin_interfaces::msg::Time>(node_.now())));
    }
    targetMarkersPublished_ = false;
    lastPublishedTargetFrame_.clear();
  }

  core::CapturePolicy capturePolicy(const bool arucoTarget) const {
    core::CapturePolicyOptions options;
    options.maximumTimeDeltaMilliseconds = config_.maximumTimeDeltaMs;
    options.requireStampedSamples = config_.requireStamped;
    options.requireCameraInfo = arucoTarget && config_.intrinsicsPath.empty();
    options.robotBaseFrame = config_.baseFrame;
    options.robotEffectorFrame = config_.effectorFrame;
    options.cameraFrame = config_.cameraFrame;
    options.settlingDurationMilliseconds = config_.settlingMs;
    options.settlingTranslationMeters = config_.stationaryTranslationM;
    options.settlingRotationDegrees = config_.stationaryRotationDeg;
    options.duplicateTranslationMeters = config_.duplicateTranslationM;
    options.duplicateRotationDegrees = config_.duplicateRotationDeg;
    return core::CapturePolicy(options);
  }

  void configure(const Response &response) {
    configured_ = false;
    configuredIntrinsics_.reset();
    try {
      refreshConfig();
      poseFormat();
      std::optional<core::PatternConfig> configuredPattern;
      if (!config_.patternInfo.empty()) {
        configuredPattern = pattern();
      }
      if (!config_.intrinsicsPath.empty()) {
        configuredIntrinsics_ =
            io::CalibrationIO::readIntrinsics(config_.intrinsicsPath);
      }
      validateVisualizationFrames(configuredPattern);
      (void)core::parseCalibrationMethod(config_.method);
      (void)capturePolicy(configuredPattern &&
                          configuredPattern->type == core::PatternType::ARUCO);
      configured_ = true;
      lastError_.clear();
      armTopicType_ = topicType(node_, config_.armTopic);
      imageTopicType_ = topicType(node_, config_.imageTopic);
      response->success = true;
      response->message = "Configured backend";
      if (armTopicType_.empty() || imageTopicType_.empty()) {
        response->message += "; one or more topics are not visible yet";
      }
    } catch (const std::exception &error) {
      fail(response, error.what());
    }
  }

  void prepareDataset(const Response &response) {
    try {
      refreshConfig();
      auto result =
          repository().prepare(config_.resume, config_.overwrite, poseFormat());
      datasetPrepared_ = true;
      preparedPath_ = normalizedPath(config_.datasetPath);
      preparedFormat_ = config_.poseFormat;
      nextSampleIndex_ = result.nextSampleIndex;
      statistics_ = result.statistics;
      lastResult_.reset();
      lastValidationStatus_ = "dataset prepared; waiting for calibration";
      lastError_.clear();
      response->success = true;
      response->message =
          std::string(result.created ? "Dataset created" : "Dataset resumed") +
          "; next sample index is " + std::to_string(nextSampleIndex_);
    } catch (const std::exception &error) {
      fail(response, error.what());
    }
  }

  void requirePrepared() const {
    if (!datasetPrepared_ || preparedFormat_ != config_.poseFormat ||
        preparedPath_ != normalizedPath(config_.datasetPath)) {
      throw std::runtime_error(
          "Dataset is not prepared. Choose Continue Dataset or Start Over.");
    }
  }

  void startCollection(const Response &response) {
    try {
      refreshConfig();
      if (!configured_) {
        throw std::runtime_error(
            "Backend configuration is not validated. Run configure first.");
      }
      requirePrepared();
      if (collecting_) {
        response->success = true;
        response->message = "Collection is already running";
        return;
      }
      if (config_.armTopic.empty() || config_.imageTopic.empty()) {
        throw std::invalid_argument("arm_topic and image_topic are required");
      }
      source_.start({config_.armTopic, config_.imageTopic,
                     config_.cameraInfoTopic, config_.baseFrame,
                     config_.effectorFrame, 1000});
      lastVisualizedImageStamp_ = 0;
      lastVisualizedGripperStamp_ = 0;
      lastGripperPoseSeen_ = {};
      targetVisible_ = false;
      clearTargetMarkersRequested_ = true;
      resetTargetPoseFilter();
      if (cachedIntrinsics_ &&
          cachedIntrinsics_->imageTopicType != source_.imageTopicType()) {
        cachedIntrinsics_.reset();
      }
      collecting_ = true;
      response->success = true;
      response->message = "Collection started";
    } catch (const std::exception &error) {
      fail(response, error.what());
    }
  }

  void rememberIntrinsics(const ros::CaptureSnapshot &snapshot) {
    if (snapshot.cameraMatrix.empty() || !snapshot.imageMetadata ||
        !snapshot.cameraInfo)
      return;

    const auto &image = *snapshot.imageMetadata;
    const auto &cameraInfo = *snapshot.cameraInfo;
    if (!cameraInfo.calibrated || image.width <= 0 || image.height <= 0 ||
        cameraInfo.width != image.width || cameraInfo.height != image.height ||
        (!image.frameId.empty() && !cameraInfo.frameId.empty() &&
         image.frameId != cameraInfo.frameId)) {
      return;
    }

    cachedIntrinsics_ = CachedIntrinsics{
        {snapshot.cameraMatrix.clone(), snapshot.distortionCoefficients.clone(),
         image.width, image.height},
        config_.imageTopic,
        config_.cameraInfoTopic,
        source_.imageTopicType(),
        config_.cameraFrame,
        normalizedPath(config_.datasetPath),
        image.frameId,
        cameraInfo.frameId,
        image.width,
        image.height};
  }

  bool cachedIntrinsicsMatchConfig() const {
    if (!cachedIntrinsics_ ||
        cachedIntrinsics_->imageTopic != config_.imageTopic ||
        cachedIntrinsics_->cameraInfoTopic != config_.cameraInfoTopic ||
        cachedIntrinsics_->configuredCameraFrame != config_.cameraFrame ||
        cachedIntrinsics_->datasetPath != normalizedPath(config_.datasetPath) ||
        cachedIntrinsics_->width <= 0 || cachedIntrinsics_->height <= 0) {
      return false;
    }
    return config_.cameraFrame.empty() ||
           (cachedIntrinsics_->acceptedImageFrame == config_.cameraFrame &&
            cachedIntrinsics_->acceptedCameraInfoFrame == config_.cameraFrame);
  }

  void captureSample(const Response &response) {
    try {
      refreshConfig();
      if (!configured_) {
        throw std::runtime_error(
            "Backend configuration changed. Run configure again.");
      }
      requirePrepared();
      if (!collecting_)
        throw std::runtime_error("Collection is not running");
      std::optional<core::PatternConfig> target;
      if (!config_.patternInfo.empty())
        target = pattern();
      if (config_.requirePattern && !target) {
        throw std::invalid_argument("pattern_info parameter is required when "
                                    "pattern detection is enabled");
      }
      const auto snapshot = source_.snapshot();
      core::CaptureRequest request;
      request.image = snapshot.imageMetadata;
      request.robotHistory = snapshot.robotHistory;
      request.cameraInfo = snapshot.cameraInfo;
      for (const auto &row : repository().acceptedPoseRows(poseFormat())) {
        request.acceptedRobotPoses.push_back(row.transform);
      }
      const bool arucoTarget =
          target && target->type == core::PatternType::ARUCO;
      const auto decision = capturePolicy(arucoTarget).evaluate(request);
      if (!decision.accepted || !decision.matchedRobotPose) {
        throw std::runtime_error(core::toString(decision.reason) + ": " +
                                 decision.message);
      }
      if (config_.requirePattern &&
          !core::TargetDetector(*target).detect(snapshot.image)) {
        throw std::runtime_error(
            "calibration target was not found; sample was not saved");
      }
      const auto values =
          io::encodePose(decision.matchedRobotPose->pose, poseFormat());
      const std::size_t index = repository().appendSample(
          snapshot.image, values,
          {snapshot.imageMetadata->timestampNanoseconds,
           decision.matchedRobotPose->timestampNanoseconds},
          poseFormat());
      rememberIntrinsics(snapshot);
      nextSampleIndex_ = index + 1;
      updateStatistics();
      updateLiveCalibration();
      lastError_.clear();
      response->success = true;
      response->message =
          "Sample " + std::to_string(index) + " saved; timestamp delta " +
          std::to_string(decision.timestampDeltaMilliseconds) + " ms";
    } catch (const std::exception &error) {
      fail(response, error.what());
    }
  }

  void removeSample(const Response &response) {
    try {
      refreshConfig();
      requirePrepared();
      repository().removeLastSample(poseFormat());
      updateStatistics();
      lastResult_.reset();
      lastValidationStatus_ = "sample removed; calibration metrics are stale";
      response->success = true;
      response->message = "Last sample removed";
    } catch (const std::exception &error) {
      fail(response, error.what());
    }
  }

  void stopCollection(const Response &response) {
    source_.stop();
    collecting_ = false;
    targetVisible_ = false;
    clearTargetMarkersRequested_ = true;
    resetTargetPoseFilter();
    clearTargetMarkers();
    updateStatistics();
    response->success = true;
    response->message = "Collection stopped";
  }

  io::DatasetCalibrationResult calculate() const {
    io::DatasetCalibrationOptions options;
    options.datasetPath = config_.datasetPath;
    options.poseFormat = poseFormat();
    options.pattern = pattern();
    if (!config_.intrinsicsPath.empty()) {
      options.intrinsics =
          io::CalibrationIO::readIntrinsics(config_.intrinsicsPath);
    } else if (cachedIntrinsicsMatchConfig()) {
      options.intrinsics = cachedIntrinsics_->value;
    }
    options.targetOptions.maximumReprojectionErrorPixels =
        config_.maximumReprojectionPx;
    options.calibrationOptions.mode = config_.eyeToHand
                                          ? core::CalibrationMode::EYE_TO_HAND
                                          : core::CalibrationMode::EYE_IN_HAND;
    options.calibrationOptions.method =
        core::parseCalibrationMethod(config_.method);
    options.calibrationOptions.minimumSamples =
        static_cast<std::size_t>(config_.minimumSamples);
    options.calibrationOptions.minimumRotationSpanDegrees =
        config_.minimumRotationDeg;
    options.calibrationOptions.minimumRotationAxisSeparationDegrees =
        config_.minimumRotationAxisSeparationDeg;
    options.parentFrame =
        config_.eyeToHand ? config_.baseFrame : config_.effectorFrame;
    options.childFrame = config_.cameraFrame;
    return io::calibrateDataset(options);
  }

  void updateLiveCalibration() {
    if (!statistics_ || statistics_->validPoseRowCount <
                            static_cast<std::size_t>(config_.minimumSamples)) {
      lastValidationStatus_ =
          "waiting for " + std::to_string(config_.minimumSamples) + " samples";
      return;
    }
    try {
      lastResult_ = calculate();
      lastValidationStatus_ = "updated";
    } catch (const std::exception &error) {
      lastValidationStatus_ = readable(error.what());
    }
  }

  void updateLiveTransforms() {
    refreshConfig();
    cameraTfActive_ = false;
    cameraTfSource_ = "none";
    cameraTfParentFrame_.clear();
    gripperTfActive_ = false;

    if (clearTargetMarkersRequested_) {
      clearTargetMarkers();
    }
    if (!config_.publishLiveTf) {
      return;
    }

    try {
      if (cameraFramesConfigured() &&
          (lastResult_.has_value() || initialCameraTfConfigured())) {
        const std::string parent = cameraParentFrame();
        const core::RigidTransform cameraTransform =
            lastResult_ ? lastResult_->calibration.transform
                        : initialCameraTransform();
        tfBroadcaster_->sendTransform(transformMessage(
            cameraTransform, parent, config_.cameraFrame,
            static_cast<builtin_interfaces::msg::Time>(node_.now())));
        cameraTfActive_ = true;
        cameraTfSource_ = lastResult_ ? "online" : "initial";
        cameraTfParentFrame_ = parent;
      }
    } catch (const std::exception &error) {
      RCLCPP_DEBUG_THROTTLE(node_.get_logger(), *node_.get_clock(), 5000,
                            "Live camera TF update skipped: %s", error.what());
    }

    if (!configured_ || !collecting_ || !source_.running() ||
        config_.cameraFrame.empty()) {
      return;
    }

    try {
      const ros::CaptureSnapshot snapshot = source_.snapshot();
      const auto now = std::chrono::steady_clock::now();
      if (config_.publishGripperTf && snapshot.latestRobotPose &&
          !config_.gripperVisualizationFrame.empty()) {
        const core::TimedRobotPose &gripper = *snapshot.latestRobotPose;
        if (gripper.timestampNanoseconds != lastVisualizedGripperStamp_) {
          lastVisualizedGripperStamp_ = gripper.timestampNanoseconds;
          lastGripperPoseSeen_ = now;
        }
        const bool fresh =
            lastVisualizedGripperStamp_ != 0 &&
            now - lastGripperPoseSeen_ <= std::chrono::seconds(1);
        if (fresh) {
          if (gripper.parentFrame == config_.baseFrame &&
              gripper.childFrame == config_.effectorFrame) {
            const rclcpp::Time gripperStamp(
                gripper.timestampNanoseconds,
                node_.get_clock()->get_clock_type());
            tfBroadcaster_->sendTransform(transformMessage(
                gripper.pose, config_.baseFrame,
                config_.gripperVisualizationFrame,
                static_cast<builtin_interfaces::msg::Time>(gripperStamp)));
            gripperTfActive_ = true;
          } else {
            RCLCPP_DEBUG_THROTTLE(
                node_.get_logger(), *node_.get_clock(), 5000,
                "Gripper visualization TF skipped: message frames are '%s' "
                "-> '%s', expected '%s' -> '%s'",
                gripper.parentFrame.c_str(), gripper.childFrame.c_str(),
                config_.baseFrame.c_str(), config_.effectorFrame.c_str());
          }
        }
      }

      if (snapshot.image.empty() || !snapshot.imageMetadata) {
        return;
      }
      if (snapshot.imageMetadata->timestampNanoseconds ==
          lastVisualizedImageStamp_) {
        if (now - lastTargetSeen_ > std::chrono::seconds(1)) {
          const bool wasVisible = targetVisible_;
          targetVisible_ = false;
          resetTargetPoseFilter();
          if (wasVisible) {
            clearTargetMarkers();
          }
        }
        return;
      }
      lastVisualizedImageStamp_ = snapshot.imageMetadata->timestampNanoseconds;
      targetVisible_ = false;
      const auto hideTarget = [this, now]() {
        targetVisible_ = false;
        clearTargetMarkers();
        if (now - lastTargetSeen_ > std::chrono::seconds(1)) {
          resetTargetPoseFilter();
        }
      };
      const core::ImageSampleMetadata &image = *snapshot.imageMetadata;
      if ((!config_.cameraFrame.empty() &&
           config_.cameraFrame != image.frameId) ||
          image.width != snapshot.image.cols ||
          image.height != snapshot.image.rows) {
        hideTarget();
        return;
      }
      const rclcpp::Time stamp(snapshot.imageMetadata->timestampNanoseconds,
                               node_.get_clock()->get_clock_type());

      const core::PatternConfig target = pattern();
      const auto observation =
          core::TargetDetector(target).detect(snapshot.image);
      if (!observation) {
        hideTarget();
        return;
      }

      cv::Mat cameraMatrix;
      cv::Mat distortion;
      if (configuredIntrinsics_) {
        if (configuredIntrinsics_->width > 0 &&
            (configuredIntrinsics_->width != image.width ||
             configuredIntrinsics_->height != image.height)) {
          hideTarget();
          return;
        }
        cameraMatrix = configuredIntrinsics_->cameraMatrix;
        distortion = configuredIntrinsics_->distortionCoefficients;
      } else {
        if (!snapshot.cameraInfo || !snapshot.cameraInfo->calibrated ||
            snapshot.cameraInfo->width != image.width ||
            snapshot.cameraInfo->height != image.height ||
            (!config_.cameraFrame.empty() &&
             snapshot.cameraInfo->frameId != config_.cameraFrame) ||
            (!image.frameId.empty() && !snapshot.cameraInfo->frameId.empty() &&
             image.frameId != snapshot.cameraInfo->frameId)) {
          hideTarget();
          return;
        }
        cameraMatrix = snapshot.cameraMatrix;
        distortion = snapshot.distortionCoefficients;
      }
      if (cameraMatrix.empty()) {
        hideTarget();
        return;
      }

      core::TargetPoseOptions options;
      options.maximumReprojectionErrorPixels = config_.maximumReprojectionPx;
      const core::TargetEstimationResult estimate =
          core::TargetPoseEstimator::estimate(
              target, {*observation}, snapshot.image.size(), cameraMatrix,
              distortion, options);
      if (estimate.poses.empty() || !estimate.poses.front()) {
        hideTarget();
        return;
      }

      const std::string targetFrame = resolvedTargetFrame(target);
      const core::RigidTransform targetPose =
          filteredTargetPose(estimate.poses.front()->targetToCamera,
                             snapshot.imageMetadata->timestampNanoseconds);
      tfBroadcaster_->sendTransform(
          transformMessage(targetPose, config_.cameraFrame, targetFrame,
                           static_cast<builtin_interfaces::msg::Time>(stamp)));
      if (target.type == core::PatternType::ARUCO) {
        targetMarkerPublisher_->publish(ros::makeArucoMarkerArray(
            target, targetFrame,
            static_cast<builtin_interfaces::msg::Time>(stamp)));
        targetMarkersPublished_ = true;
        lastPublishedTargetFrame_ = targetFrame;
      } else {
        clearTargetMarkers();
      }
      lastTargetReprojectionPx_ =
          estimate.poses.front()->reprojectionErrorPixels;
      targetVisible_ = true;
      lastTargetSeen_ = now;

    } catch (const std::exception &error) {
      RCLCPP_DEBUG_THROTTLE(node_.get_logger(), *node_.get_clock(), 5000,
                            "Live TF update skipped: %s", error.what());
    }
  }

  void runCalibration(const Response &response) {
    try {
      refreshConfig();
      lastResult_ = calculate();
      io::CalibrationIO::write(calibrationFile(), lastResult_->document());
      lastValidationStatus_ = "updated";
      lastError_.clear();
      response->success = true;
      response->message =
          "Calibration saved to " + calibrationFile().string() +
          "; method=" + core::toString(lastResult_->calibration.method) +
          "; translation RMS=" +
          std::to_string(
              lastResult_->calibration.consistency.translationRmsMeters) +
          " m; rotation RMS=" +
          std::to_string(
              lastResult_->calibration.consistency.rotationRmsDegrees) +
          " deg";
      response->message +=
          "; rejected=" +
          indexList(lastResult_->calibration.rejectedSampleIndices);
    } catch (const std::exception &error) {
      fail(response, error.what());
    }
  }

  void updateStatistics() {
    statistics_.reset();
    try {
      const auto inspection = repository().inspect(poseFormat());
      statistics_ = inspection.statistics;
      nextSampleIndex_ = inspection.nextSampleIndex;
    } catch (const std::exception &) {
    }
  }

  double progressPercent() const {
    if (!statistics_)
      return 0.0;
    std::vector<double> criteria{
        std::min(1.0, static_cast<double>(statistics_->validPoseRowCount) /
                          static_cast<double>(config_.minimumSamples))};
    if (config_.translationStats)
      criteria.push_back(statistics_->translationCoverageScore);
    if (config_.rotationStats)
      criteria.push_back(statistics_->rotationCoverageScore);
    return 100.0 * std::accumulate(criteria.begin(), criteria.end(), 0.0) /
           static_cast<double>(criteria.size());
  }

  std::string stateJson() const {
    const std::string armType =
        source_.running() ? source_.armTopicType() : armTopicType_;
    const std::string imageType =
        source_.running() ? source_.imageTopicType() : imageTopicType_;
    std::ostringstream out;
    out << "{" << "\"configured\":" << jsonBool(configured_)
        << ",\"collecting\":" << jsonBool(collecting_)
        << ",\"dataset_prepared\":" << jsonBool(datasetPrepared_)
        << ",\"arm_topic\":\"" << escapeJson(config_.armTopic) << "\""
        << ",\"image_topic\":\"" << escapeJson(config_.imageTopic) << "\""
        << ",\"camera_info_topic\":\"" << escapeJson(config_.cameraInfoTopic)
        << "\"" << ",\"dataset_path\":\"" << escapeJson(config_.datasetPath)
        << "\"" << ",\"calibration_path\":\""
        << escapeJson(calibrationFile().string()) << "\""
        << ",\"pattern_info\":\"" << escapeJson(config_.patternInfo) << "\""
        << ",\"intrinsics_path\":\"" << escapeJson(config_.intrinsicsPath)
        << "\"" << ",\"poses_format\":" << config_.poseFormat
        << ",\"eye_to_hand\":" << jsonBool(config_.eyeToHand)
        << ",\"robot_base_frame\":\"" << escapeJson(config_.baseFrame) << "\""
        << ",\"robot_effector_frame\":\"" << escapeJson(config_.effectorFrame)
        << "\"" << ",\"camera_frame\":\"" << escapeJson(config_.cameraFrame)
        << "\"" << ",\"target_frame\":\"" << escapeJson(stateTargetFrame())
        << "\"" << ",\"target_markers_topic\":\"" << ros::kTargetMarkersTopic
        << "\"" << ",\"target_tf_visible\":" << jsonBool(targetVisible_)
        << ",\"target_reprojection_px\":"
        << (targetVisible_ ? std::to_string(lastTargetReprojectionPx_) : "null")
        << ",\"target_pose_filter_time_constant_s\":"
        << (targetPoseFilterSettingValid()
                ? std::to_string(config_.targetPoseFilterTimeConstantS)
                : "null")
        << ",\"target_pose_filter_enabled\":"
        << jsonBool(targetPoseFilterEnabled())
        << ",\"target_pose_filter_initialized\":"
        << jsonBool(targetPoseFilterEnabled() &&
                    targetPoseFilter_.initialized())
        << ",\"calibration_tf_available\":" << jsonBool(lastResult_.has_value())
        << ",\"camera_tf_source\":\"" << escapeJson(cameraTfSource_) << "\""
        << ",\"camera_tf_active\":" << jsonBool(cameraTfActive_)
        << ",\"camera_tf_parent_frame\":\"" << escapeJson(cameraTfParentFrame_)
        << "\"" << ",\"initial_camera_tf_configured\":"
        << jsonBool(initialCameraTfConfigured())
        << ",\"gripper_tf_active\":" << jsonBool(gripperTfActive_)
        << ",\"gripper_visualization_frame\":\""
        << escapeJson(config_.gripperVisualizationFrame) << "\""
        << ",\"min_samples\":" << config_.minimumSamples
        << ",\"progress_percent\":" << progressPercent()
        << ",\"next_sample_index\":" << nextSampleIndex_ << ",\"image_count\":"
        << (statistics_ ? std::to_string(statistics_->imageCount) : "null")
        << ",\"pose_count\":"
        << (statistics_ ? std::to_string(statistics_->poseRowCount) : "null")
        << ",\"arm_topic_type\":\"" << escapeJson(armType) << "\""
        << ",\"image_topic_type\":\"" << escapeJson(imageType) << "\""
        << ",\"statistics_enabled\":{\"translation_spread\":"
        << jsonBool(config_.translationStats)
        << ",\"rotation_spread\":" << jsonBool(config_.rotationStats)
        << ",\"validation_rms\":" << jsonBool(config_.validationStats) << "}"
        << ",\"statistics\":";
    if (statistics_) {
      out << "{\"samples\":" << statistics_->imageCount
          << ",\"valid_pose_rows\":" << statistics_->validPoseRowCount
          << ",\"translation_spread_x\":" << statistics_->translation[0].span()
          << ",\"translation_spread_y\":" << statistics_->translation[1].span()
          << ",\"translation_spread_z\":" << statistics_->translation[2].span()
          << ",\"rotation_spread_rad\":"
          << statistics_->rotationSpreadDeg * CV_PI / 180.0
          << ",\"rotation_spread_deg\":" << statistics_->rotationSpreadDeg
          << ",\"translation_coverage_score\":"
          << statistics_->translationCoverageScore
          << ",\"rotation_coverage_score\":"
          << statistics_->rotationCoverageScore << "}";
    } else
      out << "null";
    out << ",\"last_validation_rms\":"
        << (lastResult_
                ? std::to_string(
                      lastResult_->calibration.consistency.translationRmsMeters)
                : "null")
        << ",\"last_validation_rms_status\":\""
        << escapeJson(lastValidationStatus_) << "\"" << ",\"last_error\":\""
        << escapeJson(lastError_) << "\"" << ",\"calibration_method\":\""
        << (lastResult_ ? core::toString(lastResult_->calibration.method) : "")
        << "\"" << ",\"translation_rms_m\":"
        << (lastResult_
                ? std::to_string(
                      lastResult_->calibration.consistency.translationRmsMeters)
                : "null")
        << ",\"rotation_rms_deg\":"
        << (lastResult_
                ? std::to_string(
                      lastResult_->calibration.consistency.rotationRmsDegrees)
                : "null")
        << ",\"rejected_sample_count\":"
        << (lastResult_
                ? std::to_string(
                      lastResult_->calibration.rejectedSampleIndices.size())
                : "null")
        << ",\"rejected_sample_indices\":[";
    if (lastResult_) {
      for (std::size_t position = 0;
           position < lastResult_->calibration.rejectedSampleIndices.size();
           ++position) {
        if (position != 0)
          out << ',';
        out << lastResult_->calibration.rejectedSampleIndices[position];
      }
    }
    out << "]}";
    return out.str();
  }

  void fail(const Response &response, const std::string &message) {
    lastError_ = readable(message);
    response->success = false;
    response->message = lastError_;
    RCLCPP_WARN(node_.get_logger(), "%s", lastError_.c_str());
  }

  BackendNode &node_;
  ros::CaptureSource source_;
  bool configured_{false}, collecting_{false}, datasetPrepared_{false};
  fs::path preparedPath_;
  int preparedFormat_{0};
  std::size_t nextSampleIndex_{0};
  std::string armTopicType_, imageTopicType_, lastError_, lastValidationStatus_;
  std::optional<io::DatasetStatistics> statistics_;
  std::optional<CachedIntrinsics> cachedIntrinsics_;
  std::optional<io::CameraIntrinsics> configuredIntrinsics_;
  std::optional<io::DatasetCalibrationResult> lastResult_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      targetMarkerPublisher_;
  rclcpp::TimerBase::SharedPtr visualizationTimer_;
  std::int64_t lastVisualizedImageStamp_{0};
  std::int64_t lastVisualizedGripperStamp_{0};
  std::chrono::steady_clock::time_point lastGripperPoseSeen_{};
  std::chrono::steady_clock::time_point lastTargetSeen_{};
  core::ExponentialPoseFilter targetPoseFilter_{0.25};
  std::int64_t lastFilteredTargetStamp_{0};
  double lastTargetReprojectionPx_{0.0};
  bool targetVisible_{false};
  bool targetMarkersPublished_{false};
  bool clearTargetMarkersRequested_{false};
  std::string lastPublishedTargetFrame_;
  bool cameraTfActive_{false};
  std::string cameraTfSource_{"none"};
  std::string cameraTfParentFrame_;
  bool gripperTfActive_{false};
  rclcpp::Service<Trigger>::SharedPtr configureService_, getStateService_;
  rclcpp::Service<Trigger>::SharedPtr prepareService_, startService_;
  rclcpp::Service<Trigger>::SharedPtr captureService_, removeService_;
  rclcpp::Service<Trigger>::SharedPtr stopService_, calibrationService_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      parameterValidationCallback_;
};

BackendNode::BackendNode()
    : rclcpp::Node("hand_eye_backend"), impl_(std::make_unique<Impl>(*this)) {}

BackendNode::~BackendNode() = default;

} // namespace hand_eye::app
