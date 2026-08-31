#include "lib/ros/capture_source.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ros_babel_fish/babel_fish.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace hand_eye::ros {
namespace {

std::int64_t stampNanoseconds(const builtin_interfaces::msg::Time &stamp) {
  return static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

bool hasStamp(const builtin_interfaces::msg::Time &stamp) {
  return stamp.sec != 0 || stamp.nanosec != 0;
}

core::RigidTransform transform(double x, double y, double z, double qx,
                               double qy, double qz, double qw) {
  const double norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (!std::isfinite(norm) || norm <= std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error("robot pose contains an invalid quaternion");
  }
  qx /= norm;
  qy /= norm;
  qz /= norm;
  qw /= norm;
  const cv::Mat rotation =
      (cv::Mat_<double>(3, 3) << 1.0 - 2.0 * (qy * qy + qz * qz),
       2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw),
       2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz),
       2.0 * (qy * qz - qx * qw), 2.0 * (qx * qz - qy * qw),
       2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy));
  return {rotation, (cv::Mat_<double>(3, 1) << x, y, z)};
}

std::optional<std::string> topicType(const rclcpp::Node &node,
                                     const std::string &topic) {
  if (topic.empty()) {
    return std::nullopt;
  }
  const auto topics = node.get_topic_names_and_types();
  const auto it = topics.find(topic);
  if (it == topics.end() || it->second.empty()) {
    return std::nullopt;
  }
  return it->second.front();
}

bool hasValidIntrinsics(const sensor_msgs::msg::CameraInfo &message) {
  const bool validSize = message.width > 0 && message.height > 0;
  const bool validMatrix =
      std::ranges::all_of(
          message.k, [](const double value) { return std::isfinite(value); }) &&
      message.k[0] > 0.0 && message.k[4] > 0.0;
  const bool validDistortion =
      (message.d.empty() || message.d.size() >= 4) &&
      std::ranges::all_of(
          message.d, [](const double value) { return std::isfinite(value); });
  return validSize && validMatrix && validDistortion;
}

} // namespace

class CaptureSource::Impl {
public:
  explicit Impl(rclcpp::Node &node)
      : node_(node), fish_(ros_babel_fish::BabelFish::make_shared()) {}

  void start(const CaptureSourceConfig &config) {
    if (config.armTopic.empty() || config.imageTopic.empty()) {
      throw std::invalid_argument("arm_topic and image_topic are required");
    }
    stop();
    config_ = config;
    if (config_.maximumBufferedRobotPoses == 0) {
      config_.maximumBufferedRobotPoses = 1;
    }

    subscribeImage();
    subscribeCameraInfo();
    armSubscription_ = fish_->create_subscription(
        node_, config_.armTopic, rclcpp::SensorDataQoS(),
        [this](ros_babel_fish::CompoundMessage::SharedPtr message) {
          receiveRobotPose(std::move(message));
        },
        nullptr, {}, std::chrono::seconds(5));
    if (!armSubscription_) {
      stop();
      throw std::runtime_error("arm topic is not available: " +
                               config_.armTopic);
    }
    running_ = true;
  }

  void stop() {
    rawImageSubscription_.reset();
    compressedImageSubscription_.reset();
    cameraInfoSubscription_.reset();
    armSubscription_.reset();
    std::scoped_lock lock(mutex_);
    image_.release();
    imageMetadata_.reset();
    robotHistory_.clear();
    cameraInfo_.reset();
    cameraMatrix_.release();
    distortionCoefficients_.release();
    armTopicType_.clear();
    imageTopicType_.clear();
    running_ = false;
  }

  [[nodiscard]] CaptureSnapshot snapshot() const {
    CaptureSnapshot result;
    std::scoped_lock lock(mutex_);
    result.image = image_.clone();
    result.imageMetadata = imageMetadata_;
    result.robotHistory.assign(robotHistory_.begin(), robotHistory_.end());
    if (!robotHistory_.empty()) {
      result.latestRobotPose = robotHistory_.back();
    }
    result.cameraInfo = cameraInfo_;
    result.cameraMatrix = cameraMatrix_.clone();
    result.distortionCoefficients = distortionCoefficients_.clone();
    if (imageMetadata_) {
      result.closestRobotPose = closestPoseLocked(
          imageMetadata_->timestampNanoseconds, robotHistory_);
    }
    return result;
  }

  [[nodiscard]] std::optional<core::TimedRobotPose>
  closestPose(std::int64_t timestampNanoseconds) const {
    std::scoped_lock lock(mutex_);
    return closestPoseLocked(timestampNanoseconds, robotHistory_);
  }

  [[nodiscard]] std::string armTopicType() const {
    std::scoped_lock lock(mutex_);
    return armTopicType_;
  }

  [[nodiscard]] std::string imageTopicType() const {
    std::scoped_lock lock(mutex_);
    return imageTopicType_;
  }

  [[nodiscard]] bool running() const { return running_.load(); }

private:
  using RobotHistory = std::deque<core::TimedRobotPose>;

  static std::optional<core::TimedRobotPose>
  closestPoseLocked(std::int64_t timestampNanoseconds,
                    const RobotHistory &history) {
    if (history.empty()) {
      return std::nullopt;
    }
    const auto closest = std::min_element(
        history.begin(), history.end(),
        [timestampNanoseconds](const auto &left, const auto &right) {
          const auto leftDelta =
              std::llabs(left.timestampNanoseconds - timestampNanoseconds);
          const auto rightDelta =
              std::llabs(right.timestampNanoseconds - timestampNanoseconds);
          return leftDelta < rightDelta;
        });
    return *closest;
  }

  void subscribeImage() {
    std::string topic = config_.imageTopic;
    std::string type = topicType(node_, topic).value_or("");
    if (type != "sensor_msgs/msg/CompressedImage") {
      const std::string compressedTopic = topic + "/compressed";
      if (topicType(node_, compressedTopic) ==
          "sensor_msgs/msg/CompressedImage") {
        topic = compressedTopic;
        type = "sensor_msgs/msg/CompressedImage";
      }
    }

    if (type.empty()) {
      throw std::runtime_error("image topic is not available: " +
                               config_.imageTopic);
    }
    if (type != "sensor_msgs/msg/Image" &&
        type != "sensor_msgs/msg/CompressedImage") {
      throw std::runtime_error("unsupported image topic type: " + type);
    }

    if (type == "sensor_msgs/msg/CompressedImage") {
      compressedImageSubscription_ =
          node_.create_subscription<sensor_msgs::msg::CompressedImage>(
              topic, rclcpp::SensorDataQoS(),
              [this](const sensor_msgs::msg::CompressedImage::ConstSharedPtr
                         message) {
                try {
                  const cv::Mat encoded(message->data, true);
                  cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
                  if (!image.empty()) {
                    receiveImage(image, message->header, true);
                  }
                } catch (const std::exception &error) {
                  RCLCPP_WARN(node_.get_logger(),
                              "Cannot decode compressed image: %s",
                              error.what());
                }
              });
      std::scoped_lock lock(mutex_);
      imageTopicType_ = "sensor_msgs/msg/CompressedImage";
      return;
    }

    rawImageSubscription_ = node_.create_subscription<sensor_msgs::msg::Image>(
        topic, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::Image::ConstSharedPtr message) {
          try {
            const cv::Mat image =
                cv_bridge::toCvCopy(*message,
                                    sensor_msgs::image_encodings::BGR8)
                    ->image;
            receiveImage(image, message->header, false);
          } catch (const std::exception &error) {
            RCLCPP_WARN(node_.get_logger(), "Cannot convert image: %s",
                        error.what());
          }
        });
    std::scoped_lock lock(mutex_);
    imageTopicType_ = "sensor_msgs/msg/Image";
  }

  void subscribeCameraInfo() {
    if (config_.cameraInfoTopic.empty()) {
      return;
    }
    cameraInfoSubscription_ =
        node_.create_subscription<sensor_msgs::msg::CameraInfo>(
            config_.cameraInfoTopic, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
              const bool calibrated = hasValidIntrinsics(*message);
              if (!calibrated) {
                std::scoped_lock lock(mutex_);
                cameraInfo_ = core::CameraInfoMetadata{
                    message->header.frame_id, static_cast<int>(message->width),
                    static_cast<int>(message->height), false};
                cameraMatrix_.release();
                distortionCoefficients_.release();
                RCLCPP_WARN(node_.get_logger(),
                            "CameraInfo is uncalibrated or malformed");
                return;
              }
              cv::Mat matrix(3, 3, CV_64F);
              for (std::size_t index = 0; index < message->k.size(); ++index) {
                matrix.at<double>(static_cast<int>(index / 3),
                                  static_cast<int>(index % 3)) =
                    message->k[index];
              }
              cv::Mat distortion(1, static_cast<int>(message->d.size()),
                                 CV_64F);
              for (std::size_t index = 0; index < message->d.size(); ++index) {
                distortion.at<double>(0, static_cast<int>(index)) =
                    message->d[index];
              }
              std::scoped_lock lock(mutex_);
              cameraInfo_ = core::CameraInfoMetadata{
                  message->header.frame_id, static_cast<int>(message->width),
                  static_cast<int>(message->height), true};
              cameraMatrix_ = std::move(matrix);
              distortionCoefficients_ = std::move(distortion);
            });
  }

  void receiveImage(const cv::Mat &image, const std_msgs::msg::Header &header,
                    bool /* compressed */) {
    const bool stamped = hasStamp(header.stamp);
    const auto timestamp = stamped ? stampNanoseconds(header.stamp)
                                   : node_.get_clock()->now().nanoseconds();
    std::scoped_lock lock(mutex_);
    image_ = image.clone();
    imageMetadata_ = core::ImageSampleMetadata{timestamp, header.frame_id,
                                               image.cols, image.rows, stamped};
  }

  void receiveRobotPose(ros_babel_fish::CompoundMessage::SharedPtr message) {
    try {
      core::TimedRobotPose sample;
      const auto receiptTime = node_.get_clock()->now().nanoseconds();
      const std::string type = message->name();
      if (type == "geometry_msgs/msg/PoseStamped") {
        const auto typed = message->message<geometry_msgs::msg::PoseStamped>();
        sample.pose =
            transform(typed->pose.position.x, typed->pose.position.y,
                      typed->pose.position.z, typed->pose.orientation.x,
                      typed->pose.orientation.y, typed->pose.orientation.z,
                      typed->pose.orientation.w);
        sample.hasSourceTimestamp = hasStamp(typed->header.stamp);
        sample.timestampNanoseconds =
            sample.hasSourceTimestamp ? stampNanoseconds(typed->header.stamp)
                                      : receiptTime;
        sample.parentFrame = typed->header.frame_id;
        sample.childFrame = config_.robotEffectorFrame;
      } else if (type == "geometry_msgs/msg/TransformStamped") {
        const auto typed =
            message->message<geometry_msgs::msg::TransformStamped>();
        sample.pose = transform(
            typed->transform.translation.x, typed->transform.translation.y,
            typed->transform.translation.z, typed->transform.rotation.x,
            typed->transform.rotation.y, typed->transform.rotation.z,
            typed->transform.rotation.w);
        sample.hasSourceTimestamp = hasStamp(typed->header.stamp);
        sample.timestampNanoseconds =
            sample.hasSourceTimestamp ? stampNanoseconds(typed->header.stamp)
                                      : receiptTime;
        sample.parentFrame = typed->header.frame_id;
        sample.childFrame = typed->child_frame_id;
      } else if (type == "geometry_msgs/msg/Pose") {
        const auto typed = message->message<geometry_msgs::msg::Pose>();
        sample.pose =
            transform(typed->position.x, typed->position.y, typed->position.z,
                      typed->orientation.x, typed->orientation.y,
                      typed->orientation.z, typed->orientation.w);
        sample.timestampNanoseconds = receiptTime;
        sample.parentFrame = config_.robotBaseFrame;
        sample.childFrame = config_.robotEffectorFrame;
      } else if (type == "geometry_msgs/msg/Transform") {
        const auto typed = message->message<geometry_msgs::msg::Transform>();
        sample.pose =
            transform(typed->translation.x, typed->translation.y,
                      typed->translation.z, typed->rotation.x,
                      typed->rotation.y, typed->rotation.z, typed->rotation.w);
        sample.timestampNanoseconds = receiptTime;
        sample.parentFrame = config_.robotBaseFrame;
        sample.childFrame = config_.robotEffectorFrame;
      } else {
        RCLCPP_WARN(node_.get_logger(),
                    "Unsupported arm message type '%s'; expected Pose, "
                    "PoseStamped, Transform or TransformStamped",
                    type.c_str());
        return;
      }

      std::scoped_lock lock(mutex_);
      armTopicType_ = type;
      robotHistory_.push_back(std::move(sample));
      while (robotHistory_.size() > config_.maximumBufferedRobotPoses) {
        robotHistory_.pop_front();
      }
    } catch (const std::exception &error) {
      RCLCPP_WARN(node_.get_logger(), "Cannot normalize robot pose: %s",
                  error.what());
    }
  }

  rclcpp::Node &node_;
  CaptureSourceConfig config_;
  ros_babel_fish::BabelFish::SharedPtr fish_;
  ros_babel_fish::BabelFishSubscription::SharedPtr armSubscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      rawImageSubscription_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
      compressedImageSubscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      cameraInfoSubscription_;

  mutable std::mutex mutex_;
  cv::Mat image_;
  std::optional<core::ImageSampleMetadata> imageMetadata_;
  RobotHistory robotHistory_;
  std::optional<core::CameraInfoMetadata> cameraInfo_;
  cv::Mat cameraMatrix_;
  cv::Mat distortionCoefficients_;
  std::string armTopicType_;
  std::string imageTopicType_;
  std::atomic_bool running_{false};
};

CaptureSource::CaptureSource(rclcpp::Node &node)
    : impl_(std::make_unique<Impl>(node)) {}

CaptureSource::~CaptureSource() = default;

void CaptureSource::start(const CaptureSourceConfig &config) {
  impl_->start(config);
}

void CaptureSource::stop() { impl_->stop(); }

bool CaptureSource::running() const { return impl_->running(); }

CaptureSnapshot CaptureSource::snapshot() const { return impl_->snapshot(); }

std::optional<core::TimedRobotPose>
CaptureSource::closestPose(std::int64_t timestampNanoseconds) const {
  return impl_->closestPose(timestampNanoseconds);
}

std::string CaptureSource::armTopicType() const {
  return impl_->armTopicType();
}

std::string CaptureSource::imageTopicType() const {
  return impl_->imageTopicType();
}

} // namespace hand_eye::ros
