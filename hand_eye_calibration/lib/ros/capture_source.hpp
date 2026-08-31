#pragma once

#include "lib/core/capture_policy.hpp"

#include <opencv2/core.hpp>
#include <rclcpp/node.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hand_eye::ros {

struct CaptureSourceConfig {
  std::string armTopic;
  std::string imageTopic;
  std::string cameraInfoTopic;
  std::string robotBaseFrame;
  std::string robotEffectorFrame;
  std::size_t maximumBufferedRobotPoses{1000};
};

struct CaptureSnapshot {
  cv::Mat image;
  std::optional<core::ImageSampleMetadata> imageMetadata;
  std::vector<core::TimedRobotPose> robotHistory;
  std::optional<core::TimedRobotPose> latestRobotPose;
  std::optional<core::TimedRobotPose> closestRobotPose;
  std::optional<core::CameraInfoMetadata> cameraInfo;
  cv::Mat cameraMatrix;
  cv::Mat distortionCoefficients;
};

class CaptureSource {
public:
  explicit CaptureSource(rclcpp::Node &node);
  ~CaptureSource();

  CaptureSource(const CaptureSource &) = delete;
  CaptureSource &operator=(const CaptureSource &) = delete;

  void start(const CaptureSourceConfig &config);
  void stop();

  [[nodiscard]] bool running() const;
  [[nodiscard]] CaptureSnapshot snapshot() const;
  [[nodiscard]] std::optional<core::TimedRobotPose>
  closestPose(std::int64_t timestampNanoseconds) const;

  [[nodiscard]] std::string armTopicType() const;
  [[nodiscard]] std::string imageTopicType() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace hand_eye::ros
