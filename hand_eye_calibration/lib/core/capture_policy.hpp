#pragma once

#include "lib/core/calibration.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hand_eye::core {

struct ImageSampleMetadata {
  std::int64_t timestampNanoseconds{0};
  std::string frameId;
  int width{0};
  int height{0};
  bool hasSourceTimestamp{false};
};

struct TimedRobotPose {
  std::int64_t timestampNanoseconds{0};
  RigidTransform pose;
  std::string parentFrame;
  std::string childFrame;
  bool hasSourceTimestamp{false};
};

struct CameraInfoMetadata {
  std::string frameId;
  int width{0};
  int height{0};
  bool calibrated{true};
};

struct CapturePolicyOptions {
  double maximumTimeDeltaMilliseconds{30.0};
  bool requireStampedSamples{true};
  bool requireCameraInfo{false};
  std::string robotBaseFrame;
  std::string robotEffectorFrame;
  std::string cameraFrame;
  double settlingDurationMilliseconds{500.0};
  double settlingTranslationMeters{0.0005};
  double settlingRotationDegrees{0.25};
  double duplicateTranslationMeters{0.005};
  double duplicateRotationDegrees{5.0};
};

struct CaptureRequest {
  std::optional<ImageSampleMetadata> image;
  std::vector<TimedRobotPose> robotHistory;
  std::optional<CameraInfoMetadata> cameraInfo;
  std::vector<RigidTransform> acceptedRobotPoses;
};

enum class CaptureRejectionReason {
  NONE,
  MISSING_IMAGE,
  MISSING_ROBOT_POSE,
  UNSTAMPED_SAMPLE,
  TIMESTAMP_MISMATCH,
  FRAME_MISMATCH,
  CAMERA_INFO_MISSING,
  CAMERA_INFO_INVALID,
  CAMERA_INFO_SIZE_MISMATCH,
  ROBOT_NOT_SETTLED,
  DUPLICATE_POSE,
};

struct CaptureDecision {
  bool accepted{false};
  CaptureRejectionReason reason{CaptureRejectionReason::NONE};
  std::string message;
  std::optional<TimedRobotPose> matchedRobotPose;
  double timestampDeltaMilliseconds{0.0};
};

class CapturePolicy {
public:
  explicit CapturePolicy(CapturePolicyOptions options = {});

  [[nodiscard]] const CapturePolicyOptions &options() const;
  [[nodiscard]] CaptureDecision evaluate(const CaptureRequest &request) const;

private:
  CapturePolicyOptions options_;
};

[[nodiscard]] std::string toString(CaptureRejectionReason reason);

} // namespace hand_eye::core
