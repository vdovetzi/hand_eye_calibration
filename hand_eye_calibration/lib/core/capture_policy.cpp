#include "lib/core/capture_policy.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace hand_eye::core {
namespace {

constexpr double kNanosecondsPerMillisecond = 1'000'000.0;
constexpr double kRadiansToDegrees = 180.0 / CV_PI;

double rotationDistanceDegrees(const RigidTransform &left,
                               const RigidTransform &right) {
  const cv::Mat relative = left.rotation.t() * right.rotation;
  const double cosine =
      std::clamp((cv::trace(relative)[0] - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(cosine) * kRadiansToDegrees;
}

bool frameMatches(const std::string &expected, const std::string &actual) {
  return expected.empty() || expected == actual;
}

CaptureDecision reject(const CaptureRejectionReason reason, std::string message,
                       const std::optional<TimedRobotPose> &matched = {},
                       const double deltaMilliseconds = 0.0) {
  return {false, reason, std::move(message), matched, deltaMilliseconds};
}

void validateNonNegative(const double value, const char *name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and non-negative");
  }
}

} // namespace

CapturePolicy::CapturePolicy(CapturePolicyOptions options)
    : options_(std::move(options)) {
  validateNonNegative(options_.maximumTimeDeltaMilliseconds,
                      "maximumTimeDeltaMilliseconds");
  validateNonNegative(options_.settlingDurationMilliseconds,
                      "settlingDurationMilliseconds");
  validateNonNegative(options_.settlingTranslationMeters,
                      "settlingTranslationMeters");
  validateNonNegative(options_.settlingRotationDegrees,
                      "settlingRotationDegrees");
  validateNonNegative(options_.duplicateTranslationMeters,
                      "duplicateTranslationMeters");
  validateNonNegative(options_.duplicateRotationDegrees,
                      "duplicateRotationDegrees");
}

const CapturePolicyOptions &CapturePolicy::options() const { return options_; }

CaptureDecision CapturePolicy::evaluate(const CaptureRequest &request) const {
  if (!request.image) {
    return reject(CaptureRejectionReason::MISSING_IMAGE,
                  "no camera image is available");
  }
  const ImageSampleMetadata &image = *request.image;
  if (options_.requireStampedSamples && !image.hasSourceTimestamp) {
    return reject(CaptureRejectionReason::UNSTAMPED_SAMPLE,
                  "the image does not contain a source timestamp");
  }
  if (request.robotHistory.empty()) {
    return reject(CaptureRejectionReason::MISSING_ROBOT_POSE,
                  "no robot pose is available");
  }

  std::vector<const TimedRobotPose *> candidates;
  candidates.reserve(request.robotHistory.size());
  for (const TimedRobotPose &pose : request.robotHistory) {
    if (!options_.requireStampedSamples || pose.hasSourceTimestamp) {
      candidates.push_back(&pose);
    }
  }
  if (candidates.empty()) {
    return reject(CaptureRejectionReason::UNSTAMPED_SAMPLE,
                  "the robot pose does not contain a source timestamp");
  }

  const TimedRobotPose *matched = nullptr;
  std::vector<const TimedRobotPose *> comparableCandidates;
  if (image.hasSourceTimestamp) {
    std::ranges::copy_if(
        candidates, std::back_inserter(comparableCandidates),
        [](const TimedRobotPose *pose) { return pose->hasSourceTimestamp; });
  }
  if (!comparableCandidates.empty()) {
    matched = *std::min_element(
        comparableCandidates.begin(), comparableCandidates.end(),
        [&](const auto *left, const auto *right) {
          const auto leftDelta =
              std::abs(left->timestampNanoseconds - image.timestampNanoseconds);
          const auto rightDelta = std::abs(right->timestampNanoseconds -
                                           image.timestampNanoseconds);
          return leftDelta < rightDelta;
        });
  } else {
    // Receipt-time and source-time clocks are not necessarily comparable.
    // Compatibility mode therefore uses the latest pose when either source
    // lacks a stamp instead of manufacturing a misleading time delta.
    matched = candidates.back();
  }

  double deltaMilliseconds = 0.0;
  if (image.hasSourceTimestamp && matched->hasSourceTimestamp) {
    deltaMilliseconds =
        static_cast<double>(std::abs(matched->timestampNanoseconds -
                                     image.timestampNanoseconds)) /
        kNanosecondsPerMillisecond;
    if (deltaMilliseconds > options_.maximumTimeDeltaMilliseconds) {
      return reject(
          CaptureRejectionReason::TIMESTAMP_MISMATCH,
          "the nearest robot pose is too far from the image timestamp",
          *matched, deltaMilliseconds);
    }
  }

  if (!frameMatches(options_.cameraFrame, image.frameId) ||
      !frameMatches(options_.robotBaseFrame, matched->parentFrame) ||
      !frameMatches(options_.robotEffectorFrame, matched->childFrame)) {
    return reject(CaptureRejectionReason::FRAME_MISMATCH,
                  "the image or robot pose uses an unexpected frame", *matched,
                  deltaMilliseconds);
  }

  if (options_.requireCameraInfo && !request.cameraInfo) {
    return reject(CaptureRejectionReason::CAMERA_INFO_MISSING,
                  "CameraInfo is required before capturing", *matched,
                  deltaMilliseconds);
  }
  if (request.cameraInfo) {
    const CameraInfoMetadata &info = *request.cameraInfo;
    if (!info.calibrated) {
      return reject(CaptureRejectionReason::CAMERA_INFO_INVALID,
                    "CameraInfo does not contain valid calibrated intrinsics",
                    *matched, deltaMilliseconds);
    }
    if (info.width != image.width || info.height != image.height) {
      return reject(CaptureRejectionReason::CAMERA_INFO_SIZE_MISMATCH,
                    "CameraInfo resolution does not match the image", *matched,
                    deltaMilliseconds);
    }
    if (!frameMatches(options_.cameraFrame, info.frameId) ||
        (!image.frameId.empty() && !info.frameId.empty() &&
         image.frameId != info.frameId)) {
      return reject(CaptureRejectionReason::FRAME_MISMATCH,
                    "CameraInfo and image frame IDs do not match", *matched,
                    deltaMilliseconds);
    }
  }

  if (options_.settlingDurationMilliseconds > 0.0) {
    const auto settlingNanoseconds = static_cast<std::int64_t>(
        options_.settlingDurationMilliseconds * kNanosecondsPerMillisecond);
    const std::int64_t windowStart =
        matched->timestampNanoseconds - settlingNanoseconds;
    const TimedRobotPose *windowAnchor = nullptr;
    for (const TimedRobotPose &pose : request.robotHistory) {
      if (pose.timestampNanoseconds <= windowStart &&
          (windowAnchor == nullptr ||
           pose.timestampNanoseconds > windowAnchor->timestampNanoseconds)) {
        windowAnchor = &pose;
      }
    }
    if (windowAnchor == nullptr) {
      return reject(CaptureRejectionReason::ROBOT_NOT_SETTLED,
                    "not enough robot-pose history to prove settling", *matched,
                    deltaMilliseconds);
    }
    const auto movedFromMatched = [&](const TimedRobotPose &pose) {
      return cv::norm(pose.pose.translation - matched->pose.translation) >
                 options_.settlingTranslationMeters ||
             rotationDistanceDegrees(pose.pose, matched->pose) >
                 options_.settlingRotationDegrees;
    };
    if (movedFromMatched(*windowAnchor)) {
      return reject(CaptureRejectionReason::ROBOT_NOT_SETTLED,
                    "the robot moved during the settling window", *matched,
                    deltaMilliseconds);
    }
    for (const TimedRobotPose &pose : request.robotHistory) {
      if (pose.timestampNanoseconds < windowStart ||
          pose.timestampNanoseconds > matched->timestampNanoseconds) {
        continue;
      }
      if (movedFromMatched(pose)) {
        return reject(CaptureRejectionReason::ROBOT_NOT_SETTLED,
                      "the robot moved during the settling window", *matched,
                      deltaMilliseconds);
      }
    }
  }

  for (const RigidTransform &accepted : request.acceptedRobotPoses) {
    const double translation =
        cv::norm(accepted.translation - matched->pose.translation);
    const double rotation = rotationDistanceDegrees(accepted, matched->pose);
    if (translation < options_.duplicateTranslationMeters &&
        rotation < options_.duplicateRotationDegrees) {
      return reject(CaptureRejectionReason::DUPLICATE_POSE,
                    "the robot pose is too similar to an accepted sample",
                    *matched, deltaMilliseconds);
    }
  }

  return {true, CaptureRejectionReason::NONE, "sample accepted", *matched,
          deltaMilliseconds};
}

std::string toString(const CaptureRejectionReason reason) {
  switch (reason) {
  case CaptureRejectionReason::NONE:
    return "none";
  case CaptureRejectionReason::MISSING_IMAGE:
    return "missing_image";
  case CaptureRejectionReason::MISSING_ROBOT_POSE:
    return "missing_robot_pose";
  case CaptureRejectionReason::UNSTAMPED_SAMPLE:
    return "unstamped_sample";
  case CaptureRejectionReason::TIMESTAMP_MISMATCH:
    return "timestamp_mismatch";
  case CaptureRejectionReason::FRAME_MISMATCH:
    return "frame_mismatch";
  case CaptureRejectionReason::CAMERA_INFO_MISSING:
    return "camera_info_missing";
  case CaptureRejectionReason::CAMERA_INFO_INVALID:
    return "camera_info_invalid";
  case CaptureRejectionReason::CAMERA_INFO_SIZE_MISMATCH:
    return "camera_info_size_mismatch";
  case CaptureRejectionReason::ROBOT_NOT_SETTLED:
    return "robot_not_settled";
  case CaptureRejectionReason::DUPLICATE_POSE:
    return "duplicate_pose";
  }
  throw std::invalid_argument("unknown capture rejection reason");
}

} // namespace hand_eye::core
