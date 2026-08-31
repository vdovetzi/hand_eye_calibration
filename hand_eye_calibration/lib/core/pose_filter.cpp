#include "lib/core/pose_filter.hpp"

#include <opencv2/core/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hand_eye::core {
namespace {

constexpr double kNanosecondsPerSecond = 1'000'000'000.0;

RigidTransform copyOf(const RigidTransform &value) {
  return {value.rotation, value.translation};
}

} // namespace

ExponentialPoseFilter::ExponentialPoseFilter(const double timeConstantSeconds) {
  setTimeConstantSeconds(timeConstantSeconds);
}

void ExponentialPoseFilter::setTimeConstantSeconds(const double value) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(
        "pose-filter time constant must be finite and non-negative");
  }
  timeConstantSeconds_ = value;
  reset();
}

double ExponentialPoseFilter::timeConstantSeconds() const {
  return timeConstantSeconds_;
}

RigidTransform
ExponentialPoseFilter::update(const RigidTransform &measurement,
                              const std::int64_t timestampNanoseconds) {
  if (!measurement.isFinite()) {
    throw std::invalid_argument("pose-filter measurement must be finite");
  }
  if (!value_ || timestampNanoseconds <= lastTimestampNanoseconds_ ||
      timeConstantSeconds_ == 0.0) {
    value_ = copyOf(measurement);
    lastTimestampNanoseconds_ = timestampNanoseconds;
    return copyOf(*value_);
  }

  const double elapsedSeconds =
      static_cast<double>(timestampNanoseconds - lastTimestampNanoseconds_) /
      kNanosecondsPerSecond;
  const double alpha =
      std::clamp(-std::expm1(-elapsedSeconds / timeConstantSeconds_), 0.0, 1.0);
  const cv::Mat translation =
      value_->translation +
      alpha * (measurement.translation - value_->translation);

  const cv::Quatd previous =
      cv::Quatd::createFromRotMat(value_->rotation).normalize();
  cv::Quatd current =
      cv::Quatd::createFromRotMat(measurement.rotation).normalize();
  // OpenCV 4.6 checks its near-identical-quaternion branch before applying
  // directChange. createFromRotMat() may return opposite signs for nearly
  // identical rotations around a dominant-diagonal branch boundary, making
  // the internal dot product round below -1 and producing NaNs. Align the
  // equivalent quaternion signs explicitly before calling slerp.
  if (previous.dot(current) < 0.0) {
    current = -current;
  }
  const cv::Quatd rotation =
      cv::Quatd::slerp(previous, current, alpha, cv::QUAT_ASSUME_UNIT, false);
  const cv::Matx33d rotationMatrix = rotation.toRotMat3x3(cv::QUAT_ASSUME_UNIT);

  value_ = RigidTransform(cv::Mat(rotationMatrix), translation);
  lastTimestampNanoseconds_ = timestampNanoseconds;
  return copyOf(*value_);
}

void ExponentialPoseFilter::reset() {
  value_.reset();
  lastTimestampNanoseconds_ = 0;
}

bool ExponentialPoseFilter::initialized() const { return value_.has_value(); }

} // namespace hand_eye::core
