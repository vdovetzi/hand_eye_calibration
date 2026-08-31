#pragma once

#include "lib/core/calibration.hpp"

#include <cstdint>
#include <optional>

namespace hand_eye::core {

// Time-aware exponential low-pass filter for visualization poses. Translation
// is blended linearly and rotation is blended on SO(3) with quaternion SLERP.
class ExponentialPoseFilter final {
public:
  explicit ExponentialPoseFilter(double timeConstantSeconds = 0.0);

  void setTimeConstantSeconds(double value);
  [[nodiscard]] double timeConstantSeconds() const;

  [[nodiscard]] RigidTransform update(const RigidTransform &measurement,
                                      std::int64_t timestampNanoseconds);
  void reset();
  [[nodiscard]] bool initialized() const;

private:
  double timeConstantSeconds_{0.0};
  std::optional<RigidTransform> value_;
  std::int64_t lastTimestampNanoseconds_{0};
};

} // namespace hand_eye::core
