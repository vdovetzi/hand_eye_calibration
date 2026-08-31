#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace hand_eye::core {

inline constexpr std::size_t kMinimumCalibrationSamples = 15;

struct RigidTransform {
  cv::Mat rotation;
  cv::Mat translation;

  RigidTransform();
  RigidTransform(const cv::Mat &rotation, const cv::Mat &translation);

  static RigidTransform identity();
  static RigidTransform fromMatrix(const cv::Mat &matrix);

  [[nodiscard]] cv::Mat matrix() const;
  [[nodiscard]] RigidTransform inverse() const;
  [[nodiscard]] bool isFinite() const;
};

RigidTransform operator*(const RigidTransform &left,
                         const RigidTransform &right);

enum class CalibrationMode { EYE_IN_HAND, EYE_TO_HAND };

enum class CalibrationMethod {
  AUTO,
  TSAI,
  PARK,
  HORAUD,
  ANDREFF,
  DANIILIDIS,
};

struct CalibrationSample {
  // T_base_gripper: the measured robot pose for both calibration modes.
  RigidTransform robotPose;
  // T_camera_target: the target pose measured by the camera.
  RigidTransform targetToCamera;
};

struct SampleResidual {
  std::size_t sampleIndex{0};
  double translationMeters{0.0};
  double rotationDegrees{0.0};
};

struct ConsistencyMetrics {
  double translationRmsMeters{0.0};
  double rotationRmsDegrees{0.0};
  double translationMaxMeters{0.0};
  double rotationMaxDegrees{0.0};
  std::vector<SampleResidual> residuals;
};

struct CalibrationOptions {
  CalibrationMode mode{CalibrationMode::EYE_IN_HAND};
  CalibrationMethod method{CalibrationMethod::AUTO};
  std::size_t minimumSamples{kMinimumCalibrationSamples};
  double minimumRotationSpanDegrees{35.0};
  double minimumRotationAxisSeparationDegrees{10.0};
  bool rejectOutliers{true};
  std::size_t maximumOutlierIterations{3};
  double outlierMadScale{3.5};
  double minimumTranslationOutlierThresholdMeters{0.001};
  double minimumRotationOutlierThresholdDegrees{0.25};
};

struct CalibrationResult {
  // EYE_IN_HAND: T_gripper_camera. EYE_TO_HAND: T_base_camera.
  RigidTransform transform;
  CalibrationMethod method{CalibrationMethod::AUTO};
  CalibrationMode mode{CalibrationMode::EYE_IN_HAND};
  ConsistencyMetrics consistency;
  std::vector<std::size_t> usedSampleIndices;
  std::vector<std::size_t> rejectedSampleIndices;
};

[[nodiscard]] CalibrationResult
calibrate(const std::vector<CalibrationSample> &samples,
          const CalibrationOptions &options = {});

[[nodiscard]] ConsistencyMetrics
evaluateConsistency(const std::vector<CalibrationSample> &samples,
                    const RigidTransform &calibration, CalibrationMode mode,
                    const std::vector<std::size_t> &sampleIndices = {});

[[nodiscard]] double
rotationSpanDegrees(const std::vector<CalibrationSample> &samples,
                    const std::vector<std::size_t> &sampleIndices = {});

[[nodiscard]] double rotationAxisSeparationDegrees(
    const std::vector<CalibrationSample> &samples,
    const std::vector<std::size_t> &sampleIndices = {});

[[nodiscard]] std::string toString(CalibrationMethod method);
[[nodiscard]] std::string toString(CalibrationMode mode);
[[nodiscard]] CalibrationMethod parseCalibrationMethod(std::string_view text);
[[nodiscard]] CalibrationMode parseCalibrationMode(std::string_view text);

} // namespace hand_eye::core
