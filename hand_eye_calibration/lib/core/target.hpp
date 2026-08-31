#pragma once

#include "lib/core/calibration.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hand_eye::core {

enum class PatternType { ARUCO = 1, CHESSBOARD = 2, CHARUCO = 3 };

struct PatternConfig {
  PatternType type{PatternType::CHESSBOARD};
  int rows{0};
  int columns{0};
  double squareSizeMeters{0.0};
  double markerSizeMeters{0.0};
  int dictionarySize{4};
  int markerId{0};
};

[[nodiscard]] PatternConfig parsePattern(std::string_view patternInfo);
[[nodiscard]] std::string describePattern(const PatternConfig &config);

// Returns the canonical binary marker modules, including a one-module black
// border. The result is CV_8UC1 with 0 for black and 255 for white.
[[nodiscard]] cv::Mat canonicalArucoMarker(const PatternConfig &config);
[[nodiscard]] int arucoDictionaryCapacity(const PatternConfig &config);

struct TargetObservation {
  std::vector<cv::Point2f> imagePoints;
  std::vector<cv::Point3f> objectPoints;
};

class TargetDetector {
public:
  explicit TargetDetector(PatternConfig config);

  [[nodiscard]] const PatternConfig &config() const;
  [[nodiscard]] std::optional<TargetObservation>
  detect(const cv::Mat &image) const;
  void draw(cv::Mat &image, const TargetObservation &observation) const;

private:
  PatternConfig config_;
};

struct TargetPoseOptions {
  double maximumReprojectionErrorPixels{1.0};
  std::size_t minimumViewsForIntrinsicCalibration{8};
  double arucoAmbiguityRatio{1.25};
};

struct TargetPose {
  RigidTransform targetToCamera;
  double reprojectionErrorPixels{0.0};
};

struct TargetEstimationResult {
  cv::Mat cameraMatrix;
  cv::Mat distortionCoefficients;
  double intrinsicCalibrationRmsPixels{0.0};
  bool estimatedIntrinsics{false};
  std::vector<std::optional<TargetPose>> poses;
  std::vector<std::size_t> rejectedViewIndices;
};

class TargetPoseEstimator {
public:
  [[nodiscard]] static TargetEstimationResult
  estimate(const PatternConfig &config,
           const std::vector<TargetObservation> &observations,
           const cv::Size &imageSize, const cv::Mat &cameraMatrix = {},
           const cv::Mat &distortionCoefficients = {},
           const TargetPoseOptions &options = {});
};

} // namespace hand_eye::core
