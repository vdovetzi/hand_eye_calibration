#pragma once

#include "lib/core/calibration.hpp"
#include "lib/core/target.hpp"
#include "lib/io/dataset_repository.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace hand_eye::io {

struct CameraIntrinsics {
  cv::Mat cameraMatrix;
  cv::Mat distortionCoefficients;
  int width{0};
  int height{0};
};

struct CalibrationDocument {
  core::RigidTransform transform;
  cv::Mat cameraMatrix;
  cv::Mat distortionCoefficients;
  std::string method;
  std::string calibrationType;
  std::string parentFrame;
  std::string childFrame;
  double translationRmsM = std::numeric_limits<double>::quiet_NaN();
  double rotationRmsDeg = std::numeric_limits<double>::quiet_NaN();
  std::vector<std::size_t> rejectedSampleIndices;
};

struct DatasetCalibrationOptions {
  std::filesystem::path datasetPath;
  PoseFormat poseFormat{PoseFormat::QuaternionXyzw};
  core::PatternConfig pattern;
  std::optional<CameraIntrinsics> intrinsics;
  core::TargetPoseOptions targetOptions;
  core::CalibrationOptions calibrationOptions;
  std::string parentFrame;
  std::string childFrame;
};

struct DatasetCalibrationResult {
  core::CalibrationResult calibration;
  cv::Mat cameraMatrix;
  cv::Mat distortionCoefficients;
  double intrinsicCalibrationRmsPixels{0.0};
  bool estimatedIntrinsics{false};
  std::vector<std::size_t> targetRejectedSampleIndices;
  std::string parentFrame;
  std::string childFrame;

  [[nodiscard]] CalibrationDocument document() const;
};

// Shared online/offline dataset pipeline. It detects every target, preserves
// original dataset indices through filtering, then calls core::calibrate().
[[nodiscard]] DatasetCalibrationResult
calibrateDataset(const DatasetCalibrationOptions &options);

class CalibrationIO {
public:
  // Throws std::runtime_error when the file cannot be opened or K/D are
  // missing, non-finite or have invalid dimensions.
  static CameraIntrinsics readIntrinsics(const std::filesystem::path &path);

  // Metadata is optional while reading, keeping calibration files created by
  // older package versions usable. T, K and D remain mandatory.
  static CalibrationDocument read(const std::filesystem::path &path);

  // Writes through a temporary file and atomically replaces an existing YAML.
  static void write(const std::filesystem::path &path,
                    const CalibrationDocument &document);
};

} // namespace hand_eye::io
