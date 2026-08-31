#include "lib/io/calibration_io.hpp"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hand_eye::io {
namespace {

namespace fs = std::filesystem;

cv::Mat readMatrix(const cv::FileStorage &storage, const char *key,
                   const fs::path &path) {
  cv::Mat matrix;
  storage[key] >> matrix;
  if (matrix.empty()) {
    throw std::runtime_error("Calibration file " + path.string() +
                             " does not contain " + key);
  }
  cv::Mat result;
  matrix.convertTo(result, CV_64F);
  if (!cv::checkRange(result)) {
    throw std::runtime_error(std::string(key) +
                             " contains non-finite values in " + path.string());
  }
  return result;
}

CameraIntrinsics validateIntrinsics(CameraIntrinsics intrinsics,
                                    const fs::path &path) {
  intrinsics.cameraMatrix.convertTo(intrinsics.cameraMatrix, CV_64F);
  intrinsics.distortionCoefficients.convertTo(intrinsics.distortionCoefficients,
                                              CV_64F);
  if (!cv::checkRange(intrinsics.cameraMatrix)) {
    throw std::runtime_error("K contains non-finite values in " +
                             path.string());
  }
  if (!cv::checkRange(intrinsics.distortionCoefficients)) {
    throw std::runtime_error("D contains non-finite values in " +
                             path.string());
  }
  if (intrinsics.cameraMatrix.rows != 3 || intrinsics.cameraMatrix.cols != 3) {
    throw std::runtime_error("K must be a 3x3 matrix in " + path.string());
  }
  if (intrinsics.distortionCoefficients.total() < 4 ||
      (intrinsics.distortionCoefficients.rows != 1 &&
       intrinsics.distortionCoefficients.cols != 1)) {
    throw std::runtime_error(
        "D must be a row or column vector with at least four coefficients in " +
        path.string());
  }
  if (intrinsics.cameraMatrix.at<double>(0, 0) <= 0.0 ||
      intrinsics.cameraMatrix.at<double>(1, 1) <= 0.0) {
    throw std::runtime_error("K focal lengths must be positive in " +
                             path.string());
  }
  if ((intrinsics.width == 0) != (intrinsics.height == 0) ||
      intrinsics.width < 0 || intrinsics.height < 0) {
    throw std::runtime_error("intrinsics width and height must both be "
                             "positive or both omitted in " +
                             path.string());
  }
  return intrinsics;
}

CameraIntrinsics readAndValidateIntrinsics(const cv::FileStorage &storage,
                                           const fs::path &path) {
  return validateIntrinsics(CameraIntrinsics{readMatrix(storage, "K", path),
                                             readMatrix(storage, "D", path)},
                            path);
}

std::optional<YAML::Node> findRosParameters(const YAML::Node &node) {
  if (!node || !node.IsMap()) {
    return std::nullopt;
  }
  const YAML::Node parameters = node["ros__parameters"];
  if (parameters && parameters.IsMap()) {
    return parameters;
  }
  for (const auto &entry : node) {
    const auto nested = findRosParameters(entry.second);
    if (nested) {
      return nested;
    }
  }
  return std::nullopt;
}

cv::Mat readYamlVector(const YAML::Node &node, const char *key,
                       const fs::path &path) {
  const YAML::Node valuesNode = node[key];
  if (!valuesNode || !valuesNode.IsSequence()) {
    throw std::runtime_error(std::string(key) +
                             " must be a numeric sequence in " + path.string());
  }
  try {
    const std::vector<double> values = valuesNode.as<std::vector<double>>();
    if (values.empty()) {
      throw std::runtime_error(std::string(key) + " is empty in " +
                               path.string());
    }
    return cv::Mat(1, static_cast<int>(values.size()), CV_64F,
                   const_cast<double *>(values.data()))
        .clone();
  } catch (const YAML::Exception &error) {
    throw std::runtime_error("Cannot parse " + std::string(key) + " in " +
                             path.string() + ": " + error.what());
  }
}

int readYamlDimension(const YAML::Node &node, const char *key,
                      const fs::path &path) {
  if (!node[key]) {
    return 0;
  }
  try {
    const int value = node[key].as<int>();
    if (value <= 0) {
      throw std::runtime_error(std::string(key) + " must be positive in " +
                               path.string());
    }
    return value;
  } catch (const YAML::Exception &error) {
    throw std::runtime_error("Cannot parse " + std::string(key) + " in " +
                             path.string() + ": " + error.what());
  }
}

std::optional<CameraIntrinsics>
readRosParameterIntrinsics(const fs::path &path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const YAML::Exception &) {
    return std::nullopt;
  }

  const auto rosParameters = findRosParameters(root);
  YAML::Node parameters;
  if (rosParameters) {
    parameters = *rosParameters;
  } else if (root.IsMap() && root["K"] && root["D"] && root["K"].IsSequence() &&
             root["D"].IsSequence()) {
    parameters = root;
  } else {
    return std::nullopt;
  }

  cv::Mat cameraMatrix = readYamlVector(parameters, "K", path);
  if (cameraMatrix.total() != 9) {
    throw std::runtime_error("K must contain exactly nine values in " +
                             path.string());
  }
  cameraMatrix = cameraMatrix.reshape(1, 3).clone();
  return validateIntrinsics(
      CameraIntrinsics{std::move(cameraMatrix),
                       readYamlVector(parameters, "D", path),
                       readYamlDimension(parameters, "width", path),
                       readYamlDimension(parameters, "height", path)},
      path);
}

std::string readOptionalString(const cv::FileStorage &storage,
                               const std::initializer_list<const char *> keys) {
  for (const char *key : keys) {
    const cv::FileNode node = storage[key];
    if (!node.empty()) {
      std::string value;
      node >> value;
      return value;
    }
  }
  return {};
}

double readOptionalDouble(const cv::FileStorage &storage,
                          const std::initializer_list<const char *> keys) {
  for (const char *key : keys) {
    const cv::FileNode node = storage[key];
    if (!node.empty()) {
      double value = std::numeric_limits<double>::quiet_NaN();
      node >> value;
      return value;
    }
  }
  return std::numeric_limits<double>::quiet_NaN();
}

void validateDocument(const CalibrationDocument &document) {
  if (!document.transform.isFinite()) {
    throw std::invalid_argument("Calibration transform is not finite");
  }
  if (document.cameraMatrix.empty() || document.cameraMatrix.rows != 3 ||
      document.cameraMatrix.cols != 3 ||
      !cv::checkRange(document.cameraMatrix)) {
    throw std::invalid_argument("Calibration K must be a finite 3x3 matrix");
  }
  if (document.distortionCoefficients.empty() ||
      document.distortionCoefficients.total() < 4 ||
      (document.distortionCoefficients.rows != 1 &&
       document.distortionCoefficients.cols != 1) ||
      !cv::checkRange(document.distortionCoefficients)) {
    throw std::invalid_argument(
        "Calibration D must be a finite vector with at least four values");
  }
}

fs::path temporaryYamlPath(const fs::path &path, const char *purpose) {
  static std::atomic<std::uint64_t> counter{0};
  const auto ticks =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path parent =
      path.has_parent_path() ? path.parent_path() : fs::current_path();
  return parent / (path.filename().string() + "." + purpose + "." +
                   std::to_string(ticks) + "." +
                   std::to_string(counter.fetch_add(1)) + ".yaml");
}

void removeIgnoringErrors(const fs::path &path) {
  std::error_code error;
  fs::remove(path, error);
}

} // namespace

CalibrationDocument DatasetCalibrationResult::document() const {
  CalibrationDocument value;
  value.transform = calibration.transform;
  value.cameraMatrix = cameraMatrix;
  value.distortionCoefficients = distortionCoefficients;
  value.method = core::toString(calibration.method);
  value.calibrationType = core::toString(calibration.mode);
  value.parentFrame = parentFrame;
  value.childFrame = childFrame;
  value.translationRmsM = calibration.consistency.translationRmsMeters;
  value.rotationRmsDeg = calibration.consistency.rotationRmsDegrees;
  value.rejectedSampleIndices = calibration.rejectedSampleIndices;
  return value;
}

DatasetCalibrationResult
calibrateDataset(const DatasetCalibrationOptions &options) {
  DatasetRepository repository(options.datasetPath);
  std::string validationError;
  if (!repository.validate(options.poseFormat, &validationError, true)) {
    throw std::runtime_error("Invalid calibration dataset: " + validationError);
  }

  const auto inspection = repository.inspect(options.poseFormat);
  const auto rows = repository.poseRows(options.poseFormat);
  std::map<std::size_t, core::RigidTransform> robotPoses;
  for (const PoseRow &row : rows) {
    robotPoses.emplace(row.sampleIndex, row.transform);
  }

  core::TargetDetector detector(options.pattern);
  std::vector<core::TargetObservation> observations;
  std::vector<std::size_t> observationSampleIndices;
  std::vector<std::size_t> targetRejected;
  cv::Size imageSize;
  observations.reserve(inspection.statistics.imageCount);
  observationSampleIndices.reserve(inspection.statistics.imageCount);

  for (std::size_t index = 0; index < inspection.statistics.imageCount;
       ++index) {
    const fs::path imagePath = options.datasetPath / kImagesDirectory /
                               (std::to_string(index) + ".png");
    const cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("Cannot read calibration image " +
                               imagePath.string());
    }
    if (imageSize.empty()) {
      imageSize = image.size();
    } else if (image.size() != imageSize) {
      throw std::runtime_error(
          "Calibration images do not all have the same resolution");
    }
    auto observation = detector.detect(image);
    if (!observation) {
      targetRejected.push_back(index);
      continue;
    }
    observations.push_back(std::move(*observation));
    observationSampleIndices.push_back(index);
  }
  if (observations.empty()) {
    throw std::runtime_error(
        "Calibration target was not detected in any image");
  }

  if (options.intrinsics && options.intrinsics->width > 0 &&
      (options.intrinsics->width != imageSize.width ||
       options.intrinsics->height != imageSize.height)) {
    throw std::runtime_error(
        "Camera intrinsics resolution does not match calibration images");
  }

  const cv::Mat cameraMatrix =
      options.intrinsics ? options.intrinsics->cameraMatrix : cv::Mat{};
  const cv::Mat distortion = options.intrinsics
                                 ? options.intrinsics->distortionCoefficients
                                 : cv::Mat{};
  core::TargetEstimationResult target = core::TargetPoseEstimator::estimate(
      options.pattern, observations, imageSize, cameraMatrix, distortion,
      options.targetOptions);

  std::vector<core::CalibrationSample> samples;
  std::vector<std::size_t> calibrationSampleIndices;
  samples.reserve(target.poses.size());
  calibrationSampleIndices.reserve(target.poses.size());
  for (std::size_t observationIndex = 0; observationIndex < target.poses.size();
       ++observationIndex) {
    const std::size_t sampleIndex = observationSampleIndices[observationIndex];
    if (!target.poses[observationIndex]) {
      targetRejected.push_back(sampleIndex);
      continue;
    }
    const auto pose = robotPoses.find(sampleIndex);
    if (pose == robotPoses.end()) {
      throw std::runtime_error("No robot pose for sample " +
                               std::to_string(sampleIndex));
    }
    samples.push_back(core::CalibrationSample{
        pose->second, target.poses[observationIndex]->targetToCamera});
    calibrationSampleIndices.push_back(sampleIndex);
  }

  core::CalibrationResult calibration =
      core::calibrate(samples, options.calibrationOptions);
  for (core::SampleResidual &residual : calibration.consistency.residuals) {
    residual.sampleIndex = calibrationSampleIndices.at(residual.sampleIndex);
  }
  for (std::size_t &index : calibration.usedSampleIndices) {
    index = calibrationSampleIndices.at(index);
  }
  std::vector<std::size_t> solverRejected;
  solverRejected.reserve(calibration.rejectedSampleIndices.size());
  for (const std::size_t index : calibration.rejectedSampleIndices) {
    solverRejected.push_back(calibrationSampleIndices.at(index));
  }

  std::sort(targetRejected.begin(), targetRejected.end());
  targetRejected.erase(
      std::unique(targetRejected.begin(), targetRejected.end()),
      targetRejected.end());
  calibration.rejectedSampleIndices = targetRejected;
  calibration.rejectedSampleIndices.insert(
      calibration.rejectedSampleIndices.end(), solverRejected.begin(),
      solverRejected.end());
  std::sort(calibration.rejectedSampleIndices.begin(),
            calibration.rejectedSampleIndices.end());
  calibration.rejectedSampleIndices.erase(
      std::unique(calibration.rejectedSampleIndices.begin(),
                  calibration.rejectedSampleIndices.end()),
      calibration.rejectedSampleIndices.end());

  DatasetCalibrationResult result;
  result.calibration = std::move(calibration);
  result.cameraMatrix = std::move(target.cameraMatrix);
  result.distortionCoefficients = std::move(target.distortionCoefficients);
  result.intrinsicCalibrationRmsPixels = target.intrinsicCalibrationRmsPixels;
  result.estimatedIntrinsics = target.estimatedIntrinsics;
  result.targetRejectedSampleIndices = std::move(targetRejected);
  result.parentFrame = options.parentFrame;
  result.childFrame = options.childFrame;
  return result;
}

CameraIntrinsics
CalibrationIO::readIntrinsics(const std::filesystem::path &path) {
  if (const auto rosIntrinsics = readRosParameterIntrinsics(path)) {
    return *rosIntrinsics;
  }
  cv::FileStorage storage(path.string(), cv::FileStorage::READ);
  if (!storage.isOpened()) {
    throw std::runtime_error("Cannot open camera intrinsics " + path.string());
  }
  return readAndValidateIntrinsics(storage, path);
}

CalibrationDocument CalibrationIO::read(const std::filesystem::path &path) {
  cv::FileStorage storage(path.string(), cv::FileStorage::READ);
  if (!storage.isOpened()) {
    throw std::runtime_error("Cannot open calibration " + path.string());
  }

  CameraIntrinsics intrinsics = readAndValidateIntrinsics(storage, path);
  const cv::Mat transform = readMatrix(storage, "T", path);
  if (transform.rows != 4 || transform.cols != 4) {
    throw std::runtime_error("T must be a 4x4 matrix in " + path.string());
  }

  CalibrationDocument document;
  document.transform = core::RigidTransform::fromMatrix(transform);
  document.cameraMatrix = std::move(intrinsics.cameraMatrix);
  document.distortionCoefficients =
      std::move(intrinsics.distortionCoefficients);
  document.method =
      readOptionalString(storage, {"method", "calibration_method"});
  document.calibrationType =
      readOptionalString(storage, {"calibration_type", "type"});
  document.parentFrame = readOptionalString(storage, {"parent_frame"});
  document.childFrame = readOptionalString(storage, {"child_frame"});
  document.translationRmsM =
      readOptionalDouble(storage, {"translation_rms_m", "translation_rms"});
  document.rotationRmsDeg =
      readOptionalDouble(storage, {"rotation_rms_deg", "rotation_rms"});
  const cv::FileNode rejected = storage["rejected_sample_indices"];
  if (!rejected.empty() && rejected.isSeq()) {
    for (const cv::FileNode &entry : rejected) {
      const int index = static_cast<int>(entry);
      if (index >= 0) {
        document.rejectedSampleIndices.push_back(
            static_cast<std::size_t>(index));
      }
    }
  }
  return document;
}

void CalibrationIO::write(const std::filesystem::path &path,
                          const CalibrationDocument &document) {
  validateDocument(document);
  if (path.empty()) {
    throw std::invalid_argument("Calibration output path is empty");
  }
  if (path.has_parent_path()) {
    fs::create_directories(path.parent_path());
  }

  const fs::path staged = temporaryYamlPath(path, "write");
  const fs::path backup = temporaryYamlPath(path, "backup");
  const bool replacing = fs::exists(path);
  try {
    {
      cv::FileStorage storage(staged.string(),
                              cv::FileStorage::WRITE |
                                  cv::FileStorage::FORMAT_YAML);
      if (!storage.isOpened()) {
        throw std::runtime_error("Cannot open " + staged.string() +
                                 " for writing");
      }
      storage << "T" << document.transform.matrix();
      storage << "K" << document.cameraMatrix;
      storage << "D" << document.distortionCoefficients;
      if (!document.method.empty()) {
        storage << "method" << document.method;
      }
      if (!document.calibrationType.empty()) {
        storage << "calibration_type" << document.calibrationType;
      }
      if (!document.parentFrame.empty()) {
        storage << "parent_frame" << document.parentFrame;
      }
      if (!document.childFrame.empty()) {
        storage << "child_frame" << document.childFrame;
      }
      if (std::isfinite(document.translationRmsM)) {
        storage << "translation_rms_m" << document.translationRmsM;
      }
      if (std::isfinite(document.rotationRmsDeg)) {
        storage << "rotation_rms_deg" << document.rotationRmsDeg;
      }
      if (!document.rejectedSampleIndices.empty()) {
        storage << "rejected_sample_indices" << "[";
        for (const std::size_t index : document.rejectedSampleIndices) {
          storage << static_cast<int>(index);
        }
        storage << "]";
      }
      storage.release();
    }

    if (replacing) {
      fs::rename(path, backup);
    }
    try {
      fs::rename(staged, path);
    } catch (...) {
      if (replacing) {
        std::error_code error;
        fs::rename(backup, path, error);
      }
      throw;
    }
    removeIgnoringErrors(backup);
  } catch (...) {
    removeIgnoringErrors(staged);
    throw;
  }
}

} // namespace hand_eye::io
