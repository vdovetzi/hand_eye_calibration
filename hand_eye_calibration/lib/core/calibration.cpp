#include "lib/core/calibration.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

namespace hand_eye::core {
namespace {

constexpr double kRadiansToDegrees = 180.0 / CV_PI;

cv::Mat asDoubleMatrix(const cv::Mat &value, const int rows, const int cols,
                       const char *name) {
  if (value.empty()) {
    throw std::invalid_argument(std::string(name) + " must not be empty");
  }
  cv::Mat shaped = value;
  if (rows == 3 && cols == 1 && value.rows == 1 && value.cols == 3) {
    shaped = value.t();
  }
  if (shaped.rows != rows || shaped.cols != cols || shaped.channels() != 1) {
    throw std::invalid_argument(std::string(name) + " has an invalid shape");
  }
  cv::Mat result;
  shaped.convertTo(result, CV_64F);
  if (!cv::checkRange(result, true, nullptr)) {
    throw std::invalid_argument(std::string(name) +
                                " contains non-finite values");
  }
  return result;
}

double rotationDistanceDegrees(const cv::Mat &left, const cv::Mat &right) {
  const cv::Mat relative = left.t() * right;
  const double cosine =
      std::clamp((cv::trace(relative)[0] - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(cosine) * kRadiansToDegrees;
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const auto middle =
      values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  const double upper = *middle;
  if (values.size() % 2 != 0) {
    return upper;
  }
  const auto lower = std::max_element(values.begin(), middle);
  return (upper + *lower) / 2.0;
}

std::vector<std::size_t> allIndices(const std::size_t count) {
  std::vector<std::size_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0);
  return indices;
}

const std::vector<std::size_t> &
resolveIndices(const std::vector<CalibrationSample> &samples,
               const std::vector<std::size_t> &requested,
               std::vector<std::size_t> &generated) {
  if (requested.empty()) {
    generated = allIndices(samples.size());
    return generated;
  }
  for (const std::size_t index : requested) {
    if (index >= samples.size()) {
      throw std::out_of_range("calibration sample index is out of range");
    }
  }
  return requested;
}

RigidTransform normalizedRobot(const CalibrationSample &sample,
                               const CalibrationMode mode) {
  return mode == CalibrationMode::EYE_IN_HAND ? sample.robotPose
                                              : sample.robotPose.inverse();
}

cv::HandEyeCalibrationMethod openCvMethod(const CalibrationMethod method) {
  switch (method) {
  case CalibrationMethod::TSAI:
    return cv::CALIB_HAND_EYE_TSAI;
  case CalibrationMethod::PARK:
    return cv::CALIB_HAND_EYE_PARK;
  case CalibrationMethod::HORAUD:
    return cv::CALIB_HAND_EYE_HORAUD;
  case CalibrationMethod::ANDREFF:
    return cv::CALIB_HAND_EYE_ANDREFF;
  case CalibrationMethod::DANIILIDIS:
    return cv::CALIB_HAND_EYE_DANIILIDIS;
  case CalibrationMethod::AUTO:
    break;
  }
  throw std::invalid_argument("AUTO is not an OpenCV hand-eye method");
}

RigidTransform solveOne(const std::vector<CalibrationSample> &samples,
                        const std::vector<std::size_t> &indices,
                        const CalibrationMode mode,
                        const CalibrationMethod method) {
  std::vector<cv::Mat> robotRotations;
  std::vector<cv::Mat> robotTranslations;
  std::vector<cv::Mat> targetRotations;
  std::vector<cv::Mat> targetTranslations;
  robotRotations.reserve(indices.size());
  robotTranslations.reserve(indices.size());
  targetRotations.reserve(indices.size());
  targetTranslations.reserve(indices.size());

  for (const std::size_t index : indices) {
    const RigidTransform robot = normalizedRobot(samples[index], mode);
    robotRotations.push_back(robot.rotation);
    robotTranslations.push_back(robot.translation);
    targetRotations.push_back(samples[index].targetToCamera.rotation);
    targetTranslations.push_back(samples[index].targetToCamera.translation);
  }

  cv::Mat resultRotation;
  cv::Mat resultTranslation;
  cv::calibrateHandEye(robotRotations, robotTranslations, targetRotations,
                       targetTranslations, resultRotation, resultTranslation,
                       openCvMethod(method));
  RigidTransform result(resultRotation, resultTranslation);
  const double determinant = cv::determinant(result.rotation);
  if (!result.isFinite() || determinant < 0.5 || determinant > 1.5) {
    throw std::runtime_error("hand-eye method produced an invalid transform");
  }
  return result;
}

double candidateScore(const ConsistencyMetrics &metrics) {
  return metrics.translationRmsMeters +
         metrics.rotationRmsDegrees / kRadiansToDegrees;
}

std::pair<CalibrationMethod, RigidTransform>
solveBest(const std::vector<CalibrationSample> &samples,
          const std::vector<std::size_t> &indices,
          const CalibrationOptions &options) {
  static constexpr std::array kMethods{
      CalibrationMethod::TSAI, CalibrationMethod::PARK,
      CalibrationMethod::HORAUD, CalibrationMethod::ANDREFF,
      CalibrationMethod::DANIILIDIS};

  std::vector<CalibrationMethod> methods;
  if (options.method == CalibrationMethod::AUTO) {
    methods.assign(kMethods.begin(), kMethods.end());
  } else {
    methods.push_back(options.method);
  }

  double bestScore = std::numeric_limits<double>::infinity();
  std::optional<std::pair<CalibrationMethod, RigidTransform>> best;
  std::string failures;
  for (const CalibrationMethod method : methods) {
    try {
      RigidTransform transform =
          solveOne(samples, indices, options.mode, method);
      const ConsistencyMetrics metrics =
          evaluateConsistency(samples, transform, options.mode, indices);
      const double score = candidateScore(metrics);
      if (std::isfinite(score) && score < bestScore) {
        bestScore = score;
        best = std::make_pair(method, std::move(transform));
      }
    } catch (const std::exception &error) {
      if (!failures.empty()) {
        failures += "; ";
      }
      failures += toString(method) + ": " + error.what();
    }
  }
  if (!best) {
    throw std::runtime_error("all hand-eye methods failed: " + failures);
  }
  return *best;
}

std::vector<std::size_t> findOutliers(const ConsistencyMetrics &metrics,
                                      const CalibrationOptions &options) {
  std::vector<double> translations;
  std::vector<double> rotations;
  translations.reserve(metrics.residuals.size());
  rotations.reserve(metrics.residuals.size());
  for (const SampleResidual &residual : metrics.residuals) {
    translations.push_back(residual.translationMeters);
    rotations.push_back(residual.rotationDegrees);
  }
  const double translationMedian = median(translations);
  const double rotationMedian = median(rotations);
  for (double &value : translations) {
    value = std::abs(value - translationMedian);
  }
  for (double &value : rotations) {
    value = std::abs(value - rotationMedian);
  }
  constexpr double kMadToSigma = 1.4826;
  const double translationThreshold =
      std::max(options.minimumTranslationOutlierThresholdMeters,
               translationMedian + options.outlierMadScale * kMadToSigma *
                                       median(translations));
  const double rotationThreshold =
      std::max(options.minimumRotationOutlierThresholdDegrees,
               rotationMedian +
                   options.outlierMadScale * kMadToSigma * median(rotations));

  std::vector<std::size_t> outliers;
  for (const SampleResidual &residual : metrics.residuals) {
    if (residual.translationMeters > translationThreshold ||
        residual.rotationDegrees > rotationThreshold) {
      outliers.push_back(residual.sampleIndex);
    }
  }
  return outliers;
}

std::string normalizedText(const std::string_view text) {
  std::string result;
  result.reserve(text.size());
  for (const char character : text) {
    const unsigned char value = static_cast<unsigned char>(character);
    result.push_back(character == '-' ? '_'
                                      : static_cast<char>(std::tolower(value)));
  }
  return result;
}

} // namespace

RigidTransform::RigidTransform()
    : rotation(cv::Mat::eye(3, 3, CV_64F)),
      translation(cv::Mat::zeros(3, 1, CV_64F)) {}

RigidTransform::RigidTransform(const cv::Mat &rotationValue,
                               const cv::Mat &translationValue)
    : rotation(asDoubleMatrix(rotationValue, 3, 3, "rotation")),
      translation(asDoubleMatrix(translationValue, 3, 1, "translation")) {}

RigidTransform RigidTransform::identity() { return {}; }

RigidTransform RigidTransform::fromMatrix(const cv::Mat &value) {
  const cv::Mat matrixValue = asDoubleMatrix(value, 4, 4, "transform");
  const cv::Mat expected = (cv::Mat_<double>(1, 4) << 0.0, 0.0, 0.0, 1.0);
  if (cv::norm(matrixValue.row(3), expected, cv::NORM_INF) > 1e-9) {
    throw std::invalid_argument("transform has an invalid homogeneous row");
  }
  return {matrixValue(cv::Rect(0, 0, 3, 3)), matrixValue(cv::Rect(3, 0, 1, 3))};
}

cv::Mat RigidTransform::matrix() const {
  cv::Mat result = cv::Mat::eye(4, 4, CV_64F);
  rotation.copyTo(result(cv::Rect(0, 0, 3, 3)));
  translation.copyTo(result(cv::Rect(3, 0, 1, 3)));
  return result;
}

RigidTransform RigidTransform::inverse() const {
  const cv::Mat inverseRotation = rotation.t();
  return {inverseRotation, -inverseRotation * translation};
}

bool RigidTransform::isFinite() const {
  return rotation.rows == 3 && rotation.cols == 3 && translation.rows == 3 &&
         translation.cols == 1 && cv::checkRange(rotation, true, nullptr) &&
         cv::checkRange(translation, true, nullptr);
}

RigidTransform operator*(const RigidTransform &left,
                         const RigidTransform &right) {
  return {left.rotation * right.rotation,
          left.rotation * right.translation + left.translation};
}

ConsistencyMetrics
evaluateConsistency(const std::vector<CalibrationSample> &samples,
                    const RigidTransform &calibration,
                    const CalibrationMode mode,
                    const std::vector<std::size_t> &sampleIndices) {
  std::vector<std::size_t> generated;
  const auto &indices = resolveIndices(samples, sampleIndices, generated);
  ConsistencyMetrics metrics;
  if (indices.empty()) {
    return metrics;
  }

  std::vector<RigidTransform> invariants;
  invariants.reserve(indices.size());
  for (const std::size_t index : indices) {
    invariants.push_back(normalizedRobot(samples[index], mode) * calibration *
                         samples[index].targetToCamera);
  }

  std::size_t medoid = 0;
  double lowestCost = std::numeric_limits<double>::infinity();
  for (std::size_t candidate = 0; candidate < invariants.size(); ++candidate) {
    double cost = 0.0;
    for (const RigidTransform &value : invariants) {
      cost += cv::norm(invariants[candidate].translation - value.translation) +
              rotationDistanceDegrees(invariants[candidate].rotation,
                                      value.rotation) /
                  kRadiansToDegrees;
    }
    if (cost < lowestCost) {
      lowestCost = cost;
      medoid = candidate;
    }
  }

  double translationSquares = 0.0;
  double rotationSquares = 0.0;
  metrics.residuals.reserve(indices.size());
  for (std::size_t position = 0; position < indices.size(); ++position) {
    const double translation = cv::norm(invariants[position].translation -
                                        invariants[medoid].translation);
    const double rotation = rotationDistanceDegrees(
        invariants[medoid].rotation, invariants[position].rotation);
    metrics.residuals.push_back({indices[position], translation, rotation});
    translationSquares += translation * translation;
    rotationSquares += rotation * rotation;
    metrics.translationMaxMeters =
        std::max(metrics.translationMaxMeters, translation);
    metrics.rotationMaxDegrees = std::max(metrics.rotationMaxDegrees, rotation);
  }
  metrics.translationRmsMeters =
      std::sqrt(translationSquares / static_cast<double>(indices.size()));
  metrics.rotationRmsDegrees =
      std::sqrt(rotationSquares / static_cast<double>(indices.size()));
  return metrics;
}

double rotationSpanDegrees(const std::vector<CalibrationSample> &samples,
                           const std::vector<std::size_t> &sampleIndices) {
  std::vector<std::size_t> generated;
  const auto &indices = resolveIndices(samples, sampleIndices, generated);
  double span = 0.0;
  for (std::size_t left = 0; left < indices.size(); ++left) {
    for (std::size_t right = left + 1; right < indices.size(); ++right) {
      span = std::max(span, rotationDistanceDegrees(
                                samples[indices[left]].robotPose.rotation,
                                samples[indices[right]].robotPose.rotation));
    }
  }
  return span;
}

double
rotationAxisSeparationDegrees(const std::vector<CalibrationSample> &samples,
                              const std::vector<std::size_t> &sampleIndices) {
  std::vector<std::size_t> generated;
  const auto &indices = resolveIndices(samples, sampleIndices, generated);
  std::vector<cv::Vec3d> axes;
  constexpr double kMinimumMotionRadians = CV_PI / 180.0;
  for (std::size_t left = 0; left < indices.size(); ++left) {
    for (std::size_t right = left + 1; right < indices.size(); ++right) {
      const cv::Mat relative = samples[indices[left]].robotPose.rotation.t() *
                               samples[indices[right]].robotPose.rotation;
      cv::Mat rotationVector;
      cv::Rodrigues(relative, rotationVector);
      const double angle = cv::norm(rotationVector);
      if (angle < kMinimumMotionRadians) {
        continue;
      }
      axes.emplace_back(rotationVector.at<double>(0) / angle,
                        rotationVector.at<double>(1) / angle,
                        rotationVector.at<double>(2) / angle);
    }
  }

  double maximumSine = 0.0;
  for (std::size_t left = 0; left < axes.size(); ++left) {
    for (std::size_t right = left + 1; right < axes.size(); ++right) {
      maximumSine =
          std::max(maximumSine, cv::norm(axes[left].cross(axes[right])));
    }
  }
  return std::asin(std::clamp(maximumSine, 0.0, 1.0)) * kRadiansToDegrees;
}

CalibrationResult calibrate(const std::vector<CalibrationSample> &samples,
                            const CalibrationOptions &options) {
  if (options.minimumSamples < kMinimumCalibrationSamples) {
    throw std::invalid_argument("minimumSamples must be at least " +
                                std::to_string(kMinimumCalibrationSamples));
  }
  if (samples.size() < options.minimumSamples) {
    throw std::invalid_argument(
        "not enough calibration samples: need at least " +
        std::to_string(options.minimumSamples));
  }
  if (!std::isfinite(options.minimumRotationSpanDegrees) ||
      options.minimumRotationSpanDegrees < 0.0 ||
      !std::isfinite(options.minimumRotationAxisSeparationDegrees) ||
      options.minimumRotationAxisSeparationDegrees < 0.0 ||
      options.minimumRotationAxisSeparationDegrees > 90.0 ||
      !std::isfinite(options.outlierMadScale) ||
      options.outlierMadScale < 0.0 ||
      !std::isfinite(options.minimumTranslationOutlierThresholdMeters) ||
      options.minimumTranslationOutlierThresholdMeters < 0.0 ||
      !std::isfinite(options.minimumRotationOutlierThresholdDegrees) ||
      options.minimumRotationOutlierThresholdDegrees < 0.0) {
    throw std::invalid_argument(
        "calibration thresholds must be finite and non-negative");
  }
  std::vector<std::size_t> used = allIndices(samples.size());
  std::vector<std::size_t> rejected;

  CalibrationMethod selectedMethod = CalibrationMethod::AUTO;
  RigidTransform selectedTransform;
  ConsistencyMetrics metrics;
  for (std::size_t iteration = 0;; ++iteration) {
    const double span = rotationSpanDegrees(samples, used);
    if (span < options.minimumRotationSpanDegrees) {
      throw std::invalid_argument(
          "robot poses are not observable: rotation span is " +
          std::to_string(span) + " deg, required " +
          std::to_string(options.minimumRotationSpanDegrees) + " deg");
    }
    const double axisSeparation = rotationAxisSeparationDegrees(samples, used);
    if (axisSeparation < options.minimumRotationAxisSeparationDegrees) {
      throw std::invalid_argument(
          "robot poses are not observable: rotation axes differ by only " +
          std::to_string(axisSeparation) + " deg, required " +
          std::to_string(options.minimumRotationAxisSeparationDegrees) +
          " deg");
    }
    auto solution = solveBest(samples, used, options);
    selectedMethod = solution.first;
    selectedTransform = std::move(solution.second);
    metrics =
        evaluateConsistency(samples, selectedTransform, options.mode, used);

    if (!options.rejectOutliers ||
        iteration >= options.maximumOutlierIterations) {
      break;
    }
    const std::vector<std::size_t> outliers = findOutliers(metrics, options);
    if (outliers.empty()) {
      break;
    }
    if (used.size() - outliers.size() < options.minimumSamples) {
      throw std::runtime_error(
          "outlier rejection left fewer than the required samples");
    }
    for (const std::size_t index : outliers) {
      rejected.push_back(index);
    }
    used.erase(std::remove_if(used.begin(), used.end(),
                              [&](const auto index) {
                                return std::find(outliers.begin(),
                                                 outliers.end(),
                                                 index) != outliers.end();
                              }),
               used.end());
  }

  std::sort(rejected.begin(), rejected.end());
  rejected.erase(std::unique(rejected.begin(), rejected.end()), rejected.end());
  return {selectedTransform, selectedMethod, options.mode,
          metrics,           used,           rejected};
}

std::string toString(const CalibrationMethod method) {
  switch (method) {
  case CalibrationMethod::AUTO:
    return "auto";
  case CalibrationMethod::TSAI:
    return "tsai";
  case CalibrationMethod::PARK:
    return "park";
  case CalibrationMethod::HORAUD:
    return "horaud";
  case CalibrationMethod::ANDREFF:
    return "andreff";
  case CalibrationMethod::DANIILIDIS:
    return "daniilidis";
  }
  throw std::invalid_argument("unknown calibration method");
}

std::string toString(const CalibrationMode mode) {
  return mode == CalibrationMode::EYE_IN_HAND ? "eye_in_hand" : "eye_to_hand";
}

CalibrationMethod parseCalibrationMethod(const std::string_view text) {
  const std::string value = normalizedText(text);
  if (value == "auto") {
    return CalibrationMethod::AUTO;
  }
  if (value == "tsai") {
    return CalibrationMethod::TSAI;
  }
  if (value == "park") {
    return CalibrationMethod::PARK;
  }
  if (value == "horaud") {
    return CalibrationMethod::HORAUD;
  }
  if (value == "andreff") {
    return CalibrationMethod::ANDREFF;
  }
  if (value == "daniilidis") {
    return CalibrationMethod::DANIILIDIS;
  }
  throw std::invalid_argument("unsupported calibration method: " +
                              std::string(text));
}

CalibrationMode parseCalibrationMode(const std::string_view text) {
  const std::string value = normalizedText(text);
  if (value == "eye_in_hand" || value == "eih") {
    return CalibrationMode::EYE_IN_HAND;
  }
  if (value == "eye_to_hand" || value == "eth") {
    return CalibrationMode::EYE_TO_HAND;
  }
  throw std::invalid_argument("unsupported calibration mode: " +
                              std::string(text));
}

} // namespace hand_eye::core
