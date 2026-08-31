#include "lib/core/target.hpp"

#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace hand_eye::core {
namespace {

#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR <= 6
using DictionaryType = cv::aruco::PREDEFINED_DICTIONARY_NAME;
#else
using DictionaryType = cv::aruco::PredefinedDictionaryType;
#endif

std::vector<std::string> split(const std::string_view text) {
  std::istringstream input{std::string(text)};
  std::vector<std::string> values;
  for (std::string value; input >> value;) {
    values.push_back(std::move(value));
  }
  return values;
}

int parseInt(const std::string &text, const char *name) {
  std::size_t consumed = 0;
  int value = 0;
  try {
    value = std::stoi(text, &consumed);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string(name) + " must be an integer");
  }
  if (consumed != text.size()) {
    throw std::invalid_argument(std::string(name) + " must be an integer");
  }
  return value;
}

double parsePositive(const std::string &text, const char *name) {
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string(name) + " must be a number");
  }
  if (consumed != text.size() || !std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}

void validateConfig(const PatternConfig &config) {
  if (config.dictionarySize < 4 || config.dictionarySize > 7) {
    throw std::invalid_argument("dictionary size must be between 4 and 7");
  }
  switch (config.type) {
  case PatternType::ARUCO:
    if (config.markerId < 0 || config.markerId >= 1000 ||
        config.markerSizeMeters <= 0.0) {
      throw std::invalid_argument("invalid ArUco marker ID or size");
    }
    break;
  case PatternType::CHESSBOARD:
    if (config.rows < 2 || config.columns < 2 ||
        config.squareSizeMeters <= 0.0) {
      throw std::invalid_argument("invalid chessboard dimensions or cell size");
    }
    break;
  case PatternType::CHARUCO:
    if (config.rows < 2 || config.columns < 2 ||
        config.squareSizeMeters <= 0.0 || config.markerSizeMeters <= 0.0 ||
        config.markerSizeMeters >= config.squareSizeMeters) {
      throw std::invalid_argument("invalid ChArUco board dimensions or sizes");
    }
    break;
  }
}

cv::Ptr<cv::aruco::Dictionary> dictionary(const int bits,
                                          const int maximumMarkerId) {
  if (maximumMarkerId < 0 || maximumMarkerId >= 1000) {
    throw std::invalid_argument("dictionary requires fewer than 1000 markers");
  }
  const int capacityOffset = maximumMarkerId < 50    ? 0
                             : maximumMarkerId < 100 ? 1
                             : maximumMarkerId < 250 ? 2
                                                     : 3;
  const int enumValue = (bits - 4) * 4 + capacityOffset;
  return cv::aruco::getPredefinedDictionary(
      static_cast<DictionaryType>(enumValue));
}

cv::Mat grayscale(const cv::Mat &image) {
  if (image.empty()) {
    throw std::invalid_argument("target image must not be empty");
  }
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    throw std::invalid_argument(
        "target image has an unsupported channel count");
  }
  return gray;
}

std::vector<cv::Point3f> chessboardPoints(const PatternConfig &config) {
  std::vector<cv::Point3f> points;
  points.reserve(static_cast<std::size_t>(config.rows * config.columns));
  const float size = static_cast<float>(config.squareSizeMeters);
  for (int row = 0; row < config.rows; ++row) {
    for (int column = 0; column < config.columns; ++column) {
      points.emplace_back(column * size, row * size, 0.0F);
    }
  }
  return points;
}

std::vector<cv::Point3f> markerPoints(const double markerSize) {
  const float half = static_cast<float>(markerSize / 2.0);
  return {{-half, half, 0.0F},
          {half, half, 0.0F},
          {half, -half, 0.0F},
          {-half, -half, 0.0F}};
}

void detectMarkers(const cv::Mat &image,
                   const cv::Ptr<cv::aruco::Dictionary> &arucoDictionary,
                   std::vector<std::vector<cv::Point2f>> &corners,
                   std::vector<int> &ids) {
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
  cv::aruco::ArucoDetector detector(*arucoDictionary);
  detector.detectMarkers(image, corners, ids);
#else
  cv::aruco::detectMarkers(image, arucoDictionary, corners, ids,
                           cv::aruco::DetectorParameters::create());
#endif
}

double reprojectionRms(const TargetObservation &observation,
                       const cv::Mat &rotationVector,
                       const cv::Mat &translationVector,
                       const cv::Mat &cameraMatrix, const cv::Mat &distortion) {
  std::vector<cv::Point2f> projected;
  cv::projectPoints(observation.objectPoints, rotationVector, translationVector,
                    cameraMatrix, distortion, projected);
  double squared = 0.0;
  for (std::size_t index = 0; index < projected.size(); ++index) {
    const cv::Point2f delta = projected[index] - observation.imagePoints[index];
    squared += delta.dot(delta);
  }
  return std::sqrt(squared / static_cast<double>(projected.size()));
}

std::optional<TargetPose> solvePose(const PatternConfig &config,
                                    const TargetObservation &observation,
                                    const cv::Mat &cameraMatrix,
                                    const cv::Mat &distortion,
                                    const TargetPoseOptions &options) {
  if (observation.imagePoints.size() != observation.objectPoints.size() ||
      observation.imagePoints.size() < 4) {
    throw std::invalid_argument(
        "target observation requires at least four paired points");
  }
  cv::Mat rotationVector;
  cv::Mat translationVector;
  double error = std::numeric_limits<double>::infinity();

  if (config.type == PatternType::ARUCO) {
    std::vector<cv::Mat> rotationVectors;
    std::vector<cv::Mat> translationVectors;
    cv::solvePnPGeneric(observation.objectPoints, observation.imagePoints,
                        cameraMatrix, distortion, rotationVectors,
                        translationVectors, false, cv::SOLVEPNP_IPPE_SQUARE);
    std::vector<std::pair<double, std::size_t>> candidates;
    for (std::size_t index = 0; index < rotationVectors.size(); ++index) {
      if (translationVectors[index].at<double>(2) <= 0.0) {
        continue;
      }
      candidates.emplace_back(
          reprojectionRms(observation, rotationVectors[index],
                          translationVectors[index], cameraMatrix, distortion),
          index);
    }
    std::sort(candidates.begin(), candidates.end());
    if (candidates.empty()) {
      return std::nullopt;
    }
    if (candidates.size() > 1 &&
        candidates[1].first <=
            candidates[0].first * options.arucoAmbiguityRatio + 1e-9) {
      return std::nullopt;
    }
    error = candidates[0].first;
    rotationVector = rotationVectors[candidates[0].second];
    translationVector = translationVectors[candidates[0].second];
  } else {
    if (!cv::solvePnP(observation.objectPoints, observation.imagePoints,
                      cameraMatrix, distortion, rotationVector,
                      translationVector, false, cv::SOLVEPNP_ITERATIVE)) {
      return std::nullopt;
    }
    error = reprojectionRms(observation, rotationVector, translationVector,
                            cameraMatrix, distortion);
  }

  if (!std::isfinite(error) || error > options.maximumReprojectionErrorPixels) {
    return std::nullopt;
  }
  cv::Mat rotation;
  cv::Rodrigues(rotationVector, rotation);
  return TargetPose{RigidTransform(rotation, translationVector), error};
}

} // namespace

PatternConfig parsePattern(const std::string_view patternInfo) {
  const std::vector<std::string> values = split(patternInfo);
  if (values.empty()) {
    throw std::invalid_argument("pattern configuration must not be empty");
  }
  PatternConfig config;
  const int type = parseInt(values[0], "pattern type");
  if (type < 1 || type > 3) {
    throw std::invalid_argument("pattern type must be 1, 2 or 3");
  }
  config.type = static_cast<PatternType>(type);
  if (config.type == PatternType::ARUCO) {
    if (values.size() != 4) {
      throw std::invalid_argument(
          "ArUco format is: 1 dictionary id marker_size");
    }
    config.dictionarySize = parseInt(values[1], "dictionary size");
    config.markerId = parseInt(values[2], "marker ID");
    config.markerSizeMeters = parsePositive(values[3], "marker size");
  } else if (config.type == PatternType::CHESSBOARD) {
    if (values.size() != 4) {
      throw std::invalid_argument(
          "chessboard format is: 2 rows columns cell_size");
    }
    config.rows = parseInt(values[1], "rows");
    config.columns = parseInt(values[2], "columns");
    config.squareSizeMeters = parsePositive(values[3], "cell size");
  } else {
    if (values.size() != 6 && values.size() != 7) {
      throw std::invalid_argument("ChArUco format is: 3 rows columns "
                                  "square_size marker_size dictionary");
    }
    config.rows = parseInt(values[1], "rows");
    config.columns = parseInt(values[2], "columns");
    config.squareSizeMeters = parsePositive(values[3], "square size");
    config.markerSizeMeters = parsePositive(values[4], "marker size");
    config.dictionarySize = parseInt(values[5], "dictionary size");
    if (values.size() == 7) {
      config.markerId = parseInt(values[6], "legacy marker ID");
    }
  }
  validateConfig(config);
  return config;
}

std::string describePattern(const PatternConfig &config) {
  validateConfig(config);
  std::ostringstream output;
  if (config.type == PatternType::ARUCO) {
    output << "ArUco " << config.dictionarySize << "x" << config.dictionarySize
           << " id=" << config.markerId << " size=" << config.markerSizeMeters;
  } else if (config.type == PatternType::CHESSBOARD) {
    output << "Chessboard " << config.rows << "x" << config.columns
           << " cell=" << config.squareSizeMeters;
  } else {
    output << "ChArUco " << config.rows << "x" << config.columns
           << " square=" << config.squareSizeMeters
           << " marker=" << config.markerSizeMeters
           << " dictionary=" << config.dictionarySize << "x"
           << config.dictionarySize;
  }
  return output.str();
}

int arucoDictionaryCapacity(const PatternConfig &config) {
  validateConfig(config);
  if (config.type != PatternType::ARUCO) {
    throw std::invalid_argument("dictionary capacity requires an ArUco target");
  }
  if (config.markerId < 50) {
    return 50;
  }
  if (config.markerId < 100) {
    return 100;
  }
  if (config.markerId < 250) {
    return 250;
  }
  return 1000;
}

cv::Mat canonicalArucoMarker(const PatternConfig &config) {
  validateConfig(config);
  if (config.type != PatternType::ARUCO) {
    throw std::invalid_argument("canonical marker requires an ArUco target");
  }
  const auto arucoDictionary =
      dictionary(config.dictionarySize, config.markerId);
  cv::Mat modules;
  cv::aruco::drawMarker(arucoDictionary, config.markerId,
                        arucoDictionary->markerSize + 2, modules, 1);
  if (modules.type() != CV_8UC1 || modules.rows != config.dictionarySize + 2 ||
      modules.cols != config.dictionarySize + 2) {
    throw std::runtime_error("OpenCV returned an invalid ArUco marker image");
  }
  return modules;
}

TargetDetector::TargetDetector(PatternConfig config)
    : config_(std::move(config)) {
  validateConfig(config_);
}

const PatternConfig &TargetDetector::config() const { return config_; }

std::optional<TargetObservation>
TargetDetector::detect(const cv::Mat &image) const {
  cv::Mat gray = grayscale(image);
  TargetObservation observation;
  if (config_.type == PatternType::CHESSBOARD) {
    const cv::Size dimensions(config_.columns, config_.rows);
    const int flags = cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE |
                      cv::CALIB_CB_ACCURACY;
    if (!cv::findChessboardCorners(gray, dimensions, observation.imagePoints,
                                   flags)) {
      return std::nullopt;
    }
    cv::cornerSubPix(
        gray, observation.imagePoints, cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30,
                         0.001));
    observation.objectPoints = chessboardPoints(config_);
    return observation;
  }

  const int maximumId = config_.type == PatternType::ARUCO
                            ? config_.markerId
                            : (config_.rows * config_.columns + 1) / 2 - 1;
  const auto arucoDictionary = dictionary(config_.dictionarySize, maximumId);
  std::vector<std::vector<cv::Point2f>> markerCorners;
  std::vector<int> markerIds;
  detectMarkers(gray, arucoDictionary, markerCorners, markerIds);
  if (markerIds.empty()) {
    return std::nullopt;
  }

  if (config_.type == PatternType::ARUCO) {
    const auto marker =
        std::find(markerIds.begin(), markerIds.end(), config_.markerId);
    if (marker == markerIds.end()) {
      return std::nullopt;
    }
    observation.imagePoints =
        markerCorners[static_cast<std::size_t>(marker - markerIds.begin())];
    cv::cornerSubPix(
        gray, observation.imagePoints, cv::Size(3, 3), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 20,
                         0.01));
    observation.objectPoints = markerPoints(config_.markerSizeMeters);
    return observation;
  }

  auto board = cv::aruco::CharucoBoard::create(
      config_.columns, config_.rows,
      static_cast<float>(config_.squareSizeMeters),
      static_cast<float>(config_.markerSizeMeters), arucoDictionary);
  std::vector<cv::Point2f> charucoCorners;
  std::vector<int> charucoIds;
  if (cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, gray,
                                           board, charucoCorners,
                                           charucoIds) < 4) {
    return std::nullopt;
  }
  cv::cornerSubPix(
      gray, charucoCorners, cv::Size(3, 3), cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 20,
                       0.01));
  observation.imagePoints = std::move(charucoCorners);
  observation.objectPoints.reserve(charucoIds.size());
  for (const int id : charucoIds) {
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
    observation.objectPoints.push_back(
        board->getChessboardCorners().at(static_cast<std::size_t>(id)));
#else
    observation.objectPoints.push_back(
        board->chessboardCorners.at(static_cast<std::size_t>(id)));
#endif
  }
  return observation;
}

void TargetDetector::draw(cv::Mat &image,
                          const TargetObservation &observation) const {
  if (config_.type == PatternType::CHESSBOARD) {
    cv::drawChessboardCorners(image, cv::Size(config_.columns, config_.rows),
                              observation.imagePoints, true);
  } else if (config_.type == PatternType::ARUCO) {
    const std::vector<std::vector<cv::Point2f>> corners{
        observation.imagePoints};
    cv::aruco::drawDetectedMarkers(image, corners,
                                   std::vector<int>{config_.markerId});
  } else {
    for (const cv::Point2f &point : observation.imagePoints) {
      cv::circle(image, point, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
  }
}

TargetEstimationResult TargetPoseEstimator::estimate(
    const PatternConfig &config,
    const std::vector<TargetObservation> &observations,
    const cv::Size &imageSize, const cv::Mat &cameraMatrix,
    const cv::Mat &distortionCoefficients, const TargetPoseOptions &options) {
  validateConfig(config);
  if (observations.empty()) {
    throw std::invalid_argument("at least one target observation is required");
  }
  if (imageSize.width <= 0 || imageSize.height <= 0) {
    throw std::invalid_argument("image size must be positive");
  }
  if (!std::isfinite(options.maximumReprojectionErrorPixels) ||
      options.maximumReprojectionErrorPixels <= 0.0 ||
      options.arucoAmbiguityRatio <= 1.0) {
    throw std::invalid_argument("target-pose thresholds are invalid");
  }

  TargetEstimationResult result;
  if (cameraMatrix.empty()) {
    if (config.type == PatternType::ARUCO) {
      throw std::invalid_argument(
          "ArUco pose estimation requires known camera intrinsics");
    }
    if (observations.size() < options.minimumViewsForIntrinsicCalibration) {
      throw std::invalid_argument(
          "at least " +
          std::to_string(options.minimumViewsForIntrinsicCalibration) +
          " views are required to estimate camera intrinsics");
    }
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    for (const TargetObservation &observation : observations) {
      if (observation.imagePoints.size() != observation.objectPoints.size() ||
          observation.imagePoints.size() < 4) {
        throw std::invalid_argument(
            "target observations contain unpaired points");
      }
      objectPoints.push_back(observation.objectPoints);
      imagePoints.push_back(observation.imagePoints);
    }
    std::vector<cv::Mat> rotationVectors;
    std::vector<cv::Mat> translationVectors;
    result.intrinsicCalibrationRmsPixels = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize, result.cameraMatrix,
        result.distortionCoefficients, rotationVectors, translationVectors);
    result.estimatedIntrinsics = true;
  } else {
    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3 ||
        cameraMatrix.channels() != 1) {
      throw std::invalid_argument(
          "camera matrix must be a single-channel 3x3 matrix");
    }
    cameraMatrix.convertTo(result.cameraMatrix, CV_64F);
    if (!cv::checkRange(result.cameraMatrix, true, nullptr) ||
        result.cameraMatrix.at<double>(0, 0) <= 0.0 ||
        result.cameraMatrix.at<double>(1, 1) <= 0.0) {
      throw std::invalid_argument(
          "camera matrix must contain finite positive focal lengths");
    }
    if (distortionCoefficients.empty()) {
      result.distortionCoefficients = cv::Mat::zeros(5, 1, CV_64F);
    } else {
      if ((distortionCoefficients.rows != 1 &&
           distortionCoefficients.cols != 1) ||
          distortionCoefficients.total() < 4) {
        throw std::invalid_argument("distortion coefficients must be a vector "
                                    "with at least four values");
      }
      distortionCoefficients.convertTo(result.distortionCoefficients, CV_64F);
      if (!cv::checkRange(result.distortionCoefficients, true, nullptr)) {
        throw std::invalid_argument(
            "distortion coefficients must contain only finite values");
      }
    }
  }

  result.poses.reserve(observations.size());
  for (std::size_t index = 0; index < observations.size(); ++index) {
    auto pose = solvePose(config, observations[index], result.cameraMatrix,
                          result.distortionCoefficients, options);
    if (!pose) {
      result.rejectedViewIndices.push_back(index);
    }
    result.poses.push_back(std::move(pose));
  }
  return result;
}

} // namespace hand_eye::core
