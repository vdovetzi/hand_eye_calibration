#include "hand_eye_calibration/pattern_detector.hpp"

#include "hand_eye_calibration/helpers/parse_helpers.hpp"
#include <algorithm>
#include <magic_enum.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>

namespace pattern_detector {

#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR <= 6
using LocalDictionaryEnumType = cv::aruco::PREDEFINED_DICTIONARY_NAME;
#else
using LocalDictionaryEnumType = cv::aruco::PredefinedDictionaryType;
#endif

int32_t calculateArucoDictK(const int32_t id) {
  if (id < 0 || id >= 1000) {
    throw std::runtime_error(
        "Error: aruco marker id must be between 0 and 999 inclusive");
  }
  const int c1 = (id >= 50);
  const int c2 = (id >= 100);
  const int c3 = (id >= 250);
  return 50 + c1 * 50 + c2 * 150 + c3 * 750;
}

PatternDetector
PatternDetector::fromPatternInfo(const std::string &patternInfo) {
  std::istringstream stream(patternInfo);
  std::vector<std::string> params;
  std::string param;
  while (stream >> param) {
    params.emplace_back(param);
  }
  if (params.empty()) {
    throw std::runtime_error("Error: pattern detector requires a pattern type");
  }

  const int32_t kindValue = parse_helpers::parseInt(params[0], "pattern type");
  PatternDetector detector;

  switch (static_cast<PatternKind>(kindValue)) {
  case PatternKind::ARUCO: {
    if (params.size() < 4) {
      throw std::runtime_error(
          "Error: wrong number of params to aruco calibration pattern");
    }
    const int32_t dictNum =
        parse_helpers::parseInt(params[1], "aruco dictionary");
    if (dictNum < 4 || dictNum > 7) {
      throw std::runtime_error(
          "Error: wrong dictionary number. Choose one from 4-7");
    }
    detector.kind = PatternKind::ARUCO;
    detector.arucoDictNum = dictNum;
    detector.arucoId = parse_helpers::parseInt(params[2], "aruco marker id");
    detector.arucoMarkerSize =
        parse_helpers::parsePositiveDouble(params[3], "aruco marker size");
    detector.arucoDictK = calculateArucoDictK(detector.arucoId);

    const std::string dictName = "DICT_" + std::to_string(dictNum) + "X" +
                                 std::to_string(dictNum) + "_" +
                                 std::to_string(detector.arucoDictK);
    const std::optional<LocalDictionaryEnumType> dictType =
        magic_enum::enum_cast<LocalDictionaryEnumType>(dictName);
    if (!dictType) {
      throw std::runtime_error("Error: unsupported aruco dictionary " +
                               dictName);
    }

    auto dict = cv::aruco::getPredefinedDictionary(*dictType);
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
    cv::aruco::DetectorParameters detectorParams;
    detector.arucoDetector = cv::aruco::ArucoDetector(dict, detectorParams);
#else
    detector.arucoDict = dict;
    detector.arucoParams = cv::aruco::DetectorParameters::create();
#endif
    const float markerSize = static_cast<float>(detector.arucoMarkerSize);
    detector.objectPoints.emplace_back(-markerSize / 2, markerSize / 2, 0.0f);
    detector.objectPoints.emplace_back(markerSize / 2, markerSize / 2, 0.0f);
    detector.objectPoints.emplace_back(markerSize / 2, -markerSize / 2, 0.0f);
    detector.objectPoints.emplace_back(-markerSize / 2, -markerSize / 2,
                                       0.0f);
    return detector;
  }
  case PatternKind::CHESSBOARD: {
    if (params.size() < 4) {
      throw std::runtime_error(
          "Error: wrong number of params to chessboard calibration pattern");
    }
    const int32_t rows =
        parse_helpers::parsePositiveInt(params[1], "chessboard rows");
    const int32_t cols =
        parse_helpers::parsePositiveInt(params[2], "chessboard cols");
    detector.chessboardCellSize =
        parse_helpers::parsePositiveDouble(params[3], "chessboard cell size");

    detector.kind = PatternKind::CHESSBOARD;
    detector.chessboardDims = cv::Size(cols, rows);
    detector.objectPoints.reserve(detector.chessboardDims.area());
    const float cellSize = static_cast<float>(detector.chessboardCellSize);
    for (int32_t row = 0; row < detector.chessboardDims.height; ++row) {
      for (int32_t col = 0; col < detector.chessboardDims.width; ++col) {
        detector.objectPoints.emplace_back(col * cellSize, row * cellSize,
                                           0.0f);
      }
    }
    return detector;
  }
  default:
    throw std::runtime_error(
        "Error: pattern detection is implemented only for ArUco and "
        "chessboard patterns");
  }
}

std::string PatternDetector::summary() const {
  switch (kind) {
  case PatternKind::ARUCO:
    return "ArUco dict=" + std::to_string(arucoDictNum) + "x" +
           std::to_string(arucoDictNum) + "_" + std::to_string(arucoDictK) +
           " id=" + std::to_string(arucoId) +
           " marker_size=" + std::to_string(arucoMarkerSize);
  case PatternKind::CHESSBOARD:
    return "Chessboard rows=" + std::to_string(chessboardDims.height) +
           " cols=" + std::to_string(chessboardDims.width) +
           " cell_size=" + std::to_string(chessboardCellSize);
  }
  return "Unknown pattern detector";
}

const std::vector<cv::Point3f> &PatternDetector::getObjectPoints() const {
  return objectPoints;
}

cv::Size PatternDetector::getChessboardDims() const {
  return chessboardDims;
}

bool PatternDetector::detectImagePoints(
    const cv::Mat &image, std::vector<cv::Point2f> &imagePoints) const {
  imagePoints.clear();
  switch (kind) {
  case PatternKind::ARUCO: {
    std::vector<int32_t> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
    std::vector<std::vector<cv::Point2f>> rejectedCandidates;
    arucoDetector->detectMarkers(image, markerCorners, markerIds,
                                 rejectedCandidates);
#else
    cv::aruco::detectMarkers(image, arucoDict, markerCorners, markerIds,
                             arucoParams);
#endif
    const auto it = std::find(markerIds.begin(), markerIds.end(), arucoId);
    if (it == markerIds.end()) {
      return false;
    }

    imagePoints = markerCorners[std::distance(markerIds.begin(), it)];
    return true;
  }
  case PatternKind::CHESSBOARD: {
    cv::Mat gray;
    if (image.channels() == 1) {
      gray = image;
    } else if (image.channels() == 4) {
      cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    const int32_t flags = cv::CALIB_CB_NORMALIZE_IMAGE |
                          cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY;
    const bool found =
        cv::findChessboardCorners(gray, chessboardDims, imagePoints, flags);
    if (!found) {
      imagePoints.clear();
      return false;
    }

    cv::cornerSubPix(
        gray, imagePoints, cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                         30, 0.001));
    return true;
  }
  }
  return false;
}

bool PatternDetector::detect(const cv::Mat &image) const {
  std::vector<cv::Point2f> imagePoints;
  return detectImagePoints(image, imagePoints);
}

} // namespace pattern_detector
