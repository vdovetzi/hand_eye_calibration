#pragma once

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <optional>

namespace pattern_detector {

enum class PatternKind { ARUCO = 1, CHESSBOARD = 2 };

int32_t calculateArucoDictK(int32_t id);

struct PatternDetector {
  PatternKind kind;
  int32_t arucoId = 0;
  int32_t arucoDictNum = 0;
  int32_t arucoDictK = 0;
  double arucoMarkerSize = 0.0;
  double chessboardCellSize = 0.0;
  cv::Size chessboardDims;
  std::vector<cv::Point3f> objectPoints;
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
  std::optional<cv::aruco::ArucoDetector> arucoDetector;
#else
  cv::Ptr<cv::aruco::Dictionary> arucoDict;
  cv::Ptr<cv::aruco::DetectorParameters> arucoParams;
#endif

  static PatternDetector fromPatternInfo(const std::string &patternInfo);

  std::string summary() const;
  const std::vector<cv::Point3f> &getObjectPoints() const;
  cv::Size getChessboardDims() const;
  bool detectImagePoints(const cv::Mat &image,
                         std::vector<cv::Point2f> &imagePoints) const;
  bool detect(const cv::Mat &image) const;
};

} // namespace pattern_detector
