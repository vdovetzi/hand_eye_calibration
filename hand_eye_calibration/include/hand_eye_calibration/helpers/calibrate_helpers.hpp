#pragma once

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "hand_eye_calibration/helpers/dataset_helpers.hpp"
#include "hand_eye_calibration/pattern_detector.hpp"
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Transform.h>

using geometry_msgs::msg::PoseStamped;

namespace calibration_helpers {

constexpr const char *HELPER_LOGGERNAME = "helper";

struct CalibrationData {
  cv::Mat K;
  cv::Mat D;

  std::vector<cv::Mat> R_target2cam;
  std::vector<cv::Mat> t_target2cam;
  std::vector<cv::Mat> R_gripper2base;
  std::vector<cv::Mat> t_gripper2base;

  cv::Mat R_cam2gripper;
  cv::Mat t_cam2gripper;
  cv::Mat T;

  std::unordered_set<size_t> rejectedImages;

  bool readIntrinsics(const fs::path &path);
  bool read(const fs::path &path);
};

cv::Mat toTransformMatrix(const cv::Mat &R, const cv::Mat &t);

enum class PatternOption { ARUCO = 1, CHESSBOARD = 2, CHARUCO = 3 };

struct CalibrationPattern {
  CalibrationPattern();
  explicit CalibrationPattern(const std::string &patternInfo);
  explicit CalibrationPattern(const fs::path &dataset);
  CalibrationPattern(const fs::path &dataset, const std::string &patternInfo);

  void setPatternInfo(const std::string &patternInfo);
  std::optional<PoseStamped> estimatePose(const CalibrationData &data,
                                          const CalibrationPattern &pattern,
                                          size_t index);
  bool detectOn(const cv::Mat &image);

  PatternOption getPatternName() const;
  const fs::path &getDatasetPath() const;
  cv::Size getChessboardDims() const;
  const std::vector<cv::Point2f> &getImgPoints() const;
  const rclcpp::Logger &getLogger() const;
  const std::vector<cv::Point3f> &getObjPoints() const;

private:
  const fs::path dataset_;
  const rclcpp::Logger logger_ = rclcpp::get_logger(HELPER_LOGGERNAME);

  PatternOption option_;
  std::optional<pattern_detector::PatternDetector> patternDetector_;

  std::optional<int32_t> id_;
  std::optional<double> markerSize_;
  std::optional<int32_t> rows_;
  std::optional<int32_t> cols_;
  std::optional<double> cellSize_;
  std::vector<cv::Point2f> corners_;
};

std::optional<size_t> getImageNumber(const std::string &imageName);
tf2::Transform cvMatToTF2Transform(const cv::Mat &T_cv);
void findTarget2Cam(CalibrationPattern &pattern, CalibrationData &data,
                    bool useKnownIntrinsics = false);

enum class PosesOption {
  ROT_XYZW = 1,
  ROT_WXYZ = 2,
  ROT_RPY_RAD = 3,
  ROT_RPY_DEG = 4,
  ROT_YPR_RAD = 5,
  ROT_YPR_DEG = 6,
  JOINTS = 7
};

void readPoses(std::vector<std::vector<double>> &poses,
               const std::unordered_set<size_t> &skipIndexes,
               const fs::path &posesFile);
void findGripper2Base(const fs::path &datasetPath, int32_t posesFormat,
                      CalibrationData &data);
void dumpToYAML(const CalibrationData &data);
double calculateRMS(const CalibrationData &data);

} // namespace calibration_helpers
