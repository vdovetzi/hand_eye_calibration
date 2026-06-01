#include "hand_eye_calibration/helpers/calibrate_helpers.hpp"

#include "hand_eye_calibration/helpers/parse_helpers.hpp"
#include <algorithm>
#include <cmath>
#include <csv.hpp>
#include <eigen3/Eigen/Eigen>
#include <format>
#include <iostream>
#include <magic_enum.hpp>
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <tf2/LinearMath/Matrix3x3.h>

namespace calibration_helpers {

namespace {

constexpr double DEG2RAD = std::numbers::pi / 180.0;

} // namespace

bool CalibrationData::readIntrinsics(const fs::path &path) {
  cv::FileStorage fs(path.string(), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  fs["K"] >> K;
  if (K.empty()) {
    return false;
  }

  fs["D"] >> D;
  if (D.empty()) {
    return false;
  }

  fs.release();

  return true;
}

bool CalibrationData::read(const fs::path &path) {
  cv::FileStorage fs(path.string(), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  fs["K"] >> K;
  if (K.empty()) {
    return false;
  }

  fs["D"] >> D;
  if (D.empty()) {
    return false;
  }

  fs["T"] >> T;
  if (T.empty()) {
    return false;
  }

  fs.release();

  return true;
}

cv::Mat toTransformMatrix(const cv::Mat &R, const cv::Mat &t) {
  cv::Mat T(4, 4, CV_64F);
  T.setTo(0.0);
  R.copyTo(T(cv::Rect(0, 0, 3, 3)));
  t.copyTo(T(cv::Rect(3, 0, 1, 3)));
  T.at<double>(3, 3) = 1.0;

  return T;
}

CalibrationPattern::CalibrationPattern() = default;

CalibrationPattern::CalibrationPattern(const std::string &patternInfo) {
  setPatternInfo(patternInfo);
}

CalibrationPattern::CalibrationPattern(const fs::path &dataset)
    : dataset_(dataset) {}

CalibrationPattern::CalibrationPattern(const fs::path &dataset,
                                       const std::string &patternInfo)
    : dataset_(dataset) {
  setPatternInfo(patternInfo);
}

void CalibrationPattern::setPatternInfo(const std::string &patternInfo) {
  std::istringstream stream(patternInfo);
  std::vector<std::string> patternParams;
  std::string param;
  while (stream >> param) {
    patternParams.emplace_back(param);
  }
  if (patternParams.empty()) {
    throw std::runtime_error(
        "Error: you must choose one of the supported patterns: 1, 2 or 3");
  }

  const int32_t optionValue =
      parse_helpers::parseInt(patternParams[0], "pattern type");
  std::optional<PatternOption> option =
      magic_enum::enum_cast<PatternOption>(optionValue);

  if (!option) {
    throw std::runtime_error("Error: choosen pattern is not of the supported "
                             "ones: choose 1, 2 or 3");
  }

  option_ = *option;

  switch (option_) {
  case PatternOption::ARUCO: {
    patternDetector_ =
        pattern_detector::PatternDetector::fromPatternInfo(patternInfo);
    break;
  }
  case PatternOption::CHARUCO: {
    if (patternParams.size() < 7) {
      throw std::runtime_error(
          "Error: wrong number of params to charuco calibration pattern");
    }
    rows_ = parse_helpers::parsePositiveInt(patternParams[1], "charuco rows");
    cols_ = parse_helpers::parsePositiveInt(patternParams[2], "charuco cols");
    cellSize_ =
        parse_helpers::parsePositiveDouble(patternParams[3],
                                           "charuco cell size");
    markerSize_ =
        parse_helpers::parsePositiveDouble(patternParams[4],
                                           "charuco marker size");
    const int32_t dictNum =
        parse_helpers::parseInt(patternParams[5], "charuco dictionary number");
    if (dictNum < 4 || dictNum > 7) {
      throw std::runtime_error(
          "Error: wrong dictionary number. Choose one from 4-7");
    }
    id_ = parse_helpers::parseInt(patternParams[6], "charuco marker id");
    pattern_detector::calculateArucoDictK(*id_);
    break;
  }
  case PatternOption::CHESSBOARD: {
    patternDetector_ =
        pattern_detector::PatternDetector::fromPatternInfo(patternInfo);
    break;
  }
  default: {
    throw std::runtime_error("Not implemented!");
  }
  }
}

std::optional<PoseStamped>
CalibrationPattern::estimatePose(const CalibrationData &data,
                                 const CalibrationPattern &pattern,
                                 size_t index) {
  const auto &objPoints = pattern.getObjPoints();

  cv::Mat rvec;
  cv::Mat tvec;
  const bool success = cv::solvePnP(objPoints, pattern.getImgPoints(), data.K,
                                    data.D, rvec, tvec);

  if (!success) {
    return std::nullopt;
  }

  cv::Mat R;
  cv::Rodrigues(rvec, R);
  cv::Mat T_target2cam = toTransformMatrix(R, tvec);

  const cv::Point3f &objPoint = objPoints[index];

  cv::Mat pointInCameraFrame =
      T_target2cam * cv::Mat(std::initializer_list<double>(
                         {objPoint.x, objPoint.y, objPoint.z, 1.0}));

  PoseStamped msg;

  msg.pose.position.x =
      pointInCameraFrame.at<double>(0) / pointInCameraFrame.at<double>(3);
  msg.pose.position.y =
      pointInCameraFrame.at<double>(1) / pointInCameraFrame.at<double>(3);
  msg.pose.position.z =
      pointInCameraFrame.at<double>(2) / pointInCameraFrame.at<double>(3);

  tf2::Matrix3x3 tf_rotation;

  for (int i = 0; i < R.rows; ++i) {
    for (int j = 0; j < R.cols; ++j) {
      tf_rotation[i][j] = R.at<double>(i, j);
    }
  }

  tf2::Quaternion tf_quat;
  tf_rotation.getRotation(tf_quat);
  tf_quat.normalize();

  msg.pose.orientation.x = tf_quat.x();
  msg.pose.orientation.y = tf_quat.y();
  msg.pose.orientation.z = tf_quat.z();
  msg.pose.orientation.w = tf_quat.w();

  return msg;
}

bool CalibrationPattern::detectOn(const cv::Mat &image) {
  if (!patternDetector_) {
    corners_.clear();
    return false;
  }
  return patternDetector_->detectImagePoints(image, corners_);
}

PatternOption CalibrationPattern::getPatternName() const {
  return option_;
}

const fs::path &CalibrationPattern::getDatasetPath() const {
  return dataset_;
}

cv::Size CalibrationPattern::getChessboardDims() const {
  return patternDetector_->getChessboardDims();
}

const std::vector<cv::Point2f> &CalibrationPattern::getImgPoints() const {
  return corners_;
}

const rclcpp::Logger &CalibrationPattern::getLogger() const {
  return logger_;
}

const std::vector<cv::Point3f> &CalibrationPattern::getObjPoints() const {
  return patternDetector_->getObjectPoints();
}

std::optional<size_t> getImageNumber(const std::string &imageName) {
  if (std::smatch matches;
      std::regex_match(imageName, matches, IMAGE_FILENAME_PATTERN)) {
    const int32_t imageNumber =
        parse_helpers::parseInt(matches[1].str(), "image number");
    if (imageNumber < 0) {
      return std::nullopt;
    }
    return static_cast<size_t>(imageNumber);
  }
  return std::nullopt;
}

tf2::Transform cvMatToTF2Transform(const cv::Mat &T_cv) {
  if (T_cv.rows != 4 || T_cv.cols != 4) {
    throw std::runtime_error("Transformation matrix must be 4x4");
  }

  cv::Mat R = T_cv(cv::Rect(0, 0, 3, 3));
  cv::Mat t = T_cv(cv::Rect(3, 0, 1, 3));

  tf2::Matrix3x3 tf_rotation;
  for (int i = 0; i < R.rows; i++) {
    for (int j = 0; j < R.cols; j++) {
      tf_rotation[i][j] = R.at<double>(i, j);
    }
  }

  tf2::Vector3 tf_translation(t.at<double>(0, 0), t.at<double>(1, 0),
                              t.at<double>(2, 0));

  return tf2::Transform(tf_rotation, tf_translation);
}

void findTarget2Cam(CalibrationPattern &pattern, CalibrationData &data,
                    const bool useKnownIntrinsics) {
  const rclcpp::Logger &logger = pattern.getLogger();
  switch (pattern.getPatternName()) {
  case PatternOption::ARUCO: {
    // TODO: add aruco routine
    throw std::runtime_error("Not implemented!");
  }
  case PatternOption::CHARUCO: {
    // TODO: add charuco calibration routine
    throw std::runtime_error("Not implemented!");
  }
  case PatternOption::CHESSBOARD: {
    if (useKnownIntrinsics && (data.K.empty() || data.D.empty())) {
      throw std::runtime_error(
          "Error: known intrinsics mode requires K and D matrices");
    }

    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    cv::Mat image;
    cv::Size imageSize;

    std::unordered_set<size_t> &rejectedImages = data.rejectedImages;

    const size_t imageNum =
        *dataset_helpers::countImages(pattern.getDatasetPath());
    const fs::path &datasetPath = pattern.getDatasetPath();

    imagePoints.reserve(imageNum);
    objectPoints.reserve(imageNum);
    data.R_target2cam.reserve(imageNum);
    data.t_target2cam.reserve(imageNum);

    for (size_t i = 0; i < imageNum; ++i) {
      const std::string filename = std::format("{}.png", i);
      const fs::path imagePath = datasetPath / IMG_FOLDERNAME / filename;
#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR <= 9
      image = cv::imread(imagePath.string());
#else
      cv::imread(imagePath.string(), image);
#endif

      if (imageSize.empty()) {
        imageSize = image.size();
      }

      const bool found = pattern.detectOn(image);
      if (!found) {
        RCLCPP_WARN(logger, "Chessboard not found in %s", filename.c_str());
        rejectedImages.insert(i);
        continue;
      }

      const std::vector<cv::Point2f> &corners = pattern.getImgPoints();

      if (useKnownIntrinsics) {
        cv::Mat rvec;
        cv::Mat tvec;
        const bool solved = cv::solvePnP(pattern.getObjPoints(), corners,
                                         data.K, data.D, rvec, tvec);
        if (!solved) {
          RCLCPP_DEBUG(logger, "Pose estimation failed for %s",
                       filename.c_str());
          rejectedImages.insert(i);
          continue;
        }

        cv::Mat R;
        cv::Rodrigues(rvec, R);
        data.R_target2cam.emplace_back(R);
        data.t_target2cam.emplace_back(tvec);
      } else {
        imagePoints.emplace_back(corners);
        objectPoints.emplace_back(pattern.getObjPoints());
      }
    }

    if (useKnownIntrinsics) {
      if (data.R_target2cam.size() < 3) {
        throw std::runtime_error("Error: need more images. Need at least 3");
      }
      RCLCPP_DEBUG(logger,
                   "Estimated target-to-camera poses with known intrinsics for "
                   "%zu valid images",
                   data.R_target2cam.size());
      break;
    }

    if (imagePoints.size() < 3) {
      throw std::runtime_error("Error: need more images. Need at least 3");
    }

    std::vector<cv::Mat> rvecs;

    std::cout << std::format(
        "Calibrating camera with {} valid images. It may take some time...\n",
        imagePoints.size());

    const double rms = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize, data.K, data.D, rvecs,
        data.t_target2cam, 0,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                         1e-6));

    std::vector<cv::Mat> &Rs = data.R_target2cam;
    Rs.reserve(rvecs.size());
    for (const cv::Mat &rvec : rvecs) {
      cv::Mat R;
      cv::Rodrigues(rvec, R);
      Rs.emplace_back(R);
    }

    RCLCPP_DEBUG(logger,
                 "Camera calibration successfull! Reprojection error is %f",
                 rms);
    break;
  }
  default: {
    throw std::runtime_error("Not implemented!");
  }
  }
}

void readPoses(std::vector<std::vector<double>> &poses,
               const std::unordered_set<size_t> &skipIndexes,
               const fs::path &posesFile) {
  CSVReader reader = dataset_helpers::getReader(posesFile);
  size_t rowInd = 0;
  for (auto &pose_row : reader) {
    if (pose_row.size() > 0 && !pose_row[0].is_null() &&
        !skipIndexes.contains(rowInd++)) {
      poses.emplace_back();
      std::transform(pose_row.begin(), pose_row.end(),
                     std::back_inserter(poses.back()),
                     [](csv::CSVField &elem) { return elem.get<double>(); });
    }
  }
}

void findGripper2Base(const fs::path &datasetPath, const int32_t posesFormat,
                      CalibrationData &data) {
  std::optional<PosesOption> posesOption =
      magic_enum::enum_cast<PosesOption>(posesFormat);

  if (!posesOption) {
    throw std::runtime_error(
        "Error: provide existing pose format: choose from 1 to 7");
  }

  std::vector<std::vector<double>> poses;
  poses.reserve(*dataset_helpers::countPoses(datasetPath));

  readPoses(poses, data.rejectedImages, datasetPath / CSV_FILENAME);

  std::vector<cv::Mat> &Rs = data.R_gripper2base;
  std::vector<cv::Mat> &tvecs = data.t_gripper2base;
  Rs.reserve(poses.size());
  tvecs.reserve(poses.size());

  for (const auto &pose : poses) {
    cv::Mat R_cv(3, 3, CV_64F);
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << pose[0], pose[1], pose[2]);
    switch (*posesOption) {
    case (PosesOption::ROT_XYZW): {
      Eigen::Quaterniond q;

      q.x() = pose[3];
      q.y() = pose[4];
      q.z() = pose[5];
      q.w() = pose[6];

      Eigen::Matrix3d R_eigen = q.normalized().toRotationMatrix();

      cv::eigen2cv(R_eigen, R_cv);
      break;
    }
    case (PosesOption::ROT_WXYZ): {
      Eigen::Quaterniond q;

      q.w() = pose[3];
      q.x() = pose[4];
      q.y() = pose[5];
      q.z() = pose[6];

      Eigen::Matrix3d R_eigen = q.normalized().toRotationMatrix();

      cv::eigen2cv(R_eigen, R_cv);
      break;
    }
    case (PosesOption::ROT_RPY_RAD): {
      const double roll = pose[3];
      const double pitch = pose[4];
      const double yaw = pose[5];

      tf2::Matrix3x3 R_tf2;
      R_tf2.setEulerYPR(yaw, pitch, roll);

      for (int32_t i = 0; i < R_cv.rows; ++i) {
        const tf2::Vector3 &row = R_tf2.getRow(i);
        R_cv.at<double>(i, 0) = row[0];
        R_cv.at<double>(i, 1) = row[1];
        R_cv.at<double>(i, 2) = row[2];
      }

      break;
    }
    case (PosesOption::ROT_RPY_DEG): {
      const double roll = pose[3] * DEG2RAD;
      const double pitch = pose[4] * DEG2RAD;
      const double yaw = pose[5] * DEG2RAD;

      tf2::Matrix3x3 R_tf2;
      R_tf2.setEulerYPR(yaw, pitch, roll);

      for (int32_t i = 0; i < R_cv.rows; ++i) {
        const tf2::Vector3 &row = R_tf2.getRow(i);
        R_cv.at<double>(i, 0) = row[0];
        R_cv.at<double>(i, 1) = row[1];
        R_cv.at<double>(i, 2) = row[2];
      }

      break;
    }
    case (PosesOption::ROT_YPR_RAD): {
      const double yaw = pose[3];
      const double pitch = pose[4];
      const double roll = pose[5];

      tf2::Matrix3x3 R_tf2;
      R_tf2.setEulerYPR(yaw, pitch, roll);

      for (int32_t i = 0; i < R_cv.rows; ++i) {
        const tf2::Vector3 &row = R_tf2.getRow(i);
        R_cv.at<double>(i, 0) = row[0];
        R_cv.at<double>(i, 1) = row[1];
        R_cv.at<double>(i, 2) = row[2];
      }

      break;
    }
    case (PosesOption::ROT_YPR_DEG): {
      const double yaw = pose[3] * DEG2RAD;
      const double pitch = pose[4] * DEG2RAD;
      const double roll = pose[5] * DEG2RAD;

      tf2::Matrix3x3 R_tf2;
      R_tf2.setEulerYPR(yaw, pitch, roll);

      for (int32_t i = 0; i < R_cv.rows; ++i) {
        const tf2::Vector3 &row = R_tf2.getRow(i);
        R_cv.at<double>(i, 0) = row[0];
        R_cv.at<double>(i, 1) = row[1];
        R_cv.at<double>(i, 2) = row[2];
      }

      break;
    }
    case (PosesOption::JOINTS): {
      throw std::runtime_error("Not implemented!");
    }
    }
    Rs.emplace_back(R_cv);
    tvecs.emplace_back(tvec);
  }
}

void dumpToYAML(const CalibrationData &data) {
  cv::Mat T = toTransformMatrix(data.R_cam2gripper, data.t_cam2gripper);

  cv::FileStorage out(CALIBRATION_FILENAME, cv::FileStorage::WRITE);
  out << "T" << T;
  out << "K" << data.K;
  out << "D" << data.D;
  out.release();
}

double calculateRMS(const CalibrationData &data) {
  const cv::Mat T_cam2gripper =
      toTransformMatrix(data.R_cam2gripper, data.t_cam2gripper);

  std::vector<cv::Vec3d> targetOriginsInGripper;
  targetOriginsInGripper.reserve(data.t_target2cam.size());

  for (const cv::Mat &t_target2cam : data.t_target2cam) {
    cv::Mat targetOriginInCamera =
        (cv::Mat_<double>(4, 1) << t_target2cam.at<double>(0, 0),
         t_target2cam.at<double>(1, 0), t_target2cam.at<double>(2, 0), 1.0);
    cv::Mat targetOriginInGripper = T_cam2gripper * targetOriginInCamera;
    targetOriginsInGripper.emplace_back(targetOriginInGripper.at<double>(0, 0),
                                        targetOriginInGripper.at<double>(1, 0),
                                        targetOriginInGripper.at<double>(2, 0));
  }

  if (targetOriginsInGripper.empty()) {
    throw std::runtime_error("Error: cannot calculate validation RMS without "
                             "valid target poses");
  }

  cv::Vec3d meanOrigin(0.0, 0.0, 0.0);
  for (const cv::Vec3d &origin : targetOriginsInGripper) {
    meanOrigin += origin;
  }
  meanOrigin *= 1.0 / static_cast<double>(targetOriginsInGripper.size());

  double squaredResidualSum = 0.0;
  for (const cv::Vec3d &origin : targetOriginsInGripper) {
    const cv::Vec3d residual = origin - meanOrigin;
    squaredResidualSum += residual.dot(residual);
  }

  return std::sqrt(squaredResidualSum /
                   static_cast<double>(targetOriginsInGripper.size()));
}

} // namespace calibration_helpers
