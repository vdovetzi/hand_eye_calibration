#pragma once

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "magic_enum.hpp"
#include <csv.hpp>
#include <eigen3/Eigen/Eigen>
#include <filesystem>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ranges>
#include <rclcpp/rclcpp.hpp>
#include <regex>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>

using csv::CSVFormat;
using csv::CSVReader;
using cv::aruco::Dictionary;
using geometry_msgs::msg::PoseStamped;
#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR <= 6
using DictionaryEnumType = cv::aruco::PREDEFINED_DICTIONARY_NAME;
#else
using DictionaryEnumType = cv::aruco::PredefinedDictionaryType;
#endif
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
using cv::aruco::ArucoDetector;
#endif
namespace fs = std::filesystem;

constexpr const char *CSV_FILENAME = "poses.csv";
constexpr const char *IMG_FOLDERNAME = "images";
constexpr const char *HELPER_LOGGERNAME = "helper";
constexpr const char *DATASET_FOLDERNAME = "dataset";
constexpr const char *CALIBRATION_FILENAME = "calibration.yaml";
constexpr const double DEG2RAD = std::numbers::pi / 180.0;
const std::regex PATTERN(R"(^(\d+)\.png$)");

inline CSVReader getReader(const fs::path &path) {
  static CSVFormat format;
  format.delimiter('\t').no_header().quote(false);
  return CSVReader(path.relative_path().c_str(), format);
}

// Count poses and returns that count
inline std::optional<size_t> countPoses(const fs::path &datasetPath) {
  try {
    CSVReader reader = getReader(datasetPath / CSV_FILENAME);
    size_t count = 0;
    // Reading rows
    for (auto &row : reader) {
      if (row.size() > 0 && !row[0].is_null()) {
        count++;
      }
    }
    return count;

  } catch (std::exception &ex) {
    RCLCPP_DEBUG(rclcpp::get_logger(HELPER_LOGGERNAME), "Error: %s", ex.what());
    return std::nullopt;
  }
}

// Count images in images/ folder and returns that count
inline std::optional<size_t> countImages(const fs::path &datasetPath) {
  size_t count = 0;
  try {
    for (const auto &entry :
         fs::directory_iterator(datasetPath / IMG_FOLDERNAME)) {
      if (entry.is_regular_file()) {
        const std::string &filename = entry.path().filename().string();
        if (std::regex_match(filename, PATTERN)) {
          ++count;
        }
      }
    }

    return count;
  } catch (std::exception &ex) {
    RCLCPP_DEBUG(rclcpp::get_logger(HELPER_LOGGERNAME), "Error: %s", ex.what());
    return std::nullopt;
  }
}

inline bool validateDataset(const fs::path &dataset) {
  const fs::path imgFolder = dataset / IMG_FOLDERNAME;
  const fs::path posesFile = dataset / CSV_FILENAME;
  const rclcpp::Logger &logger = rclcpp::get_logger(HELPER_LOGGERNAME);
  std::initializer_list<bool> conds = {
      fs::exists(dataset),   fs::is_directory(dataset),
      fs::exists(imgFolder), fs::is_directory(imgFolder),
      fs::exists(posesFile), fs::is_regular_file(posesFile)};

  if (!std::ranges::all_of(conds, [](bool x) { return x; })) {
    RCLCPP_DEBUG(logger,
                 "Check existance of the following files and "
                 "directories:\ndataset/\ndataset/images/\ndataset/poses.csv");
    return false;
  }

  std::optional<size_t> imagesCountOpt = countImages(dataset);
  std::optional<size_t> posesCountOpt = countPoses(dataset);

  if (!imagesCountOpt.has_value()) {
    RCLCPP_DEBUG(logger, "Cannot count images in images/ folder");
    return false;
  }

  if (!posesCountOpt.has_value()) {
    RCLCPP_DEBUG(logger, "Cannot count poses in poses.csv");
    return false;
  }

  if (*posesCountOpt != *imagesCountOpt) {
    RCLCPP_DEBUG(logger, "Poses count is not equal to the images count");
    return false;
  }

  return true;
}

struct CalibrationData {
  // Camera calibration
  cv::Mat K;
  cv::Mat D;

  // Hand-Eye-Calibration
  std::vector<cv::Mat> R_target2cam;
  std::vector<cv::Mat> t_target2cam;
  std::vector<cv::Mat> R_gripper2base;
  std::vector<cv::Mat> t_gripper2base;

  cv::Mat R_cam2gripper;
  cv::Mat t_cam2gripper;
  cv::Mat T;

  std::unordered_set<size_t> rejectedImages;

  bool read(const fs::path &path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
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
};

inline cv::Mat toTransformMatrix(const cv::Mat &R, const cv::Mat &t) {
  cv::Mat T(4, 4, CV_64F);
  T.setTo(0.0);
  R.copyTo(T(cv::Rect(0, 0, 3, 3)));
  t.copyTo(T(cv::Rect(3, 0, 1, 3)));
  T.at<double>(3, 3) = 1.0;

  return T;
}

enum class PatternOption { ARUCO = 1, CHESSBOARD = 2, CHARUCO = 3 };

struct CalibrationPattern {
  CalibrationPattern() {}
  CalibrationPattern(const std::string &patternInfo) {
    setPatternInfo(patternInfo);
  }
  CalibrationPattern(const fs::path &dataset) : dataset_(dataset) {}
  CalibrationPattern(const fs::path &dataset, const std::string &patternInfo)
      : dataset_(dataset) {
    setPatternInfo(patternInfo);
  }

  void setPatternInfo(const std::string &patternInfo) {
    std::vector<std::string_view> patternParams;
    for (const auto &part : std::views::split(patternInfo, ' ')) {
      patternParams.emplace_back(std::string_view(part.data(), part.size()));
    }
    if (patternParams.empty()) {
      throw std::runtime_error(
          "Error: you must choose one of the supported patterns: 1, 2 or 3");
    }

    std::optional<PatternOption> option = magic_enum::enum_cast<PatternOption>(
        std::stoi(patternParams[0].data()));

    if (!option) {
      throw std::runtime_error("Error: choosen pattern is not of the supported "
                               "ones: choose 1, 2 or 3");
    }

    option_ = *option;

    switch (option_) {
    case PatternOption::ARUCO: {
      if (patternParams.size() < 4) {
        throw std::runtime_error(
            "Error: wrong number of params to aruco calibration pattern");
      }
      // TODO: add more checks on that conversions. BTW, an error will be
      // caught, but the .what() won't be user friendly
      const int32_t dictNum = std::atoi(patternParams[1].data());
      id_ = std::atoi(patternParams[2].data());
      initializeDict(dictNum, *id_);
      markerSize_ = std::atof(patternParams[3].data());
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
      cv::aruco::DetectorParameters params;
      detector_ = ArucoDetector(*dict_, params);
#else
      params_ = cv::aruco::DetectorParameters::create();
#endif

      float markerSize = *markerSize_;

      objPoints_ = std::vector<cv::Point3f>();
      objPoints_->emplace_back(
          cv::Point3f(-markerSize / 2, markerSize / 2, 0)); // top-left
      objPoints_->emplace_back(
          cv::Point3f(markerSize / 2, markerSize / 2, 0)); // top-right
      objPoints_->emplace_back(
          cv::Point3f(markerSize / 2, -markerSize / 2, 0)); // bottom-right
      objPoints_->emplace_back(
          cv::Point3f(-markerSize / 2, -markerSize / 2, 0)); // bottom-left

      break;
    }
    case PatternOption::CHARUCO: {
      if (patternParams.size() < 7) {
        throw std::runtime_error(
            "Error: wrong number of params to charuco calibration pattern");
      }
      rows_ = std::atoi(patternParams[1].data());
      cols_ = std::atoi(patternParams[2].data());
      cellSize_ = std::atof(patternParams[3].data());
      markerSize_ = std::atof(patternParams[4].data());
      const int32_t dict_num = std::atoi(patternParams[5].data());
      id_ = std::atoi(patternParams[6].data());
      initializeDict(dict_num, *id_);
      break;
    }
    case PatternOption::CHESSBOARD: {
      if (patternParams.size() < 4) {
        throw std::runtime_error(
            "Error: wrong number of params to chessboard calibration pattern");
      }
      rows_ = std::atoi(patternParams[1].data());
      cols_ = std::atoi(patternParams[2].data());
      cellSize_ = std::atof(patternParams[3].data());

      float squareSize = getCellSize();
      cv::Size dims = getChessboardDims();

      objPoints_ = std::vector<cv::Point3f>();
      objPoints_->reserve(dims.area());
      for (int32_t r = 0; r < dims.height; ++r) {
        for (int32_t c = 0; c < dims.width; ++c) {
          objPoints_->emplace_back(c * squareSize, r * squareSize, 0.0f);
        }
      }

      break;
    }
    default: {
      throw std::runtime_error("Not implemented!");
    }
    }
  }

  std::optional<PoseStamped> estimatePose(const CalibrationData &data,
                                          const CalibrationPattern &pattern,
                                          size_t index) {
    const auto &objPoints = pattern.getObjPoints();

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(objPoints, pattern.getImgPoints(), data.K,
                                data.D, rvec, tvec);

    if (!success) {
      return std::nullopt;
    }

    static cv::Mat R;
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

    static tf2::Matrix3x3 tf_rotation;

    for (int i = 0; i < R.rows; ++i) {
      for (int j = 0; j < R.cols; ++j) {
        tf_rotation[i][j] = R.at<double>(i, j);
      }
    }

    static tf2::Quaternion tf_quat;
    tf_rotation.getRotation(tf_quat);
    tf_quat.normalize();

    msg.pose.orientation.x = tf_quat.x();
    msg.pose.orientation.y = tf_quat.y();
    msg.pose.orientation.z = tf_quat.z();
    msg.pose.orientation.w = tf_quat.w();

    return msg;
  }

  bool detectOn(const cv::Mat &image) {
    switch (option_) {
    case PatternOption::ARUCO: {
      static std::vector<int32_t> markerIds;
      static std::vector<std::vector<cv::Point2f>> markerCorners,
          rejectedCandidates;
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
      detector_->detectMarkers(image, markerCorners, markerIds,
                               rejectedCandidates);
#else
      cv::aruco::detectMarkers(image, dict_, markerCorners, markerIds, params_,
                               rejectedCandidates);
#endif

      auto it = std::find(markerIds.begin(), markerIds.end(), *id_);
      if (!markerIds.empty() && it != markerIds.end()) {
        size_t ind = it - markerIds.begin();
        assert(markerIds[ind] == *id_);
        corners_ = markerCorners[ind];
      } else {
        corners_ = std::vector<cv::Point2f>();
      }
      return true;
    }
    case PatternOption::CHESSBOARD: {
      cv::Mat gray;
      const int32_t flags = cv::CALIB_CB_NORMALIZE_IMAGE |
                            cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY;
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

      const bool found = cv::findChessboardCornersSB(gray, getChessboardDims(),
                                                     corners_, flags);
      if (!found) {
        return false;
      }

      cv::cornerSubPix(
          gray, corners_, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                           0.001));

      return true;
    }
    default: {

      break;
    }
    }
    return false;
  }

  PatternOption getPatternName() const { return option_; }

  const fs::path &getDatasetPath() const { return dataset_; }

  cv::Size getChessboardDims() const { return cv::Size(*cols_, *rows_); }

  double getMarkerSize() const { return *markerSize_; }

  const std::vector<cv::Point2f> &getImgPoints() const { return corners_; }

  double getCellSize() const { return *cellSize_; }

  const Dictionary &getDictionary() const { return *dict_; }

  const rclcpp::Logger &getLogger() const { return logger_; }

  const std::vector<cv::Point3f> &getObjPoints() const { return *objPoints_; }

private:
  // Initialization stuff
  const fs::path dataset_;
  const rclcpp::Logger logger_ = rclcpp::get_logger(HELPER_LOGGERNAME);

  // Patterns stuff
  PatternOption option_;
  // ArUco & ChArUco
  std::optional<int32_t> id_;
  std::optional<double> markerSize_;
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
  std::optional<cv::aruco::Dictionary> dict_;
  std::optional<ArucoDetector> detector_;
#else
  cv::Ptr<cv::aruco::Dictionary> dict_;
  cv::Ptr<cv::aruco::DetectorParameters> params_;
#endif

  // Chessboard & ChArUco
  std::optional<int32_t> rows_;
  std::optional<int32_t> cols_;
  std::optional<double> cellSize_;
  std::vector<cv::Point2f> corners_;
  std::optional<std::vector<cv::Point3f>> objPoints_;

  void initializeDict(const int32_t dictNum, const int32_t id) {
    if (dictNum < 4 || dictNum > 7) {
      throw std::runtime_error(
          "Error: wrong dictionary number. Choose one from 4-7");
    }
    std::string K = std::to_string(calculateDictK(id));
    std::string M = std::to_string(dictNum);
    std::string dictName("DICT_" + M + "X" + M + "_" + K);

    std::optional<DictionaryEnumType> dictType =
        magic_enum::enum_cast<DictionaryEnumType>(dictName);

    dict_ = cv::aruco::getPredefinedDictionary(*dictType);
  }

  int32_t calculateDictK(const int32_t id) const {
    if (id < 50) {
      return 50;
    }
    if (id < 100) {
      return 100;
    }
    if (id < 250) {
      return 250;
    }
    if (id < 1000) {
      return 1000;
    }
    throw std::runtime_error("Error: id should be no more than 999");
  }
};

inline std::optional<size_t> getImageNumber(const std::string &imageName) {
  if (std::smatch matches; std::regex_match(imageName, matches, PATTERN)) {
    return std::stoi(matches[1].str());
  }
  return std::nullopt;
}

inline tf2::Transform cvMatToTF2Transform(const cv::Mat &T_cv) {
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

// Calibrating camera to get camera's intrinsics parameters and then get the
// extrinsics. I hope in future, there will no need in camera calibrating, if
// the intrinsics were already provided.
inline void findTarget2Cam(CalibrationPattern &pattern, CalibrationData &data) {
  const rclcpp::Logger &logger = pattern.getLogger();
  switch (pattern.getPatternName()) {
  case PatternOption::ARUCO: {
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    cv::Mat image;
    cv::Size imageSize;

    std::unordered_set<size_t> &rejectedImages = data.rejectedImages;

    const size_t imageNum = *countImages(pattern.getDatasetPath());
    const fs::path &datasetPath = pattern.getDatasetPath();

    imagePoints.reserve(imageNum);
    objectPoints.reserve(imageNum);

    RCLCPP_INFO(logger,
                "Press any key to go to the next image. Press 'd' to reject "
                "the image");

    for (size_t i = 0; i < imageNum; ++i) {
      const std::string filename = std::format("{}.png", i);
#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR <= 9
      image = cv::imread(datasetPath / IMG_FOLDERNAME / filename);
#else
      cv::imread(datasetPath / IMG_FOLDERNAME / filename, image);
#endif

      if (imageSize.empty()) {
        imageSize = image.size();
      }

      pattern.detectOn(image);
      const std::vector<cv::Point2f> &corners = pattern.getImgPoints();

      if (corners.empty()) {
        RCLCPP_WARN(logger, "Aruco marker not found in %s", filename.c_str());
        rejectedImages.insert(i);
        continue;
      }

      cv::aruco::drawDetectedMarkers(
          image, std::vector<std::vector<cv::Point2f>>{corners});
      cv::imshow(filename, image);
      int32_t key = cv::waitKey(0);
      if (key == 'd') {
        RCLCPP_WARN(logger, "Image %s is rejected", filename.c_str());
        rejectedImages.insert(i);
        cv::destroyWindow(filename);
        continue;
      }

      cv::destroyWindow(filename);

      imagePoints.emplace_back(corners);
      objectPoints.emplace_back(pattern.getObjPoints());
    }

    cv::destroyAllWindows();

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
  case PatternOption::CHARUCO: {
    // TODO: add charuco calibration routine
    throw std::runtime_error("Not implemented!");
  }
  case PatternOption::CHESSBOARD: {
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    cv::Mat image;
    cv::Size imageSize;

    const cv::Size patternDims = pattern.getChessboardDims();

    std::unordered_set<size_t> &rejectedImages = data.rejectedImages;

    const size_t imageNum = *countImages(pattern.getDatasetPath());
    const fs::path &datasetPath = pattern.getDatasetPath();

    imagePoints.reserve(imageNum);

    RCLCPP_INFO(logger,
                "Press any key to go to the next image. Press 'd' to reject "
                "the image");

    for (size_t i = 0; i < imageNum; ++i) {
      const std::string filename = std::format("{}.png", i);
#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR <= 9
      image = cv::imread(datasetPath / IMG_FOLDERNAME / filename);
#else
      cv::imread(datasetPath / IMG_FOLDERNAME / filename, image);
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

      cv::drawChessboardCorners(image, patternDims, corners, found);
      cv::imshow(filename, image);
      int32_t key = cv::waitKey(0);
      if (key == 'd') {
        RCLCPP_WARN(logger, "Image %s is rejected", filename.c_str());
        rejectedImages.insert(i);
        cv::destroyWindow(filename);
        continue;
      }

      cv::destroyWindow(filename);

      imagePoints.emplace_back(corners);
      objectPoints.emplace_back(pattern.getObjPoints());
    }

    cv::destroyAllWindows();

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

enum class PosesOption {
  ROT_XYZW = 1,
  ROT_WXYZ = 2,
  ROT_RPY_RAD = 3,
  ROT_RPY_DEG = 4,
  JOINTS = 5
};

inline void readPoses(std::vector<std::vector<double>> &poses,
                      const std::unordered_set<size_t> &skipIndexes,
                      const fs::path &posesFile) {
  CSVReader reader = getReader(posesFile);
  size_t rowInd = 0;
  // Reading poses
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

inline void findGripper2Base(const fs::path &datasetPath,
                             const int32_t posesFormat, CalibrationData &data) {

  std::optional<PosesOption> posesOption =
      magic_enum::enum_cast<PosesOption>(posesFormat);

  if (!posesOption) {
    throw std::runtime_error(
        "Error: provide exising pose format: choose from 1 to 5");
  }

  std::vector<std::vector<double>> poses;
  poses.reserve(*countPoses(datasetPath));

  readPoses(poses, data.rejectedImages, datasetPath / CSV_FILENAME);

  std::vector<cv::Mat> &Rs = data.R_gripper2base;
  std::vector<cv::Mat> &tvecs = data.t_gripper2base;
  Rs.reserve(poses.size());
  tvecs.reserve(poses.size());

  Eigen::Matrix3d R_eigen;
  cv::Mat rvec;

  for (const auto &pose : poses) {
    switch (*posesOption) {
    case (PosesOption::ROT_XYZW): {
      tvecs.emplace_back(
          std::initializer_list<double>{pose[0], pose[1], pose[2]});

      Eigen::Quaterniond q;

      q.x() = pose[3];
      q.y() = pose[4];
      q.z() = pose[5];
      q.w() = pose[6];

      R_eigen = q.normalized().toRotationMatrix();

      cv::Mat R_cv;
      cv::eigen2cv(R_eigen, R_cv);

      Rs.emplace_back(R_cv);
      break;
    }
    case (PosesOption::ROT_WXYZ): {
      tvecs.emplace_back(
          std::initializer_list<double>{pose[0], pose[1], pose[2]});
      Eigen::Quaterniond q;

      q.w() = pose[3];
      q.x() = pose[4];
      q.y() = pose[5];
      q.z() = pose[6];

      R_eigen = q.normalized().toRotationMatrix();

      cv::Mat R_cv;
      cv::eigen2cv(R_eigen, R_cv);

      Rs.emplace_back(R_cv);
      break;
    }
    case (PosesOption::ROT_RPY_RAD): {
      tvecs.emplace_back(
          std::initializer_list<double>{pose[0], pose[1], pose[2]});

      double roll = pose[3];
      double pitch = pose[4];
      double yaw = pose[5];

      tf2::Matrix3x3 R_tf2;
      R_tf2.setEulerYPR(yaw, pitch, roll);
      cv::Mat R_cv(3, 3, CV_64F);

      for (int32_t i = 0; i < R_cv.rows; ++i) {
        const tf2::Vector3 &row = R_tf2.getRow(i);
        R_cv.at<double>(i, 0) = row[0];
        R_cv.at<double>(i, 1) = row[1];
        R_cv.at<double>(i, 2) = row[2];
      }

      Rs.emplace_back(R_cv);

      break;
    }
    case (PosesOption::ROT_RPY_DEG): {
      tvecs.emplace_back(
          std::initializer_list<double>{pose[0], pose[1], pose[2]});

      double roll = pose[3] * DEG2RAD;
      double pitch = pose[4] * DEG2RAD;
      double yaw = pose[5] * DEG2RAD;

      tf2::Matrix3x3 R_tf2;
      R_tf2.setEulerYPR(yaw, pitch, roll);
      cv::Mat R_cv(3, 3, CV_64F);

      for (int32_t i = 0; i < R_cv.rows; ++i) {
        const tf2::Vector3 &row = R_tf2.getRow(i);
        R_cv.at<double>(i, 0) = row[0];
        R_cv.at<double>(i, 1) = row[1];
        R_cv.at<double>(i, 2) = row[2];
      }

      Rs.emplace_back(R_cv);

      break;
    }
    case (PosesOption::JOINTS): {
      throw std::runtime_error("Not implemented!");
    }
    }
  }
}

inline void dumpToYAML(const CalibrationData &data) {
  cv::Mat T = toTransformMatrix(data.R_cam2gripper, data.t_cam2gripper);

  cv::FileStorage out(CALIBRATION_FILENAME, cv::FileStorage::WRITE);
  out << "T" << T;
  out << "K" << data.K;
  out << "D" << data.D;
  out.release();
}
