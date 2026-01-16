#pragma once

#include "magic_enum.hpp"
#include <csv.hpp>
#include <eigen3/Eigen/Eigen>
#include <filesystem>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ranges>
#include <rclcpp/rclcpp.hpp>
#include <regex>
#include <tf2/LinearMath/Matrix3x3.h>

using csv::CSVFormat;
using csv::CSVReader;
using cv::aruco::Dictionary;
using cv::aruco::PredefinedDictionaryType;
namespace fs = std::filesystem;

constexpr const char *CSV_FILENAME = "poses.csv";
constexpr const char *IMG_FOLDERNAME = "images";
constexpr const char *HELPER_LOGGERNAME = "helper";
constexpr const char *DATASET_FOLDERNAME = "dataset";
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

  if (!std::all_of(conds.begin(), conds.end(), [](bool x) { return x; })) {
    RCLCPP_DEBUG(logger,
                 "Check existance of the following files and "
                 "directories:\ndataset/\ndataset/images/\ndataset/poses.csv");
    return false;
  }

  std::optional<size_t> imagesCountOpt = countImages(dataset);
  std::optional<size_t> posesCountOpt = countPoses(dataset);

  if (!imagesCountOpt) {
    RCLCPP_DEBUG(logger, "Cannot count images in images/ folder");
    return false;
  }

  if (!posesCountOpt) {
    RCLCPP_DEBUG(logger, "Cannot count poses in poses.csv");
    return false;
  }

  if (*posesCountOpt != *posesCountOpt) {
    RCLCPP_DEBUG(logger, "Poses count is not equal to the images count");
    return false;
  }

  return true;
}

enum class PatternOption { ARUCO = 1, CHESSBOARD = 2, CHARUCO = 3 };

struct CalibrationPattern {
  CalibrationPattern(const fs::path &dataset) : dataset_(dataset) {};
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
      break;
    }
    default: {
      throw std::runtime_error("Not implemented!");
    }
    }
  }

  PatternOption getPatternName() const { return option_; }

  const fs::path &getDatasetPath() const { return dataset_; }

  std::optional<cv::Size> getChessboardDims() const {
    if (!rows_ || !cols_) {
      return std::nullopt;
    }
    return cv::Size(*cols_, *rows_);
  }

  std::optional<double> getMarkerSize() const { return markerSize_; }

  std::optional<double> getCellSize() const { return cellSize_; }

  std::optional<Dictionary> getDictionary() const { return dict_; }

  const rclcpp::Logger &getLogger() const { return logger_; }

private:
  // Initialization stuff
  const fs::path &dataset_;
  const rclcpp::Logger logger_ = rclcpp::get_logger(HELPER_LOGGERNAME);

  // Patterns stuff
  PatternOption option_;
  // ArUco & ChArUco
  std::optional<cv::aruco::Dictionary> dict_;
  std::optional<int32_t> id_;
  std::optional<double> markerSize_;

  // Chessboard & ChArUco
  std::optional<int32_t> rows_;
  std::optional<int32_t> cols_;
  std::optional<double> cellSize_;

  void initializeDict(const int32_t dictNum, const int32_t id) {
    if (dictNum < 4 || dictNum > 7) {
      throw std::runtime_error(
          "Error: wrong dictionary number. Choose one from 4-7");
    }
    std::string K = std::to_string(calculateDictK(id));
    std::string M = std::to_string(dictNum);
    std::string dictName("DICT_" + M + "X" + M + "_" + K);

    std::optional<PredefinedDictionaryType> dictType =
        magic_enum::enum_cast<PredefinedDictionaryType>(dictName);

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

struct CalibrationData {
  // Camera calibration
  cv::Mat K;
  cv::Mat D;

  // Hand-Eye-Calibration
  std::vector<cv::Mat> rvecsTarget2Cam;
  std::vector<cv::Mat> tvecsTarget2Cam;
  std::vector<cv::Mat> rvecsGripper2Base;
  std::vector<cv::Mat> tvecsGripper2Base;

  std::unordered_set<size_t> rejectedImages;
};

inline std::optional<size_t> getImageNumber(const std::string imageName) {
  std::smatch matches;
  if (std::regex_match(imageName, matches, PATTERN)) {
    return std::stoi(matches[1].str());
  }
  return std::nullopt;
}

// Calibrating camera to get camera's intrinsics parameters and then get the
// extrinsics. I hope in future, there will no need in camera calibrating, if
// the intrinsics were already provided.
inline void findTarget2Cam(CalibrationPattern &pattern, CalibrationData &data) {
  switch (pattern.getPatternName()) {
  case PatternOption::ARUCO: {
    // TODO: add aruco calibration routine
    throw std::runtime_error("Not implemented!");
  }
  case PatternOption::CHARUCO: {
    // TODO: add charuco calibration routine
    throw std::runtime_error("Not implemented!");
  }
  case PatternOption::CHESSBOARD: {
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<cv::Point2f> corners;
    cv::Mat gray;
    cv::Mat image;

    cv::Size imageSize;

    int32_t flags = cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE |
                cv::CALIB_CB_ACCURACY;

    cv::Size boardSize = *pattern.getChessboardDims();
    double squareSize = *pattern.getCellSize();

    static std::vector<cv::Point3f> objTemplate;
    if (objTemplate.empty()) {
      objTemplate.reserve(boardSize.area());
      for (int32_t r = 0; r < boardSize.height; ++r)
        for (int32_t c = 0; c < boardSize.width; ++c)
          objTemplate.emplace_back(c * squareSize, r * squareSize, 0.0f);
    }

    std::unordered_set<size_t> &rejectedImages = data.rejectedImages;

    for (const auto &entry :
         fs::directory_iterator(pattern.getDatasetPath() / IMG_FOLDERNAME)) {
      cv::imread(entry.path().string(), image);

      std::string filename = entry.path().filename();

      if (imageSize.empty()) {
        imageSize = image.size();
      }

      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

      bool found = cv::findChessboardCornersSB(
          gray, *pattern.getChessboardDims(), corners, flags);
      if (!found) {
        RCLCPP_WARN(pattern.getLogger(), "Chessboard not found in %s",
                    entry.path().string().data());
        rejectedImages.insert(*getImageNumber(filename));
        continue;
      }

      cv::cornerSubPix(
          gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                           0.001));

      cv::drawChessboardCorners(image, *pattern.getChessboardDims(), corners,
                                found);
      cv::imshow(filename, image);
      int32_t key = cv::waitKey(0);
      if (key == 'd') {
        RCLCPP_WARN(pattern.getLogger(), "Image %s is rejected",
                    filename.c_str());
        rejectedImages.insert(*getImageNumber(filename));
        cv::destroyWindow(filename);
        continue;
      }

      cv::destroyWindow(filename);

      imagePoints.emplace_back(corners);
      objectPoints.emplace_back(objTemplate);
    }

    if (imagePoints.size() < 3) {
      throw std::runtime_error("Error: need more images. Need at least 3");
    }

    double rms = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize, data.K, data.D,
        data.rvecsTarget2Cam, data.tvecsTarget2Cam, 0,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                         1e-6));

    RCLCPP_DEBUG(pattern.getLogger(),
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
                             const int32_t posesFormat,
                             CalibrationData &data) {

  std::optional<PosesOption> posesOption =
      magic_enum::enum_cast<PosesOption>(posesFormat);

  if (!posesOption) {
    throw std::runtime_error(
        "Error: provide exising pose format: choose from 1 to 5");
  }

  std::vector<std::vector<double>> poses;
  poses.reserve(*countPoses(datasetPath));

  readPoses(poses, data.rejectedImages, datasetPath / CSV_FILENAME);

  std::vector<cv::Mat> &rvecs = data.rvecsGripper2Base;
  std::vector<cv::Mat> &tvecs = data.tvecsGripper2Base;
  rvecs.reserve(poses.size());
  tvecs.reserve(poses.size());

  Eigen::Matrix3d R_eigen;
  cv::Mat rvec;
  cv::Mat R_cv(3, 3, CV_64F);

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

      q.normalize();
      R_eigen = std::move(q).toRotationMatrix();

      for (int32_t i = 0; i < 3; ++i) {
        for (int32_t j = 0; j < 3; ++j) {
          R_cv.at<double>(i, j) = R_eigen(i, j);
        }
      }

      cv::Rodrigues(R_cv, rvec);
      rvecs.emplace_back(rvec);
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

      q.normalize();
      R_eigen = std::move(q).toRotationMatrix();

      for (int32_t i = 0; i < 3; ++i) {
        for (int32_t j = 0; j < 3; ++j) {
          R_cv.at<double>(i, j) = R_eigen(i, j);
        }
      }

      cv::Rodrigues(R_cv, rvec);
      rvecs.emplace_back(rvec);
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

      for (int32_t i = 0; i < R_cv.rows; ++i) {
        const tf2::Vector3 &row = R_tf2.getRow(i);
        R_cv.at<cv::Vec3d>(i)[0] = row[0];
        R_cv.at<cv::Vec3d>(i)[1] = row[1];
        R_cv.at<cv::Vec3d>(i)[2] = row[2];
      }

      cv::Rodrigues(R_cv, rvec);
      rvecs.emplace_back(rvec);

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

      for (int32_t i = 0; i < R_cv.rows; ++i) {
        const tf2::Vector3 &row = R_tf2.getRow(i);
        R_cv.at<cv::Vec3d>(i)[0] = row[0];
        R_cv.at<cv::Vec3d>(i)[1] = row[1];
        R_cv.at<cv::Vec3d>(i)[2] = row[2];
      }

      cv::Rodrigues(R_cv, rvec);
      rvecs.emplace_back(rvec);

      break;
    }
    case (PosesOption::JOINTS): {
      throw std::runtime_error("Not implemented!");
    }
    }
  }
}