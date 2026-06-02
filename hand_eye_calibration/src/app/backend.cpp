#include <cv_bridge/cv_bridge.hpp>
#include <hand_eye_calibration/helpers/calibrate_helpers.hpp>
#include <hand_eye_calibration/helpers/collect_helpers.hpp>
#include <hand_eye_calibration/helpers/dataset_helpers.hpp>
#include <hand_eye_calibration/pattern_detector.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ros_babel_fish/babel_fish.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <csv.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <numbers>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <vector>

namespace {

using Trigger = std_srvs::srv::Trigger;
using TriggerRequest = std::shared_ptr<Trigger::Request>;
using TriggerResponse = std::shared_ptr<Trigger::Response>;
namespace fs = std::filesystem;

std::string jsonEscape(const std::string &text) {
  std::ostringstream escaped;
  for (const char ch : text) {
    switch (ch) {
    case '\\':
      escaped << "\\\\";
      break;
    case '"':
      escaped << "\\\"";
      break;
    case '\n':
      escaped << "\\n";
      break;
    case '\r':
      escaped << "\\r";
      break;
    case '\t':
      escaped << "\\t";
      break;
    default:
      escaped << ch;
      break;
    }
  }
  return escaped.str();
}

const char *jsonBool(const bool value) { return value ? "true" : "false"; }

void replaceAll(std::string &text, const std::string &from,
                const std::string &to) {
  size_t pos = 0;
  while ((pos = text.find(from, pos)) != std::string::npos) {
    text.replace(pos, from.size(), to);
    pos += to.size();
  }
}

std::string userMessage(std::string message) {
  replaceAll(message, "dataset_path", "dataset folder");
  replaceAll(message, "image_topic", "camera topic");
  replaceAll(message, "arm_topic", "arm topic");
  replaceAll(message, "pattern_info", "pattern settings");
  replaceAll(message, "intrinsics_path", "intrinsics YAML");
  replaceAll(message, "resume_dataset=true", "Continue Dataset");
  replaceAll(message, "overwrite_dataset=true", "Start Over");
  replaceAll(message, "last_validation_rms", "target lock RMS");
  replaceAll(message, "validation_rms", "target lock RMS");
  replaceAll(message, "target_lock_rms", "Target lock RMS");
  return message;
}

struct SpanMetric {
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = -std::numeric_limits<double>::infinity();

  void add(const double value) {
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }

  bool empty() const {
    return !std::isfinite(minimum) || !std::isfinite(maximum);
  }

  double span() const { return empty() ? 0.0 : maximum - minimum; }
};

struct LocalDatasetStatistics {
  size_t samples = 0;
  size_t validPoseRows = 0;
  std::array<SpanMetric, 3> translation;
  double rotationSpreadRad = 0.0;
  double translationCoverageScore = 0.0;
  double rotationCoverageScore = 0.0;
};

double sampleCoverageFactor(const size_t samples) {
  constexpr double kFullCoverageSamples = 8.0;
  if (samples <= 2) {
    return 0.0;
  }
  return std::min(1.0, (static_cast<double>(samples) - 2.0) /
                           (kFullCoverageSamples - 2.0));
}

double translationRange(const LocalDatasetStatistics &stats) {
  return std::sqrt(std::pow(stats.translation[0].span(), 2.0) +
                   std::pow(stats.translation[1].span(), 2.0) +
                   std::pow(stats.translation[2].span(), 2.0));
}

double rotationSpreadDeg(const LocalDatasetStatistics &stats) {
  return stats.rotationSpreadRad * 180.0 / std::numbers::pi;
}

void updateCoverageScores(LocalDatasetStatistics &stats) {
  const double density = sampleCoverageFactor(stats.validPoseRows);
  stats.translationCoverageScore =
      density * std::min(1.0, translationRange(stats) / 0.10);
  stats.rotationCoverageScore =
      density * std::min(1.0, rotationSpreadDeg(stats) / 25.0);
}

bool isBlankLine(const std::string &line) {
  return std::ranges::all_of(
      line, [](const unsigned char ch) { return std::isspace(ch); });
}

size_t minimumPoseColumns(const int32_t posesFormat) {
  switch (posesFormat) {
  case 1:
  case 2:
    return 7;
  case 3:
  case 4:
  case 5:
  case 6:
    return 6;
  case 7:
    return 1;
  default:
    return 1;
  }
}

// poses.csv is intentionally a TSV file: captureSample() writes it with
// csv::make_tsv_writer(). Reading numeric values as whitespace-separated tokens
// accepts existing tab-delimited datasets and does not depend on CSV
// autodetect.
std::optional<std::vector<double>> parsePoseTsvLine(const std::string &line) {
  if (isBlankLine(line)) {
    return std::nullopt;
  }

  std::string normalised = line;
  std::replace(normalised.begin(), normalised.end(), ',', ' ');
  std::replace(normalised.begin(), normalised.end(), ';', ' ');

  std::istringstream stream(normalised);
  std::vector<double> values;
  double value = 0.0;
  while (stream >> value) {
    if (!std::isfinite(value)) {
      return std::nullopt;
    }
    values.push_back(value);
  }
  stream >> std::ws;
  if (!stream.eof() || values.empty()) {
    return std::nullopt;
  }
  return values;
}

cv::Matx33d quaternionRotation(double qx, double qy, double qz, double qw) {
  const double norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (norm <= std::numeric_limits<double>::epsilon()) {
    return cv::Matx33d::eye();
  }
  qx /= norm;
  qy /= norm;
  qz /= norm;
  qw /= norm;

  return cv::Matx33d(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),
                     2.0 * (qx * qz + qy * qw), 2.0 * (qx * qy + qz * qw),
                     1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw),
                     2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw),
                     1.0 - 2.0 * (qx * qx + qy * qy));
}

cv::Matx33d rpyRotation(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);

  return cv::Matx33d(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                     sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                     -sp, cp * sr, cp * cr);
}

std::optional<cv::Matx33d> rowRotation(const std::vector<double> &row,
                                       const int32_t posesFormat) {
  switch (posesFormat) {
  case 1:
    return quaternionRotation(row[3], row[4], row[5], row[6]);
  case 2:
    return quaternionRotation(row[4], row[5], row[6], row[3]);
  case 3:
    return rpyRotation(row[3], row[4], row[5]);
  case 4:
    return rpyRotation(row[3] * std::numbers::pi / 180.0,
                       row[4] * std::numbers::pi / 180.0,
                       row[5] * std::numbers::pi / 180.0);
  case 5:
    return rpyRotation(row[5], row[4], row[3]);
  case 6:
    return rpyRotation(row[5] * std::numbers::pi / 180.0,
                       row[4] * std::numbers::pi / 180.0,
                       row[3] * std::numbers::pi / 180.0);
  default:
    return std::nullopt;
  }
}

double rotationDistanceRad(const cv::Matx33d &left, const cv::Matx33d &right) {
  const cv::Matx33d relative = left.t() * right;
  const double trace = relative(0, 0) + relative(1, 1) + relative(2, 2);
  const double cosAngle = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(cosAngle);
}

void findGripper2BaseTsv(const fs::path &dataset, const int32_t posesFormat,
                         calibration_helpers::CalibrationData &data) {
  if (posesFormat == 7) {
    throw std::runtime_error(
        "JointState pose format is not implemented for hand-eye calibration");
  }
  if (posesFormat < 1 || posesFormat > 6) {
    throw std::runtime_error(
        "Unsupported pose format; choose a value from 1 to 6");
  }

  const fs::path posesPath = dataset / CSV_FILENAME;
  std::ifstream poses(posesPath);
  if (!poses.is_open()) {
    throw std::runtime_error("Cannot open TSV poses file: " +
                             posesPath.string());
  }

  data.R_gripper2base.clear();
  data.t_gripper2base.clear();

  std::string line;
  size_t lineNumber = 0;
  size_t sampleIndex = 0;
  while (std::getline(poses, line)) {
    ++lineNumber;
    if (isBlankLine(line)) {
      continue;
    }

    const size_t currentSampleIndex = sampleIndex++;
    if (data.rejectedImages.contains(currentSampleIndex)) {
      continue;
    }

    const auto row = parsePoseTsvLine(line);
    if (!row || row->size() < minimumPoseColumns(posesFormat)) {
      throw std::runtime_error("Invalid TSV pose row " +
                               std::to_string(lineNumber) + " in " +
                               posesPath.string());
    }

    const auto rotation = rowRotation(*row, posesFormat);
    if (!rotation) {
      throw std::runtime_error("Cannot calculate pose rotation for row " +
                               std::to_string(lineNumber));
    }

    cv::Mat R(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        R.at<double>(r, c) = (*rotation)(r, c);
      }
    }
    cv::Mat t = (cv::Mat_<double>(3, 1) << (*row)[0], (*row)[1], (*row)[2]);
    data.R_gripper2base.emplace_back(std::move(R));
    data.t_gripper2base.emplace_back(std::move(t));
  }

  if (data.R_gripper2base.empty()) {
    throw std::runtime_error("No valid TSV pose rows remain for calibration");
  }
  if (data.R_gripper2base.size() != data.R_target2cam.size()) {
    throw std::runtime_error(
        "Pose/image mismatch after rejected images were removed: robot poses=" +
        std::to_string(data.R_gripper2base.size()) +
        ", detected target poses=" + std::to_string(data.R_target2cam.size()));
  }
}

double
calculateValidationRms(const calibration_helpers::CalibrationData &data) {
  if (data.R_target2cam.empty() ||
      data.R_target2cam.size() != data.R_gripper2base.size()) {
    throw std::runtime_error("Cannot calculate validation RMS without matched "
                             "robot and target poses");
  }

  const cv::Mat T_calibrated = calibration_helpers::toTransformMatrix(
      data.R_cam2gripper, data.t_cam2gripper);
  std::vector<cv::Vec3d> targetOrigins;
  targetOrigins.reserve(data.R_target2cam.size());

  for (size_t i = 0; i < data.R_target2cam.size(); ++i) {
    const cv::Mat T_robot = calibration_helpers::toTransformMatrix(
        data.R_gripper2base[i], data.t_gripper2base[i]);
    const cv::Mat T_target2cam = calibration_helpers::toTransformMatrix(
        data.R_target2cam[i], data.t_target2cam[i]);

    // The transform supplied to calibrateHandEye is the moving frame pose.
    // In eye-in-hand this gives target coordinates in base; after the
    // eye-to-hand inversion it gives target coordinates in gripper.
    const cv::Mat T_target_fixed = T_robot * T_calibrated * T_target2cam;
    targetOrigins.emplace_back(T_target_fixed.at<double>(0, 3),
                               T_target_fixed.at<double>(1, 3),
                               T_target_fixed.at<double>(2, 3));
  }

  cv::Vec3d mean(0.0, 0.0, 0.0);
  for (const cv::Vec3d &origin : targetOrigins) {
    mean += origin;
  }
  mean *= 1.0 / static_cast<double>(targetOrigins.size());

  double squaredResidualSum = 0.0;
  for (const cv::Vec3d &origin : targetOrigins) {
    const cv::Vec3d residual = origin - mean;
    squaredResidualSum += residual.dot(residual);
  }
  return std::sqrt(squaredResidualSum /
                   static_cast<double>(targetOrigins.size()));
}

std::string liveRmsWaitingMessage(const int32_t minSamples) {
  return "waiting for at least " + std::to_string(minSamples) + " samples";
}

std::optional<LocalDatasetStatistics>
readDatasetStatisticsTsv(const fs::path &dataset, const int32_t posesFormat,
                         std::string *error = nullptr) {
  const fs::path posesPath = dataset / CSV_FILENAME;
  std::ifstream poses(posesPath);
  if (!poses.is_open()) {
    if (error) {
      *error = "cannot open " + posesPath.string();
    }
    return std::nullopt;
  }

  LocalDatasetStatistics stats;
  std::vector<cv::Matx33d> rotations;
  std::string line;
  size_t lineNumber = 0;
  while (std::getline(poses, line)) {
    ++lineNumber;
    if (isBlankLine(line)) {
      continue;
    }
    ++stats.samples;
    const auto row = parsePoseTsvLine(line);
    if (!row || row->size() < minimumPoseColumns(posesFormat)) {
      if (error) {
        *error = "invalid TSV pose row " + std::to_string(lineNumber) + " in " +
                 posesPath.string();
      }
      continue;
    }

    ++stats.validPoseRows;
    if (posesFormat >= 1 && posesFormat <= 6) {
      stats.translation[0].add((*row)[0]);
      stats.translation[1].add((*row)[1]);
      stats.translation[2].add((*row)[2]);
      if (const auto rotation = rowRotation(*row, posesFormat)) {
        rotations.push_back(*rotation);
      }
    }
  }

  for (size_t i = 0; i < rotations.size(); ++i) {
    for (size_t j = i + 1; j < rotations.size(); ++j) {
      stats.rotationSpreadRad =
          std::max(stats.rotationSpreadRad,
                   rotationDistanceRad(rotations[i], rotations[j]));
    }
  }
  updateCoverageScores(stats);
  return stats;
}

std::optional<size_t> countImagesLocal(const fs::path &dataset) {
  const fs::path imageDir = dataset / IMG_FOLDERNAME;
  if (!fs::exists(imageDir) || !fs::is_directory(imageDir)) {
    return size_t{0};
  }

  const std::regex imagePattern(R"(^(\d+)\.png$)");
  size_t count = 0;
  for (const auto &entry : fs::directory_iterator(imageDir)) {
    if (entry.is_regular_file() &&
        std::regex_match(entry.path().filename().string(), imagePattern)) {
      ++count;
    }
  }
  return count;
}

std::optional<size_t> countPosesTsv(const fs::path &dataset,
                                    const int32_t posesFormat) {
  const auto stats = readDatasetStatisticsTsv(dataset, posesFormat);
  return stats ? std::optional<size_t>(stats->validPoseRows) : std::nullopt;
}

std::optional<size_t> resumeDatasetIndex(const fs::path &dataset,
                                         const int32_t posesFormat,
                                         std::string &error) {
  const fs::path imageDir = dataset / IMG_FOLDERNAME;
  if (!fs::exists(imageDir) || !fs::is_directory(imageDir)) {
    error = "images folder is missing";
    return std::nullopt;
  }

  const std::regex imagePattern(R"(^(\d+)\.png$)");
  std::set<size_t> indices;
  for (const auto &entry : fs::directory_iterator(imageDir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::smatch match;
    const std::string filename = entry.path().filename().string();
    if (std::regex_match(filename, match, imagePattern)) {
      indices.insert(static_cast<size_t>(std::stoull(match[1].str())));
    }
  }

  if (indices.empty()) {
    const fs::path posesPath = dataset / CSV_FILENAME;
    if (fs::exists(posesPath) && fs::file_size(posesPath) > 0) {
      error = "poses.csv contains rows, but images folder contains no samples";
      return std::nullopt;
    }
    return size_t{0};
  }

  std::string poseError;
  const auto stats = readDatasetStatisticsTsv(dataset, posesFormat, &poseError);
  if (!stats) {
    error = "poses.csv is missing or cannot be opened as TSV: " + poseError;
    return std::nullopt;
  }
  if (stats->validPoseRows != indices.size()) {
    error = "poses.csv is TSV but does not contain one valid pose row per "
            "image: images=" +
            std::to_string(indices.size()) +
            ", valid pose rows=" + std::to_string(stats->validPoseRows);
    return std::nullopt;
  }

  size_t expected = 0;
  for (const size_t index : indices) {
    if (index != expected) {
      error = "image numbering is not contiguous; missing image index " +
              std::to_string(expected);
      return std::nullopt;
    }
    ++expected;
  }
  return *indices.rbegin() + 1;
}

bool validateDatasetTsv(const fs::path &dataset, const int32_t posesFormat,
                        std::string &error) {
  const auto imageCount = countImagesLocal(dataset);
  if (!imageCount || *imageCount == 0) {
    error = "no calibration images found";
    return false;
  }

  std::string poseError;
  const auto stats = readDatasetStatisticsTsv(dataset, posesFormat, &poseError);
  if (!stats) {
    error = "poses.csv is missing or cannot be opened as TSV: " + poseError;
    return false;
  }
  if (stats->validPoseRows == 0) {
    error = poseError.empty() ? "poses.csv contains no valid TSV pose rows"
                              : poseError;
    return false;
  }
  if (stats->validPoseRows != *imageCount) {
    error = "image/pose count mismatch: images=" + std::to_string(*imageCount) +
            ", valid TSV pose rows=" + std::to_string(stats->validPoseRows);
    return false;
  }
  return true;
}

} // namespace

class HandEyeBackendNode final : public rclcpp::Node {
public:
  HandEyeBackendNode() : Node("hand_eye_backend") {
    declareParameters();
    refreshConfigFromParameters();

    fish_ = ros_babel_fish::BabelFish::make_shared();

    configureSrv_ = create_service<Trigger>(
        "~/configure", [this](const TriggerRequest, TriggerResponse response) {
          configure(response);
        });
    getStateSrv_ = create_service<Trigger>(
        "~/get_state", [this](const TriggerRequest, TriggerResponse response) {
          refreshConfigFromParameters();
          updateDatasetStatistics();
          response->success = true;
          response->message = stateJson();
        });
    prepareDatasetSrv_ = create_service<Trigger>(
        "~/prepare_dataset",
        [this](const TriggerRequest, TriggerResponse response) {
          prepareDataset(response);
        });
    startCollectionSrv_ = create_service<Trigger>(
        "~/start_collection",
        [this](const TriggerRequest, TriggerResponse response) {
          startCollection(response);
        });
    captureSampleSrv_ = create_service<Trigger>(
        "~/capture_sample",
        [this](const TriggerRequest, TriggerResponse response) {
          captureSample(response);
        });
    removeSampleSrv_ = create_service<Trigger>(
        "~/remove_sample",
        [this](const TriggerRequest, TriggerResponse response) {
          removeSample(response);
        });
    stopCollectionSrv_ = create_service<Trigger>(
        "~/stop_collection",
        [this](const TriggerRequest, TriggerResponse response) {
          stopCollection(response);
        });
    runCalibrationSrv_ = create_service<Trigger>(
        "~/run_calibration",
        [this](const TriggerRequest, TriggerResponse response) {
          runCalibration(response);
        });

    RCLCPP_INFO(get_logger(),
                "Hand-eye backend is ready. Use parameters for arguments and "
                "std_srvs/Trigger services for commands.");
  }

private:
  void declareParameters() {
    declare_parameter<std::string>("arm_topic", "");
    declare_parameter<std::string>("image_topic", "");
    declare_parameter<std::string>("dataset_path", "dataset");
    declare_parameter<std::string>("calibration_path", ".");
    declare_parameter<std::string>("pattern_info", "");
    declare_parameter<int32_t>("poses_format", 1);
    declare_parameter<bool>("eye_to_hand", true);
    declare_parameter<std::string>("intrinsics_path", "");
    declare_parameter<bool>("require_pattern_detection", true);
    declare_parameter<bool>("overwrite_dataset", false);
    declare_parameter<bool>("resume_dataset", true);
    declare_parameter<int32_t>("min_samples", 3);
    declare_parameter<bool>("stat_translation_spread", true);
    declare_parameter<bool>("stat_rotation_spread", true);
    declare_parameter<bool>("stat_validation_rms", false);
  }

  void refreshConfigFromParameters() {
    armTopic_ = get_parameter("arm_topic").as_string();
    imageTopic_ = get_parameter("image_topic").as_string();
    datasetPath_ = get_parameter("dataset_path").as_string();
    calibrationPath_ = get_parameter("calibration_path").as_string();
    patternInfo_ = get_parameter("pattern_info").as_string();
    posesFormat_ = get_parameter("poses_format").as_int();
    eyeToHand_ = get_parameter("eye_to_hand").as_bool();
    intrinsicsPath_ = get_parameter("intrinsics_path").as_string();
    requirePatternDetection_ =
        get_parameter("require_pattern_detection").as_bool();
    overwriteDataset_ = get_parameter("overwrite_dataset").as_bool();
    resumeDataset_ = get_parameter("resume_dataset").as_bool();
    minSamples_ = std::max<int32_t>(1, get_parameter("min_samples").as_int());
    statTranslationSpread_ = get_parameter("stat_translation_spread").as_bool();
    statRotationSpread_ = get_parameter("stat_rotation_spread").as_bool();
    statValidationRms_ = get_parameter("stat_validation_rms").as_bool();
  }

  fs::path datasetDir() const { return fs::path(datasetPath_); }

  fs::path calibrationFile() const {
    fs::path path(calibrationPath_);
    if (path.empty() || path == ".") {
      return fs::path(CALIBRATION_FILENAME);
    }
    if (path.extension() == ".yaml" || path.extension() == ".yml") {
      return path;
    }
    return path / CALIBRATION_FILENAME;
  }

  std::optional<std::string> findTopicType(const std::string &topicName) const {
    if (topicName.empty()) {
      return std::nullopt;
    }
    const auto topics = get_topic_names_and_types();
    const auto it =
        std::ranges::find_if(topics, [&topicName](const auto &entry) {
          return entry.first == topicName;
        });
    if (it == topics.end() || it->second.empty()) {
      return std::nullopt;
    }
    return it->second.front();
  }

  bool topicHasType(const std::string &topicName,
                    const std::string &topicType) const {
    const auto topics = get_topic_names_and_types();
    const auto it =
        std::ranges::find_if(topics, [&topicName](const auto &entry) {
          return entry.first == topicName;
        });
    if (it == topics.end()) {
      return false;
    }
    return std::ranges::find(it->second, topicType) != it->second.end();
  }

  bool ensurePatternDetector(std::string &error) {
    if (patternInfo_.empty()) {
      error = "pattern_info parameter is empty";
      return false;
    }
    if (patternDetector_ && detectorPatternInfo_ == patternInfo_) {
      return true;
    }
    try {
      patternDetector_ =
          pattern_detector::PatternDetector::fromPatternInfo(patternInfo_);
      detectorPatternInfo_ = patternInfo_;
      return true;
    } catch (const std::exception &ex) {
      error = ex.what();
      patternDetector_.reset();
      detectorPatternInfo_.clear();
      return false;
    }
  }

  void configure(const TriggerResponse &response) {
    refreshConfigFromParameters();
    lastError_.clear();

    if (!patternInfo_.empty()) {
      std::string error;
      if (!ensurePatternDetector(error)) {
        fail(response, error);
        return;
      }
    }

    armTopicType_ = findTopicType(armTopic_).value_or("");
    imageTopicType_ = findTopicType(imageTopic_).value_or("");
    configured_ = true;

    std::ostringstream message;
    message << "Configured backend";
    if (armTopicType_.empty() && !armTopic_.empty()) {
      message << "; arm topic is not visible yet";
    }
    if (imageTopicType_.empty() && !imageTopic_.empty()) {
      message << "; image topic is not visible yet";
    }

    response->success = true;
    response->message = message.str();
  }

  void prepareDataset(const TriggerResponse &response) {
    refreshConfigFromParameters();
    const fs::path dataset = datasetDir();

    try {
      if (dataset.empty()) {
        fail(response, "dataset_path parameter is empty");
        return;
      }

      if (fs::exists(dataset)) {
        if (!fs::is_directory(dataset)) {
          if (!overwriteDataset_) {
            fail(response, "dataset_path exists and is not a directory; set "
                           "overwrite_dataset=true to replace it");
            return;
          }
          fs::remove_all(dataset);
        } else if (overwriteDataset_) {
          fs::remove_all(dataset);
        } else if (resumeDataset_) {
          std::string error;
          const std::optional<size_t> nextIndex =
              resumeDatasetIndex(dataset, posesFormat_, error);
          if (!nextIndex) {
            fail(response, "cannot resume dataset: " + error);
            return;
          }
          nextSampleIndex_ = *nextIndex;
          datasetPrepared_ = true;
          lastError_.clear();
          updateDatasetStatistics();
          response->success = true;
          response->message = "Dataset resume prepared; next sample index is " +
                              std::to_string(nextSampleIndex_);
          return;
        } else {
          fail(response,
               "Dataset folder already exists. Choose Continue Dataset to "
               "append samples or Start Over to replace it.");
          return;
        }
      }

      fs::create_directories(dataset / IMG_FOLDERNAME);
      nextSampleIndex_ = 0;
      datasetPrepared_ = true;
      lastError_.clear();
      updateDatasetStatistics();
      response->success = true;
      response->message = "Dataset directory prepared at " + dataset.string();
    } catch (const std::exception &ex) {
      fail(response, ex.what());
    }
  }

  void startCollection(const TriggerResponse &response) {
    refreshConfigFromParameters();
    if (armTopic_.empty() || imageTopic_.empty()) {
      fail(response, "arm_topic and image_topic parameters are required");
      return;
    }

    if (!datasetPrepared_) {
      fail(response, "Dataset is not prepared. Check dataset folder settings.");
      return;
    }

    if (collecting_) {
      response->success = true;
      response->message = "Collection is already running";
      return;
    }

    try {
      std::string imageSubscriptionTopic = imageTopic_;
      if (topicHasType(imageTopic_, "sensor_msgs/msg/CompressedImage")) {
        imageTopicType_ = "sensor_msgs/msg/CompressedImage";
      } else if (topicHasType(imageTopic_ + "/compressed",
                              "sensor_msgs/msg/CompressedImage")) {
        imageSubscriptionTopic = imageTopic_ + "/compressed";
        imageTopicType_ = "sensor_msgs/msg/CompressedImage";
      } else {
        imageTopicType_ = findTopicType(imageTopic_).value_or(imageTopicType_);
      }
      imageSub_.reset();
      compressedImageSub_.reset();
      if (imageTopicType_ == "sensor_msgs/msg/CompressedImage") {
        compressedImageSub_ =
            create_subscription<sensor_msgs::msg::CompressedImage>(
                imageSubscriptionTopic, rclcpp::SensorDataQoS(),
                [this](const sensor_msgs::msg::CompressedImage::ConstSharedPtr
                           msg) {
                  try {
                    const cv::Mat encoded(msg->data, true);
                    cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
                    if (image.empty()) {
                      RCLCPP_WARN(get_logger(),
                                  "Cannot decode compressed image frame");
                      return;
                    }
                    std::lock_guard<std::mutex> lock(imageMutex_);
                    latestImage_ = image.clone();
                  } catch (const std::exception &ex) {
                    RCLCPP_WARN(get_logger(), "Cannot decode image: %s",
                                ex.what());
                  }
                });
      } else {
        imageSub_ = create_subscription<sensor_msgs::msg::Image>(
            imageSubscriptionTopic, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
              try {
                cv::Mat image = cv_bridge::toCvCopy(
                                    *msg, sensor_msgs::image_encodings::BGR8)
                                    ->image;
                std::lock_guard<std::mutex> lock(imageMutex_);
                latestImage_ = image.clone();
              } catch (const std::exception &ex) {
                RCLCPP_WARN(get_logger(), "Cannot convert image: %s",
                            ex.what());
              }
            });
      }

      armSub_ = fish_->create_subscription(
          *this, armTopic_, 1,
          [this](ros_babel_fish::CompoundMessage::SharedPtr msg) {
            {
              std::lock_guard<std::mutex> lock(armMutex_);
              latestArmMsg_ = std::move(msg);
              armTopicType_ = latestArmMsg_->name();
            }
            RCLCPP_DEBUG(get_logger(), "Got arm message of type %s",
                         armTopicType_.c_str());
          });

      collecting_ = true;
      configured_ = true;
      response->success = true;
      response->message = "Collection started";
    } catch (const std::exception &ex) {
      fail(response, ex.what());
    }
  }

  void captureSample(const TriggerResponse &response) {
    refreshConfigFromParameters();

    cv::Mat image;
    {
      std::lock_guard<std::mutex> lock(imageMutex_);
      image = latestImage_.clone();
    }

    ros_babel_fish::CompoundMessage::SharedPtr armMsg;
    {
      std::lock_guard<std::mutex> lock(armMutex_);
      armMsg = latestArmMsg_;
    }

    if (image.empty()) {
      fail(response, "no image has been received yet");
      return;
    }
    if (!armMsg) {
      fail(response, "no arm message has been received yet");
      return;
    }

    if (requirePatternDetection_) {
      std::string error;
      if (!ensurePatternDetector(error)) {
        fail(response, error);
        return;
      }
      if (!patternDetector_->detect(image)) {
        fail(response,
             "calibration target was not found; sample was not saved");
        return;
      }
    }

    try {
      const fs::path dataset = datasetDir();
      fs::create_directories(dataset / IMG_FOLDERNAME);

      const fs::path imagePath = dataset / IMG_FOLDERNAME /
                                 (std::to_string(nextSampleIndex_) + ".png");
      if (!cv::imwrite(imagePath.string(), image)) {
        fail(response, "failed to write image " + imagePath.string());
        return;
      }

      std::vector<std::string> poseDump;
      collect_helpers::dumpMessageContent(*armMsg, poseDump);

      std::ofstream outFile(dataset / CSV_FILENAME, std::ios::app);
      if (!outFile.is_open()) {
        fail(response, "failed to open " + (dataset / CSV_FILENAME).string());
        return;
      }
      auto writer = csv::make_tsv_writer(outFile);
      writer << poseDump;
      outFile.flush();
      if (!outFile.good()) {
        fail(response, "failed to write " + (dataset / CSV_FILENAME).string());
        return;
      }
      outFile.close();

      const size_t savedIndex = nextSampleIndex_++;
      lastError_.clear();
      updateDatasetStatistics();
      updateLiveTargetLockRms();
      response->success = true;
      response->message = "Sample " + std::to_string(savedIndex) + " saved";
    } catch (const std::exception &ex) {
      fail(response, ex.what());
    }
  }

  void removeSample(const TriggerResponse &response) {
    refreshConfigFromParameters();
    if (!datasetPrepared_) {
      fail(response, "Dataset is not prepared. Check dataset folder settings.");
      return;
    }

    const fs::path dataset = datasetDir();
    if (nextSampleIndex_ == 0) {
      std::string error;
      const std::optional<size_t> nextIndex =
          resumeDatasetIndex(dataset, posesFormat_, error);
      if (!nextIndex || *nextIndex == 0) {
        fail(response, "dataset has no samples to remove");
        return;
      }
      nextSampleIndex_ = *nextIndex;
    }

    const size_t removedIndex = nextSampleIndex_ - 1;
    const fs::path imagePath =
        dataset / IMG_FOLDERNAME / (std::to_string(removedIndex) + ".png");
    const fs::path posesPath = dataset / CSV_FILENAME;

    try {
      if (!fs::exists(imagePath) || !fs::is_regular_file(imagePath)) {
        fail(response, "cannot remove sample: image " + imagePath.string() +
                           " is missing");
        return;
      }

      std::ifstream input(posesPath);
      if (!input.is_open()) {
        fail(response,
             "cannot remove sample: cannot open " + posesPath.string());
        return;
      }

      std::vector<std::string> lines;
      std::string line;
      std::optional<size_t> lastPoseLine;
      while (std::getline(input, line)) {
        if (!isBlankLine(line)) {
          lastPoseLine = lines.size();
        }
        lines.emplace_back(line);
      }
      input.close();

      if (!lastPoseLine) {
        fail(response, "cannot remove sample: poses.csv has no pose rows");
        return;
      }

      lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(*lastPoseLine));

      std::ofstream output(posesPath, std::ios::trunc);
      if (!output.is_open()) {
        fail(response,
             "cannot remove sample: cannot write " + posesPath.string());
        return;
      }
      for (const std::string &row : lines) {
        output << row << '\n';
      }
      output.flush();
      if (!output.good()) {
        fail(response,
             "cannot remove sample: failed to update " + posesPath.string());
        return;
      }

      if (!fs::remove(imagePath)) {
        fail(response,
             "cannot remove sample: failed to delete " + imagePath.string());
        return;
      }

      nextSampleIndex_ = removedIndex;
      lastError_.clear();
      lastValidationRms_.reset();
      updateDatasetStatistics();
      updateLiveTargetLockRms();

      response->success = true;
      response->message = "Sample " + std::to_string(removedIndex) + " removed";
    } catch (const std::exception &ex) {
      fail(response, ex.what());
    }
  }

  void stopCollection(const TriggerResponse &response) {
    imageSub_.reset();
    compressedImageSub_.reset();
    armSub_.reset();
    collecting_ = false;

    updateDatasetStatistics();
    if (datasetStats_) {
      RCLCPP_INFO(get_logger(),
                  "Statistics: samples=%zu valid_pose_rows=%zu "
                  "translation coverage=%.0f%% "
                  "rotation coverage=%.0f%% "
                  "translation span=[%.5f %.5f %.5f] m "
                  "rotation span=%.2f deg",
                  datasetStats_->samples, datasetStats_->validPoseRows,
                  datasetStats_->translationCoverageScore * 100.0,
                  datasetStats_->rotationCoverageScore * 100.0,
                  datasetStats_->translation[0].span(),
                  datasetStats_->translation[1].span(),
                  datasetStats_->translation[2].span(),
                  rotationSpreadDeg(*datasetStats_));
    }

    response->success = true;
    response->message = "Collection stopped";
  }

  void runCalibration(const TriggerResponse &response) {
    refreshConfigFromParameters();
    std::string datasetError;
    if (!validateDatasetTsv(datasetDir(), posesFormat_, datasetError)) {
      fail(response, "dataset is invalid: " + datasetError);
      return;
    }
    if (patternInfo_.empty()) {
      fail(response, "pattern_info parameter is required");
      return;
    }

    try {
      calibration_helpers::CalibrationData data = calculateCalibration();
      lastValidationRms_ = calculateValidationRms(data);
      lastValidationRmsStatus_ = "updated";
      writeCalibration(data, calibrationFile());
      lastError_.clear();

      response->success = true;
      response->message =
          "Calibration saved to " + calibrationFile().string() +
          "; Target lock RMS=" + std::to_string(*lastValidationRms_);
    } catch (const std::exception &ex) {
      fail(response, ex.what());
    }
  }

  calibration_helpers::CalibrationData calculateCalibration() {
    calibration_helpers::CalibrationPattern pattern(datasetDir(), patternInfo_);
    calibration_helpers::CalibrationData data;

    const bool useKnownIntrinsics = !intrinsicsPath_.empty();
    if (useKnownIntrinsics && !data.readIntrinsics(fs::path(intrinsicsPath_))) {
      throw std::runtime_error("cannot read intrinsics from " +
                               intrinsicsPath_);
    }

    calibration_helpers::findTarget2Cam(pattern, data, useKnownIntrinsics);
    findGripper2BaseTsv(datasetDir(), posesFormat_, data);

    if (eyeToHand_) {
      invertGripperBase(data);
    }

    cv::calibrateHandEye(data.R_gripper2base, data.t_gripper2base,
                         data.R_target2cam, data.t_target2cam,
                         data.R_cam2gripper, data.t_cam2gripper);
    data.T = calibration_helpers::toTransformMatrix(data.R_cam2gripper,
                                                    data.t_cam2gripper);
    return data;
  }

  void updateLiveTargetLockRms() {
    if (!statValidationRms_) {
      lastValidationRmsStatus_.clear();
      return;
    }

    lastValidationRms_.reset();
    const auto imageCount = countImagesLocal(datasetDir());
    const auto poseCount = countPosesTsv(datasetDir(), posesFormat_);
    const size_t pairedSamples =
        imageCount && poseCount ? std::min(*imageCount, *poseCount) : 0;
    if (pairedSamples < static_cast<size_t>(minSamples_)) {
      lastValidationRmsStatus_ = liveRmsWaitingMessage(minSamples_);
      return;
    }
    if (patternInfo_.empty()) {
      lastValidationRmsStatus_ = "waiting for pattern settings";
      return;
    }

    try {
      calibration_helpers::CalibrationData data = calculateCalibration();
      lastValidationRms_ = calculateValidationRms(data);
      lastValidationRmsStatus_ = "updated";
      RCLCPP_DEBUG(get_logger(), "Target lock RMS updated: %.6f m",
                   *lastValidationRms_);
    } catch (const std::exception &ex) {
      lastValidationRmsStatus_ = userMessage(ex.what());
      RCLCPP_DEBUG(get_logger(), "Cannot update target lock RMS: %s",
                   lastValidationRmsStatus_.c_str());
    }
  }

  void invertGripperBase(calibration_helpers::CalibrationData &data) const {
    std::vector<cv::Mat> R_base2gripper;
    std::vector<cv::Mat> t_base2gripper;
    const size_t count = data.R_gripper2base.size();
    R_base2gripper.reserve(count);
    t_base2gripper.reserve(count);

    for (size_t i = 0; i < data.t_gripper2base.size(); ++i) {
      cv::Mat &R = data.R_gripper2base[i];
      cv::Mat &t = data.t_gripper2base[i];
      cv::Mat R_inv = R.t();
      cv::Mat t_inv = -R_inv * t;
      R_base2gripper.emplace_back(R_inv);
      t_base2gripper.emplace_back(t_inv);
    }

    data.R_gripper2base = std::move(R_base2gripper);
    data.t_gripper2base = std::move(t_base2gripper);
  }

  void writeCalibration(const calibration_helpers::CalibrationData &data,
                        const fs::path &path) const {
    if (path.has_parent_path()) {
      fs::create_directories(path.parent_path());
    }

    cv::FileStorage out(path.string(), cv::FileStorage::WRITE);
    out << "T" << data.T;
    out << "K" << data.K;
    out << "D" << data.D;
    out.release();
  }

  void fail(const TriggerResponse &response, const std::string &message) {
    const std::string readableMessage = userMessage(message);
    lastError_ = readableMessage;
    response->success = false;
    response->message = readableMessage;
    RCLCPP_WARN(get_logger(), "%s", readableMessage.c_str());
  }

  void updateDatasetStatistics() {
    datasetStats_ = readDatasetStatisticsTsv(datasetDir(), posesFormat_);
  }

  std::string stateJson() const {
    const std::optional<size_t> imageCount = countImagesLocal(datasetDir());
    const std::optional<size_t> poseCount =
        countPosesTsv(datasetDir(), posesFormat_);
    const double progressPercent = datasetQualityPercent(imageCount, poseCount);

    std::ostringstream out;
    out << "{"
        << "\"configured\":" << jsonBool(configured_) << ","
        << "\"collecting\":" << jsonBool(collecting_) << ","
        << "\"dataset_prepared\":" << jsonBool(datasetPrepared_) << ","
        << "\"arm_topic\":\"" << jsonEscape(armTopic_) << "\","
        << "\"image_topic\":\"" << jsonEscape(imageTopic_) << "\","
        << "\"dataset_path\":\"" << jsonEscape(datasetPath_) << "\","
        << "\"calibration_path\":\"" << jsonEscape(calibrationFile().string())
        << "\","
        << "\"min_samples\":" << minSamples_ << ","
        << "\"progress_percent\":" << progressPercent << ","
        << "\"next_sample_index\":" << nextSampleIndex_ << ","
        << "\"image_count\":"
        << (imageCount ? std::to_string(*imageCount) : "null") << ","
        << "\"pose_count\":"
        << (poseCount ? std::to_string(*poseCount) : "null") << ","
        << "\"arm_topic_type\":\"" << jsonEscape(armTopicType_) << "\","
        << "\"image_topic_type\":\"" << jsonEscape(imageTopicType_) << "\","
        << "\"statistics_enabled\":{\"translation_spread\":"
        << jsonBool(statTranslationSpread_)
        << ",\"rotation_spread\":" << jsonBool(statRotationSpread_)
        << ",\"validation_rms\":" << jsonBool(statValidationRms_) << "},"
        << "\"statistics\":";
    if (datasetStats_) {
      out << "{"
          << "\"samples\":" << datasetStats_->samples << ","
          << "\"valid_pose_rows\":" << datasetStats_->validPoseRows << ","
          << "\"translation_spread_x\":" << datasetStats_->translation[0].span()
          << ","
          << "\"translation_spread_y\":" << datasetStats_->translation[1].span()
          << ","
          << "\"translation_spread_z\":" << datasetStats_->translation[2].span()
          << ","
          << "\"rotation_spread_rad\":" << datasetStats_->rotationSpreadRad
          << ","
          << "\"rotation_spread_deg\":" << rotationSpreadDeg(*datasetStats_)
          << ","
          << "\"translation_coverage_score\":"
          << datasetStats_->translationCoverageScore << ","
          << "\"rotation_coverage_score\":"
          << datasetStats_->rotationCoverageScore << "}";
    } else {
      out << "null";
    }
    out << ","
        << "\"last_validation_rms\":";
    if (lastValidationRms_) {
      out << *lastValidationRms_;
    } else {
      out << "null";
    }
    out << ",\"last_validation_rms_status\":\""
        << jsonEscape(lastValidationRmsStatus_) << "\""
        << ",\"last_error\":\"" << jsonEscape(lastError_) << "\"}";
    return out.str();
  }

  double datasetQualityPercent(const std::optional<size_t> imageCount,
                               const std::optional<size_t> poseCount) const {
    const size_t pairedSamples =
        imageCount && poseCount ? std::min(*imageCount, *poseCount) : 0;
    std::vector<double> criteria;
    criteria.push_back(std::min(1.0, static_cast<double>(pairedSamples) /
                                         static_cast<double>(minSamples_)));

    if (datasetStats_ && datasetStats_->validPoseRows > 0) {
      if (statTranslationSpread_) {
        criteria.push_back(datasetStats_->translationCoverageScore);
      }
      if (statRotationSpread_) {
        criteria.push_back(datasetStats_->rotationCoverageScore);
      }
    } else {
      if (statTranslationSpread_) {
        criteria.push_back(0.0);
      }
      if (statRotationSpread_) {
        criteria.push_back(0.0);
      }
    }

    const double total = std::accumulate(criteria.begin(), criteria.end(), 0.0);
    return 100.0 * total / static_cast<double>(criteria.size());
  }

  std::string armTopic_;
  std::string imageTopic_;
  std::string datasetPath_;
  std::string calibrationPath_;
  std::string patternInfo_;
  std::string intrinsicsPath_;
  std::string armTopicType_;
  std::string imageTopicType_;
  std::string lastError_;

  int32_t posesFormat_ = 1;
  int32_t minSamples_ = 3;
  bool eyeToHand_ = true;
  bool requirePatternDetection_ = true;
  bool overwriteDataset_ = false;
  bool resumeDataset_ = true;
  bool statTranslationSpread_ = true;
  bool statRotationSpread_ = true;
  bool statValidationRms_ = false;
  bool configured_ = false;
  bool collecting_ = false;
  bool datasetPrepared_ = false;
  size_t nextSampleIndex_ = 0;
  std::optional<double> lastValidationRms_;
  std::string lastValidationRmsStatus_;

  cv::Mat latestImage_;
  mutable std::mutex imageMutex_;
  ros_babel_fish::CompoundMessage::SharedPtr latestArmMsg_;
  mutable std::mutex armMutex_;

  ros_babel_fish::BabelFish::SharedPtr fish_;
  ros_babel_fish::BabelFishSubscription::SharedPtr armSub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
      compressedImageSub_;

  std::optional<LocalDatasetStatistics> datasetStats_;
  std::optional<pattern_detector::PatternDetector> patternDetector_;
  std::string detectorPatternInfo_;

  rclcpp::Service<Trigger>::SharedPtr configureSrv_;
  rclcpp::Service<Trigger>::SharedPtr getStateSrv_;
  rclcpp::Service<Trigger>::SharedPtr prepareDatasetSrv_;
  rclcpp::Service<Trigger>::SharedPtr startCollectionSrv_;
  rclcpp::Service<Trigger>::SharedPtr captureSampleSrv_;
  rclcpp::Service<Trigger>::SharedPtr removeSampleSrv_;
  rclcpp::Service<Trigger>::SharedPtr stopCollectionSrv_;
  rclcpp::Service<Trigger>::SharedPtr runCalibrationSrv_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HandEyeBackendNode>());
  rclcpp::shutdown();
  return 0;
}
