#include "lib/io/dataset_repository.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace hand_eye::io {
namespace {

namespace fs = std::filesystem;

constexpr const char *kAuditHeader =
    "index\timage_timestamp_ns\trobot_timestamp_ns\tdelta_ms\n";
constexpr double kFullTranslationRangeM = 0.10;
constexpr double kFullRotationSpreadDeg = 35.0;
constexpr double kFullCoverageSamples = 15.0;

struct ParsedPoses {
  std::size_t rowCount = 0;
  std::vector<PoseRow> rows;
  std::vector<std::string> errors;
};

struct AuditRecord {
  std::size_t index = 0;
  std::int64_t imageTimestampNs = 0;
  std::int64_t robotTimestampNs = 0;
  double deltaMs = 0.0;
};

struct ParsedAudit {
  std::vector<AuditRecord> records;
  std::vector<std::string> errors;
};

bool isBlank(const std::string &line) {
  return std::ranges::all_of(line, [](const unsigned char character) {
    return std::isspace(character) != 0;
  });
}

std::string readText(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Cannot open " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (input.bad()) {
    throw std::runtime_error("Cannot read " + path.string());
  }
  return contents.str();
}

void writeText(const fs::path &path, const std::string &contents) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("Cannot open " + path.string() + " for writing");
  }
  output << contents;
  output.flush();
  if (!output) {
    throw std::runtime_error("Cannot write " + path.string());
  }
}

fs::path temporaryPath(const fs::path &path, const std::string &purpose,
                       const std::string &suffix = {}) {
  static std::atomic<std::uint64_t> counter{0};
  const auto ticks =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return fs::path(path.string() + "." + purpose + "." + std::to_string(ticks) +
                  "." + std::to_string(counter.fetch_add(1)) + suffix);
}

void removeIgnoringErrors(const fs::path &path) {
  std::error_code error;
  fs::remove_all(path, error);
}

struct FileReplacement {
  fs::path target;
  fs::path staged;
  fs::path backup;
  bool hadTarget = false;
  bool backedUp = false;
  bool installed = false;

  FileReplacement(fs::path targetPath, fs::path stagedPath)
      : target(std::move(targetPath)), staged(std::move(stagedPath)) {}

  void install() {
    hadTarget = fs::exists(target);
    if (hadTarget) {
      backup = temporaryPath(target, "backup");
      fs::rename(target, backup);
      backedUp = true;
    }
    fs::rename(staged, target);
    installed = true;
  }

  void rollback() noexcept {
    std::error_code error;
    if (installed) {
      fs::remove_all(target, error);
      error.clear();
    }
    if (backedUp) {
      fs::rename(backup, target, error);
    }
    removeIgnoringErrors(staged);
  }

  void finish() noexcept {
    removeIgnoringErrors(backup);
    removeIgnoringErrors(staged);
  }
};

void installAll(std::vector<FileReplacement> &replacements) {
  std::size_t current = 0;
  try {
    for (; current < replacements.size(); ++current) {
      replacements[current].install();
    }
  } catch (...) {
    replacements[current].rollback();
    while (current > 0) {
      replacements[--current].rollback();
    }
    throw;
  }
  for (auto &replacement : replacements) {
    replacement.finish();
  }
}

std::vector<double> parseNumbers(const std::string &line) {
  std::istringstream stream(line);
  std::vector<double> values;
  double value = 0.0;
  while (stream >> value) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("pose contains a non-finite value");
    }
    values.push_back(value);
  }
  stream.clear();
  stream >> std::ws;
  if (!stream.eof()) {
    throw std::runtime_error("pose contains a non-numeric value");
  }
  return values;
}

cv::Mat quaternionRotation(double qx, double qy, double qz, double qw) {
  const double norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (!std::isfinite(norm) || norm <= std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error("pose contains a zero-length quaternion");
  }
  qx /= norm;
  qy /= norm;
  qz /= norm;
  qw /= norm;

  return (cv::Mat_<double>(3, 3) << 1.0 - 2.0 * (qy * qy + qz * qz),
          2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw),
          2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz),
          2.0 * (qy * qz - qx * qw), 2.0 * (qx * qz - qy * qw),
          2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy));
}

cv::Mat rpyRotation(const double roll, const double pitch, const double yaw) {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);

  return (cv::Mat_<double>(3, 3) << cy * cp, cy * sp * sr - sy * cr,
          cy * sp * cr + sy * sr, sy * cp, sy * sp * sr + cy * cr,
          sy * sp * cr - cy * sr, -sp, cp * sr, cp * cr);
}

PoseRow makePoseRow(const std::size_t sampleIndex,
                    const std::vector<double> &values,
                    const PoseFormat format) {
  if (values.size() < poseColumnCount(format)) {
    throw std::runtime_error(
        "expected at least " + std::to_string(poseColumnCount(format)) +
        " pose columns, got " + std::to_string(values.size()));
  }
  if (!std::ranges::all_of(
          values, [](const double value) { return std::isfinite(value); })) {
    throw std::runtime_error("pose contains a non-finite value");
  }

  cv::Mat rotation;
  switch (format) {
  case PoseFormat::QuaternionXyzw:
    rotation = quaternionRotation(values[3], values[4], values[5], values[6]);
    break;
  case PoseFormat::QuaternionWxyz:
    rotation = quaternionRotation(values[4], values[5], values[6], values[3]);
    break;
  case PoseFormat::RollPitchYawRadians:
    rotation = rpyRotation(values[3], values[4], values[5]);
    break;
  case PoseFormat::RollPitchYawDegrees:
    rotation = rpyRotation(values[3] * std::numbers::pi / 180.0,
                           values[4] * std::numbers::pi / 180.0,
                           values[5] * std::numbers::pi / 180.0);
    break;
  case PoseFormat::YawPitchRollRadians:
    rotation = rpyRotation(values[5], values[4], values[3]);
    break;
  case PoseFormat::YawPitchRollDegrees:
    rotation = rpyRotation(values[5] * std::numbers::pi / 180.0,
                           values[4] * std::numbers::pi / 180.0,
                           values[3] * std::numbers::pi / 180.0);
    break;
  }

  cv::Mat translation =
      (cv::Mat_<double>(3, 1) << values[0], values[1], values[2]);
  return PoseRow{sampleIndex, values,
                 core::RigidTransform(rotation, translation)};
}

ParsedPoses readPoses(const fs::path &path, const PoseFormat format) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Cannot open " + path.string());
  }

  ParsedPoses parsed;
  std::string line;
  std::size_t lineNumber = 0;
  while (std::getline(input, line)) {
    ++lineNumber;
    if (isBlank(line)) {
      continue;
    }
    const std::size_t sampleIndex = parsed.rowCount++;
    try {
      parsed.rows.push_back(
          makePoseRow(sampleIndex, parseNumbers(line), format));
    } catch (const std::exception &error) {
      parsed.errors.push_back("invalid pose row " + std::to_string(lineNumber) +
                              ": " + error.what());
    }
  }
  if (input.bad()) {
    throw std::runtime_error("Cannot read " + path.string());
  }
  return parsed;
}

std::vector<std::string> splitTsv(const std::string &line) {
  std::vector<std::string> fields;
  std::istringstream stream(line);
  std::string field;
  while (std::getline(stream, field, '\t')) {
    fields.push_back(field);
  }
  return fields;
}

ParsedAudit readAudit(const fs::path &path) {
  ParsedAudit parsed;
  if (!fs::exists(path)) {
    return parsed;
  }
  std::ifstream input(path);
  if (!input.is_open()) {
    parsed.errors.push_back("cannot open " + path.string());
    return parsed;
  }

  std::string line;
  std::size_t lineNumber = 0;
  bool firstDataOrHeader = true;
  while (std::getline(input, line)) {
    ++lineNumber;
    if (isBlank(line)) {
      continue;
    }
    if (firstDataOrHeader && line.starts_with("index\t")) {
      firstDataOrHeader = false;
      continue;
    }
    firstDataOrHeader = false;
    try {
      const auto fields = splitTsv(line);
      if (fields.size() < 4) {
        throw std::runtime_error("expected four columns");
      }
      if (fields[0].starts_with('-')) {
        throw std::runtime_error("sample index cannot be negative");
      }
      std::size_t parsedCharacters = 0;
      const auto index = std::stoull(fields[0], &parsedCharacters);
      if (parsedCharacters != fields[0].size()) {
        throw std::runtime_error("invalid sample index");
      }
      const auto imageTimestamp = std::stoll(fields[1], &parsedCharacters);
      if (parsedCharacters != fields[1].size()) {
        throw std::runtime_error("invalid image timestamp");
      }
      const auto robotTimestamp = std::stoll(fields[2], &parsedCharacters);
      if (parsedCharacters != fields[2].size()) {
        throw std::runtime_error("invalid robot timestamp");
      }
      const double delta = std::stod(fields[3], &parsedCharacters);
      if (parsedCharacters != fields[3].size() || !std::isfinite(delta)) {
        throw std::runtime_error("invalid timestamp delta");
      }
      parsed.records.push_back(AuditRecord{static_cast<std::size_t>(index),
                                           imageTimestamp, robotTimestamp,
                                           delta});
    } catch (const std::exception &error) {
      parsed.errors.push_back("invalid audit row " +
                              std::to_string(lineNumber) + ": " + error.what());
    }
  }
  return parsed;
}

std::string auditText(std::vector<AuditRecord> records) {
  std::ostringstream output;
  output << kAuditHeader << std::setprecision(17);
  for (const auto &record : records) {
    output << record.index << '\t' << record.imageTimestampNs << '\t'
           << record.robotTimestampNs << '\t' << record.deltaMs << '\n';
  }
  return output.str();
}

std::string appendPoseText(std::string contents,
                           const std::vector<double> &values) {
  if (!contents.empty() && contents.back() != '\n') {
    contents.push_back('\n');
  }
  std::ostringstream row;
  row << std::setprecision(17);
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      row << '\t';
    }
    row << values[i];
  }
  row << '\n';
  contents += row.str();
  return contents;
}

std::string removeLastPoseRow(const std::string &contents) {
  std::istringstream input(contents);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) {
    lines.push_back(line);
  }
  const auto last = std::find_if(
      lines.rbegin(), lines.rend(),
      [](const std::string &candidate) { return !isBlank(candidate); });
  if (last == lines.rend()) {
    throw std::runtime_error("poses.csv contains no pose rows");
  }
  lines.erase(std::next(last).base());

  std::ostringstream output;
  for (const auto &remaining : lines) {
    output << remaining << '\n';
  }
  return output.str();
}

double rotationDistanceDeg(const cv::Mat &left, const cv::Mat &right) {
  const cv::Mat relative = left.t() * right;
  const double trace = relative.at<double>(0, 0) + relative.at<double>(1, 1) +
                       relative.at<double>(2, 2);
  const double cosine = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(cosine) * 180.0 / std::numbers::pi;
}

void updateStatistics(DatasetStatistics &statistics,
                      const std::vector<PoseRow> &poses) {
  for (const auto &pose : poses) {
    for (int axis = 0; axis < 3; ++axis) {
      const double value = pose.transform.translation.at<double>(axis, 0);
      statistics.translation[axis].minimum =
          std::min(statistics.translation[axis].minimum, value);
      statistics.translation[axis].maximum =
          std::max(statistics.translation[axis].maximum, value);
    }
  }
  statistics.translationRangeM =
      std::sqrt(std::pow(statistics.translation[0].span(), 2.0) +
                std::pow(statistics.translation[1].span(), 2.0) +
                std::pow(statistics.translation[2].span(), 2.0));

  for (std::size_t i = 0; i < poses.size(); ++i) {
    for (std::size_t j = i + 1; j < poses.size(); ++j) {
      statistics.rotationSpreadDeg =
          std::max(statistics.rotationSpreadDeg,
                   rotationDistanceDeg(poses[i].transform.rotation,
                                       poses[j].transform.rotation));
    }
  }

  const double density =
      poses.size() <= 2
          ? 0.0
          : std::min(1.0, (poses.size() - 2.0) / (kFullCoverageSamples - 2.0));
  statistics.translationCoverageScore =
      density *
      std::min(1.0, statistics.translationRangeM / kFullTranslationRangeM);
  statistics.rotationCoverageScore =
      density *
      std::min(1.0, statistics.rotationSpreadDeg / kFullRotationSpreadDeg);
}

std::string joinErrors(const std::vector<std::string> &errors) {
  std::ostringstream joined;
  for (std::size_t i = 0; i < errors.size(); ++i) {
    if (i != 0) {
      joined << "; ";
    }
    joined << errors[i];
  }
  return joined.str();
}

void checkSafeDatasetPath(const fs::path &path) {
  if (path.empty()) {
    throw std::invalid_argument("Dataset path must not be empty");
  }

  std::error_code error;
  const fs::path absolute = fs::absolute(path, error);
  if (error) {
    throw std::invalid_argument("Cannot resolve dataset path " + path.string() +
                                ": " + error.message());
  }
  fs::path resolved = fs::weakly_canonical(absolute, error);
  if (error) {
    throw std::invalid_argument("Cannot resolve dataset path " + path.string() +
                                ": " + error.message());
  }
  resolved = resolved.lexically_normal();

  const fs::path current = fs::weakly_canonical(fs::current_path());
  const auto isAncestorOrSame = [](const fs::path &candidate,
                                   const fs::path &descendant) {
    auto candidatePart = candidate.begin();
    auto descendantPart = descendant.begin();
    while (candidatePart != candidate.end()) {
      if (descendantPart == descendant.end() ||
          *candidatePart != *descendantPart) {
        return false;
      }
      ++candidatePart;
      ++descendantPart;
    }
    return true;
  };

  const fs::path temporaryRoot =
      fs::weakly_canonical(fs::temp_directory_path());
  if (resolved == resolved.root_path() || resolved == temporaryRoot ||
      isAncestorOrSame(resolved, current)) {
    throw std::invalid_argument(
        "Dataset path must not be a filesystem root, the temporary root, "
        "the current directory, or an ancestor of the current directory: " +
        resolved.string());
  }
}

} // namespace

PoseFormat parsePoseFormat(const std::int32_t value) {
  if (value < static_cast<std::int32_t>(PoseFormat::QuaternionXyzw) ||
      value > static_cast<std::int32_t>(PoseFormat::YawPitchRollDegrees)) {
    throw std::invalid_argument("Unsupported pose format " +
                                std::to_string(value) +
                                "; expected a value from 1 to 6");
  }
  return static_cast<PoseFormat>(value);
}

std::size_t poseColumnCount(const PoseFormat format) {
  switch (format) {
  case PoseFormat::QuaternionXyzw:
  case PoseFormat::QuaternionWxyz:
    return 7;
  case PoseFormat::RollPitchYawRadians:
  case PoseFormat::RollPitchYawDegrees:
  case PoseFormat::YawPitchRollRadians:
  case PoseFormat::YawPitchRollDegrees:
    return 6;
  }
  throw std::invalid_argument("Unsupported pose format");
}

std::vector<double> encodePose(const core::RigidTransform &transform,
                               const PoseFormat format) {
  if (!transform.isFinite()) {
    throw std::invalid_argument("Cannot encode a non-finite transform");
  }
  const cv::Mat &rotation = transform.rotation;
  const cv::Mat &translation = transform.translation;

  const double trace = rotation.at<double>(0, 0) + rotation.at<double>(1, 1) +
                       rotation.at<double>(2, 2);
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double qw = 1.0;
  if (trace > 0.0) {
    const double scale = std::sqrt(trace + 1.0) * 2.0;
    qw = 0.25 * scale;
    qx = (rotation.at<double>(2, 1) - rotation.at<double>(1, 2)) / scale;
    qy = (rotation.at<double>(0, 2) - rotation.at<double>(2, 0)) / scale;
    qz = (rotation.at<double>(1, 0) - rotation.at<double>(0, 1)) / scale;
  } else if (rotation.at<double>(0, 0) > rotation.at<double>(1, 1) &&
             rotation.at<double>(0, 0) > rotation.at<double>(2, 2)) {
    const double scale =
        std::sqrt(1.0 + rotation.at<double>(0, 0) - rotation.at<double>(1, 1) -
                  rotation.at<double>(2, 2)) *
        2.0;
    qw = (rotation.at<double>(2, 1) - rotation.at<double>(1, 2)) / scale;
    qx = 0.25 * scale;
    qy = (rotation.at<double>(0, 1) + rotation.at<double>(1, 0)) / scale;
    qz = (rotation.at<double>(0, 2) + rotation.at<double>(2, 0)) / scale;
  } else if (rotation.at<double>(1, 1) > rotation.at<double>(2, 2)) {
    const double scale =
        std::sqrt(1.0 + rotation.at<double>(1, 1) - rotation.at<double>(0, 0) -
                  rotation.at<double>(2, 2)) *
        2.0;
    qw = (rotation.at<double>(0, 2) - rotation.at<double>(2, 0)) / scale;
    qx = (rotation.at<double>(0, 1) + rotation.at<double>(1, 0)) / scale;
    qy = 0.25 * scale;
    qz = (rotation.at<double>(1, 2) + rotation.at<double>(2, 1)) / scale;
  } else {
    const double scale =
        std::sqrt(1.0 + rotation.at<double>(2, 2) - rotation.at<double>(0, 0) -
                  rotation.at<double>(1, 1)) *
        2.0;
    qw = (rotation.at<double>(1, 0) - rotation.at<double>(0, 1)) / scale;
    qx = (rotation.at<double>(0, 2) + rotation.at<double>(2, 0)) / scale;
    qy = (rotation.at<double>(1, 2) + rotation.at<double>(2, 1)) / scale;
    qz = 0.25 * scale;
  }
  if (qw < 0.0) {
    qx = -qx;
    qy = -qy;
    qz = -qz;
    qw = -qw;
  }

  const double pitch =
      std::asin(std::clamp(-rotation.at<double>(2, 0), -1.0, 1.0));
  double roll = 0.0;
  double yaw = 0.0;
  if (std::abs(std::cos(pitch)) > 1e-9) {
    roll = std::atan2(rotation.at<double>(2, 1), rotation.at<double>(2, 2));
    yaw = std::atan2(rotation.at<double>(1, 0), rotation.at<double>(0, 0));
  } else {
    yaw = std::atan2(-rotation.at<double>(0, 1), rotation.at<double>(1, 1));
  }

  const std::vector<double> xyz{translation.at<double>(0, 0),
                                translation.at<double>(1, 0),
                                translation.at<double>(2, 0)};
  switch (format) {
  case PoseFormat::QuaternionXyzw:
    return {xyz[0], xyz[1], xyz[2], qx, qy, qz, qw};
  case PoseFormat::QuaternionWxyz:
    return {xyz[0], xyz[1], xyz[2], qw, qx, qy, qz};
  case PoseFormat::RollPitchYawRadians:
    return {xyz[0], xyz[1], xyz[2], roll, pitch, yaw};
  case PoseFormat::RollPitchYawDegrees:
    return {xyz[0],
            xyz[1],
            xyz[2],
            roll * 180.0 / std::numbers::pi,
            pitch * 180.0 / std::numbers::pi,
            yaw * 180.0 / std::numbers::pi};
  case PoseFormat::YawPitchRollRadians:
    return {xyz[0], xyz[1], xyz[2], yaw, pitch, roll};
  case PoseFormat::YawPitchRollDegrees:
    return {xyz[0],
            xyz[1],
            xyz[2],
            yaw * 180.0 / std::numbers::pi,
            pitch * 180.0 / std::numbers::pi,
            roll * 180.0 / std::numbers::pi};
  }
  throw std::invalid_argument("Unsupported pose format");
}

bool AxisRange::empty() const {
  return !std::isfinite(minimum) || !std::isfinite(maximum);
}

double AxisRange::span() const { return empty() ? 0.0 : maximum - minimum; }

bool DatasetInspection::empty() const {
  return statistics.imageCount == 0 && statistics.poseRowCount == 0;
}

std::string DatasetInspection::errorMessage() const {
  return joinErrors(errors);
}

double SampleAudit::timeDeltaMs() const {
  const long double difference = static_cast<long double>(imageTimestampNs) -
                                 static_cast<long double>(robotTimestampNs);
  return static_cast<double>(std::abs(difference) / 1.0e6L);
}

DatasetRepository::DatasetRepository(fs::path datasetPath)
    : datasetPath_(std::move(datasetPath)) {
  checkSafeDatasetPath(datasetPath_);
}

const fs::path &DatasetRepository::path() const { return datasetPath_; }

DatasetPreparation DatasetRepository::prepare(const bool resume,
                                              const bool overwrite,
                                              const PoseFormat format) {
  if (resume && overwrite) {
    throw std::invalid_argument("resume and overwrite cannot both be enabled");
  }

  if (fs::exists(datasetPath_) && !overwrite) {
    if (!resume) {
      throw std::runtime_error(
          "Dataset already exists; explicitly resume or overwrite it");
    }
    const DatasetInspection existing = inspect(format);
    if (!existing.valid) {
      throw std::runtime_error("Cannot resume corrupted dataset: " +
                               existing.errorMessage());
    }
    return DatasetPreparation{false, existing.nextSampleIndex,
                              existing.statistics};
  }

  fs::path backup;
  const bool replacing = fs::exists(datasetPath_);
  if (replacing) {
    backup = temporaryPath(datasetPath_, "replaced");
    fs::rename(datasetPath_, backup);
  }

  try {
    fs::create_directories(datasetPath_ / kImagesDirectory);
    writeText(datasetPath_ / kPosesFilename, "");
    writeText(datasetPath_ / kSamplesFilename, kAuditHeader);
  } catch (...) {
    removeIgnoringErrors(datasetPath_);
    if (replacing) {
      std::error_code error;
      fs::rename(backup, datasetPath_, error);
    }
    throw;
  }
  removeIgnoringErrors(backup);
  return DatasetPreparation{true, 0, {}};
}

DatasetInspection DatasetRepository::inspect(const PoseFormat format) const {
  DatasetInspection inspection;
  if (!fs::exists(datasetPath_) || !fs::is_directory(datasetPath_)) {
    inspection.errors.push_back("dataset directory is missing");
    return inspection;
  }

  const fs::path imageDirectory = datasetPath_ / kImagesDirectory;
  if (!fs::exists(imageDirectory) || !fs::is_directory(imageDirectory)) {
    inspection.errors.push_back("images directory is missing");
  } else {
    const std::regex imagePattern(R"(^(\d+)\.png$)");
    std::set<std::size_t> indices;
    for (const auto &entry : fs::directory_iterator(imageDirectory)) {
      if (!entry.is_regular_file()) {
        inspection.errors.push_back("images contains a non-file entry: " +
                                    entry.path().filename().string());
        continue;
      }
      std::smatch match;
      const std::string filename = entry.path().filename().string();
      if (!std::regex_match(filename, match, imagePattern)) {
        inspection.errors.push_back("images contains an unexpected file: " +
                                    filename);
        continue;
      }
      std::size_t index = 0;
      try {
        index = static_cast<std::size_t>(std::stoull(match[1].str()));
      } catch (const std::exception &) {
        inspection.errors.push_back("invalid image index: " + filename);
        continue;
      }
      const std::string canonicalFilename = std::to_string(index) + ".png";
      if (filename != canonicalFilename) {
        inspection.errors.push_back("image name is not canonical: " + filename);
      }
      if (!indices.insert(index).second) {
        inspection.errors.push_back("duplicate image index: " +
                                    std::to_string(index));
      }
      try {
        if (cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED).empty()) {
          inspection.errors.push_back("cannot decode image: " + filename);
        }
      } catch (const std::exception &) {
        inspection.errors.push_back("cannot decode image: " + filename);
      }
    }
    std::size_t expected = 0;
    for (const std::size_t index : indices) {
      if (index != expected) {
        inspection.errors.push_back(
            "image numbering is not contiguous; expected " +
            std::to_string(expected) + ", found " + std::to_string(index));
        break;
      }
      ++expected;
    }
    inspection.statistics.imageCount = indices.size();
    inspection.nextSampleIndex = indices.size();
  }

  const fs::path posesPath = datasetPath_ / kPosesFilename;
  if (!fs::exists(posesPath) || !fs::is_regular_file(posesPath)) {
    inspection.errors.push_back("poses.csv is missing");
  } else {
    try {
      const ParsedPoses poses = readPoses(posesPath, format);
      inspection.statistics.poseRowCount = poses.rowCount;
      inspection.statistics.validPoseRowCount = poses.rows.size();
      inspection.errors.insert(inspection.errors.end(), poses.errors.begin(),
                               poses.errors.end());
      updateStatistics(inspection.statistics, poses.rows);
    } catch (const std::exception &error) {
      inspection.errors.push_back(error.what());
    }
  }

  if (inspection.statistics.imageCount != inspection.statistics.poseRowCount) {
    inspection.errors.push_back(
        "image/pose count mismatch: images=" +
        std::to_string(inspection.statistics.imageCount) +
        ", pose rows=" + std::to_string(inspection.statistics.poseRowCount));
  }

  const ParsedAudit audit = readAudit(datasetPath_ / kSamplesFilename);
  inspection.errors.insert(inspection.errors.end(), audit.errors.begin(),
                           audit.errors.end());
  for (std::size_t i = 0; i < audit.records.size(); ++i) {
    if (audit.records[i].index != i) {
      inspection.errors.push_back(
          "samples.tsv indices are not contiguous at row " +
          std::to_string(i + 1));
      break;
    }
  }
  if (audit.records.size() > inspection.statistics.poseRowCount) {
    inspection.errors.push_back(
        "samples.tsv contains more rows than poses.csv");
  }

  inspection.valid = inspection.errors.empty();
  return inspection;
}

bool DatasetRepository::validate(const PoseFormat format, std::string *error,
                                 const bool requireSamples) const {
  const DatasetInspection inspection = inspect(format);
  if (!inspection.valid) {
    if (error != nullptr) {
      *error = inspection.errorMessage();
    }
    return false;
  }
  if (requireSamples && inspection.empty()) {
    if (error != nullptr) {
      *error = "dataset contains no samples";
    }
    return false;
  }
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

std::size_t DatasetRepository::nextSampleIndex(const PoseFormat format) const {
  const DatasetInspection inspection = inspect(format);
  if (!inspection.valid) {
    throw std::runtime_error("Invalid dataset: " + inspection.errorMessage());
  }
  return inspection.nextSampleIndex;
}

std::vector<PoseRow>
DatasetRepository::poseRows(const PoseFormat format) const {
  const DatasetInspection inspection = inspect(format);
  if (!inspection.valid) {
    throw std::runtime_error("Invalid dataset: " + inspection.errorMessage());
  }
  return readPoses(datasetPath_ / kPosesFilename, format).rows;
}

std::vector<PoseRow>
DatasetRepository::acceptedPoseRows(const PoseFormat format) const {
  return poseRows(format);
}

std::vector<core::RigidTransform> DatasetRepository::loadRobotTransforms(
    const PoseFormat format,
    const std::unordered_set<std::size_t> &rejectedImageIndices) const {
  std::vector<core::RigidTransform> transforms;
  for (const auto &row : poseRows(format)) {
    if (!rejectedImageIndices.contains(row.sampleIndex)) {
      transforms.push_back(row.transform);
    }
  }
  if (!rejectedImageIndices.empty() && transforms.empty()) {
    throw std::runtime_error("No robot poses remain after rejected images were "
                             "removed");
  }
  return transforms;
}

std::size_t DatasetRepository::appendSample(
    const cv::Mat &image, const std::vector<double> &poseValues,
    const SampleAudit &audit, const PoseFormat format) {
  if (image.empty()) {
    throw std::invalid_argument("Cannot append an empty image");
  }
  makePoseRow(0, poseValues, format);

  const DatasetInspection inspection = inspect(format);
  if (!inspection.valid) {
    throw std::runtime_error("Cannot append to invalid dataset: " +
                             inspection.errorMessage());
  }
  const std::size_t index = inspection.nextSampleIndex;

  const fs::path posesPath = datasetPath_ / kPosesFilename;
  const fs::path auditPath = datasetPath_ / kSamplesFilename;
  const fs::path imagePath =
      datasetPath_ / kImagesDirectory / (std::to_string(index) + ".png");
  if (fs::exists(imagePath)) {
    throw std::runtime_error("Refusing to overwrite " + imagePath.string());
  }

  const fs::path stagedPoses = temporaryPath(posesPath, "append");
  const fs::path stagedAudit = temporaryPath(auditPath, "append");
  const fs::path stagedImage =
      temporaryPath(datasetPath_ / kImagesDirectory / std::to_string(index),
                    "append", ".png");

  try {
    writeText(stagedPoses, appendPoseText(readText(posesPath), poseValues));

    ParsedAudit parsedAudit = readAudit(auditPath);
    if (!parsedAudit.errors.empty()) {
      throw std::runtime_error(joinErrors(parsedAudit.errors));
    }
    while (parsedAudit.records.size() < index) {
      const std::size_t missing = parsedAudit.records.size();
      parsedAudit.records.push_back(AuditRecord{missing, 0, 0, 0.0});
    }
    parsedAudit.records.push_back(AuditRecord{index, audit.imageTimestampNs,
                                              audit.robotTimestampNs,
                                              audit.timeDeltaMs()});
    writeText(stagedAudit, auditText(std::move(parsedAudit.records)));

    if (!cv::imwrite(stagedImage.string(), image)) {
      throw std::runtime_error("Cannot encode calibration image as PNG");
    }

    std::vector<FileReplacement> replacements{{posesPath, stagedPoses},
                                              {auditPath, stagedAudit},
                                              {imagePath, stagedImage}};
    installAll(replacements);
  } catch (...) {
    removeIgnoringErrors(stagedPoses);
    removeIgnoringErrors(stagedAudit);
    removeIgnoringErrors(stagedImage);
    throw;
  }
  return index;
}

void DatasetRepository::removeLastSample(const PoseFormat format) {
  const DatasetInspection inspection = inspect(format);
  if (!inspection.valid) {
    throw std::runtime_error("Cannot remove from invalid dataset: " +
                             inspection.errorMessage());
  }
  if (inspection.empty()) {
    throw std::runtime_error("Cannot remove a sample from an empty dataset");
  }
  const std::size_t index = inspection.nextSampleIndex - 1;
  const fs::path posesPath = datasetPath_ / kPosesFilename;
  const fs::path auditPath = datasetPath_ / kSamplesFilename;
  const fs::path imagePath =
      datasetPath_ / kImagesDirectory / (std::to_string(index) + ".png");

  const fs::path stagedPoses = temporaryPath(posesPath, "remove");
  const fs::path stagedAudit = temporaryPath(auditPath, "remove");
  const fs::path removedImage = temporaryPath(imagePath, "removed", ".png");
  bool imageMoved = false;
  try {
    writeText(stagedPoses, removeLastPoseRow(readText(posesPath)));

    ParsedAudit parsedAudit = readAudit(auditPath);
    if (!parsedAudit.errors.empty()) {
      throw std::runtime_error(joinErrors(parsedAudit.errors));
    }
    if (!parsedAudit.records.empty() &&
        parsedAudit.records.back().index == index) {
      parsedAudit.records.pop_back();
    }
    writeText(stagedAudit, auditText(std::move(parsedAudit.records)));

    fs::rename(imagePath, removedImage);
    imageMoved = true;
    std::vector<FileReplacement> replacements{{posesPath, stagedPoses},
                                              {auditPath, stagedAudit}};
    installAll(replacements);
    removeIgnoringErrors(removedImage);
  } catch (...) {
    removeIgnoringErrors(stagedPoses);
    removeIgnoringErrors(stagedAudit);
    if (imageMoved && fs::exists(removedImage)) {
      std::error_code error;
      fs::rename(removedImage, imagePath, error);
    }
    throw;
  }
}

} // namespace hand_eye::io
