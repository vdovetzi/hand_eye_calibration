#pragma once

#include "lib/core/calibration.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace hand_eye::io {

inline constexpr const char *kImagesDirectory = "images";
inline constexpr const char *kPosesFilename = "poses.csv";
inline constexpr const char *kSamplesFilename = "samples.tsv";

enum class PoseFormat : std::int32_t {
  QuaternionXyzw = 1,
  QuaternionWxyz = 2,
  RollPitchYawRadians = 3,
  RollPitchYawDegrees = 4,
  YawPitchRollRadians = 5,
  YawPitchRollDegrees = 6,
};

PoseFormat parsePoseFormat(std::int32_t value);
std::size_t poseColumnCount(PoseFormat format);
std::vector<double> encodePose(const core::RigidTransform &transform,
                               PoseFormat format);

struct AxisRange {
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = -std::numeric_limits<double>::infinity();

  [[nodiscard]] bool empty() const;
  [[nodiscard]] double span() const;
};

struct DatasetStatistics {
  std::size_t imageCount = 0;
  std::size_t poseRowCount = 0;
  std::size_t validPoseRowCount = 0;
  std::array<AxisRange, 3> translation;
  double translationRangeM = 0.0;
  double rotationSpreadDeg = 0.0;
  double translationCoverageScore = 0.0;
  double rotationCoverageScore = 0.0;
};

struct DatasetInspection {
  bool valid = false;
  std::size_t nextSampleIndex = 0;
  DatasetStatistics statistics;
  std::vector<std::string> errors;

  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::string errorMessage() const;
};

struct DatasetPreparation {
  bool created = false;
  std::size_t nextSampleIndex = 0;
  DatasetStatistics statistics;
};

struct SampleAudit {
  std::int64_t imageTimestampNs = 0;
  std::int64_t robotTimestampNs = 0;

  [[nodiscard]] double timeDeltaMs() const;
};

struct PoseRow {
  std::size_t sampleIndex = 0;
  std::vector<double> values;
  core::RigidTransform transform;
};

class DatasetRepository {
public:
  explicit DatasetRepository(std::filesystem::path datasetPath);

  [[nodiscard]] const std::filesystem::path &path() const;

  // An existing dataset must be explicitly resumed or overwritten. A newly
  // prepared dataset is valid for collection even though it contains no rows.
  DatasetPreparation prepare(bool resume, bool overwrite, PoseFormat format);

  [[nodiscard]] DatasetInspection inspect(PoseFormat format) const;
  [[nodiscard]] bool validate(PoseFormat format, std::string *error = nullptr,
                              bool requireSamples = true) const;
  [[nodiscard]] std::size_t nextSampleIndex(PoseFormat format) const;

  [[nodiscard]] std::vector<PoseRow> poseRows(PoseFormat format) const;
  [[nodiscard]] std::vector<PoseRow> acceptedPoseRows(PoseFormat format) const;
  [[nodiscard]] std::vector<core::RigidTransform> loadRobotTransforms(
      PoseFormat format,
      const std::unordered_set<std::size_t> &rejectedImageIndices = {}) const;

  // poseValues are written as one TSV row. Values after the columns required by
  // the selected legacy format are retained as numeric metadata. The image,
  // pose and timestamp audit either all become visible or the previous dataset
  // is restored.
  std::size_t appendSample(const cv::Mat &image,
                           const std::vector<double> &poseValues,
                           const SampleAudit &audit, PoseFormat format);

  // Samples are append-only, so removing the last sample is the only removal
  // operation exposed by the repository. It keeps numbering contiguous.
  void removeLastSample(PoseFormat format);

private:
  std::filesystem::path datasetPath_;
};

} // namespace hand_eye::io
