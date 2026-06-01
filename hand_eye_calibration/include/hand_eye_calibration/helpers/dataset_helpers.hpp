#pragma once

#include <csv.hpp>
#include <filesystem>
#include <optional>
#include <regex>

using csv::CSVFormat;
using csv::CSVReader;
namespace fs = std::filesystem;

inline constexpr const char *CSV_FILENAME = "poses.csv";
inline constexpr const char *IMG_FOLDERNAME = "images";
inline constexpr const char *DATASET_FOLDERNAME = "dataset";
inline constexpr const char *CALIBRATION_FILENAME = "calibration.yaml";

inline const std::regex IMAGE_FILENAME_PATTERN(R"(^(\d+)\.png$)");

namespace dataset_helpers {

CSVReader getReader(const fs::path &path);
std::optional<size_t> countPoses(const fs::path &datasetPath);
std::optional<size_t> countImages(const fs::path &datasetPath);
bool validateDataset(const fs::path &dataset, std::string *error = nullptr);

}; // namespace dataset_helpers
