#include "hand_eye_calibration/helpers/dataset_helpers.hpp"

#include <exception>
#include <fstream>
#include <sstream>

namespace dataset_helpers {

CSVReader getReader(const fs::path &path) {
  CSVFormat format;
  format.delimiter('\t').no_header().quote(false);
  return CSVReader(path.relative_path().string(), format);
}

std::optional<size_t> countPoses(const fs::path &datasetPath) {
  try {
    std::ifstream poses(datasetPath / CSV_FILENAME);
    if (!poses.is_open()) {
      return std::nullopt;
    }

    size_t count = 0;
    std::string line;
    while (std::getline(poses, line)) {
      std::istringstream row(line);
      double value = 0.0;
      if (row >> value) {
        ++count;
      }
    }

    if (poses.bad()) {
      return std::nullopt;
    }
    return count;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

std::optional<size_t> countImages(const fs::path &datasetPath) {
  size_t count = 0;
  try {
    for (const auto &entry :
         fs::directory_iterator(datasetPath / IMG_FOLDERNAME)) {
      if (!entry.is_regular_file()) {
        continue;
      }

      const std::string filename = entry.path().filename().string();
      if (std::regex_match(filename, IMAGE_FILENAME_PATTERN)) {
        ++count;
      }
    }
    return count;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

bool validateDataset(const fs::path &dataset, std::string *error) {
  const fs::path imgFolder = dataset / IMG_FOLDERNAME;
  const fs::path posesFile = dataset / CSV_FILENAME;

  if (!fs::exists(dataset) || !fs::is_directory(dataset)) {
    if (error) {
      *error = "dataset directory is missing";
    }
    return false;
  }
  if (!fs::exists(imgFolder) || !fs::is_directory(imgFolder)) {
    if (error) {
      *error = "images directory is missing";
    }
    return false;
  }
  if (!fs::exists(posesFile) || !fs::is_regular_file(posesFile)) {
    if (error) {
      *error = "poses.csv is missing";
    }
    return false;
  }

  const std::optional<size_t> imagesCount = countImages(dataset);
  if (!imagesCount) {
    if (error) {
      *error = "cannot count images in images directory";
    }
    return false;
  }

  const std::optional<size_t> posesCount = countPoses(dataset);
  if (!posesCount) {
    if (error) {
      *error = "cannot count poses in poses.csv";
    }
    return false;
  }

  if (*posesCount != *imagesCount) {
    if (error) {
      *error = "poses count is not equal to images count";
    }
    return false;
  }

  return true;
}

} // namespace dataset_helpers
