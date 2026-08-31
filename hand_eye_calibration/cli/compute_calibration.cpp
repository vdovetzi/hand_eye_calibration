#include "lib/core/calibration.hpp"
#include "lib/core/target.hpp"
#include "lib/io/calibration_io.hpp"
#include "lib/io/dataset_repository.hpp"

#include <boost/program_options.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace po = boost::program_options;
namespace fs = std::filesystem;

namespace {

struct Options {
  fs::path dataset;
  fs::path output{"calibration.yaml"};
  std::optional<fs::path> intrinsics;
  std::string pattern;
  std::string method{"auto"};
  std::string baseFrame;
  std::string effectorFrame;
  std::string cameraFrame;
  int poseFormat{1};
  bool eyeToHand{true};
  bool rejectOutliers{true};
  std::size_t minimumSamples{15};
  std::size_t minimumIntrinsicViews{8};
  double minimumRotationSpanDeg{35.0};
  double minimumRotationAxisSeparationDeg{10.0};
  double maximumReprojectionPx{1.0};
};

Options parseOptions(const int argc, const char *const argv[]) {
  po::options_description description("Offline hand-eye calibration options");
  description.add_options()("help,h", "Show this help")(
      "dataset-path,dp", po::value<std::string>()->required(),
      "Dataset containing images/ and tab-separated poses.csv")(
      "eye-to-hand,eth", po::value<bool>()->default_value(true),
      "true for fixed camera; false for eye-in-hand")(
      "calibration-pattern,cp", po::value<std::string>()->required(),
      "'1 dict id marker_size', '2 rows cols cell_size', or "
      "'3 rows cols square marker dict [legacy_id]'")(
      "poses-format,pf", po::value<int>()->required(),
      "1=xyzw, 2=wxyz, 3/4=RPY rad/deg, 5/6=YPR rad/deg")(
      "intrinsics-path,ip", po::value<std::string>(),
      "YAML containing K and D; mandatory for ArUco")(
      "output,o", po::value<std::string>()->default_value("calibration.yaml"),
      "Output calibration YAML")(
      "method,m", po::value<std::string>()->default_value("auto"),
      "auto, tsai, park, horaud, andreff or daniilidis")(
      "min-samples", po::value<std::size_t>()->default_value(15),
      "Minimum accepted observations (hard floor: 15)")(
      "min-rotation-deg", po::value<double>()->default_value(35.0),
      "Minimum robot rotational excitation")(
      "min-axis-separation-deg", po::value<double>()->default_value(10.0),
      "Minimum separation between robot rotation axes")(
      "max-reprojection-px", po::value<double>()->default_value(1.0),
      "Maximum per-view target reprojection RMS")(
      "minimum-intrinsic-views", po::value<std::size_t>()->default_value(8),
      "Minimum views when camera intrinsics must be estimated")(
      "no-outlier-rejection", po::bool_switch(),
      "Disable robust MAD pose-outlier removal")(
      "base-frame", po::value<std::string>()->default_value(""),
      "Robot base frame metadata")("effector-frame",
                                   po::value<std::string>()->default_value(""),
                                   "Robot effector frame metadata")(
      "camera-frame", po::value<std::string>()->default_value(""),
      "Camera frame metadata")("log-level,ll",
                               po::value<std::string>()->default_value("Info"),
                               "Accepted for v1 command compatibility");

  po::variables_map variables;
  po::store(po::parse_command_line(argc, argv, description), variables);
  if (variables.count("help")) {
    std::cout << description << '\n';
    std::exit(0);
  }
  po::notify(variables);

  Options options;
  options.dataset = variables["dataset-path"].as<std::string>();
  options.output = variables["output"].as<std::string>();
  options.pattern = variables["calibration-pattern"].as<std::string>();
  options.poseFormat = variables["poses-format"].as<int>();
  options.eyeToHand = variables["eye-to-hand"].as<bool>();
  options.method = variables["method"].as<std::string>();
  options.minimumSamples = variables["min-samples"].as<std::size_t>();
  options.minimumRotationSpanDeg = variables["min-rotation-deg"].as<double>();
  options.minimumRotationAxisSeparationDeg =
      variables["min-axis-separation-deg"].as<double>();
  options.maximumReprojectionPx = variables["max-reprojection-px"].as<double>();
  options.minimumIntrinsicViews =
      variables["minimum-intrinsic-views"].as<std::size_t>();
  options.rejectOutliers = !variables["no-outlier-rejection"].as<bool>();
  options.baseFrame = variables["base-frame"].as<std::string>();
  options.effectorFrame = variables["effector-frame"].as<std::string>();
  options.cameraFrame = variables["camera-frame"].as<std::string>();
  if (variables.count("intrinsics-path")) {
    options.intrinsics = variables["intrinsics-path"].as<std::string>();
  }
  return options;
}

fs::path outputFile(fs::path path) {
  if ((fs::exists(path) && fs::is_directory(path)) ||
      path.extension().empty()) {
    path /= "calibration.yaml";
  }
  return path;
}

void printIndices(const std::vector<std::size_t> &indices) {
  if (indices.empty()) {
    std::cout << "none";
    return;
  }
  for (std::size_t position = 0; position < indices.size(); ++position) {
    if (position != 0) {
      std::cout << ',';
    }
    std::cout << indices[position];
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    hand_eye::io::DatasetCalibrationOptions request;
    request.datasetPath = options.dataset;
    request.poseFormat = hand_eye::io::parsePoseFormat(options.poseFormat);
    request.pattern = hand_eye::core::parsePattern(options.pattern);
    if (options.intrinsics) {
      request.intrinsics =
          hand_eye::io::CalibrationIO::readIntrinsics(*options.intrinsics);
    }
    request.targetOptions.maximumReprojectionErrorPixels =
        options.maximumReprojectionPx;
    request.targetOptions.minimumViewsForIntrinsicCalibration =
        options.minimumIntrinsicViews;
    request.calibrationOptions.mode =
        options.eyeToHand ? hand_eye::core::CalibrationMode::EYE_TO_HAND
                          : hand_eye::core::CalibrationMode::EYE_IN_HAND;
    request.calibrationOptions.method =
        hand_eye::core::parseCalibrationMethod(options.method);
    request.calibrationOptions.minimumSamples = options.minimumSamples;
    request.calibrationOptions.minimumRotationSpanDegrees =
        options.minimumRotationSpanDeg;
    request.calibrationOptions.minimumRotationAxisSeparationDegrees =
        options.minimumRotationAxisSeparationDeg;
    request.calibrationOptions.rejectOutliers = options.rejectOutliers;
    request.parentFrame =
        options.eyeToHand ? options.baseFrame : options.effectorFrame;
    request.childFrame = options.cameraFrame;

    auto result = hand_eye::io::calibrateDataset(request);
    const fs::path output = outputFile(options.output);
    hand_eye::io::CalibrationIO::write(output, result.document());

    std::cout << std::fixed << std::setprecision(6)
              << "Calibration saved to: " << output << '\n'
              << "Mode: " << hand_eye::core::toString(result.calibration.mode)
              << '\n'
              << "Selected method: "
              << hand_eye::core::toString(result.calibration.method) << '\n'
              << "Translation consistency RMS [m]: "
              << result.calibration.consistency.translationRmsMeters << '\n'
              << "Rotation consistency RMS [deg]: "
              << result.calibration.consistency.rotationRmsDegrees << '\n'
              << "Used samples: " << result.calibration.usedSampleIndices.size()
              << '\n'
              << "Rejected sample indices: ";
    printIndices(result.calibration.rejectedSampleIndices);
    std::cout << '\n';
    if (result.estimatedIntrinsics) {
      std::cout << "Camera calibration RMS [px]: "
                << result.intrinsicCalibrationRmsPixels << '\n';
    }
    std::cout << "Transform T:\n"
              << result.calibration.transform.matrix() << '\n';
  } catch (const po::error &error) {
    std::cerr << "Argument error: " << error.what() << '\n';
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "Calibration failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
