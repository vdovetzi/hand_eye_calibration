#include "lib/core/capture_policy.hpp"
#include "lib/core/target.hpp"
#include "lib/io/calibration_io.hpp"
#include "lib/io/dataset_repository.hpp"
#include "lib/ros/capture_source.hpp"

#include <boost/program_options.hpp>
#include <opencv2/highgui.hpp>
#include <rclcpp/rclcpp.hpp>

#include <csignal>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace po = boost::program_options;
namespace fs = std::filesystem;
using hand_eye::core::CapturePolicy;
using hand_eye::core::CapturePolicyOptions;
using hand_eye::core::CaptureRequest;
using hand_eye::core::TargetDetector;
using hand_eye::io::DatasetRepository;
using hand_eye::io::PoseFormat;
using hand_eye::ros::CaptureSource;

namespace {

volatile std::sig_atomic_t interrupted = 0;

void signalHandler(int) { interrupted = 1; }

struct Options {
  std::string armTopic;
  std::string imageTopic;
  std::string cameraInfoTopic;
  std::string intrinsicsPath;
  std::string patternInfo;
  std::string baseFrame;
  std::string effectorFrame;
  std::string cameraFrame;
  fs::path dataset;
  PoseFormat poseFormat{PoseFormat::QuaternionXyzw};
  bool overwrite{false};
  bool resume{true};
  bool requireStamped{true};
  double maximumTimeDeltaMs{30.0};
  double settlingMs{500.0};
  double stationaryTranslationM{0.0005};
  double stationaryRotationDeg{0.25};
  double duplicateTranslationM{0.005};
  double duplicateRotationDeg{5.0};
};

Options parseOptions(const int argc, const char *const argv[]) {
  po::options_description description("Dataset collector options");
  description.add_options()("help,h", "Show this help")(
      "arm-topic,at", po::value<std::string>()->required(),
      "Stamped robot Pose/Transform topic")(
      "image-topic,it", po::value<std::string>()->required(),
      "Image or CompressedImage topic")(
      "camera-info-topic,ci", po::value<std::string>()->default_value(""),
      "Optional CameraInfo topic")(
      "intrinsics-path,ip", po::value<std::string>()->default_value(""),
      "Intrinsics YAML; required for standalone ArUco datasets")(
      "output,o", po::value<std::string>()->default_value("."),
      "Output directory; the compatible dataset/ folder is created inside")(
      "pattern-type,pt", po::value<std::string>()->default_value(""),
      "Optional target: '1 dict id size', '2 rows cols cell', or ChArUco")(
      "poses-format,pf", po::value<int>()->default_value(1),
      "Pose format 1..6")("base-frame",
                          po::value<std::string>()->default_value(""),
                          "Expected robot base frame")(
      "effector-frame", po::value<std::string>()->default_value(""),
      "Expected robot effector frame")(
      "camera-frame", po::value<std::string>()->default_value(""),
      "Expected camera frame")("require-stamped",
                               po::value<bool>()->default_value(true),
                               "Reject messages without source timestamps")(
      "max-time-delta-ms", po::value<double>()->default_value(30.0),
      "Maximum image/robot timestamp difference")(
      "settle-window-ms", po::value<double>()->default_value(500.0),
      "Required stationary history")("stationary-translation-m",
                                     po::value<double>()->default_value(0.0005),
                                     "Maximum motion during settling")(
      "stationary-rotation-deg", po::value<double>()->default_value(0.25),
      "Maximum angular motion during settling")(
      "duplicate-translation-m", po::value<double>()->default_value(0.005),
      "Near-duplicate translation threshold")(
      "duplicate-rotation-deg", po::value<double>()->default_value(5.0),
      "Near-duplicate rotation threshold")(
      "resume", po::value<bool>()->default_value(true),
      "Resume a valid existing dataset")("overwrite", po::bool_switch(),
                                         "Replace an existing dataset");

  po::variables_map variables;
  po::store(po::parse_command_line(argc, argv, description), variables);
  if (variables.count("help")) {
    std::cout << description << '\n';
    std::exit(0);
  }
  po::notify(variables);

  Options result;
  result.armTopic = variables["arm-topic"].as<std::string>();
  result.imageTopic = variables["image-topic"].as<std::string>();
  result.cameraInfoTopic = variables["camera-info-topic"].as<std::string>();
  result.intrinsicsPath = variables["intrinsics-path"].as<std::string>();
  result.patternInfo = variables["pattern-type"].as<std::string>();
  result.baseFrame = variables["base-frame"].as<std::string>();
  result.effectorFrame = variables["effector-frame"].as<std::string>();
  result.cameraFrame = variables["camera-frame"].as<std::string>();
  result.dataset = fs::path(variables["output"].as<std::string>()) / "dataset";
  result.poseFormat =
      hand_eye::io::parsePoseFormat(variables["poses-format"].as<int>());
  result.overwrite = variables["overwrite"].as<bool>();
  // Overwrite is an explicit replacement request and must not be combined
  // with the resume default.
  result.resume = result.overwrite ? false : variables["resume"].as<bool>();
  result.requireStamped = variables["require-stamped"].as<bool>();
  result.maximumTimeDeltaMs = variables["max-time-delta-ms"].as<double>();
  result.settlingMs = variables["settle-window-ms"].as<double>();
  result.stationaryTranslationM =
      variables["stationary-translation-m"].as<double>();
  result.stationaryRotationDeg =
      variables["stationary-rotation-deg"].as<double>();
  result.duplicateTranslationM =
      variables["duplicate-translation-m"].as<double>();
  result.duplicateRotationDeg =
      variables["duplicate-rotation-deg"].as<double>();
  return result;
}

CapturePolicy
makePolicy(const Options &options,
           const std::optional<hand_eye::core::PatternConfig> &pattern) {
  CapturePolicyOptions policy;
  policy.maximumTimeDeltaMilliseconds = options.maximumTimeDeltaMs;
  policy.requireStampedSamples = options.requireStamped;
  policy.requireCameraInfo =
      pattern && pattern->type == hand_eye::core::PatternType::ARUCO &&
      options.intrinsicsPath.empty();
  policy.robotBaseFrame = options.baseFrame;
  policy.robotEffectorFrame = options.effectorFrame;
  policy.cameraFrame = options.cameraFrame;
  policy.settlingDurationMilliseconds = options.settlingMs;
  policy.settlingTranslationMeters = options.stationaryTranslationM;
  policy.settlingRotationDegrees = options.stationaryRotationDeg;
  policy.duplicateTranslationMeters = options.duplicateTranslationM;
  policy.duplicateRotationDegrees = options.duplicateRotationDeg;
  return CapturePolicy(policy);
}

} // namespace

int main(int argc, char **argv) {
  std::signal(SIGINT, signalHandler);
  try {
    const Options options = parseOptions(argc, argv);
    rclcpp::init(argc, argv);
    if (!options.intrinsicsPath.empty()) {
      hand_eye::io::CalibrationIO::readIntrinsics(options.intrinsicsPath);
    }

    std::optional<hand_eye::core::PatternConfig> pattern;
    std::unique_ptr<TargetDetector> detector;
    if (!options.patternInfo.empty()) {
      pattern = hand_eye::core::parsePattern(options.patternInfo);
      detector = std::make_unique<TargetDetector>(*pattern);
    }
    if (pattern && pattern->type == hand_eye::core::PatternType::ARUCO &&
        options.intrinsicsPath.empty()) {
      throw std::invalid_argument(
          "ArUco collection requires --intrinsics-path so the offline "
          "calibration can reuse the same K and D");
    }
    CapturePolicy policy = makePolicy(options, pattern);
    DatasetRepository repository(options.dataset);
    const auto prepared = repository.prepare(options.resume, options.overwrite,
                                             options.poseFormat);

    auto node = rclcpp::Node::make_shared("dataset_collector");
    CaptureSource source(*node);
    source.start({options.armTopic, options.imageTopic, options.cameraInfoTopic,
                  options.baseFrame, options.effectorFrame, 1000});

    RCLCPP_INFO(node->get_logger(),
                "Dataset ready at %s (next sample %zu). Enter=capture, q=quit",
                options.dataset.string().c_str(), prepared.nextSampleIndex);

    while (rclcpp::ok() && !interrupted) {
      rclcpp::spin_some(node);
      const auto snapshot = source.snapshot();
      if (!snapshot.image.empty()) {
        cv::imshow("Hand-eye dataset collector", snapshot.image);
      }
      const int key = cv::waitKey(1) & 0xff;
      if (key == 'q' || key == 27) {
        break;
      }
      if (key != 13 && key != 10) {
        continue;
      }

      CaptureRequest request;
      request.image = snapshot.imageMetadata;
      request.robotHistory = snapshot.robotHistory;
      request.cameraInfo = snapshot.cameraInfo;
      for (const auto &row : repository.acceptedPoseRows(options.poseFormat)) {
        request.acceptedRobotPoses.push_back(row.transform);
      }
      const auto decision = policy.evaluate(request);
      if (!decision.accepted || !decision.matchedRobotPose) {
        RCLCPP_WARN(node->get_logger(), "Sample rejected (%s): %s",
                    hand_eye::core::toString(decision.reason).c_str(),
                    decision.message.c_str());
        continue;
      }
      if (detector && !detector->detect(snapshot.image)) {
        RCLCPP_WARN(node->get_logger(),
                    "Sample rejected: calibration target was not detected");
        continue;
      }

      const std::vector<double> pose = hand_eye::io::encodePose(
          decision.matchedRobotPose->pose, options.poseFormat);
      const auto index = repository.appendSample(
          snapshot.image, pose,
          {snapshot.imageMetadata->timestampNanoseconds,
           decision.matchedRobotPose->timestampNanoseconds},
          options.poseFormat);
      RCLCPP_INFO(node->get_logger(),
                  "Sample %zu committed (timestamp delta %.3f ms)", index,
                  decision.timestampDeltaMilliseconds);
    }

    source.stop();
    cv::destroyAllWindows();
    const auto stats = repository.inspect(options.poseFormat).statistics;
    RCLCPP_INFO(node->get_logger(),
                "Finished with %zu samples; rotation span %.2f deg",
                stats.validPoseRowCount, stats.rotationSpreadDeg);
  } catch (const po::error &error) {
    std::cerr << "Argument error: " << error.what() << '\n';
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "Collection failed: " << error.what() << '\n';
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return 1;
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
