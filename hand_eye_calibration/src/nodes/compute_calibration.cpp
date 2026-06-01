// https://github.com/ros/class_loader/pull/199
#include <console_bridge/console.h>

#include <boost/program_options.hpp>
#include <hand_eye_calibration/helpers/calibrate_helpers.hpp>
#include <magic_enum.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

using Level = rclcpp::Logger::Level;
using namespace calibration_helpers;
namespace po = boost::program_options;

std::shared_ptr<rclcpp::Logger> logger;
rclcpp::Node::SharedPtr node;

void ret(int32_t code) {
  logger.reset();
  node.reset();
  rclcpp::shutdown();
  exit(code);
}

void signalHandler([[maybe_unused]] int32_t signal) {
  std::cout << "\nSIGINT (Ctrl+C) received. Shutting down gracefully..."
            << std::endl;
  ret(0);
}

int32_t main(int32_t argc, char **argv) {
  signal(SIGINT, signalHandler);
  rclcpp::init(argc, argv);

  node = rclcpp::Node::make_shared("calibrator");
  logger = std::make_shared<rclcpp::Logger>(node->get_logger());

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Shows help message")(
      "dataset-path,dp", po::value<std::string>()->required(),
      "Path to dataset directory containing images/ and poses.csv")(
      "eye-to-hand,eth", po::value<bool>()->default_value(true),
      "Whether it is eye-to-hand calibration or not (i.e. eye-in-hand)")(
      "calibration-pattern,cp", po::value<std::string>()->required(),
      "Supported patterns:\n1. ArUco '<dict in {4,5,6,7}> <id> <size in m>'\n"
      "2. Chessboard '<rows> <cols> <size in m>'\n"
      "3. ChArUco '<rows> <cols> <cell size in m> <marker size in m> <dict "
      "in {4,5,6,7}> <id>'")(
      "poses-format,pf", po::value<int32_t>()->required(),
      "Supported formats:\n1. translation(x y z) rotation(x y z w)\n"
      "2. translation(x y z) rotation(w x y z)\n"
      "3. translation(x y z) rotation(r p y) in rad\n"
      "4. translation(x y z) rotation(r p y) in deg\n"
      "5. translation(x y z) rotation(y p r) in rad\n"
      "6. translation(x y z) rotation(y p r) in deg\n"
      "7. joints(j0 j1 j2 ... jn)")(
      "intrinsics-path,ip", po::value<std::string>(),
      "Optional YAML path with camera intrinsics K and D. If provided, "
      "intrinsics are loaded instead of recalibrated")(
      "log-level,ll", po::value<std::string>()->default_value("Info"),
      "Log-level: Info, Debug, Error, Fatal");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      ret(0);
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "Error: " << e.what() << "\n" << desc << "\n";
    ret(1);
  }

  // Getting parsed input
  const fs::path datasetPath = fs::path(vm["dataset-path"].as<std::string>());
  const std::string calibrationPattern =
      vm["calibration-pattern"].as<std::string>();
  const int32_t posesFormat = vm["poses-format"].as<int32_t>();
  const std::string logLevel = vm["log-level"].as<std::string>();
  const bool eye2hand = vm["eye-to-hand"].as<bool>();
  std::optional<fs::path> intrinsicsPath;
  if (vm.count("intrinsics-path")) {
    intrinsicsPath = fs::path(vm["intrinsics-path"].as<std::string>());
  }

  // Setting logger level;
  std::optional<Level> level = magic_enum::enum_cast<Level>(logLevel);

  if (!level) {
    RCLCPP_ERROR(*logger, "Error: there is no such logger level");
    ret(1);
  }

  // Setting ROS log-level
  logger->set_level(*level);
  rcutils_ret_t res = rcutils_logging_set_logger_level(
      HELPER_LOGGERNAME, static_cast<int32_t>(*level));
  if (RCUTILS_RET_OK != res) {
    RCLCPP_WARN(*logger, "Error: unable to set desired logging level");
  }
  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_ERROR);

  // Checks dataset existance and its correctness
  std::string datasetError;
  if (!dataset_helpers::validateDataset(datasetPath, &datasetError)) {
    RCLCPP_ERROR(*logger, "Error: dataset doesn't exist or is corrupted. Set "
                          "debug level for more details.");
    RCLCPP_DEBUG(*logger, "Dataset validation failed: %s",
                 datasetError.c_str());
    ret(1);
  }

  auto pattern = std::make_unique<CalibrationPattern>(datasetPath);

  // Getting stuff for calibration based on the choosen pattern
  try {
    pattern->setPatternInfo(calibrationPattern);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(*logger, "%s", e.what());
    ret(1);
  }

  // Contains all data for cv::calibrateHandEye()
  auto data = std::make_unique<CalibrationData>();

  try {
    if (intrinsicsPath) {
      if (!data->readIntrinsics(*intrinsicsPath)) {
        RCLCPP_ERROR(*logger, "Error: cannot read camera intrinsics from %s",
                     intrinsicsPath->string().c_str());
        ret(1);
      }
      RCLCPP_INFO(*logger, "Loaded camera intrinsics from %s",
                  intrinsicsPath->string().c_str());
    }

    findTarget2Cam(*pattern, *data, intrinsicsPath.has_value());
    findGripper2Base(datasetPath, posesFormat, *data);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(*logger, "%s", e.what());
    ret(1);
  }

  if (eye2hand) {
    std::vector<cv::Mat> R_base2gripper;
    std::vector<cv::Mat> t_base2gripper;
    const size_t count = data->R_gripper2base.size();
    R_base2gripper.reserve(count);
    t_base2gripper.reserve(count);

    for (size_t i = 0; i < data->t_gripper2base.size(); ++i) {
      cv::Mat &R = data->R_gripper2base[i];
      cv::Mat &t = data->t_gripper2base[i];

      cv::Mat R_inv = R.t();
      cv::Mat t_inv = -R_inv * t;

      R_base2gripper.emplace_back(R_inv);
      t_base2gripper.emplace_back(t_inv);
    }

    data->R_gripper2base = R_base2gripper;
    data->t_gripper2base = t_base2gripper;
  }

  try {
    cv::calibrateHandEye(data->R_gripper2base, data->t_gripper2base,
                         data->R_target2cam, data->t_target2cam,
                         data->R_cam2gripper, data->t_cam2gripper);

    const double rms = calculateRMS(*data);

    dumpToYAML(*data);

    RCLCPP_INFO(*logger, "Calibration completed! Saved to %s",
                CALIBRATION_FILENAME);
    RCLCPP_INFO(*logger,
                "Validation RMS chessboard origin spread in gripper frame [m]: "
                "%.6f",
                rms);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(*logger, "%s", e.what());
    ret(1);
  }

  std::cout << "Rotation matrix:\n"
            << cv::format(data->R_cam2gripper, cv::Formatter::FMT_DEFAULT)
            << '\n';

  std::cout << "Translation vector:\n"
            << cv::format(data->t_cam2gripper, cv::Formatter::FMT_DEFAULT)
            << '\n';

  ret(0);
}
