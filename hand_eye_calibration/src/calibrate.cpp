// https://github.com/ros/class_loader/pull/199
#include <console_bridge/console.h>

#include <boost/program_options.hpp>
#include <hand_eye_calibration/calibrate_helpers.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

using Level = rclcpp::Logger::Level;
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

// TODO: add opportunity to provide intrinsics in cli and thus not to find them
// TODO: add YPR order for rotation
int32_t main(int32_t argc, char **argv) {
  signal(SIGINT, signalHandler);
  rclcpp::init(argc, argv);

  node = rclcpp::Node::make_shared("calibrator");
  logger = std::make_shared<rclcpp::Logger>(node->get_logger());

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Shows help message")(
      "dataset-path,dp", po::value<std::string>()->required(),
      "Input dataset path")(
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
      "5. joints(j0 j1 j2 ... jn)")(
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
  const fs::path dataset_path =
      fs::path(vm["dataset-path"].as<std::string>()) / DATASET_FOLDERNAME;
  const std::string calibration_pattern =
      vm["calibration-pattern"].as<std::string>();
  const int32_t poses_format = vm["poses-format"].as<int32_t>();
  const std::string logLevel = vm["log-level"].as<std::string>();
  const bool eye2hand = vm["eye-to-hand"].as<bool>();

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
  if (!validateDataset(dataset_path)) {
    RCLCPP_ERROR(*logger, "Error: dataset doesn't exist or is corrupted. Set "
                          "debug level for more details.");
    ret(1);
  }

  auto pattern = std::make_unique<CalibrationPattern>(dataset_path);

  // Getting stuff for calibration based on the choosen pattern
  try {
    pattern->setPatternInfo(calibration_pattern);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(*logger, "%s", e.what());
    ret(1);
  }

  // Contains all data for cv::calibrateHandEye()
  auto data = std::make_unique<CalibrationData>();

  try {
    findTarget2Cam(*pattern, *data);
    findGripper2Base(dataset_path, poses_format, *data);
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

      cv::Mat R_T = R.t();

      R_base2gripper.emplace_back(R_T);
      t_base2gripper.emplace_back(-R_T * t);
    }

    data->R_gripper2base = R_base2gripper;
    data->t_gripper2base = t_base2gripper;
  }

  cv::Mat R_cam2gripper;
  cv::Mat t_cam2gripper;

  try {
    cv::calibrateHandEye(data->R_gripper2base, data->t_gripper2base,
                         data->R_target2cam, data->t_target2cam, R_cam2gripper,
                         t_cam2gripper);

    // Combining into homogeneous transformation matrix and dump to yaml
    dumpToYAML(R_cam2gripper, t_cam2gripper);

    RCLCPP_INFO(*logger, "Calibration completed! Saved to %s",
                CALIBRATION_FILENAME);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(*logger, "%s", e.what());
    ret(1);
  }

  std::cout << "Rotation matrix:\n"
            << cv::format(R_cam2gripper, cv::Formatter::FMT_DEFAULT) << '\n';

  std::cout << "Translation vector:\n"
            << cv::format(t_cam2gripper, cv::Formatter::FMT_DEFAULT) << '\n';

  ret(0);
}