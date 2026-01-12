// https://github.com/ros/class_loader/pull/199
#include <console_bridge/console.h>

#include <boost/program_options.hpp>
#include <hand_eye_calibration/calibrate_helpers.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

using Level = rclcpp::Logger::Level;
using cv::utils::logging::LogLevel;
namespace po = boost::program_options;

std::shared_ptr<rclcpp::Logger> logger;
rclcpp::Node::SharedPtr node;

void ret(int32_t code) {
  logger.reset();
  node.reset();
  rclcpp::shutdown();
  exit(code);
}

// TODO: add opportunity to provide intrinsics in cli and thus not to find them
// TODO: add YPR order for rotation
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  node = rclcpp::Node::make_shared("dataset_collector");
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
  std::string log_level = vm["log-level"].as<std::string>();
  const bool eye2hand = vm["eye-to-hand"].as<bool>();

  // Setting logger level;
  std::optional<Level> level = magic_enum::enum_cast<Level>(log_level);

  if (!level) {
    RCLCPP_ERROR(*logger, "Error: there is no such logger level");
    ret(1);
  }

  // Setting ROS log-level
  logger->set_level(*level);
  rcutils_ret_t res = rcutils_logging_set_logger_level(
      HELPER_LOGGERNAME, static_cast<int>(*level));
  if (RCUTILS_RET_OK != res) {
    RCLCPP_WARN(*logger, "Error: unable to set desired logging level");
  }
  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_ERROR);

  // Setting cv log-level
  std::transform(log_level.begin(), log_level.end(), log_level.begin(),
                 ::toupper);
  cv::utils::logging::setLogLevel(*magic_enum::enum_cast<LogLevel>(
      "LOG_LEVEL_" +
      ((log_level.find("WARN") != std::string::npos) ? "WARNING" : log_level)));

  // Checks dataset existance and its correctness
  if (!validate_dataset(dataset_path)) {
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
    FindTarget2Cam(*pattern, *data);
    FindGripper2Base(dataset_path, poses_format, *data);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(*logger, "%s", e.what());
    ret(1);
  }

  if (eye2hand) {

    std::vector<cv::Mat> rvecs_base2gripper;
    std::vector<cv::Mat> tvecs_base2gripper;

    for (size_t i = 0; i < data->tvecs_gripper2base.size(); ++i) {
      cv::Mat &rvec = data->rvecs_gripper2base[i];
      cv::Mat &tvec = data->tvecs_gripper2base[i];

      cv::Mat rvec_t = rvec.t();

      rvecs_base2gripper.emplace_back(rvec_t);
      tvecs_base2gripper.emplace_back(-rvec_t * tvec);
    }

    data->rvecs_gripper2base = rvecs_base2gripper;
    data->tvecs_gripper2base = tvecs_base2gripper;
  }

  cv::Mat R_cam2gripper;
  cv::Mat t_cam2gripper;

  try {
    cv::calibrateHandEye(data->rvecs_gripper2base, data->tvecs_gripper2base,
                         data->rvecs_target2cam, data->tvecs_target2cam,
                         R_cam2gripper, t_cam2gripper);

    RCLCPP_INFO(*logger, "Calibration completed!");
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