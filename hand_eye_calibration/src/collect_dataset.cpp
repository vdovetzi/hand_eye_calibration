// https://github.com/ros/class_loader/pull/199
#include <console_bridge/console.h>

#include <boost/program_options.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <hand_eye_calibration/collect_helpers.hpp>
#include <image_transport/image_transport.hpp>
#include <magic_enum.hpp>
#include <opencv2/opencv.hpp>

using Level = rclcpp::Logger::Level;
using namespace std::chrono_literals;
using namespace ros_babel_fish;
using csv::TSVWriter;
using image_transport::ImageTransport;
using image_transport::TransportHints;
using sensor_msgs::image_encodings::BGR8;
using sensor_msgs::msg::Image;
namespace po = boost::program_options;

std::shared_ptr<rclcpp::Logger> logger;
rclcpp::Node::SharedPtr node;
std::shared_ptr<std::ofstream> outFile;
BabelFishSubscription::SharedPtr armSub;
std::shared_ptr<TSVWriter<std::ofstream>> writer;
fs::path outputPath;
size_t imgCount = 0;

void ret(int32_t code) {
  writer.reset();
  outFile.reset();
  logger.reset();
  armSub.reset();
  node.reset();
  rclcpp::shutdown();
  cv::destroyAllWindows();
  exit(code);
}

void signalHandler([[maybe_unused]] int32_t signal) {
  std::cout << "\nSIGINT (Ctrl+C) received. Shutting down gracefully..."
            << std::endl;
  ret(0);
}

enum class KEYS { ENTER = 13, Q = 113 };

void SavePoseToFile(const Message &msg) {
  std::vector<std::string> dump;
  dumpMessageContent(msg, dump);
  *writer << dump;
  RCLCPP_DEBUG(*logger, "Pose saved!");
}

void SaveImageToFolder(cv::Mat &image) {
  // TODO: Add marker checking
  cv::imwrite(outputPath / IMG_FOLDERNAME /
                  (std::to_string(imgCount++) + ".png"),
              image);
  RCLCPP_DEBUG(*logger, "Image saved!");
}

int32_t main(int32_t argc, const char **argv) {
  signal(SIGINT, signalHandler);
  rclcpp::init(argc, argv);

  node = rclcpp::Node::make_shared("dataset_collector");
  logger = std::make_shared<rclcpp::Logger>(node->get_logger());

  // Parser settings
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Shows help message")(
      "arm-topic,at", po::value<std::string>()->required(),
      "Input arm topic")("image-topic,it", po::value<std::string>()->required(),
                         "Input image topic")(
      "output,o", po::value<std::string>()->default_value("."), "Output path")(
      "log-level,ll", po::value<std::string>()->default_value("Info"),
      "Log-level: Info, Debug, Error");

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
  const std::string armTopicName = vm["arm-topic"].as<std::string>();
  const std::string imageTopicName = vm["image-topic"].as<std::string>();
  outputPath = fs::path(vm["output"].as<std::string>()) / "dataset";
  const std::string logLevel = vm["log-level"].as<std::string>();

  // Setting logger level;
  std::optional<Level> level = magic_enum::enum_cast<Level>(logLevel);

  if (!level) {
    RCLCPP_ERROR(*logger, "Error: there is no such logger level");
    ret(1);
  }

  logger->set_level(level.value());
  rcutils_ret_t res = rcutils_logging_set_logger_level(
      DUMPER_LOGGERNAME, static_cast<int32_t>(*level));
  if (RCUTILS_RET_OK != res) {
    RCLCPP_WARN(*logger, "Error: unable to set desired logging level");
  }
  res = rcutils_logging_set_logger_level(HELPER_LOGGERNAME,
                                         static_cast<int32_t>(*level));
  if (RCUTILS_RET_OK != res) {
    RCLCPP_WARN(*logger, "Error: unable to set desired logging level");
  }
  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_ERROR);

  // Ensure existing topics
  // TODO: get rid off sleep. Wait for topic to be availible with a time limit.
  // When exceeds it, exit with an error.
  bool armTopicFound = false;
  bool imageTopicFound = false;
  rclcpp::sleep_for(2s); // Small sleep for fetching actual topics
  auto topics = node->get_topic_names_and_types();
  for (const auto &[topic_name, topic_types] : topics) {
    RCLCPP_DEBUG(*logger, "Topic: %s", topic_name.c_str());
    if (topic_name == armTopicName) {
      armTopicFound = true;
    } else if (topic_name == imageTopicName) {
      imageTopicFound = true;
    }
  }

  if (!armTopicFound) {
    RCLCPP_ERROR(*logger, "Error: provide existing arm-topic");
    ret(1);
  }

  if (!imageTopicFound) {
    RCLCPP_ERROR(*logger, "Error: provide existing image-topic");
    ret(1);
  }

  // Creating output directories
  // TODO: think better about create: probably it's not necessary to put create
  // in each branch
  // TODO: remove duplicated code and make a desired function for that
  bool create = true;
  if (fs::exists(outputPath) && fs::is_directory(outputPath)) {
    bool stop = false;
    while (!stop) {
      RCLCPP_WARN(*logger, "Output directory already exits, delete it?[Y/n]");
      std::string decision;
      std::cin >> decision;
      switch (decision[0]) {
      case 'Y':
      case 'y': {
        fs::remove_all(outputPath);
        stop = true;
        create = true;
        break;
      }
      case 'N':
      case 'n': {
        create = false;
        // Checking images/
        std::regex pattern(R"(^(\d+)\.png$)");
        std::smatch matches;
        size_t cnt = 0;
        std::set<int32_t> files;
        for (const auto &entry :
             fs::directory_iterator(outputPath / IMG_FOLDERNAME)) {
          if (entry.is_regular_file()) {
            const std::string &filename = entry.path().filename().string();
            if (std::regex_match(filename, matches, pattern)) {
              size_t number = std::stoi(matches[1].str());
              files.insert(number);
            }
          }
        }
        for (const int32_t filenum : files) {
          const std::string filename = std::to_string(filenum) + ".png";
          if (!std::regex_match(filename, matches, pattern)) {
            bool stop = false;
            while (!stop) {
              RCLCPP_WARN(*logger,
                          "Existing image folder contains wrong filename: %s. "
                          "Dataset is corrupted, remove it?[Y/n]",
                          filename.c_str());
              std::string decision;
              std::cin >> decision;
              switch (decision[0]) {
              case 'Y':
              case 'y': {
                fs::remove_all(outputPath);
                stop = true;
                create = true;
                break;
              }
              case 'N':
              case 'n': {
                RCLCPP_ERROR(
                    *logger,
                    "Error: remove dataset manually or run programm again.");
                ret(1);
                break;
              }
              default: {
                RCLCPP_WARN(*logger, "No such option %c", decision[0]);
                break;
              }
              }
            }
          } else {
            create = false;
            const size_t number = std::stoi(matches[1].str());
            if (number - cnt > 1) {
              bool stop = false;
              while (!stop) {
                RCLCPP_WARN(
                    *logger,
                    "Existing image folder contains wrong filename: %s. "
                    "Dataset is corrupted, remove it?[Y/n]",
                    filename.c_str());
                std::string decision;
                std::cin >> decision;
                switch (decision[0]) {
                case 'Y':
                case 'y': {
                  fs::remove_all(outputPath);
                  stop = true;
                  create = true;
                  break;
                }
                case 'N':
                case 'n': {
                  RCLCPP_ERROR(*logger, "Error: remove dataset manually or "
                                        "run programm again.");
                  ret(1);
                  break;
                }
                default: {
                  RCLCPP_WARN(*logger, "No such option %c", decision[0]);
                  break;
                }
                }
              }
            } else {
              ++cnt;
            }
          }
        }
        // Checking poses.csv
        std::optional<size_t> nRowsOpt = countPoses(outputPath);
        if (!nRowsOpt.has_value()) {
          bool stop = false;
          while (!stop) {
            RCLCPP_WARN(*logger, "Existing poses.csv is corrupted or empty. "
                                 "Dataset is corrupted, remove it?[Y/n]");
            std::string decision;
            std::cin >> decision;
            switch (decision[0]) {
            case 'Y':
            case 'y': {
              fs::remove_all(outputPath);
              stop = true;
              create = true;
              break;
            }
            case 'N':
            case 'n': {
              RCLCPP_ERROR(*logger, "Error: remove dataset manually or "
                                    "run programm again.");
              ret(1);
              break;
            }
            default: {
              RCLCPP_WARN(*logger, "No such option %c", decision[0]);
              break;
            }
            }
          }
        } else {
          const size_t nRows = *nRowsOpt;
          if (nRows == cnt) {
            imgCount = cnt;
          } else {
            bool stop = false;
            while (!stop) {
              RCLCPP_WARN(*logger,
                          "Existing poses.csv contains wrong rows number: %ld, "
                          "instead of %ld. "
                          "Dataset is corrupted, remove it?[Y/n]",
                          nRows, cnt);
              std::string decision;
              std::cin >> decision;
              switch (decision[0]) {
              case 'Y':
              case 'y': {
                fs::remove_all(outputPath);
                stop = true;
                create = true;
                break;
              }
              case 'N':
              case 'n': {
                RCLCPP_ERROR(*logger, "Error: remove dataset manually or "
                                      "run programm again.");
                ret(1);
                break;
              }
              default: {
                RCLCPP_WARN(*logger, "No such option %c", decision[0]);
                break;
              }
              }
            }
          }
        }
        stop = true;
        break;
      }
      default: {
        RCLCPP_WARN(*logger, "No such option %c", decision[0]);
        break;
      }
      }
    }
  }
  if (create) {
    try {
      fs::create_directory(outputPath);
      fs::create_directory(outputPath / IMG_FOLDERNAME);
      RCLCPP_INFO(*logger, "Folder created successfully: %s",
                  outputPath.string().data());
    } catch (const fs::filesystem_error &e) {
      RCLCPP_ERROR(*logger, "Error: %s", e.what());
      ret(1);
    }
  }

  // Create tsv writer for gripper poses
  outFile =
      std::make_shared<std::ofstream>(outputPath / CSV_FILENAME, std::ios::app);
  writer = std::make_shared<TSVWriter<std::ofstream>>(
      csv::make_tsv_writer(*outFile));
  csv::set_decimal_places(5);

  // Create sync atomic
  std::atomic<uint32_t> n = 0;

  // Image topic subscription
  cv::Mat image;
  auto it = std::make_unique<ImageTransport>(node);
  auto th = std::make_unique<TransportHints>(node.get(), "compressed");
  auto image_sub = it->subscribe(
      imageTopicName, 1,
      [&image, &n](const Image::ConstSharedPtr &img_msg) {
        image = cv_bridge::toCvCopy(*img_msg, BGR8)->image;
        if (((n.load() & 0b10) >> 1) == 1) {
          SaveImageToFolder(image);
          n.store(n.load() & 0b01);
        }
      },
      nullptr, th.get());

  // Arm topic subscription
  std::string armTopicType;
  auto fish = BabelFish::make_shared();
  armSub = fish->create_subscription(
      *node, armTopicName, 1,
      [&armTopicType, &n](ros_babel_fish::CompoundMessage::SharedPtr msg) {
        // DEBUG info
        if (armTopicType != msg->name()) {
          armTopicType = msg->name();
          RCLCPP_DEBUG(*logger, "Pose topic type: %s", armTopicType.c_str());
        }

        if ((n.load() & 0b01) == 1) {
          SavePoseToFile(*msg);
          n.store(n.load() & 0b10);
        }
      });

  while (rclcpp::ok()) {
    rclcpp::spin_some(node);
    if (image.empty())
      continue;

    cv::imshow("Image Viewer", image);
    int32_t key = cv::waitKey(1) & 0xFF;

    switch (key) {

    case static_cast<int32_t>(KEYS::ENTER): {
      RCLCPP_DEBUG(*logger, "Pressed Enter!");
      n.store(0b11);
      while (n.load() != 0b00) {
        rclcpp::spin_some(node);
      }
      RCLCPP_INFO(*logger, "Sample %ld saved!", imgCount);
      continue;
    }

    case static_cast<int32_t>(KEYS::Q): {
      RCLCPP_INFO(*logger, "Collecting dataset finished!");

      // Counting total number of images and poses
      std::optional<size_t> posesCountOpt = countPoses(outputPath);
      if (!posesCountOpt.has_value()) {
        RCLCPP_ERROR(*logger, "Error: cannot count rows in poses.csv. Maybe "
                              "file is empty or corrupted?");
        ret(1);
      }
      std::optional<size_t> imagesCountOpt = countImages(outputPath);
      if (!imagesCountOpt.has_value()) {
        RCLCPP_ERROR(*logger, "Error: cannot count images in images/. Maybe "
                              "directory was corrupted?");
        ret(1);
      }
      const size_t imagesCount = *imagesCountOpt;
      const size_t posesCount = *posesCountOpt;

      // Should be equal
      if (imagesCount != posesCount) {
        RCLCPP_ERROR(
            *logger,
            "Error: amount of images and poses are not equal. Something "
            "happend during dataset collection. Try to delete previous dataset "
            "and start collection from scratch.");
        ret(1);
      }

      RCLCPP_INFO(*logger, "Dataset length is %ld", imagesCount);

      // TODO: calculate some statistics on collected dataset.

      ret(0);
    }
    }
  }

  ret(0);
}