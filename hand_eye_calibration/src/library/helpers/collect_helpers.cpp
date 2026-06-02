#include "hand_eye_calibration/helpers/collect_helpers.hpp"

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <csv.hpp>
#include <eigen3/Eigen/Geometry>
#include <fstream>
#include <hand_eye_calibration/helpers/dataset_helpers.hpp>
#include <iostream>
#include <locale>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>

namespace collect_helpers {

namespace {

double quaternionDistanceRad(const Eigen::Quaterniond &a,
                             const Eigen::Quaterniond &b) {
  const double dot = std::clamp(std::abs(a.dot(b)), 0.0, 1.0);
  return 2.0 * std::acos(dot);
}

double sampleCoverageFactor(const size_t samples) {
  constexpr double kFullCoverageSamples = 8.0;
  if (samples <= 2) {
    return 0.0;
  }
  return std::min(1.0, (static_cast<double>(samples) - 2.0) /
                           (kFullCoverageSamples - 2.0));
}

double translationRange(const DatasetStatistics &stats) {
  return std::sqrt(std::pow(stats.translation[0].span(), 2.0) +
                   std::pow(stats.translation[1].span(), 2.0) +
                   std::pow(stats.translation[2].span(), 2.0));
}

double rotationSpreadDeg(const DatasetStatistics &stats) {
  return stats.rotationSpreadRad * 180.0 / std::numbers::pi;
}

void updateCoverageScores(DatasetStatistics &stats) {
  const double density = sampleCoverageFactor(stats.validPoseRows);
  stats.translationCoverageScore =
      density * std::min(1.0, translationRange(stats) / 0.10);
  stats.rotationCoverageScore =
      density * std::min(1.0, rotationSpreadDeg(stats) / 25.0);
}

} // namespace

SerializedMsg catchMsg(std::string ros_topic_name, std::string rosTopicType,
                       rclcpp::GenericSubscription::SharedPtr &subscription,
                       rclcpp::Node::SharedPtr node) {
  auto promise = std::make_shared<std::promise<SerializedMsg>>();
  auto future = promise->get_future();
  subscription = rclcpp::create_generic_subscription(
      node->get_node_topics_interface(), ros_topic_name, rosTopicType, 1,
      [promise, node](SerializedMsg msg) {
        promise->set_value(msg);
        RCLCPP_DEBUG(node->get_logger(), "Got msg!");
      });

  rclcpp::spin_until_future_complete(node, future);
  return future.get();
}

void dumpValue(const std::string &val, std::vector<std::string> &dump) {
  dump.emplace_back(val);
}

void dumpValue(const std::wstring &val, std::vector<std::string> &dump) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  dump.emplace_back(converter.to_bytes(val));
}

void dumpValue(const uint8_t &val, std::vector<std::string> &dump) {
  dump.emplace_back(std::to_string(static_cast<int32_t>(val)));
}

void dumpValue(const char16_t &val, std::vector<std::string> &dump) {
  dump.emplace_back(std::to_string(static_cast<int32_t>(val)));
}

void dumpValue(const char *val, std::vector<std::string> &dump) {
  dump.emplace_back(val);
}

void dumpPose(const ros_babel_fish::CompoundMessage &pose,
              std::vector<std::string> &dump, bool stamped) {
  if (stamped) {
    auto poseMsg = pose.message<geometry_msgs::msg::PoseStamped>();
    dump.emplace_back(std::to_string(poseMsg->pose.position.x));
    dump.emplace_back(std::to_string(poseMsg->pose.position.y));
    dump.emplace_back(std::to_string(poseMsg->pose.position.z));
    dump.emplace_back(std::to_string(poseMsg->pose.orientation.x));
    dump.emplace_back(std::to_string(poseMsg->pose.orientation.y));
    dump.emplace_back(std::to_string(poseMsg->pose.orientation.z));
    dump.emplace_back(std::to_string(poseMsg->pose.orientation.w));
    return;
  }
  auto poseMsg = pose.message<geometry_msgs::msg::Pose>();
  dump.emplace_back(std::to_string(poseMsg->position.x));
  dump.emplace_back(std::to_string(poseMsg->position.y));
  dump.emplace_back(std::to_string(poseMsg->position.z));
  dump.emplace_back(std::to_string(poseMsg->orientation.x));
  dump.emplace_back(std::to_string(poseMsg->orientation.y));
  dump.emplace_back(std::to_string(poseMsg->orientation.z));
  dump.emplace_back(std::to_string(poseMsg->orientation.w));
}

void dumpTransform(const ros_babel_fish::CompoundMessage &tf,
                   std::vector<std::string> &dump, bool stamped) {
  if (stamped) {
    auto tfMsg = tf.message<geometry_msgs::msg::TransformStamped>();
    dump.emplace_back(std::to_string(tfMsg->transform.translation.x));
    dump.emplace_back(std::to_string(tfMsg->transform.translation.y));
    dump.emplace_back(std::to_string(tfMsg->transform.translation.z));
    dump.emplace_back(std::to_string(tfMsg->transform.rotation.x));
    dump.emplace_back(std::to_string(tfMsg->transform.rotation.y));
    dump.emplace_back(std::to_string(tfMsg->transform.rotation.z));
    dump.emplace_back(std::to_string(tfMsg->transform.rotation.w));
    return;
  }
  auto tfMsg = tf.message<geometry_msgs::msg::Transform>();
  dump.emplace_back(std::to_string(tfMsg->translation.x));
  dump.emplace_back(std::to_string(tfMsg->translation.y));
  dump.emplace_back(std::to_string(tfMsg->translation.z));
  dump.emplace_back(std::to_string(tfMsg->rotation.x));
  dump.emplace_back(std::to_string(tfMsg->rotation.y));
  dump.emplace_back(std::to_string(tfMsg->rotation.z));
  dump.emplace_back(std::to_string(tfMsg->rotation.w));
}

void dumpJoints(const ros_babel_fish::CompoundMessage &joints,
                std::vector<std::string> &dump) {
  auto jointsMsg = joints.message<sensor_msgs::msg::JointState>();
  for (const double &p : jointsMsg->position) {
    dump.emplace_back(std::to_string(p));
  }
}

void dumpMessageContent(const ros_babel_fish::Message &message,
                        std::vector<std::string> &dump) {
  using namespace ros_babel_fish;
  if (message.type() == MessageTypes::Compound) {
    auto &compound = message.as<CompoundMessage>();
    if (compound.isTime()) {
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"),
                   "Skipping time, because it is not a pose");
      return;
    }
    if (compound.isDuration()) {
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"),
                   "Skipping duration, because it is not a pose");
      return;
    }
    if (compound.name() == "geometry_msgs/msg/PoseStamped") {
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"), "Dumping PoseStemped");
      dumpPose(compound, dump, true);
      return;
    }
    if (compound.name() == "geometry_msgs/msg/Pose") {
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"), "Dumping Pose");
      dumpPose(compound, dump);
      return;
    }
    if (compound.name() == "geometry_msgs/msg/TransformStamped") {
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"), "Dumping TransformStemped");
      dumpTransform(compound, dump, true);
      return;
    }
    if (compound.name() == "geometry_msgs/msg/Transform") {
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"), "Dumping Transform");
      dumpTransform(compound, dump);
      return;
    }
    if (compound.name() == "sensor_msgs/msg/JointState") {
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"), "Dumping JointState");
      dumpJoints(compound, dump);
      return;
    }
    for (size_t i = 0; i < compound.keys().size(); ++i) {
      dumpMessageContent(*compound.values()[i], dump);
    }
  } else if (message.type() == MessageTypes::Array) {
    auto &base = message.as<ArrayMessageBase>();
    RCLCPP_DEBUG(rclcpp::get_logger("dumper"), "Dumping array");
    RBF2_TEMPLATE_CALL_ARRAY_TYPES(dumpArray, base, dump);
  } else {
    switch (message.type()) {
    case MessageTypes::Array:
    case MessageTypes::Compound:
    case MessageTypes::None:
      break;
    case MessageTypes::Bool:
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"), "Dumping bool");
      dump.emplace_back(message.as<ValueMessage<bool>>().getValue() ? "true"
                                                                    : "false");
      break;
    case MessageTypes::UInt8:
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"), "Dumping uint8");
      dump.emplace_back(std::to_string(static_cast<uint32_t>(
          message.as<ValueMessage<uint8_t>>().getValue())));
      break;
    case MessageTypes::UInt16:
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"), "Dumping uint16");
      dump.emplace_back(std::to_string(message.value<uint16_t>()));
      break;
    default:
      RCLCPP_DEBUG(rclcpp::get_logger("dumper"), "Dumping other value");
      RBF2_TEMPLATE_CALL_VALUE_TYPES(dumpMessageValue, message.type(), message,
                                     dump);
    }
  }
}

bool askYesNo(const rclcpp::Logger &logger, const std::string &question) {
  while (true) {
    RCLCPP_WARN(logger, "%s[Y/n]", question.c_str());
    std::string decision;
    std::cin >> decision;
    if (decision.empty()) {
      continue;
    }

    switch (decision[0]) {
    case 'Y':
    case 'y':
      return true;
    case 'N':
    case 'n':
      return false;
    default:
      RCLCPP_WARN(logger, "No such option %c", decision[0]);
      break;
    }
  }
}

std::optional<size_t> getResumeImageIndex(const fs::path &datasetPath,
                                          std::string &error) {
  const fs::path imagePath = datasetPath / IMG_FOLDERNAME;
  if (!fs::exists(imagePath) || !fs::is_directory(imagePath)) {
    error = "images/ directory is missing";
    return std::nullopt;
  }

  std::set<size_t> imageNumbers;
  for (const auto &entry : fs::directory_iterator(imagePath)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    const std::string filename = entry.path().filename().string();
    std::smatch matches;
    if (!std::regex_match(filename, matches, IMAGE_FILENAME_PATTERN)) {
      error = "images/ contains unexpected file: " + filename;
      return std::nullopt;
    }
    imageNumbers.insert(std::stoul(matches[1].str()));
  }

  size_t expected = 0;
  for (const size_t imageNumber : imageNumbers) {
    if (imageNumber != expected) {
      error = "images/ numbering is not contiguous";
      return std::nullopt;
    }
    ++expected;
  }

  std::optional<size_t> posesCount = dataset_helpers::countPoses(datasetPath);
  if (!posesCount) {
    error = "poses.csv is missing, empty, or corrupted";
    return std::nullopt;
  }
  if (*posesCount != imageNumbers.size()) {
    error = "images/ count does not match poses.csv row count";
    return std::nullopt;
  }

  return imageNumbers.size();
}

DatasetPreparation prepareDatasetDirectory(const fs::path &datasetPath,
                                           const rclcpp::Logger &logger) {
  bool shouldCreate = true;

  if (fs::exists(datasetPath)) {
    if (!fs::is_directory(datasetPath)) {
      if (!askYesNo(logger,
                    "Output path exists and is not a directory, delete it?")) {
        throw std::runtime_error(
            "Error: remove output path manually or run program again.");
      }
      fs::remove_all(datasetPath);
    } else if (askYesNo(logger,
                        "Output directory already exists, delete it?")) {
      fs::remove_all(datasetPath);
    } else {
      shouldCreate = false;
      std::string error;
      std::optional<size_t> nextImageIndex =
          getResumeImageIndex(datasetPath, error);

      if (!nextImageIndex) {
        const std::string question =
            "Existing dataset is corrupted: " + error + ". Remove it?";
        if (!askYesNo(logger, question)) {
          throw std::runtime_error(
              "Error: remove dataset manually or run program again.");
        }
        fs::remove_all(datasetPath);
        shouldCreate = true;
      } else {
        return DatasetPreparation{*nextImageIndex, false};
      }
    }
  }

  if (shouldCreate) {
    fs::create_directories(datasetPath / IMG_FOLDERNAME);
    RCLCPP_INFO(logger, "Folder created successfully: %s",
                datasetPath.string().c_str());
  }

  return DatasetPreparation{0, true};
}

bool isPoseLikeArmTopic(const std::string &armTopicType) {
  return armTopicType == "geometry_msgs/msg/Pose" ||
         armTopicType == "geometry_msgs/msg/PoseStamped" ||
         armTopicType == "geometry_msgs/msg/Transform" ||
         armTopicType == "geometry_msgs/msg/TransformStamped";
}

bool isJointStateArmTopic(const std::string &armTopicType) {
  return armTopicType == "sensor_msgs/msg/JointState";
}

std::optional<DatasetStatistics>
calculateDatasetStatistics(const fs::path &datasetPath,
                           const std::string &armTopicType) {
  DatasetStatistics stats;
  stats.armTopicType = armTopicType;
  stats.jointStateDataset = isJointStateArmTopic(armTopicType);
  stats.poseMetricsAvailable = isPoseLikeArmTopic(armTopicType);

  try {
    std::ifstream poses(datasetPath / CSV_FILENAME);
    if (!poses.is_open()) {
      return std::nullopt;
    }

    std::vector<Eigen::Vector3d> translations;
    std::vector<Eigen::Quaterniond> rotations;

    std::string line;
    while (std::getline(poses, line)) {
      std::istringstream row(line);
      std::array<double, 7> pose{};
      if (!(row >> pose[0])) {
        continue;
      }

      ++stats.poseRows;
      if (!stats.poseMetricsAvailable) {
        continue;
      }
      for (size_t i = 1; i < pose.size(); ++i) {
        row >> pose[i];
      }
      if (row.fail()) {
        continue;
      }

      Eigen::Vector3d translation(pose[0], pose[1], pose[2]);
      Eigen::Quaterniond rotation(pose[6], pose[3], pose[4], pose[5]);
      rotation.normalize();

      for (size_t axis = 0; axis < stats.translation.size(); ++axis) {
        stats.translation[axis].min =
            std::min(stats.translation[axis].min, translation[axis]);
        stats.translation[axis].max =
            std::max(stats.translation[axis].max, translation[axis]);
      }

      translations.emplace_back(translation);
      rotations.emplace_back(rotation);
    }
    if (poses.bad()) {
      return std::nullopt;
    }

    stats.samples = stats.poseRows;
    stats.validPoseRows = translations.size();
    for (size_t i = 0; i < rotations.size(); ++i) {
      for (size_t j = i + 1; j < rotations.size(); ++j) {
        stats.rotationSpreadRad =
            std::max(stats.rotationSpreadRad,
                     quaternionDistanceRad(rotations[i], rotations[j]));
      }
    }
    updateCoverageScores(stats);
    return stats;
  } catch (const std::exception &ex) {
    RCLCPP_DEBUG(rclcpp::get_logger(HELPER_LOGGERNAME), "Error: %s", ex.what());
    return std::nullopt;
  }
}

void logDatasetStatistics(const DatasetStatistics &stats,
                          const rclcpp::Logger &logger) {
  if (stats.jointStateDataset) {
    RCLCPP_INFO(logger,
                "Calibration dataset statistics\n"
                "  samples                : %zu\n"
                "  arm topic type         : %s",
                stats.samples, stats.armTopicType.c_str());
    RCLCPP_WARN(logger,
                "Arm topic type is sensor_msgs/msg/JointState; pose-space "
                "calibration quality statistics are not implemented for joint "
                "datasets yet.");
    return;
  }
  if (!stats.poseMetricsAvailable) {
    RCLCPP_INFO(logger,
                "Calibration dataset statistics\n"
                "  samples                : %zu\n"
                "  arm topic type         : %s",
                stats.samples, stats.armTopicType.c_str());
    RCLCPP_WARN(logger,
                "Arm topic type '%s' is not supported for calibration quality "
                "statistics.",
                stats.armTopicType.c_str());
    return;
  }
  if (stats.validPoseRows == 0) {
    RCLCPP_INFO(logger,
                "Calibration dataset statistics\n"
                "  samples                : %zu\n"
                "  arm topic type         : %s",
                stats.samples, stats.armTopicType.c_str());
    RCLCPP_WARN(logger,
                "No valid pose rows were found; calibration quality statistics "
                "cannot be calculated.");
    return;
  }

  const double radToDeg = 180.0 / std::numbers::pi;
  RCLCPP_INFO(logger,
              "Statistics\n"
              "  samples                : %zu\n"
              "  translation coverage   : %8.0f%%\n"
              "  rotation coverage      : %8.0f%%\n"
              "  translation span [m]   : x=%8.5f  y=%8.5f  z=%8.5f\n"
              "  rotation span          : %8.2f deg  (%8.5f rad)",
              stats.validPoseRows, stats.translationCoverageScore * 100.0,
              stats.rotationCoverageScore * 100.0, stats.translation[0].span(),
              stats.translation[1].span(), stats.translation[2].span(),
              stats.rotationSpreadRad * radToDeg, stats.rotationSpreadRad);
}

} // namespace collect_helpers
