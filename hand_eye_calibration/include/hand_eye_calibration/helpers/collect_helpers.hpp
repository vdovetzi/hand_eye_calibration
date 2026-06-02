#pragma once

// TODO: add docstrings to everywhere

// #include "hand_eye_calibration/helpers/dataset_helpers.hpp"
#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <ros_babel_fish/babel_fish.hpp>
#include <ros_babel_fish/macros.hpp>

namespace collect_helpers {

inline constexpr const char *DUMPER_LOGGERNAME = "dumper";
inline constexpr const char *HELPER_LOGGERNAME = "helper";

using SerializedMsg = std::shared_ptr<const rclcpp::SerializedMessage>;
namespace fs = std::filesystem;

enum class DatasetPosesOption {
  ROT_XYZW = 1,
  ROT_WXYZ = 2,
  ROT_RPY_RAD = 3,
  ROT_RPY_DEG = 4,
  JOINTS = 5
};

SerializedMsg catchMsg(std::string ros_topic_name, std::string rosTopicType,
                       rclcpp::GenericSubscription::SharedPtr &subscription,
                       rclcpp::Node::SharedPtr node);

void dumpMessageContent(const ros_babel_fish::Message &message,
                        std::vector<std::string> &dump);

template <bool BOUNDED, bool FIXED_LENGTH>
void dumpArray(
    const ros_babel_fish::CompoundArrayMessage_<BOUNDED, FIXED_LENGTH> &message,
    std::vector<std::string> &dump) {
  for (size_t i = 0; i < message.size(); ++i) {
    dumpMessageContent(message[i], dump);
  }
}

template <typename T>
void dumpValue(const T &val, std::vector<std::string> &dump) {
  dump.emplace_back(std::to_string(val));
}

void dumpValue(const std::string &val, std::vector<std::string> &dump);
void dumpValue(const std::wstring &val, std::vector<std::string> &dump);
void dumpValue(const uint8_t &val, std::vector<std::string> &dump);
void dumpValue(const char16_t &val, std::vector<std::string> &dump);
void dumpValue(const char *val, std::vector<std::string> &dump);

template <typename T>
void dumpMessageValue(const ros_babel_fish::Message &val,
                      std::vector<std::string> &dump) {
  dumpValue(val.value<T>(), dump);
}

template <typename T>
void dumpArray(const T &message, std::vector<std::string> &dump) {
  for (size_t i = 0; i < message.size(); ++i) {
    dumpValue(message[i], dump);
  }
}

void dumpPose(const ros_babel_fish::CompoundMessage &pose,
              std::vector<std::string> &dump, bool stamped = false);
void dumpTransform(const ros_babel_fish::CompoundMessage &tf,
                   std::vector<std::string> &dump, bool stamped = false);
void dumpJoints(const ros_babel_fish::CompoundMessage &joints,
                std::vector<std::string> &dump);

struct DatasetPreparation {
  size_t nextImageIndex = 0;
  bool created = false;
};

struct AxisRange {
  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::lowest();

  double span() const { return max - min; }
};

struct DatasetStatistics {
  size_t samples = 0;
  size_t poseRows = 0;
  size_t validPoseRows = 0;
  std::string armTopicType;
  bool poseMetricsAvailable = false;
  bool jointStateDataset = false;
  std::array<AxisRange, 3> translation;
  double rotationSpreadRad = 0.0;
  double translationCoverageScore = 0.0;
  double rotationCoverageScore = 0.0;
};

bool askYesNo(const rclcpp::Logger &logger, const std::string &question);
std::optional<size_t> getResumeImageIndex(const fs::path &datasetPath,
                                          std::string &error);
DatasetPreparation prepareDatasetDirectory(const fs::path &datasetPath,
                                           const rclcpp::Logger &logger);
bool isPoseLikeArmTopic(const std::string &armTopicType);
bool isJointStateArmTopic(const std::string &armTopicType);
std::optional<DatasetStatistics>
calculateDatasetStatistics(const fs::path &datasetPath,
                           const std::string &armTopicType);
void logDatasetStatistics(const DatasetStatistics &stats,
                          const rclcpp::Logger &logger);

} // namespace collect_helpers
