#pragma once

// TODO: add docstrings to everywhere

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <csv.hpp>
#include <filesystem>
#include <future>
#include <rclcpp/rclcpp.hpp>
#include <regex>
#include <ros_babel_fish/babel_fish.hpp>
#include <ros_babel_fish/macros.hpp>

using csv::CSVFormat;
using csv::CSVReader;
namespace fs = std::filesystem;

constexpr const char *CSV_FILENAME = "poses.csv";
constexpr const char *IMG_FOLDERNAME = "images";
constexpr const char *DUMPER_LOGGERNAME = "dumper";
constexpr const char *HELPER_LOGGERNAME = "helper";

using SerializedMsg = std::shared_ptr<const rclcpp::SerializedMessage>;

inline SerializedMsg
catch_msg(std::string ros_topic_name, std::string rosTopicType,
          rclcpp::GenericSubscription::SharedPtr &subscription,
          rclcpp::Node::SharedPtr node) {
  SerializedMsg caughtMsg;
  auto promise = std::make_shared<std::promise<SerializedMsg>>();
  auto future = promise->get_future();
  subscription = rclcpp::create_generic_subscription(
      node->get_node_topics_interface(), ros_topic_name, rosTopicType, 1,
      [promise, node](SerializedMsg msg) {
        promise->set_value(msg);
        RCLCPP_DEBUG(node->get_logger(), "Got msg!");
      });

  rclcpp::spin_until_future_complete(node, future);

  return caughtMsg;
}

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

// Generic template for everything else
template <typename T>
void dumpValue(const T &val, std::vector<std::string> &dump) {
  dump.emplace_back(std::to_string(val));
}

inline void dumpValue(const std::string &val, std::vector<std::string> &dump) {
  dump.emplace_back(val);
}

inline void dumpValue(const std::wstring &val, std::vector<std::string> &dump) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  dump.emplace_back(converter.to_bytes(val));
}

inline void dumpValue(const uint8_t &val, std::vector<std::string> &dump) {
  dump.emplace_back(std::to_string(static_cast<int>(val)));
}

inline void dumpValue(const char16_t &val, std::vector<std::string> &dump) {
  dump.emplace_back(std::to_string(static_cast<int>(val)));
}

inline void dumpValue(const char *val, std::vector<std::string> &dump) {
  dump.emplace_back(val);
}

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

inline void dumpPose(const ros_babel_fish::CompoundMessage &pose,
                     std::vector<std::string> &dump, bool stamped = false) {
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

inline void dumpTransform(const ros_babel_fish::CompoundMessage &tf,
                          std::vector<std::string> &dump,
                          bool stamped = false) {
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

inline void dumpJoints(const ros_babel_fish::CompoundMessage &joints,
                       std::vector<std::string> &dump) {
  auto jointsMsg = joints.message<sensor_msgs::msg::JointState>();
  for (const double &p : jointsMsg->position) {
    dump.emplace_back(std::to_string(p));
  }
}

inline void dumpMessageContent(const ros_babel_fish::Message &message,
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
      dump.emplace_back(std::to_string(static_cast<unsigned int>(
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

// Count poses and returns that count
inline std::optional<size_t> countPoses(const fs::path &outputPath) {
  CSVFormat format;
  format.delimiter('\t').no_header().quote(false);
  const std::string &full_path =
      (outputPath / CSV_FILENAME).relative_path().string();
  try {
    CSVReader reader(full_path, format);
    size_t count = 0;
    // Reading rows
    for (auto &row : reader) {
      if (row.size() > 0 && !row[0].is_null()) {
        count++;
      }
    }
    return count;

  } catch (std::exception &ex) {
    RCLCPP_DEBUG(rclcpp::get_logger(HELPER_LOGGERNAME), "Error: %s", ex.what());
    return std::nullopt;
  }
}

inline std::optional<size_t> countImages(const fs::path &datasetPath) {
  size_t count = 0;
  std::regex pattern(R"(^(\d+)\.png$)");
  try {
    for (const auto &entry :
         fs::directory_iterator(datasetPath / IMG_FOLDERNAME)) {
      if (entry.is_regular_file()) {
        const std::string &filename = entry.path().filename().string();
        if (std::regex_match(filename, pattern)) {
          ++count;
        }
      }
    }

    return count;
  } catch (std::exception &ex) {
    RCLCPP_DEBUG(rclcpp::get_logger(HELPER_LOGGERNAME), "Error: %s", ex.what());
    return std::nullopt;
  }
}