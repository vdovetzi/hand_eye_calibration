#include "hand_eye_calibration/helpers/ros_helpers.hpp"

namespace ros_helpers {

std::optional<std::vector<std::string>>
waitForTopic(const rclcpp::Node::SharedPtr &node,
             const std::string &topicName, std::chrono::milliseconds timeout,
             std::chrono::milliseconds pollInterval) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;

  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
    const auto topics = node->get_topic_names_and_types();
    for (const auto &[name, types] : topics) {
      RCLCPP_DEBUG(node->get_logger(), "Topic: %s", name.c_str());
      if (name == topicName) {
        return types;
      }
    }

    rclcpp::sleep_for(pollInterval);
  }

  return std::nullopt;
}

} // namespace ros_helpers
