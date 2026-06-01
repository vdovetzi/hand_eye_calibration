#pragma once

#include <chrono>
#include <rclcpp/rclcpp.hpp>

namespace ros_helpers {

std::optional<std::vector<std::string>>
waitForTopic(const rclcpp::Node::SharedPtr &node, const std::string &topicName,
             std::chrono::milliseconds timeout = std::chrono::seconds(5),
             std::chrono::milliseconds pollInterval =
                 std::chrono::milliseconds(100));

} // namespace ros_helpers
