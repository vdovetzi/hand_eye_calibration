#pragma once

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>

class QLabel;

namespace hand_eye::app {

class CameraFeed final {
public:
  CameraFeed(rclcpp::Node::SharedPtr node, std::string topic, std::string type,
             QLabel *label, const std::string &patternInfo);
  ~CameraFeed();

  CameraFeed(const CameraFeed &) = delete;
  CameraFeed &operator=(const CameraFeed &) = delete;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace hand_eye::app
