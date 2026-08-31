#pragma once

#include <rclcpp/rclcpp.hpp>

#include <memory>

namespace hand_eye::app {

class BackendNode final : public rclcpp::Node {
public:
  BackendNode();
  ~BackendNode() override;

  BackendNode(const BackendNode &) = delete;
  BackendNode &operator=(const BackendNode &) = delete;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace hand_eye::app
