#include "app/backend_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hand_eye::app::BackendNode>());
  rclcpp::shutdown();
  return 0;
}
