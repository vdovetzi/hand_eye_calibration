#include "app/frontend_window.hpp"

#include <QApplication>
#include <QTimer>

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <thread>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  QApplication application(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("hand_eye_frontend");
  node->declare_parameter<bool>("open_live_3d", false);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinThread([&executor]() { executor.spin(); });

  auto window = std::make_unique<hand_eye::app::FrontendWindow>(node);
  window->show();
  QTimer rosShutdownMonitor;
  QObject::connect(
      &rosShutdownMonitor, &QTimer::timeout, &application,
      [&application, node]() {
        if (!rclcpp::ok(node->get_node_base_interface()->get_context())) {
          application.quit();
        }
      });
  rosShutdownMonitor.start(100);
  const int result = application.exec();

  executor.cancel();
  spinThread.join();
  window.reset();
  rclcpp::shutdown();
  return result;
}
