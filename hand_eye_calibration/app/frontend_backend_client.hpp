#pragma once

#include "app/frontend_state.hpp"

#include <QObject>
#include <QString>

#include <rclcpp/rclcpp.hpp>

#include <functional>
#include <string>
#include <vector>

namespace hand_eye::app {

struct TopicDescriptor {
  QString name;
  QString type;
};

struct TopicGraph {
  std::vector<TopicDescriptor> armTopics;
  std::vector<TopicDescriptor> imageTopics;
};

struct ParameterUpdateResult {
  bool success = false;
  QString message;
};

struct TriggerResult {
  QString service;
  bool success = false;
  QString message;
};

class FrontendBackendClient final {
public:
  using TriggerCallback = std::function<void(TriggerResult)>;

  FrontendBackendClient(rclcpp::Node::SharedPtr node, QObject *callbackContext,
                        std::string backendNode = "/hand_eye_backend");

  [[nodiscard]] TopicGraph discoverTopics() const;
  [[nodiscard]] ParameterUpdateResult
  pushParameters(const FrontendSettings &settings,
                 bool overwriteDataset = false,
                 bool resumeDataset = false) const;
  void callTrigger(const QString &service, TriggerCallback callback) const;

private:
  void dispatch(std::function<void()> callback) const;

  rclcpp::Node::SharedPtr node_;
  QObject *callbackContext_;
  std::string backendNode_;
};

} // namespace hand_eye::app
