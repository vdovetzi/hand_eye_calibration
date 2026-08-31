#include "app/frontend_backend_client.hpp"

#include <QMetaObject>

#include <rclcpp/parameter_client.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <algorithm>
#include <chrono>
#include <future>
#include <map>
#include <utility>

namespace hand_eye::app {
namespace {

using namespace std::chrono_literals;
using Trigger = std_srvs::srv::Trigger;

constexpr const char *kImageType = "sensor_msgs/msg/Image";
constexpr const char *kCompressedImageType = "sensor_msgs/msg/CompressedImage";

bool isSupportedArmType(const std::string &type) {
  return type == "geometry_msgs/msg/Pose" ||
         type == "geometry_msgs/msg/PoseStamped" ||
         type == "geometry_msgs/msg/Transform" ||
         type == "geometry_msgs/msg/TransformStamped";
}

bool hasType(const std::vector<std::string> &types,
             const std::string &expected) {
  return std::ranges::find(types, expected) != types.end();
}

QString baseImageTopic(const QString &topic) {
  constexpr auto suffix = "/compressed";
  constexpr int suffixLength = sizeof("/compressed") - 1;
  return topic.endsWith(suffix) ? topic.left(topic.size() - suffixLength)
                                : topic;
}

std::string serviceName(const std::string &backendNode,
                        const std::string &service) {
  return backendNode + "/" + service;
}

std::vector<rclcpp::Parameter> makeParameters(const FrontendSettings &settings,
                                              const bool overwriteDataset,
                                              const bool resumeDataset) {
  return {
      rclcpp::Parameter("arm_topic", settings.armTopic.toStdString()),
      rclcpp::Parameter("image_topic", settings.imageTopic.toStdString()),
      rclcpp::Parameter("dataset_path", settings.datasetPath.toStdString()),
      rclcpp::Parameter("calibration_path",
                        settings.calibrationPath.toStdString()),
      rclcpp::Parameter(
          "pattern_info",
          FrontendState::serializePattern(settings.pattern).toStdString()),
      rclcpp::Parameter("poses_format", settings.posesFormat),
      rclcpp::Parameter("eye_to_hand", settings.eyeToHand),
      rclcpp::Parameter("intrinsics_path",
                        settings.intrinsicsPath.toStdString()),
      rclcpp::Parameter("require_pattern_detection", true),
      rclcpp::Parameter("overwrite_dataset", overwriteDataset),
      rclcpp::Parameter("resume_dataset", resumeDataset),
      rclcpp::Parameter("stat_translation_spread",
                        settings.translationStatistics),
      rclcpp::Parameter("stat_rotation_spread", settings.rotationStatistics),
      rclcpp::Parameter("stat_validation_rms",
                        settings.validationRmsStatistics),
  };
}

} // namespace

FrontendBackendClient::FrontendBackendClient(rclcpp::Node::SharedPtr node,
                                             QObject *callbackContext,
                                             std::string backendNode)
    : node_(std::move(node)), callbackContext_(callbackContext),
      backendNode_(std::move(backendNode)) {}

TopicGraph FrontendBackendClient::discoverTopics() const {
  TopicGraph graph;
  std::map<QString, QString> imageTopics;

  for (const auto &[name, types] : node_->get_topic_names_and_types()) {
    const QString topic = QString::fromStdString(name);
    if (hasType(types, kCompressedImageType)) {
      imageTopics[baseImageTopic(topic)] =
          QString::fromStdString(kCompressedImageType);
    } else if (hasType(types, kImageType) && !imageTopics.contains(topic)) {
      imageTopics[topic] = QString::fromStdString(kImageType);
    }

    const auto armType =
        std::ranges::find_if(types, [](const std::string &type) {
          return isSupportedArmType(type);
        });
    if (armType != types.end()) {
      graph.armTopics.push_back(
          TopicDescriptor{topic, QString::fromStdString(*armType)});
    }
  }

  graph.imageTopics.reserve(imageTopics.size());
  for (const auto &[name, type] : imageTopics) {
    graph.imageTopics.push_back(TopicDescriptor{name, type});
  }
  return graph;
}

ParameterUpdateResult
FrontendBackendClient::pushParameters(const FrontendSettings &settings,
                                      const bool overwriteDataset,
                                      const bool resumeDataset) const {
  auto client =
      std::make_shared<rclcpp::AsyncParametersClient>(node_, backendNode_);
  if (!client->wait_for_service(1s)) {
    return {false, "Parameter service is not available for " +
                       QString::fromStdString(backendNode_)};
  }

  auto future = client->set_parameters(
      makeParameters(settings, overwriteDataset, resumeDataset));
  if (future.wait_for(2s) != std::future_status::ready) {
    return {false, "Timed out while setting backend parameters"};
  }

  try {
    const auto results = future.get();
    for (const auto &result : results) {
      if (!result.successful) {
        return {false, "Backend rejected parameters: " +
                           QString::fromStdString(result.reason)};
      }
    }
  } catch (const std::exception &error) {
    return {false, "Could not set backend parameters: " +
                       QString::fromUtf8(error.what())};
  }
  return {true, "Backend parameters updated"};
}

void FrontendBackendClient::callTrigger(const QString &service,
                                        TriggerCallback callback) const {
  const std::string serviceText = service.toStdString();
  const std::string fullName = serviceName(backendNode_, serviceText);
  auto client = node_->create_client<Trigger>(fullName);
  if (!client->wait_for_service(500ms)) {
    dispatch([callback = std::move(callback), service, fullName]() mutable {
      callback(TriggerResult{service, false,
                             "Service is not available: " +
                                 QString::fromStdString(fullName)});
    });
    return;
  }

  auto request = std::make_shared<Trigger::Request>();
  client->async_send_request(
      request, [this, client, service,
                callback](rclcpp::Client<Trigger>::SharedFuture future) {
        TriggerResult result;
        result.service = service;
        try {
          const auto response = future.get();
          result.success = response->success;
          result.message = QString::fromStdString(response->message);
        } catch (const std::exception &error) {
          result.success = false;
          result.message =
              "Service call failed: " + QString::fromUtf8(error.what());
        }
        dispatch([callback, result = std::move(result)]() mutable {
          callback(std::move(result));
        });
      });
}

void FrontendBackendClient::dispatch(std::function<void()> callback) const {
  QMetaObject::invokeMethod(callbackContext_, std::move(callback),
                            Qt::QueuedConnection);
}

} // namespace hand_eye::app
